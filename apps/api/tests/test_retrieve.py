"""
tests/test_retrieve.py
======================
Test dell'endpoint `POST /retrieve` (M2 task #10a).

Strategia: tutti i test qui sono **unit** sullo strato HTTP. La
pipeline RAG (embedder + vector store + retriever) è già coperta da
integration test in `tests/test_rag/test_retriever.py` (che girano
contro Qdrant reale + OpenAI mockato). Qui ci interessa SOLO:

- Validazione Pydantic della request (422 corretti).
- Wiring della dependency injection (collection default, parametri
  passati correttamente al retriever).
- Mapping degli errori upstream (UnexpectedResponse → HTTP status
  appropriato).
- Forma della response (Match.id, score, payload preservati).

Il retriever è sostituito da `FakeRetriever` via
`app.dependency_overrides`. Vedi `tests/conftest.py` per il fake.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.rag.vector_store import Match


def test_retrieve_happy_path(client: TestClient, fake_retriever) -> None:
    """Query valida + fake con 2 Match → response 200 con 2 chunks."""
    fake_retriever.matches_to_return = [
        Match(id="chunk-a", score=0.95, payload={"text": "first", "source": "a.md"}),
        Match(id="chunk-b", score=0.82, payload={"text": "second", "source": "b.md"}),
    ]

    response = client.post(
        "/retrieve",
        json={"query": "how do I do X", "collection": "demo", "top_k": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert "chunks" in body
    assert len(body["chunks"]) == 2
    assert body["chunks"][0]["id"] == "chunk-a"
    assert body["chunks"][0]["score"] == 0.95
    assert body["chunks"][0]["payload"] == {"text": "first", "source": "a.md"}
    assert body["chunks"][1]["id"] == "chunk-b"


# ----------------------------------------------------------------------------
# Pass-through dei parametri: l'endpoint deve girare correttamente i valori
# (collection, filter, top_k) al retriever sottostante. Test "white-box leggeri":
# ispezioniamo `fake_retriever.calls` per verificare cosa è stato invocato.
# ----------------------------------------------------------------------------


def test_retrieve_uses_default_collection(client: TestClient, fake_retriever) -> None:
    """Senza 'collection' nel body → fake riceve settings.qdrant_collection_name.

    Verifica che l'endpoint NON spedisca un None al retriever, ma applichi
    il default operativo del backend (definito in app/config.py).
    """
    fake_retriever.matches_to_return = []
    response = client.post("/retrieve", json={"query": "x"})
    assert response.status_code == 200
    assert len(fake_retriever.calls) == 1
    assert fake_retriever.calls[0]["collection"] == settings.qdrant_collection_name


def test_retrieve_uses_provided_collection(client: TestClient, fake_retriever) -> None:
    """'collection': 'custom' → fake riceve 'custom' (override del default)."""
    fake_retriever.matches_to_return = []
    response = client.post("/retrieve", json={"query": "x", "collection": "custom"})
    assert response.status_code == 200
    assert fake_retriever.calls[0]["collection"] == "custom"


def test_retrieve_passes_filter(client: TestClient, fake_retriever) -> None:
    """'filter': {...} → fake riceve il dict identico.

    Il filtro viene passato al retriever come-è (shallow equality match
    sul payload Qdrant). L'endpoint non lo interpreta né lo trasforma.
    """
    fake_retriever.matches_to_return = []
    response = client.post(
        "/retrieve",
        json={"query": "x", "filter": {"source": "intro.md"}},
    )
    assert response.status_code == 200
    assert fake_retriever.calls[0]["filter"] == {"source": "intro.md"}


def test_retrieve_default_top_k_is_5(client: TestClient, fake_retriever) -> None:
    """Senza 'top_k' → fake riceve 5 (default dichiarato nello schema)."""
    fake_retriever.matches_to_return = []
    response = client.post("/retrieve", json={"query": "x"})
    assert response.status_code == 200
    assert fake_retriever.calls[0]["top_k"] == 5


# ----------------------------------------------------------------------------
# Validation 422: input fuori dai vincoli Pydantic.
# ----------------------------------------------------------------------------
# Pydantic rifiuta la request PRIMA che l'handler venga eseguito; FastAPI
# traduce automaticamente il ValidationError in HTTP 422 Unprocessable
# Entity. Verifichiamo anche che il retriever NON sia stato chiamato:
# è la conferma che la barriera è effettivamente "before-handler".


def test_retrieve_empty_query_422(client: TestClient, fake_retriever) -> None:
    """query='' → 422 (Pydantic min_length=1). Il fake NON viene chiamato."""
    response = client.post("/retrieve", json={"query": ""})
    assert response.status_code == 422
    assert fake_retriever.calls == []


def test_retrieve_top_k_zero_422(client: TestClient, fake_retriever) -> None:
    """top_k=0 → 422 (Pydantic ge=1). Il fake NON viene chiamato."""
    response = client.post("/retrieve", json={"query": "x", "top_k": 0})
    assert response.status_code == 422
    assert fake_retriever.calls == []


def test_retrieve_top_k_too_large_422(client: TestClient, fake_retriever) -> None:
    """top_k=1000 → 422 (Pydantic le=50). Il fake NON viene chiamato.

    Il tetto a 50 è una guardia di costo/latenza: un top_k enorme
    significherebbe embedding+search inutilmente cari. Se il caso d'uso
    cambierà, il vincolo si alza nello schema (single source of truth).
    """
    response = client.post("/retrieve", json={"query": "x", "top_k": 1000})
    assert response.status_code == 422
    assert fake_retriever.calls == []


# ----------------------------------------------------------------------------
# Upstream error mapping: come l'endpoint reagisce quando Qdrant fallisce.
# ----------------------------------------------------------------------------
# Il vector store è una dipendenza esterna; qualunque errore HTTP da Qdrant
# arriva al nostro codice come `qdrant_client.http.exceptions.UnexpectedResponse`.
# L'endpoint distingue:
#   - 404 da Qdrant → 404 al client, con nome della collection nel detail
#     (errore del client: ha chiesto una collection inesistente).
#   - tutto il resto → 502 Bad Gateway (errore di un servizio a valle,
#     non colpa del client e non nostra).


def _make_unexpected_response(status_code: int) -> UnexpectedResponse:
    """Costruisce un UnexpectedResponse minimo per i test.

    Il costruttore richiede (status_code, reason_phrase, content, headers).
    Lo wrappiamo in un helper perché serve a 2+ test e mette via un po' di
    rumore.
    """
    return UnexpectedResponse(
        status_code=status_code,
        reason_phrase="Mock Error",
        content=b"mock body",
        headers={},
    )


def test_retrieve_collection_not_found_404(client: TestClient, fake_retriever) -> None:
    """Qdrant ritorna 404 (collection inesistente) → endpoint risponde 404.

    Il detail include il nome della collection per aiutare il debug
    lato chiamante ("ho sbagliato il nome?").
    """
    fake_retriever.error_to_raise = _make_unexpected_response(404)

    response = client.post(
        "/retrieve",
        json={"query": "x", "collection": "does-not-exist"},
    )
    assert response.status_code == 404
    body = response.json()
    assert "does-not-exist" in body["detail"]


def test_retrieve_qdrant_500_returns_502(client: TestClient, fake_retriever) -> None:
    """Qdrant ritorna 500 (errore generico upstream) → endpoint risponde 502.

    502 Bad Gateway è la semantica HTTP corretta: "il server upstream da
    cui dipendo ha fallito". Non 500, perché 500 implicherebbe un bug
    nel NOSTRO codice.
    """
    fake_retriever.error_to_raise = _make_unexpected_response(500)

    response = client.post("/retrieve", json={"query": "x"})
    assert response.status_code == 502
    assert "vector store error" in response.json()["detail"]
