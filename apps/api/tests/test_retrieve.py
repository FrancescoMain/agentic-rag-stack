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
