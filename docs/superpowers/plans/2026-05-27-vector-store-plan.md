# Vector Store layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementare `app/rag/vector_store.py` — Protocol `VectorStore` + implementazione concreta `QdrantVectorStore` — secondo il design [docs/superpowers/specs/2026-05-27-vector-store-design.md](../specs/2026-05-27-vector-store-design.md).

**Architecture:** Singolo modulo Python con `typing.Protocol` come ABC, DTO Pydantic (`VectorPoint`, `Match`), e classe concreta che incapsula `AsyncQdrantClient`. Test integration su Qdrant reale (container già attivo via `docker compose up -d qdrant`), no mock.

**Tech Stack:** Python 3.12, qdrant-client ≥1.11, Pydantic v2, pytest + pytest-asyncio (modalità "auto"), Qdrant v1.11.0 in docker.

---

## File structure

| File | Action | Responsability |
|---|---|---|
| `apps/api/app/rag/vector_store.py` | **Create** | DTO + Protocol + QdrantVectorStore + get_vector_store singleton |
| `apps/api/app/config.py` | **Modify** | Aggiungi `qdrant_url`, `qdrant_api_key`, `qdrant_collection_name` |
| `apps/api/pyproject.toml` | **Modify** | Aggiungi dep `qdrant-client>=1.11,<2` + pytest marker `integration` |
| `apps/api/tests/test_rag/conftest.py` | **Modify** | Aggiungi fixture `qdrant_store` (session) e `unique_collection` (function) |
| `apps/api/tests/test_rag/test_vector_store.py` | **Create** | Test unit (mapping helpers) + integration (Qdrant live) |
| `docs/ROADMAP.md` | **Modify** | M2 task #5 da ⚪ a ✅, status block aggiornato |

Pattern del repo da rispettare:
- `from __future__ import annotations` in cima a ogni `.py`
- Type hints moderni (`list[T]`, `dict[K,V]`, `X | None`)
- Docstring estese in italiano didattico (vedi `app/rag/embedder.py`)
- Logger modulo-livello: `logger = logging.getLogger(__name__)`
- `asyncio_mode = "auto"` → niente `@pytest.mark.asyncio`

---

## Pre-flight check (NOT a task — just verify before starting)

```powershell
# Working tree pulito
git status

# Qdrant up & healthy
docker ps --filter "name=agentic-rag-qdrant" --format "{{.Status}}"
# expected: Up X minutes (healthy)

# Branch corrente
git branch --show-current  # expected: main
```

Se Qdrant non gira: `docker compose up -d qdrant` dalla root del repo.

---

## Task 1: Dependencies + config + pytest marker

**Files:**
- Modify: `apps/api/pyproject.toml`
- Modify: `apps/api/app/config.py`

- [ ] **Step 1.1: Aggiungi `qdrant-client` come dependency**

In [apps/api/pyproject.toml](../../../apps/api/pyproject.toml), dentro `[project] dependencies = [ ... ]`, aggiungi prima della chiusura `]`:

```toml
  # qdrant-client: client ufficiale per Qdrant (vector DB scelto in ADR-0003).
  # Versione 1.11.x allinea l'API client con l'immagine docker qdrant/qdrant:v1.11.0.
  # Forniamo sia QdrantClient sync sia AsyncQdrantClient — usiamo l'async per
  # coerenza con FastAPI/embedder. Il pacchetto include i tipi: VectorParams,
  # Distance, PointStruct, Filter, FieldCondition, MatchValue, ScoredPoint.
  "qdrant-client>=1.11,<2",
```

- [ ] **Step 1.2: Aggiungi marker `integration` a pytest**

In [apps/api/pyproject.toml](../../../apps/api/pyproject.toml), dentro `[tool.pytest.ini_options]`, sotto la riga `asyncio_mode = "auto"`, aggiungi:

```toml
# Marker custom per test che richiedono servizi esterni (Qdrant, ecc.).
# Skippati di default se il servizio non è raggiungibile (vedi conftest fixture).
# Per girare solo gli integration: `uv run pytest -m integration`
# Per girare TUTTO tranne gli integration: `uv run pytest -m "not integration"`
markers = [
  "integration: test che richiede un servizio esterno live (es. Qdrant)",
]
```

- [ ] **Step 1.3: Aggiungi i 3 campi Qdrant a `Settings`**

In [apps/api/app/config.py](../../../apps/api/app/config.py), subito dopo il blocco `openai_api_key: str = Field(default="")` (prima del commento `# SettingsConfigDict configura...`), aggiungi:

```python
    # ------------------------------------------------------------------------
    # Vector database (M2 — vedi ADR-0003 Qdrant)
    # ------------------------------------------------------------------------
    # URL del Qdrant locale lanciato via `docker compose up -d qdrant`.
    # In prod cambierà a Qdrant Cloud o a un'istanza self-hosted in cluster.
    qdrant_url: str = Field(default="http://localhost:6333")

    # API key per autenticare le query verso Qdrant. In dev locale Qdrant
    # gira senza auth (vedi docker-compose.yml), quindi può restare vuota.
    # La imposteremo a M5 / prod.
    qdrant_api_key: str = Field(default="")

    # Nome di default della collezione usata dall'ingest CLI (M2 task #6) e
    # dal retriever (task #7). Il vector store stesso NON usa questo campo:
    # accetta `collection: str` per chiamata. Lo mettiamo qui per coerenza
    # con .env / .env.example, dove la chiave QDRANT_COLLECTION_NAME esiste.
    qdrant_collection_name: str = Field(default="agentic-rag-demo")
```

- [ ] **Step 1.4: Installa la nuova dipendenza**

Run:
```powershell
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack\apps\api
uv sync
```

Expected: output che termina con `Installed N packages` (di cui `qdrant-client`, `grpcio`, `httpx`, ecc.).

- [ ] **Step 1.5: Smoke test: import qdrant-client e settings**

Run:
```powershell
uv run python -c "from qdrant_client import AsyncQdrantClient; from app.config import settings; print('OK', settings.qdrant_url)"
```

Expected output:
```
OK http://localhost:6333
```

- [ ] **Step 1.6: Verifica suite esistente ancora verde**

Run:
```powershell
uv run pytest -q
```

Expected: tutti i test esistenti (health, classify, chunker, embedder) passano. Nessun nuovo test ancora.

- [ ] **Step 1.7: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/pyproject.toml apps/api/app/config.py apps/api/uv.lock
git commit -m "feat(rag): add qdrant-client dep + Qdrant config fields (M2 task #5 prep)"
```

---

## Task 2: DTOs (VectorPoint, Match) + Protocol VectorStore

**Files:**
- Create: `apps/api/app/rag/vector_store.py`
- Create: `apps/api/tests/test_rag/test_vector_store.py`

- [ ] **Step 2.1: Crea `vector_store.py` con DTO + Protocol (no implementation yet)**

Crea il file con il seguente contenuto:

```python
"""
app/rag/vector_store.py
=======================
Astrazione di accesso al vector database.

Espone:
- `VectorPoint` (Pydantic): input di `upsert`. id stringa stabile +
  vettore float + payload arbitrario (testo + metadata).
- `Match` (Pydantic): output di `search`. id + score + payload.
- `VectorStore` (typing.Protocol): l'interfaccia astratta — 4 metodi async.
- `QdrantVectorStore`: implementazione concreta su `AsyncQdrantClient`.
- `get_vector_store()`: factory singleton modulo-livello.

------------------------------------------------------------------------
Perché un Protocol e non una `abc.ABC`?
------------------------------------------------------------------------

Python 3.8+ ha `typing.Protocol` (PEP 544): definisce "duck typing
strutturale" — una classe soddisfa il Protocol se ha i metodi giusti,
SENZA bisogno di ereditare esplicitamente. Vantaggi rispetto ad ABC:

- Niente accoppiamento di ereditarietà: domani posso fare un
  `InMemoryVectorStore` per test senza modificare nessun import.
- Più Pythonic 2026: lo usano LangChain, LlamaIndex, fastapi.
- Più test-friendly: nei test posso passare un duck-typed object.

Per i lettori Java/C#: pensa a "interface implicita".

------------------------------------------------------------------------
Mapping ID interno (dettaglio implementativo)
------------------------------------------------------------------------

Qdrant accetta come ID di un punto SOLO uint64 o UUID. Il chunker
produce `Chunk.id` come SHA-256 hex (64 char), non valido. Soluzione:
internamente `QdrantVectorStore` mappa qualsiasi stringa stabile a un
UUID v5 deterministico (namespace fisso), e preserva l'id originale
nel payload sotto chiave `__vp_id`. Il `Match.id` ritorna sempre
l'id originale, mai l'UUID Qdrant — il consumer non si accorge.

Vedi ADR-0003 per la motivazione di Qdrant e ADR-0004 per gli
embedding (text-embedding-3-small, 1536-dim).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DTO
# ---------------------------------------------------------------------------


class VectorPoint(BaseModel):
    """Punto vettoriale da upsertare.

    Attributes:
        id: identità STABILE (idempotenza). Una stringa qualunque; il
            QdrantVectorStore la normalizza internamente a UUID v5 e
            preserva l'originale nel payload. Il chunker produce sha256.
        vector: embedding, lista di float. La dimensione DEVE coincidere
            con il `vector_size` con cui è stata creata la collection
            (es. 1536 per text-embedding-3-small). Mismatch → errore
            Qdrant a upsert.
        payload: metadata + dati. Convenzione (non enforced):
            `{text, source, heading, position, token_count}`. La chiave
            `__vp_id` è RISERVATA all'uso interno — non usarla.
    """

    id: str = Field(min_length=1)
    vector: list[float] = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class Match(BaseModel):
    """Risultato di una ricerca.

    Attributes:
        id: id ORIGINALE (quello passato all'upsert come `VectorPoint.id`),
            non l'UUID Qdrant interno. La chiave `__vp_id` è già stata
            rimossa dal payload.
        score: similarity score. Per Cosine distance, range tipico
            [0, 1] su testi normalizzati; range teorico [-1, 1].
            Maggiore = più simile.
        payload: payload originale dell'upsert (depurato della chiave
            riservata interna).
    """

    id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class VectorStore(Protocol):
    """Interfaccia astratta per un vector database.

    Ogni metodo è async per coerenza con il resto dello stack (FastAPI,
    embedder) e per non bloccare l'event loop sulle chiamate di rete.

    Convenzione: gli errori "di configurazione" (vector_size mismatch su
    ensure_collection) sono `ValueError`. Gli errori di rete o del servizio
    propagano come eccezioni del client sottostante (es. `qdrant_client`
    alza `UnexpectedResponse` per gli HTTP error).
    """

    async def ensure_collection(
        self,
        name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        """Crea la collection se non esiste; no-op se esiste con stessa config.

        Args:
            name: nome della collection.
            vector_size: dimensione del vettore (es. 1536).
            distance: "Cosine" | "Dot" | "Euclid". Default "Cosine".

        Raises:
            ValueError: se la collection esiste ma con vector_size diverso.
        """
        ...

    async def upsert(
        self,
        collection: str,
        points: Sequence[VectorPoint],
    ) -> int:
        """Inserisce o aggiorna i punti. Idempotente per id.

        Args:
            collection: nome collection.
            points: sequenza di VectorPoint. Se >100, internamente
                batchato in chiamate di 100 per chiamata.

        Returns:
            Numero di punti inviati (= len(points)).
        """
        ...

    async def search(
        self,
        collection: str,
        query: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Match]:
        """Ricerca i top-k punti più simili a `query`.

        Args:
            collection: nome collection.
            query: vettore di query (stessa dimensione della collection).
            top_k: numero massimo di risultati. Default 5.
            filter: shallow equality match sul payload, es.
                `{"source": "intro.md"}` → solo punti con
                `payload["source"] == "intro.md"`. None → nessun filtro.
                Multiple chiavi → AND logico.

        Returns:
            Lista di Match ordinata per score decrescente.
            Lista vuota se collection vuota o nessun match.
        """
        ...

    async def delete_collection(self, name: str) -> None:
        """Elimina la collection. No-op se non esiste."""
        ...


# ---------------------------------------------------------------------------
# Implementazione concreta: Qdrant
# ---------------------------------------------------------------------------
# (Implementata nelle task successive.)


# ---------------------------------------------------------------------------
# Factory singleton
# ---------------------------------------------------------------------------
# (Implementato nelle task successive.)


__all__ = [
    "Match",
    "VectorPoint",
    "VectorStore",
]
```

- [ ] **Step 2.2: Crea `test_vector_store.py` con smoke test su DTO + Protocol**

Crea il file con il seguente contenuto:

```python
"""
tests/test_rag/test_vector_store.py
===================================
Test del modulo `app/rag/vector_store.py`.

Convenzioni:
- Test unit "puri" (mapping helpers, DTO): nessun marker, girano sempre.
- Test integration che richiedono Qdrant live: marker `@pytest.mark.integration`,
  skippati se Qdrant non risponde su /readyz.

Per girare solo unit:    `uv run pytest -m "not integration" tests/test_rag/test_vector_store.py`
Per girare integration:  `uv run pytest -m integration       tests/test_rag/test_vector_store.py`
"""

from __future__ import annotations

import pytest

from app.rag.vector_store import Match, VectorPoint, VectorStore


# ---------------------------------------------------------------------------
# DTO smoke tests (unit, no Qdrant)
# ---------------------------------------------------------------------------


def test_vector_point_minimal() -> None:
    """VectorPoint accetta id non vuoto + vector non vuoto, payload default {}."""
    point = VectorPoint(id="abc123", vector=[0.1, 0.2, 0.3])
    assert point.id == "abc123"
    assert point.vector == [0.1, 0.2, 0.3]
    assert point.payload == {}


def test_vector_point_with_payload() -> None:
    """VectorPoint accetta dict arbitrario come payload."""
    point = VectorPoint(
        id="x",
        vector=[1.0],
        payload={"text": "hello", "source": "intro.md", "heading": "H1"},
    )
    assert point.payload["text"] == "hello"
    assert point.payload["source"] == "intro.md"


def test_vector_point_empty_id_rejected() -> None:
    """id stringa vuota viene rifiutata da Pydantic."""
    with pytest.raises(ValueError):
        VectorPoint(id="", vector=[0.1])


def test_vector_point_empty_vector_rejected() -> None:
    """vector vuoto viene rifiutato (un punto senza embedding non ha senso)."""
    with pytest.raises(ValueError):
        VectorPoint(id="x", vector=[])


def test_match_basic() -> None:
    """Match si costruisce con id + score + payload opzionale."""
    m = Match(id="x", score=0.95, payload={"text": "hi"})
    assert m.id == "x"
    assert m.score == 0.95
    assert m.payload == {"text": "hi"}


def test_match_default_payload() -> None:
    """Match permette payload omesso → {} default."""
    m = Match(id="x", score=0.5)
    assert m.payload == {}


# ---------------------------------------------------------------------------
# Protocol smoke test (unit)
# ---------------------------------------------------------------------------


def test_protocol_is_importable() -> None:
    """Il Protocol è importabile e definisce i 4 metodi del contratto.

    Non istanziamo Protocol direttamente (è una "interfaccia"); verifichiamo
    solo che esista come simbolo e che abbia i metodi attesi.
    """
    methods = {"ensure_collection", "upsert", "search", "delete_collection"}
    assert methods.issubset(set(dir(VectorStore)))
```

- [ ] **Step 2.3: Run i test, verifica che passino**

Run:
```powershell
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack\apps\api
uv run pytest tests/test_rag/test_vector_store.py -v
```

Expected: 7 passed.

- [ ] **Step 2.4: Run ruff format + check sui file nuovi**

Run:
```powershell
uv run ruff format app/rag/vector_store.py tests/test_rag/test_vector_store.py
uv run ruff check app/rag/vector_store.py tests/test_rag/test_vector_store.py
```

Expected: `All checks passed!`.

- [ ] **Step 2.5: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/app/rag/vector_store.py apps/api/tests/test_rag/test_vector_store.py
git commit -m "feat(rag): add VectorPoint, Match DTOs + VectorStore Protocol (M2 task #5)"
```

---

## Task 3: Mapping helpers (unit, no Qdrant)

**Files:**
- Modify: `apps/api/app/rag/vector_store.py`
- Modify: `apps/api/tests/test_rag/test_vector_store.py`

Aggiungi 3 funzioni pure di mapping fra i tipi Qdrant e i nostri DTO. Sono unit-test friendly perché non toccano la rete.

- [ ] **Step 3.1: Aggiungi i test (rosso prima)**

In [apps/api/tests/test_rag/test_vector_store.py](../../../apps/api/tests/test_rag/test_vector_store.py), aggiungi alla fine del file:

```python


# ---------------------------------------------------------------------------
# Mapping helpers (unit, no Qdrant)
# ---------------------------------------------------------------------------

from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, ScoredPoint

from app.rag.vector_store import (
    _RESERVED_PAYLOAD_KEY,
    _filter_to_qdrant,
    _from_qdrant_scored_point,
    _to_qdrant_point,
)


def test_to_qdrant_point_normalizes_id_to_uuid() -> None:
    """Un id stringa arbitraria viene normalizzato a UUID v5 deterministico."""
    vp = VectorPoint(id="not-a-uuid", vector=[0.1, 0.2], payload={"text": "hi"})
    ps = _to_qdrant_point(vp)
    assert isinstance(ps, PointStruct)
    # Lo UUID prodotto è una stringa nel formato xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.
    assert isinstance(ps.id, str)
    assert len(ps.id) == 36
    assert ps.id.count("-") == 4


def test_to_qdrant_point_id_is_deterministic() -> None:
    """Stesso id originale → stesso UUID Qdrant (upsert idempotente)."""
    vp1 = VectorPoint(id="stable-id", vector=[0.1])
    vp2 = VectorPoint(id="stable-id", vector=[0.9])  # vector diverso, id uguale
    assert _to_qdrant_point(vp1).id == _to_qdrant_point(vp2).id


def test_to_qdrant_point_preserves_original_id_in_payload() -> None:
    """L'id originale viene preservato nel payload sotto chiave riservata."""
    vp = VectorPoint(id="sha256-abc", vector=[0.1], payload={"text": "hi"})
    ps = _to_qdrant_point(vp)
    assert ps.payload[_RESERVED_PAYLOAD_KEY] == "sha256-abc"
    # Il resto del payload utente è preservato:
    assert ps.payload["text"] == "hi"


def test_from_qdrant_scored_point_restores_original_id() -> None:
    """Il Match.id è l'id originale, non l'UUID Qdrant. La chiave riservata
    viene rimossa dal payload pubblico."""
    sp = ScoredPoint(
        id="00000000-0000-0000-0000-000000000001",
        version=0,
        score=0.87,
        payload={_RESERVED_PAYLOAD_KEY: "original-sha256-xyz", "text": "hi"},
    )
    m = _from_qdrant_scored_point(sp)
    assert m.id == "original-sha256-xyz"
    assert m.score == 0.87
    assert m.payload == {"text": "hi"}
    # La chiave riservata NON deve trapelare al consumer:
    assert _RESERVED_PAYLOAD_KEY not in m.payload


def test_from_qdrant_scored_point_handles_missing_reserved_key() -> None:
    """Fallback robusto: se per qualche motivo manca la chiave riservata
    (es. punto inserito direttamente bypassando il nostro upsert),
    il Match.id ricade sull'UUID Qdrant come stringa."""
    sp = ScoredPoint(
        id="00000000-0000-0000-0000-000000000002",
        version=0,
        score=0.5,
        payload={"text": "hi"},  # niente __vp_id
    )
    m = _from_qdrant_scored_point(sp)
    assert m.id == "00000000-0000-0000-0000-000000000002"
    assert m.payload == {"text": "hi"}


def test_filter_to_qdrant_none_returns_none() -> None:
    """filter=None → None (Qdrant accetta None per 'nessun filtro')."""
    assert _filter_to_qdrant(None) is None


def test_filter_to_qdrant_empty_dict_returns_none() -> None:
    """filter={} → None (nessuna condizione = nessun filtro)."""
    assert _filter_to_qdrant({}) is None


def test_filter_to_qdrant_single_key_produces_field_condition() -> None:
    """{key: value} → Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])."""
    f = _filter_to_qdrant({"source": "intro.md"})
    assert isinstance(f, Filter)
    assert f.must is not None
    assert len(f.must) == 1
    cond = f.must[0]
    assert isinstance(cond, FieldCondition)
    assert cond.key == "source"
    assert isinstance(cond.match, MatchValue)
    assert cond.match.value == "intro.md"


def test_filter_to_qdrant_multiple_keys_produces_and() -> None:
    """{k1: v1, k2: v2} → tutti come `must` (AND logico)."""
    f = _filter_to_qdrant({"source": "a.md", "heading": "H1"})
    assert isinstance(f, Filter)
    assert f.must is not None
    assert len(f.must) == 2
    keys = sorted(cond.key for cond in f.must)
    assert keys == ["heading", "source"]
```

- [ ] **Step 3.2: Run i test — devono FALLIRE (helpers non esistono ancora)**

Run:
```powershell
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack\apps\api
uv run pytest tests/test_rag/test_vector_store.py -v
```

Expected: ImportError o `cannot import name '_to_qdrant_point' from 'app.rag.vector_store'`.

- [ ] **Step 3.3: Implementa i 3 helpers + costante**

In [apps/api/app/rag/vector_store.py](../../../apps/api/app/rag/vector_store.py), sostituisci il commento `# (Implementata nelle task successive.)` (sotto "Implementazione concreta: Qdrant") con:

```python
# Namespace UUID fisso per la derivazione deterministica dell'id Qdrant
# da una stringa arbitraria (es. sha256 hex dal chunker). Usiamo il
# namespace standard "OID" da RFC 4122 § Appendix C — è un valore arbitrario
# ma stabile e riconoscibile; l'unica cosa che conta è che NON cambi mai.
_ID_NAMESPACE = uuid.UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")

# Chiave riservata nel payload Qdrant: contiene l'id originale del
# VectorPoint, in modo che `Match` possa restituirlo invece dell'UUID
# Qdrant interno. Prefisso "__" per minimizzare collisioni con metadata
# user-defined.
_RESERVED_PAYLOAD_KEY = "__vp_id"


def _normalize_id(raw: str) -> str:
    """Mappa una stringa stabile a un UUID v5 deterministico.

    Qdrant accetta come point id SOLO uint64 o UUID. Stringhe arbitrarie
    (sha256 hex, slug, ecc.) vanno tradotte. UUID v5 è perfetto: stesso
    input → stesso UUID, sempre, ovunque.
    """
    return str(uuid.uuid5(_ID_NAMESPACE, raw))


def _to_qdrant_point(point: VectorPoint) -> PointStruct:
    """VectorPoint → PointStruct di qdrant-client.

    Side-effect: arricchisce il payload con la chiave riservata
    `__vp_id` per preservare l'id originale.
    """
    qdrant_id = _normalize_id(point.id)
    payload = {**point.payload, _RESERVED_PAYLOAD_KEY: point.id}
    return PointStruct(id=qdrant_id, vector=point.vector, payload=payload)


def _from_qdrant_scored_point(sp: ScoredPoint) -> Match:
    """ScoredPoint di qdrant-client → Match nostro.

    Side-effect: rimuove la chiave riservata `__vp_id` dal payload prima
    di esporlo al consumer, e la usa come `Match.id`. Se manca, fallback
    sull'UUID Qdrant (caso "rotto" — punto inserito senza il nostro upsert).
    """
    payload = dict(sp.payload or {})
    original_id = payload.pop(_RESERVED_PAYLOAD_KEY, str(sp.id))
    return Match(id=original_id, score=sp.score, payload=payload)


def _filter_to_qdrant(filter: dict[str, Any] | None) -> Filter | None:
    """Dict shallow equality → Filter Qdrant con AND logico (clausola `must`).

    None o dict vuoto → None (Qdrant accetta None per "nessun filtro").
    Esempio: {"source": "a.md", "heading": "H1"} → Filter(must=[
        FieldCondition(key="source", match=MatchValue(value="a.md")),
        FieldCondition(key="heading", match=MatchValue(value="H1")),
    ])
    """
    if not filter:
        return None
    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in filter.items()
    ]
    return Filter(must=conditions)
```

- [ ] **Step 3.4: Aggiungi gli import qdrant_client.models in cima al file**

In [apps/api/app/rag/vector_store.py](../../../apps/api/app/rag/vector_store.py), subito dopo `from pydantic import BaseModel, Field`, aggiungi:

```python
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
)
```

- [ ] **Step 3.5: Aggiorna `__all__`**

In fondo a [apps/api/app/rag/vector_store.py](../../../apps/api/app/rag/vector_store.py), sostituisci il blocco `__all__` con:

```python
__all__ = [
    "Match",
    "VectorPoint",
    "VectorStore",
]
```

(Gli helper `_*` sono intenzionalmente non in `__all__`: sono privati al modulo. I test li importano comunque per nome — è OK per i test.)

- [ ] **Step 3.6: Run i test — devono PASSARE tutti**

Run:
```powershell
uv run pytest tests/test_rag/test_vector_store.py -v
```

Expected: 16 passed (7 dei DTO + 9 mapping).

- [ ] **Step 3.7: Run ruff format + check**

Run:
```powershell
uv run ruff format app/rag/vector_store.py tests/test_rag/test_vector_store.py
uv run ruff check app/rag/vector_store.py tests/test_rag/test_vector_store.py
```

Expected: clean.

- [ ] **Step 3.8: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/app/rag/vector_store.py apps/api/tests/test_rag/test_vector_store.py
git commit -m "feat(rag): add Qdrant <-> DTO mapping helpers with UUID id normalization"
```

---

## Task 4: Fixtures + `ensure_collection` / `delete_collection`

**Files:**
- Modify: `apps/api/app/rag/vector_store.py`
- Modify: `apps/api/tests/test_rag/conftest.py`
- Modify: `apps/api/tests/test_rag/test_vector_store.py`

Prima fixture + primi test integration. Per coerenza con il design (no mock), questi richiedono Qdrant live.

- [ ] **Step 4.1: Aggiungi fixtures al conftest**

In [apps/api/tests/test_rag/conftest.py](../../../apps/api/tests/test_rag/conftest.py), aggiungi alla fine del file (dopo la fixture `fake_openai`):

```python


# ---------------------------------------------------------------------------
# Fixture per i test integration di vector_store.py
# ---------------------------------------------------------------------------
# Pattern: session-scoped per il client (riusare la connessione),
# function-scoped per il nome collection (isolare i test fra loro).
# Se Qdrant non risponde, i test marcati @pytest.mark.integration vengono
# SKIPPATI con motivo chiaro — non falliscono.


@pytest.fixture(scope="session")
def qdrant_store() -> Any:
    """Restituisce un QdrantVectorStore configurato.

    Verifica connettività con un GET /readyz; se Qdrant è down, skippa
    tutti i test che dipendono da questa fixture.

    `session` scope: riusiamo lo stesso client per tutta la sessione
    pytest. AsyncQdrantClient mantiene un pool httpx internamente.
    """
    import httpx

    from app.config import settings
    from app.rag.vector_store import QdrantVectorStore

    try:
        r = httpx.get(f"{settings.qdrant_url}/readyz", timeout=2.0)
        if r.status_code != 200:
            pytest.skip(
                f"Qdrant non pronto su {settings.qdrant_url}/readyz "
                f"(status {r.status_code}). Avvia: docker compose up -d qdrant"
            )
    except httpx.RequestError as exc:
        pytest.skip(
            f"Qdrant non raggiungibile su {settings.qdrant_url}: {exc}. "
            f"Avvia: docker compose up -d qdrant"
        )

    return QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


@pytest.fixture
async def unique_collection(qdrant_store: Any) -> Any:
    """Yield di un nome di collection unico; cleanup automatico al teardown.

    Prefisso `_test_` per identificare a colpo d'occhio le collection
    create dai test (utili a `delete_collection` manuale di pulizia).
    """
    import uuid as _uuid

    name = f"_test_{_uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        # Best-effort cleanup. Se delete fallisce (es. collection mai
        # creata, o Qdrant nel frattempo down), non fallire il teardown.
        try:
            await qdrant_store.delete_collection(name)
        except Exception as exc:  # noqa: BLE001 — è un teardown best-effort
            logger.warning("Cleanup collection %s fallito: %s", name, exc)
```

E aggiungi in cima al conftest (se non presenti):

```python
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 4.2: Implementa `QdrantVectorStore.__init__` + `ensure_collection` + `delete_collection`**

In [apps/api/app/rag/vector_store.py](../../../apps/api/app/rag/vector_store.py), dopo il commento `# Implementazione concreta: Qdrant` (e dopo i mapping helpers), aggiungi:

```python
class QdrantVectorStore:
    """Implementazione concreta di `VectorStore` su Qdrant.

    Configurata con un URL HTTP e una API key opzionale. Mantiene
    internamente un `AsyncQdrantClient` long-lived.
    """

    def __init__(self, url: str, api_key: str | None = None) -> None:
        self._client = AsyncQdrantClient(url=url, api_key=api_key)

    async def ensure_collection(
        self,
        name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        """Crea la collection se non esiste; no-op se esiste con stessa config.

        Sui mismatch di `vector_size` alziamo ValueError invece di tentare
        un drop+recreate silenzioso (che cancellerebbe dati). Cambiare
        embedding model è una decisione esplicita: il chiamante chiama
        `delete_collection` prima.
        """
        distance_enum = self._parse_distance(distance)

        if await self._client.collection_exists(name):
            info = await self._client.get_collection(name)
            # `vectors` può essere un VectorParams (single unnamed vector,
            # nostro caso) o un dict[str, VectorParams] (multi-vector).
            # Estraiamo la size in modo robusto.
            existing_params = info.config.params.vectors
            if isinstance(existing_params, dict):
                # Per now ci aspettiamo single-vector; se è dict, prendi il
                # primo entry e confronta. Se servirà multi-vector in M2+,
                # estenderemo l'API di ensure_collection.
                existing_size = next(iter(existing_params.values())).size
            else:
                existing_size = existing_params.size
            if existing_size != vector_size:
                raise ValueError(
                    f"Collection '{name}' esiste con vector_size={existing_size}, "
                    f"richiesto={vector_size}. Drop+recreate esplicito necessario."
                )
            logger.info(
                "ensure_collection_noop",
                extra={"collection": name, "vector_size": vector_size},
            )
            return

        await self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=distance_enum),
        )
        logger.info(
            "ensure_collection_created",
            extra={"collection": name, "vector_size": vector_size, "distance": distance},
        )

    async def delete_collection(self, name: str) -> None:
        """Elimina la collection. No-op se non esiste."""
        if not await self._client.collection_exists(name):
            return
        await self._client.delete_collection(name)
        logger.info("delete_collection_done", extra={"collection": name})

    # -----------------------------------------------------------------------
    # Helpers privati
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_distance(distance: str) -> Distance:
        """Parse del nome distance ("Cosine"/"Dot"/"Euclid") al tipo enum."""
        mapping = {
            "Cosine": Distance.COSINE,
            "Dot": Distance.DOT,
            "Euclid": Distance.EUCLID,
        }
        if distance not in mapping:
            raise ValueError(
                f"Distance '{distance}' non supportata. Valori validi: {sorted(mapping)}"
            )
        return mapping[distance]
```

- [ ] **Step 4.3: Aggiungi gli import necessari**

In [apps/api/app/rag/vector_store.py](../../../apps/api/app/rag/vector_store.py), **sostituisci** il blocco `from qdrant_client.models import (...)` aggiunto in Task 3 con questa versione, che aggiunge `AsyncQdrantClient`, `Distance`, e `VectorParams`:

```python
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)
```

- [ ] **Step 4.4: Aggiorna `__all__`**

```python
__all__ = [
    "Match",
    "QdrantVectorStore",
    "VectorPoint",
    "VectorStore",
]
```

- [ ] **Step 4.5: Scrivi i test integration per ensure/delete**

In [apps/api/tests/test_rag/test_vector_store.py](../../../apps/api/tests/test_rag/test_vector_store.py), aggiungi alla fine:

```python


# ---------------------------------------------------------------------------
# Integration tests — Qdrant live (skip se /readyz down)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_ensure_collection_creates_new(qdrant_store, unique_collection) -> None:
    """Una collection nuova viene creata correttamente."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    # Verifica indirettamente via search: una collection esistente accetta
    # search (anche se vuota → []).
    results = await qdrant_store.search(unique_collection, query=[0.1] * 4, top_k=1)
    assert results == []


@pytest.mark.integration
async def test_ensure_collection_idempotent_same_config(qdrant_store, unique_collection) -> None:
    """Chiamare ensure_collection due volte con stessa config → no-op."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    # Seconda chiamata non deve sollevare nulla.
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)


@pytest.mark.integration
async def test_ensure_collection_size_mismatch_raises(qdrant_store, unique_collection) -> None:
    """Stessa collection con vector_size diverso → ValueError."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    with pytest.raises(ValueError, match="vector_size"):
        await qdrant_store.ensure_collection(unique_collection, vector_size=8)


@pytest.mark.integration
async def test_delete_collection_existing(qdrant_store, unique_collection) -> None:
    """delete_collection rimuove una collection esistente."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    await qdrant_store.delete_collection(unique_collection)
    # Dopo delete, una search deve fallire — la collection non esiste più.
    # Riusiamo ensure_collection per ricrearla pulita, così il teardown
    # della fixture trova qualcosa da pulire (no-op se non esiste è già OK,
    # ma rendiamo la sequenza esplicita).
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)


@pytest.mark.integration
async def test_delete_collection_missing_is_noop(qdrant_store) -> None:
    """delete_collection su una collection inesistente → no-op (no raise)."""
    import uuid as _uuid

    fake = f"_test_does_not_exist_{_uuid.uuid4().hex[:8]}"
    # Non sollevare. Se solleva, il test fallisce naturalmente.
    await qdrant_store.delete_collection(fake)


@pytest.mark.integration
async def test_ensure_collection_invalid_distance_raises(qdrant_store, unique_collection) -> None:
    """Distance non supportata → ValueError prima di toccare Qdrant."""
    with pytest.raises(ValueError, match="Distance"):
        await qdrant_store.ensure_collection(unique_collection, vector_size=4, distance="Manhattan")
```

- [ ] **Step 4.6: Run i test integration — devono PASSARE**

Run:
```powershell
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack\apps\api
uv run pytest tests/test_rag/test_vector_store.py -v
```

Expected: 22 passed (16 unit + 6 integration). Se Qdrant non gira, gli integration sono "skipped" — non "failed".

- [ ] **Step 4.7: Verifica suite completa**

Run:
```powershell
uv run pytest -q
```

Expected: tutto verde (incl. health, classify, chunker, embedder, vector_store).

- [ ] **Step 4.8: Run ruff**

Run:
```powershell
uv run ruff format app/rag/vector_store.py tests/test_rag/
uv run ruff check app/rag/vector_store.py tests/test_rag/
```

Expected: clean.

- [ ] **Step 4.9: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/app/rag/vector_store.py apps/api/tests/test_rag/conftest.py apps/api/tests/test_rag/test_vector_store.py
git commit -m "feat(rag): QdrantVectorStore.ensure_collection + delete_collection"
```

---

## Task 5: `upsert` con batching

**Files:**
- Modify: `apps/api/app/rag/vector_store.py`
- Modify: `apps/api/tests/test_rag/test_vector_store.py`

- [ ] **Step 5.1: Test integration (rosso prima)**

In [apps/api/tests/test_rag/test_vector_store.py](../../../apps/api/tests/test_rag/test_vector_store.py), aggiungi alla fine:

```python


@pytest.mark.integration
async def test_upsert_returns_count(qdrant_store, unique_collection) -> None:
    """upsert ritorna il numero di punti inviati."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    points = [
        VectorPoint(id="a", vector=[1.0, 0.0, 0.0, 0.0], payload={"text": "alfa"}),
        VectorPoint(id="b", vector=[0.0, 1.0, 0.0, 0.0], payload={"text": "beta"}),
        VectorPoint(id="c", vector=[0.0, 0.0, 1.0, 0.0], payload={"text": "gamma"}),
    ]
    n = await qdrant_store.upsert(unique_collection, points)
    assert n == 3


@pytest.mark.integration
async def test_upsert_empty_list_returns_zero(qdrant_store, unique_collection) -> None:
    """upsert di lista vuota → 0, niente errore, niente call."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    n = await qdrant_store.upsert(unique_collection, [])
    assert n == 0


@pytest.mark.integration
async def test_upsert_same_id_overwrites(qdrant_store, unique_collection) -> None:
    """Re-upsert dello stesso id con vector diverso → il vecchio è soppiantato."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    await qdrant_store.upsert(
        unique_collection,
        [VectorPoint(id="x", vector=[1.0, 0.0, 0.0, 0.0], payload={"v": "old"})],
    )
    await qdrant_store.upsert(
        unique_collection,
        [VectorPoint(id="x", vector=[0.0, 1.0, 0.0, 0.0], payload={"v": "new"})],
    )
    # search vicino al "nuovo" vettore deve trovare un solo punto con payload v=new.
    results = await qdrant_store.search(unique_collection, query=[0.0, 1.0, 0.0, 0.0], top_k=10)
    assert len(results) == 1
    assert results[0].payload["v"] == "new"


@pytest.mark.integration
async def test_upsert_batches_over_100(qdrant_store, unique_collection) -> None:
    """upsert di 250 punti viene batchato in 3 chiamate (100+100+50) ma il
    risultato è transparente al consumer: tutti i punti devono essere
    presenti."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    points = [
        VectorPoint(
            id=f"p{i:03d}",
            vector=[float(i % 4 == 0), float(i % 4 == 1), float(i % 4 == 2), float(i % 4 == 3)],
            payload={"i": i},
        )
        for i in range(250)
    ]
    n = await qdrant_store.upsert(unique_collection, points)
    assert n == 250
    # Verifica: 250 punti distinti devono essere recuperabili almeno via
    # search top_k=250 su un vettore qualunque.
    results = await qdrant_store.search(unique_collection, query=[1.0, 0.0, 0.0, 0.0], top_k=250)
    assert len(results) == 250
```

- [ ] **Step 5.2: Run i test — devono FALLIRE (`upsert` non esiste ancora)**

Run:
```powershell
uv run pytest tests/test_rag/test_vector_store.py -v -m integration
```

Expected: `AttributeError: 'QdrantVectorStore' object has no attribute 'upsert'`.

- [ ] **Step 5.3: Implementa `upsert` con batching interno**

In [apps/api/app/rag/vector_store.py](../../../apps/api/app/rag/vector_store.py), dentro la class `QdrantVectorStore`, dopo `delete_collection`, aggiungi:

```python
    # Soglia di batching interna. Coerente con l'embedder (batch_size=100).
    # Qdrant accetta upsert più grandi, ma 100 è uno sweet spot fra throughput
    # (meno round-trip) e robustezza (payload più piccolo → timeout più rari).
    _UPSERT_BATCH_SIZE = 100

    async def upsert(
        self,
        collection: str,
        points: Sequence[VectorPoint],
    ) -> int:
        """Insert/update di punti. Idempotente per id (overwrite su collisione).

        Batching interno a chunk di 100 per chiamata; il consumer non se ne
        accorge — vede una singola operazione.
        """
        if not points:
            return 0

        total = 0
        for start in range(0, len(points), self._UPSERT_BATCH_SIZE):
            batch = points[start : start + self._UPSERT_BATCH_SIZE]
            qdrant_points = [_to_qdrant_point(p) for p in batch]
            await self._client.upsert(collection_name=collection, points=qdrant_points)
            total += len(batch)

        logger.info(
            "upsert_done",
            extra={
                "collection": collection,
                "total_points": total,
                "batches": (len(points) + self._UPSERT_BATCH_SIZE - 1) // self._UPSERT_BATCH_SIZE,
            },
        )
        return total
```

- [ ] **Step 5.4: Run i test — devono passare (tranne quelli che dipendono da `search` non ancora implementata: 2 falliscono)**

Run:
```powershell
uv run pytest tests/test_rag/test_vector_store.py -v -m integration
```

Expected: i due test che NON usano `search` (`test_upsert_returns_count`, `test_upsert_empty_list_returns_zero`) passano. Gli altri due falliscono con `AttributeError: ... no attribute 'search'`. **Va bene**: search è il prossimo task.

- [ ] **Step 5.5: Ruff**

```powershell
uv run ruff format app/rag/vector_store.py tests/test_rag/test_vector_store.py
uv run ruff check app/rag/vector_store.py tests/test_rag/test_vector_store.py
```

- [ ] **Step 5.6: Commit (intermedio, è OK avere 2 test failing — search è next task)**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/app/rag/vector_store.py apps/api/tests/test_rag/test_vector_store.py
git commit -m "feat(rag): QdrantVectorStore.upsert with internal batching of 100"
```

Nota: in TDD-purist questo commit lascerebbe due test rossi nel main. Accettabile per il flusso bite-sized se il TASK successivo li renderà verdi (e c'è sempre un commit finale di "all green"). In alternativa, si può accorpare con Task 6.

---

## Task 6: `search` con filtri

**Files:**
- Modify: `apps/api/app/rag/vector_store.py`
- Modify: `apps/api/tests/test_rag/test_vector_store.py`

- [ ] **Step 6.1: Test integration (rosso prima)**

In [apps/api/tests/test_rag/test_vector_store.py](../../../apps/api/tests/test_rag/test_vector_store.py), aggiungi alla fine:

```python


@pytest.mark.integration
async def test_search_empty_collection_returns_empty(qdrant_store, unique_collection) -> None:
    """search su una collection vuota → []."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    results = await qdrant_store.search(unique_collection, query=[0.5] * 4, top_k=5)
    assert results == []


@pytest.mark.integration
async def test_search_returns_top_k_ordered(qdrant_store, unique_collection) -> None:
    """search ritorna top_k ordinato per score decrescente."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    # Tre punti: il primo è perfettamente vicino a [1,0,0,0], gli altri lontani.
    points = [
        VectorPoint(id="near", vector=[1.0, 0.0, 0.0, 0.0], payload={"text": "near"}),
        VectorPoint(id="mid", vector=[0.7, 0.7, 0.0, 0.0], payload={"text": "mid"}),
        VectorPoint(id="far", vector=[0.0, 0.0, 1.0, 0.0], payload={"text": "far"}),
    ]
    await qdrant_store.upsert(unique_collection, points)

    results = await qdrant_store.search(unique_collection, query=[1.0, 0.0, 0.0, 0.0], top_k=3)
    assert len(results) == 3
    # "near" ha score max (cosine = 1.0 perfetto):
    assert results[0].id == "near"
    # Ordine decrescente:
    assert results[0].score >= results[1].score >= results[2].score


@pytest.mark.integration
async def test_search_respects_top_k_limit(qdrant_store, unique_collection) -> None:
    """top_k=1 → al massimo 1 risultato."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    points = [
        VectorPoint(id=f"p{i}", vector=[1.0, 0.0, 0.0, 0.0], payload={"i": i})
        for i in range(5)
    ]
    await qdrant_store.upsert(unique_collection, points)
    results = await qdrant_store.search(unique_collection, query=[1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1


@pytest.mark.integration
async def test_search_with_filter_returns_only_matching(qdrant_store, unique_collection) -> None:
    """Filter {source: X} → solo punti con payload[source]=X."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    points = [
        VectorPoint(id="a", vector=[1.0, 0.0, 0.0, 0.0], payload={"source": "intro.md"}),
        VectorPoint(id="b", vector=[1.0, 0.0, 0.0, 0.0], payload={"source": "intro.md"}),
        VectorPoint(id="c", vector=[1.0, 0.0, 0.0, 0.0], payload={"source": "other.md"}),
    ]
    await qdrant_store.upsert(unique_collection, points)

    results = await qdrant_store.search(
        unique_collection,
        query=[1.0, 0.0, 0.0, 0.0],
        top_k=10,
        filter={"source": "intro.md"},
    )
    assert len(results) == 2
    assert all(r.payload["source"] == "intro.md" for r in results)
    ids = sorted(r.id for r in results)
    assert ids == ["a", "b"]


@pytest.mark.integration
async def test_search_with_filter_and_logic(qdrant_store, unique_collection) -> None:
    """Filter con più chiavi → AND (deve matchare TUTTE le chiavi)."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    points = [
        VectorPoint(
            id="a", vector=[1.0, 0.0, 0.0, 0.0],
            payload={"source": "intro.md", "heading": "H1"},
        ),
        VectorPoint(
            id="b", vector=[1.0, 0.0, 0.0, 0.0],
            payload={"source": "intro.md", "heading": "H2"},
        ),
        VectorPoint(
            id="c", vector=[1.0, 0.0, 0.0, 0.0],
            payload={"source": "other.md", "heading": "H1"},
        ),
    ]
    await qdrant_store.upsert(unique_collection, points)

    results = await qdrant_store.search(
        unique_collection,
        query=[1.0, 0.0, 0.0, 0.0],
        top_k=10,
        filter={"source": "intro.md", "heading": "H1"},
    )
    assert len(results) == 1
    assert results[0].id == "a"


@pytest.mark.integration
async def test_search_returns_match_with_original_id(qdrant_store, unique_collection) -> None:
    """Match.id deve essere l'id ORIGINALE (sha256-like), non l'UUID Qdrant."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    original_id = "sha256-mock-not-actually-a-hash"
    await qdrant_store.upsert(
        unique_collection,
        [VectorPoint(id=original_id, vector=[1.0, 0.0, 0.0, 0.0], payload={})],
    )
    results = await qdrant_store.search(unique_collection, query=[1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].id == original_id
    # E non deve trapelare la chiave riservata:
    assert "__vp_id" not in results[0].payload
```

- [ ] **Step 6.2: Run i test — devono FALLIRE**

```powershell
uv run pytest tests/test_rag/test_vector_store.py -v -m integration
```

Expected: failures con `AttributeError: ... no attribute 'search'`.

- [ ] **Step 6.3: Implementa `search`**

In [apps/api/app/rag/vector_store.py](../../../apps/api/app/rag/vector_store.py), dentro la class `QdrantVectorStore`, dopo `upsert`, aggiungi:

```python
    async def search(
        self,
        collection: str,
        query: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Match]:
        """Top-k semantic search. Filter shallow equality opzionale."""
        qdrant_filter = _filter_to_qdrant(filter)
        scored = await self._client.search(
            collection_name=collection,
            query_vector=query,
            limit=top_k,
            query_filter=qdrant_filter,
        )
        return [_from_qdrant_scored_point(sp) for sp in scored]
```

- [ ] **Step 6.4: Run i test — devono PASSARE tutti**

```powershell
uv run pytest tests/test_rag/test_vector_store.py -v
```

Expected: 32 passed (16 unit + 16 integration).

- [ ] **Step 6.5: Ruff**

```powershell
uv run ruff format app/rag/vector_store.py tests/test_rag/test_vector_store.py
uv run ruff check app/rag/vector_store.py tests/test_rag/test_vector_store.py
```

- [ ] **Step 6.6: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/app/rag/vector_store.py apps/api/tests/test_rag/test_vector_store.py
git commit -m "feat(rag): QdrantVectorStore.search with optional filter (AND logic)"
```

---

## Task 7: `get_vector_store()` singleton + verifica finale + roadmap

**Files:**
- Modify: `apps/api/app/rag/vector_store.py`
- Modify: `apps/api/tests/test_rag/test_vector_store.py`
- Modify: `docs/ROADMAP.md`

- [ ] **Step 7.1: Test del singleton**

In [apps/api/tests/test_rag/test_vector_store.py](../../../apps/api/tests/test_rag/test_vector_store.py), aggiungi alla fine:

```python


# ---------------------------------------------------------------------------
# Factory singleton
# ---------------------------------------------------------------------------


def test_get_vector_store_is_singleton() -> None:
    """Due chiamate a get_vector_store ritornano la stessa istanza."""
    from app.rag.vector_store import QdrantVectorStore, get_vector_store

    a = get_vector_store()
    b = get_vector_store()
    assert a is b
    assert isinstance(a, QdrantVectorStore)
```

- [ ] **Step 7.2: Run il test — deve FALLIRE (`get_vector_store` non esiste)**

```powershell
uv run pytest tests/test_rag/test_vector_store.py::test_get_vector_store_is_singleton -v
```

Expected: ImportError.

- [ ] **Step 7.3: Implementa il singleton**

In [apps/api/app/rag/vector_store.py](../../../apps/api/app/rag/vector_store.py), sostituisci il blocco `# Factory singleton ... # (Implementato nelle task successive.)` con:

```python
# ---------------------------------------------------------------------------
# Factory singleton
# ---------------------------------------------------------------------------
# Pattern coerente con `app.config.settings`: istanza modulo-livello creata
# alla prima chiamata e riusata. AsyncQdrantClient è progettato per essere
# long-lived (gestisce internamente un pool httpx); ricrearlo per ogni
# request sarebbe wasteful.

_vector_store_singleton: QdrantVectorStore | None = None


def get_vector_store() -> QdrantVectorStore:
    """Ritorna l'istanza singleton di QdrantVectorStore.

    Usa `app.config.settings` per URL e API key. Pensato per essere usato
    con `Depends(get_vector_store)` da endpoint FastAPI (dal task #10 in poi).
    """
    global _vector_store_singleton
    if _vector_store_singleton is None:
        from app.config import settings  # import lazy: evita ciclo all'avvio

        _vector_store_singleton = QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return _vector_store_singleton
```

- [ ] **Step 7.4: Aggiorna `__all__`**

In fondo a [apps/api/app/rag/vector_store.py](../../../apps/api/app/rag/vector_store.py):

```python
__all__ = [
    "Match",
    "QdrantVectorStore",
    "VectorPoint",
    "VectorStore",
    "get_vector_store",
]
```

- [ ] **Step 7.5: Run tutti i test**

```powershell
uv run pytest -q
```

Expected: tutto verde (33 nuovi test su vector_store + suite esistente).

- [ ] **Step 7.6: Run ruff sull'intera codebase modificata**

```powershell
uv run ruff format app/ tests/
uv run ruff check app/ tests/
```

Expected: clean.

- [ ] **Step 7.7: Aggiorna `docs/ROADMAP.md`**

In [docs/ROADMAP.md](../../ROADMAP.md), sezione M2:

1. Cambia il task #5 da:
   ```
   5. **Vector store client** (`app/rag/vector_store.py`): abstract base
      class + implementazione concreta (Pinecone o pgvector secondo ADR).
   ```
   a:
   ```
   5. ✅ **Vector store client** (`app/rag/vector_store.py`): Protocol
      `VectorStore` (typing.Protocol — duck typing strutturale) +
      implementazione concreta `QdrantVectorStore` su `AsyncQdrantClient`.
      Metodi: `ensure_collection` (idempotente, ValueError su size mismatch),
      `upsert` (batching interno 100/call, idempotente per id),
      `search` (top-k + filter shallow AND), `delete_collection`.
      Singleton via `get_vector_store()`. 33 test verdi (17 unit + 16
      integration su Qdrant live).
   ```

2. Nessun altro cambio nella tabella "Status d'insieme" (M2 resta ⚪ — il milestone non è chiuso, ci sono altri task).

- [ ] **Step 7.8: Commit finale**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/app/rag/vector_store.py apps/api/tests/test_rag/test_vector_store.py docs/ROADMAP.md
git commit -m "feat(rag): get_vector_store singleton; mark M2 task #5 done in roadmap"
```

- [ ] **Step 7.9: Verifica finale: git log + suite verde**

```powershell
git log --oneline -10
uv run --directory apps/api pytest -q
```

Expected: 6 commit nuovi su HEAD (uno per task), tutta la suite verde.

---

## Spec coverage map

| Spec requirement | Task |
|---|---|
| `VectorPoint`, `Match` Pydantic | Task 2 |
| `VectorStore` Protocol | Task 2 |
| `QdrantVectorStore` su `AsyncQdrantClient` | Task 4 |
| `ensure_collection` idempotente + ValueError su size mismatch | Task 4 |
| `delete_collection` no-op | Task 4 |
| `upsert` batching 100, idempotente per id | Task 5 |
| `search` top-k ordinato | Task 6 |
| `search` con filter dict shallow AND | Task 6 |
| `Match` ritorna id originale (non UUID Qdrant) | Task 3 (mapping) + Task 6 (integration) |
| `get_vector_store()` singleton | Task 7 |
| Config: 3 nuovi campi `qdrant_*` | Task 1 |
| Dep `qdrant-client` ≥1.11 | Task 1 |
| pytest marker `integration` | Task 1 |
| Test integration su Qdrant reale, no mock | Task 4-6 (sempre integration) |
| Mapping helpers come unit test puri | Task 3 |

---

## Final acceptance check

A fine Task 7, eseguire:

```powershell
# 1. Suite intera verde
uv run --directory apps/api pytest -q

# 2. Linter clean
uv run --directory apps/api ruff check .
uv run --directory apps/api ruff format --check .

# 3. La nuova API è importabile e funzionante
uv run --directory apps/api python -c "from app.rag.vector_store import VectorStore, VectorPoint, Match, QdrantVectorStore, get_vector_store; print('OK:', get_vector_store().__class__.__name__)"
```

Expected del terzo comando:
```
OK: QdrantVectorStore
```
