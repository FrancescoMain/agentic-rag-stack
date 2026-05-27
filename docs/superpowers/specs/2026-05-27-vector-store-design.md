# Design — Vector Store layer (M2 task #5)

**Status:** Approved
**Date:** 2026-05-27
**Milestone:** M2 — Knowledge base & RAG pipeline
**Task:** #5 — `app/rag/vector_store.py`
**Related ADRs:** [ADR-0003](../../adr/0003-vector-db-choice-qdrant.md) (Qdrant), [ADR-0004](../../adr/0004-embedding-model-choice.md) (text-embedding-3-small)

## Goal

Esporre un'astrazione `VectorStore` con una sola implementazione concreta `QdrantVectorStore`, capace di creare collezioni in modo idempotente, fare upsert di punti vettoriali, e recuperare i top-k più simili a una query. È il "data layer" su cui si appoggeranno il CLI di ingest (task #6) e il retriever (task #7).

## Non-goals

- Hybrid search / BM25 → task #7
- CLI di ingestion (orchestrazione load → chunk → embed → upsert) → task #6
- Reranking → task #8
- Citazioni → task #9
- Endpoint `/retrieve` → task #10
- Filter API ricca (range, geo, nested) — per ora solo equality match shallow

## Architecture

Un singolo modulo `app/rag/vector_store.py` contenente:

```
┌─────────────────────────────────────────────────────┐
│  app/rag/vector_store.py                            │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐               │
│  │ VectorPoint  │    │    Match     │  Pydantic     │
│  │ (input DTO)  │    │ (output DTO) │  DTOs         │
│  └──────────────┘    └──────────────┘               │
│                                                     │
│  ┌─────────────────────────────────────────┐        │
│  │  VectorStore (typing.Protocol)          │        │
│  │  - ensure_collection                    │        │
│  │  - upsert                               │        │
│  │  - search                               │        │
│  │  - delete_collection                    │        │
│  └─────────────────────────────────────────┘        │
│                  ▲                                  │
│                  │ implements                       │
│  ┌─────────────────────────────────────────┐        │
│  │  QdrantVectorStore                      │        │
│  │  wraps AsyncQdrantClient                │        │
│  └─────────────────────────────────────────┘        │
│                                                     │
│  get_vector_store() -> VectorStore  (singleton)     │
└─────────────────────────────────────────────────────┘
```

## Interfaces

### DTOs

```python
class VectorPoint(BaseModel):
    """Punto da upsertare nel vector store."""
    id: str               # ID deterministico (sha256 dal chunker → upsert idempotente)
    vector: list[float]   # embedding (1536 dim per text-embedding-3-small)
    payload: dict[str, Any]  # metadata + testo originale (per citazioni)

class Match(BaseModel):
    """Risultato di una ricerca."""
    id: str
    score: float          # cosine similarity, range [-1, 1] (tipicamente [0, 1] per testi normalizzati)
    payload: dict[str, Any]
```

### Protocol

```python
class VectorStore(Protocol):
    async def ensure_collection(
        self,
        name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None: ...

    async def upsert(
        self,
        collection: str,
        points: Sequence[VectorPoint],
    ) -> int: ...

    async def search(
        self,
        collection: str,
        query: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Match]: ...

    async def delete_collection(self, name: str) -> None: ...
```

## Behavior contracts

### `ensure_collection`
- Idempotente: se la collection NON esiste, la crea. Se esiste con stessa `vector_size` e `distance`, **no-op**.
- Se esiste con `vector_size` diverso, **alza `ValueError`** (segnale forte: hai cambiato embedding model, decidi tu se fare `delete_collection` + ricreare).
- `distance` accetta i valori canonici Qdrant: `"Cosine"`, `"Dot"`, `"Euclid"`.

### `upsert`
- Idempotente per `id`: se un punto con stesso `id` esiste, viene sovrascritto (questo perché il chunker produce id deterministici sha256). Comportamento garantito da Qdrant nativo.
- Ritorna il numero di punti effettivamente inviati (= `len(points)` se non c'è errore; non distingue insert vs update — Qdrant non lo espone).
- Batch interno: se `len(points) > 100`, l'implementazione fa più chiamate di upsert da 100 punti ciascuna per evitare payload troppo grossi. Soglia uniformata a quella dell'embedder.

### `search`
- `top_k` default 5. Nessun limite massimo enforced lato vector store (Qdrant lo gestisce).
- `filter`: dict shallow `{campo: valore}` con semantica equality match. Es. `{"source": "tutorial.md"}` → restituisce solo chunk con `payload["source"] == "tutorial.md"`. Internamente mappato a `qdrant_client.models.Filter` con `FieldCondition` + `MatchValue`. Multiple chiavi nel dict → AND logico.
- Ritorna `list[Match]` ordinata per `score` decrescente (default Qdrant).

### `delete_collection`
- Esposto principalmente per i test e per "drop + recreate" manuale dopo cambio modello.
- No-op se la collection non esiste (idempotente).

## Lifecycle

- **Costruzione:** `QdrantVectorStore(url, api_key=None)` — apre `AsyncQdrantClient`. Connessione lazy (Qdrant client non fa I/O al costruttore).
- **Singleton modulo-livello:** `get_vector_store() -> VectorStore` — istanzia una volta sola usando `app.config.settings.qdrant_url` e `settings.qdrant_api_key`. Pattern coerente con `app/config.py`. FastAPI può iniettarlo via `Depends`.
- **Shutdown:** non chiudiamo esplicitamente il client (i pool HTTPX gestiscono da soli). In M5 quando avremo lifespan FastAPI, valuteremo un `aclose()` deterministico.

## Configuration

Aggiunti a [apps/api/app/config.py](../../../apps/api/app/config.py):

```python
qdrant_url: str = Field(default="http://localhost:6333")
qdrant_api_key: str = Field(default="")          # vuoto = no auth (dev)
qdrant_collection_name: str = Field(default="agentic-rag-demo")
```

Le env var corrispondenti sono già in `.env.example` (`QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME`).

`qdrant_collection_name` NON è usato dal vector store stesso (che riceve `collection: str` per chiamata) — è usato dall'ingest CLI come default. Lo mettiamo qui per coerenza.

## Dependencies

Aggiunte a [apps/api/pyproject.toml](../../../apps/api/pyproject.toml):

```toml
qdrant-client = ">=1.11,<2"
```

Versione coerente con l'immagine Docker (`qdrant/qdrant:v1.11.0`).

## Testing strategy

### Integration tests (predominanti)

File: `apps/api/tests/rag/test_vector_store.py`

Marker: `@pytest.mark.integration` (definito in `pytest.ini` o `pyproject.toml` con `markers`).

Fixture session-scoped che:
1. Verifica che Qdrant risponda su `settings.qdrant_url/readyz`. Se no → skip con messaggio "start qdrant via docker compose".
2. Restituisce un `QdrantVectorStore` configurato.

Fixture function-scoped che:
1. Genera un nome collection unico (`_test_{uuid}`).
2. Yield del nome.
3. Cleanup: `delete_collection` nel teardown.

**Test cases:**
- `ensure_collection` crea una collection nuova → ok
- `ensure_collection` chiamato due volte stessa config → no-op (no error)
- `ensure_collection` con vector_size diverso su collection esistente → `ValueError`
- `upsert` di 3 punti → `search` su un vettore vicino a uno → torna quello come top-1
- `upsert` dello stesso `id` con vector diverso → secondo `search` rispecchia il nuovo vettore (overwrite)
- `upsert` di 250 punti → batch interno, tutti presenti dopo
- `search` con `filter={"source": "x"}` → ritorna solo i punti col matching source
- `delete_collection` di una non esistente → no-op
- `search` su collection vuota → lista vuota

### Unit tests (limitati)

Solo per le funzioni pure di mapping:
- `_to_qdrant_point(VectorPoint) -> PointStruct`
- `_from_qdrant_scored_point(ScoredPoint) -> Match`
- `_filter_to_qdrant(dict | None) -> Filter | None`

Niente mock dell'`AsyncQdrantClient`. Se serve testare error paths (es. timeout) lo facciamo via integration con un'istanza Qdrant fermata, in M5.

### CI considerations (per M5)

In M5 il job `test` su GitHub Actions farà `docker compose up -d qdrant` prima dei pytest, oppure useremo `testcontainers-python`. Per ora non è in scope.

## Risks & open questions

1. **`AsyncQdrantClient` su Windows in Python 3.12** — la libreria usa httpx async, dovrebbe essere fluida. Se incontriamo flakiness (raro ma documentato in alcune versioni), fallback è il `QdrantClient` sync chiamato via `asyncio.to_thread`. Lo decidiamo solo se accade.
2. **Distance metric e vettori non-normalizzati** — text-embedding-3-small produce vettori normalizzati, ma se in M4+ aggiungiamo altri embedding (es. multimodali), può saltare l'assunzione. La decisione "default Cosine, ma parametro" copre questo caso.
3. **Schema del `payload`** — non lo standardizziamo qui. Convenzione (non enforced): `{text, source, heading, position, token_count}` per chunk testuali. Se diventa scomodo, M2 task #9 introdurrà un Pydantic schema dedicato.

## Acceptance criteria

- [ ] `app/rag/vector_store.py` esiste con `VectorPoint`, `Match`, `VectorStore` Protocol, `QdrantVectorStore` concreta, `get_vector_store()` singleton.
- [ ] `app/config.py` ha i 3 nuovi campi Qdrant.
- [ ] `qdrant-client` in `pyproject.toml`.
- [ ] `uv run pytest -m integration apps/api/tests/rag/test_vector_store.py` → tutti verdi con Qdrant up.
- [ ] `uv run pytest apps/api/` → suite intera resta verde (no regressioni su chunker/embedder).
- [ ] `uv run ruff check apps/api/ && uv run ruff format --check apps/api/` → clean.
- [ ] Nessun mock di `AsyncQdrantClient` introdotto.

## Out-of-band notes

Il file [apps/web/pnpm-workspace.yaml](../../../apps/web/pnpm-workspace.yaml) creato per sbloccare i build script di sharp/unrs-resolver in pnpm 11 NON è correlato a questo task ma è stato necessario per far partire il frontend. Va committato a parte come fix indipendente.
