# Design — POST /retrieve endpoint (M2 task #10a)

**Status:** Approved
**Date:** 2026-05-27
**Milestone:** M2 — Knowledge base & RAG pipeline
**Task:** #10a — `POST /retrieve` endpoint (dense-only)
**Related:** task #10b (futuro: reranker + citations integration)
**Depends on:** task #5 (vector_store), #6 (ingest), #7a (retriever)

## Goal

Esporre via HTTP il retriever dense costruito in #7a. Questo è il "chiusura del loop" RAG end-to-end della milestone M2: a fine task, posso chiamare `curl POST /retrieve` con una query in linguaggio naturale e ricevere i top-k chunk rilevanti dal corpus indicizzato.

## Decomposizione del task #10 originale

Il task #10 del ROADMAP era "endpoint `POST /retrieve` → `{chunks, citations}`". Lo spezziamo in:

- **#10a (questo task):** endpoint che ritorna solo `{chunks: [...]}` (pass-through del retriever). Validazione input, error handling, test FastAPI.
- **#10b (futuro):** quando #8 reranker e #9 citation builder saranno fatti, modificheranno questo endpoint per aggiungere reranker step + arricchire la response con `citations` strutturate.

Questa decomposizione è coerente con [#7a/#7b](2026-05-27-dense-retriever-design.md) e con la memory feedback "incremental YAGNI".

## Non-goals

- Reranker (Cohere o BGE) → task #8
- Citation builder come oggetto separato `{chunk_id, source, snippet, score}` → task #9
- Streaming SSE della response → M3 (con il chat endpoint)
- Authentication / rate limiting → M5
- Router separato `app/routers/rag.py` → quando avremo più endpoint RAG sarà sensato
- Frontend playground UI per /retrieve → arriva in M3

## Architecture

Singolo endpoint nuovo in `app/main.py`, accanto a `/classify`. Pattern del repo è "endpoint+schemi inline in main.py, logica in service o module dedicato". La logica vera (retriever) vive già in `app/rag/retriever.py` — questo task è solo la sottile shell HTTP attorno.

```
┌────────────────────────────────────────────────────┐
│ POST /retrieve                                     │
│   ↓                                                │
│ RetrieveRequest (Pydantic validation)              │
│   ↓                                                │
│ retriever.retrieve(query, collection, top_k, ...)  │
│   ↓ (gestione exception)                           │
│ map list[Match] → RetrieveResponse                 │
│   ↓                                                │
│ JSON {chunks: [...]}                               │
└────────────────────────────────────────────────────┘
```

## HTTP contract

### Request

```http
POST /retrieve
Content-Type: application/json

{
  "query": "how do I configure CORS in FastAPI?",
  "collection": "agentic-rag-demo",
  "top_k": 5,
  "filter": {"source": "intro.md"}
}
```

- `query` (string, **required**): query in linguaggio naturale. `min_length=1` (Pydantic respinge vuota).
- `collection` (string, optional): nome collection Qdrant. Se omessa, default a `settings.qdrant_collection_name`.
- `top_k` (int, optional): numero max risultati. Default 5, range `[1, 50]`.
- `filter` (dict, optional): shallow equality match sul payload. AND fra chiavi. Es. `{"source": "intro.md", "heading": "H1"}`.

### Response 200

```json
{
  "chunks": [
    {
      "id": "sha256-...",
      "score": 0.87,
      "payload": {
        "text": "...",
        "source": "intro.md",
        "heading": "Configuration",
        "position": 3,
        "token_count": 124
      }
    }
  ]
}
```

- `chunks`: lista ordinata per `score` decrescente. Vuota se nessun match.
- `chunks[].id`: id ORIGINALE (sha256 del chunker), NON l'UUID Qdrant.
- `chunks[].payload`: pass-through del payload del chunk. Niente chiave riservata `__vp_id` (già rimossa da vector_store).

### Error responses

| Status | Quando | Detail |
|---|---|---|
| **422** | Validation Pydantic fallita (es. query vuota, top_k fuori range, body malformato) | Standard FastAPI validation error |
| **404** | Qdrant ritorna 404 (collection inesistente) | `"collection '<name>' not found"` |
| **502** | Qdrant ritorna altro errore HTTP (5xx, timeout, ecc.) | `"vector store error: <details>"` |
| **503** | `OPENAI_API_KEY` mancante (`get_retriever()` solleva ValueError al boot del singleton) | `"retriever not configured: ..."` |

## Interfaces (Pydantic, inline in app/main.py)

```python
class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, description="Domanda in linguaggio naturale.")
    collection: str | None = Field(
        default=None,
        description="Nome collection Qdrant. Default: settings.qdrant_collection_name.",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Numero max di risultati.")
    filter: dict[str, Any] | None = Field(
        default=None,
        description="Shallow equality match sul payload (AND fra chiavi).",
    )


class RetrievedChunk(BaseModel):
    id: str
    score: float
    payload: dict[str, Any]


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunk]
```

`RetrievedChunk` è strutturalmente uguale a `app.rag.vector_store.Match` ma esiste come schema HTTP separato. Convenzione: gli schemi di response sono separati dai DTO interni per evitare leak di refactor (se un giorno aggiungo un campo a `Match` per uso interno, non finisce auto nella response JSON).

## Endpoint signature

```python
@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    request: RetrieveRequest,
    retriever: Retriever = Depends(get_retriever),
) -> RetrieveResponse:
    collection = request.collection or settings.qdrant_collection_name
    try:
        matches = await retriever.retrieve(
            query=request.query,
            collection=collection,
            top_k=request.top_k,
            filter=request.filter,
        )
    except ValueError as exc:
        # Rete di sicurezza: Pydantic dovrebbe già aver respinto query="".
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"collection '{collection}' not found",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"vector store error: {exc}",
        ) from exc

    return RetrieveResponse(
        chunks=[RetrievedChunk(**m.model_dump()) for m in matches]
    )
```

Note implementative:
- `UnexpectedResponse` da `qdrant_client.http.exceptions`.
- `503` non gestito esplicitamente nell'endpoint: il `Depends(get_retriever)` solleva `ValueError` se OPENAI_API_KEY è vuota, e FastAPI lo trasforma in 500 di default. Per ottenere 503 esplicito, registriamo un `exception_handler(ValueError)` mirato — opzionalmente. Decisione: lo aggiungiamo SE l'endpoint gira con `OPENAI_API_KEY=""` (smoke manuale lo verifica). Altrimenti accettiamo 500.

## Dependency override per i test

Stesso pattern di `fake_anthropic`/`fake_openai`:

```python
class FakeRetriever:
    """Test double per app.rag.retriever.Retriever.

    Configurabile dal test:
        fake.matches_to_return = [Match(id="x", score=0.9, payload={...}), ...]
        fake.error_to_raise = UnexpectedResponse(...)
    """

    def __init__(self) -> None:
        self.matches_to_return: list[Match] = []
        self.error_to_raise: Exception | None = None
        self.calls: list[dict[str, Any]] = []

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Match]:
        self.calls.append(
            {"query": query, "collection": collection, "top_k": top_k, "filter": filter}
        )
        if self.error_to_raise is not None:
            raise self.error_to_raise
        return self.matches_to_return


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    """Fixture: istanzia FakeRetriever e lo collega via dependency_overrides."""
    fake = FakeRetriever()
    app.dependency_overrides[get_retriever] = lambda: fake
    yield fake
    app.dependency_overrides.clear()
```

Vive in `tests/conftest.py` top-level (insieme a `fake_anthropic` e `fake_openai`).

## Test plan

File: `apps/api/tests/test_retrieve.py`. Tutti unit con `TestClient` + `FakeRetriever`. Niente Qdrant, niente OpenAI — il retriever è sostituito dal fake.

| # | Test | Verifica |
|---|---|---|
| 1 | `test_retrieve_happy_path` | POST query valida, fake ritorna 2 Match → response 200 con 2 chunks correttamente serializzati |
| 2 | `test_retrieve_uses_default_collection` | Request senza `collection` → fake riceve `settings.qdrant_collection_name` |
| 3 | `test_retrieve_uses_provided_collection` | `"collection": "x"` → fake riceve `"x"` |
| 4 | `test_retrieve_passes_filter` | `"filter": {"source": "y"}` → fake riceve filter |
| 5 | `test_retrieve_default_top_k` | Request senza `top_k` → fake riceve `top_k=5` |
| 6 | `test_retrieve_empty_query_422` | `"query": ""` → 422 (Pydantic) |
| 7 | `test_retrieve_top_k_zero_422` | `"top_k": 0` → 422 |
| 8 | `test_retrieve_top_k_too_large_422` | `"top_k": 1000` → 422 |
| 9 | `test_retrieve_collection_not_found_404` | fake alza `UnexpectedResponse(status_code=404)` → 404 |
| 10 | `test_retrieve_qdrant_error_502` | fake alza `UnexpectedResponse(status_code=500)` → 502 |
| 11 | `test_retrieve_match_shape_preserved` | Match con id arbitrario + payload nested → response.chunks[0] coincide field-by-field |

Niente test integration con Qdrant live: il retriever è già testato così in `tests/test_rag/test_retriever.py`. Qui testiamo solo lo strato HTTP.

## Smoke manuale (parte del task)

```powershell
# Pre: corpus piccolo già ingestato (collection smoke_endpoint).
curl -X POST http://localhost:8000/retrieve `
     -H "Content-Type: application/json" `
     -d '{"query": "how do I build a web API", "collection": "smoke_endpoint", "top_k": 2}'
```

Expected: JSON con `chunks` non vuoto, score plausibili, source coerente.

## File touched

| File | Action |
|---|---|
| `apps/api/app/main.py` | **Modify** (aggiungi schemi + endpoint `/retrieve`) |
| `apps/api/tests/conftest.py` | **Modify** (aggiungi `FakeRetriever` + fixture `fake_retriever`) |
| `apps/api/tests/test_retrieve.py` | **Create** |
| `docs/ROADMAP.md` | **Modify** (split task #10 in #10a ✅ + #10b ⚪) |

## Risks & open questions

1. **`UnexpectedResponse.status_code` access**: verificare che il tipo di errore di `qdrant-client` 1.18 abbia effettivamente `status_code` come attributo. In caso negativo, parsare `str(exc)` o usare `e.args`. Da accertare in implementazione; se la firma è diversa, adatto.
2. **503 vs 500 quando OPENAI_API_KEY è vuota**: come notato sopra, non aggiungo handler esplicito a meno che lo smoke non lo mostri necessario. YAGNI.
3. **OpenAPI schema `dict[str, Any]`** per `filter`: FastAPI lo serializza come `Dict[str, Any]` nello swagger; per il consumer è ok ma poco documentato. Pattern del repo accetta questo livello di laxness.

## Acceptance criteria

- [ ] `POST /retrieve` registrato in `app/main.py` e visibile su `/docs`.
- [ ] 11 test verdi su `tests/test_retrieve.py`.
- [ ] Suite intera resta verde (no regressioni).
- [ ] ruff clean.
- [ ] Smoke manuale: ingest piccolo corpus → `curl POST /retrieve` → response JSON sensata.
- [ ] `docs/ROADMAP.md` aggiornato con split 10a/10b.
