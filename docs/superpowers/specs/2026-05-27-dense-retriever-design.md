# Design — Dense Retriever (M2 task #7a)

**Status:** Approved
**Date:** 2026-05-27
**Milestone:** M2 — Knowledge base & RAG pipeline
**Task:** #7a — `app/rag/retriever.py` (dense retriever)
**Related:** task #7b (hybrid sparse + RRF) — backlog, da introdurre solo se l'eval lo richiede
**Depends on:** task #4 (embedder), #5 (vector_store), #6 (ingest)

## Goal

Esporre un `Retriever` che, data una query in linguaggio naturale, embedda con OpenAI e cerca i top-k chunk più simili nella collection Qdrant. È il "motore di ricerca" del sistema RAG su cui si appoggeranno citation builder (task #9), endpoint `/retrieve` (task #10), e in M3 il chat con citazioni inline.

## Decomposizione esplicita della roadmap

Il task #7 originale del ROADMAP era "hybrid search = vector + BM25". Lo spezziamo in:

- **#7a (questo task):** dense retriever con metadata filters. Sufficiente per cominciare, semplice da testare, copre la stragrande maggioranza delle query in linguaggio naturale.
- **#7b (backlog):** sparse vectors via FastEmbed BM25 + RRF fusion. Da introdurre **se** l'eval (task #11-12) mostra che dense fallisce su query lessicali (codici errore, identifier, version numbers). Significherebbe: nuova dep `qdrant-client[fastembed]`, collection multi-vector, re-ingest, breaking change a vector_store. Lavoro consistente, da fare solo dietro evidenza.

Questa decomposizione è coerente col principio "vedi il dolore prima della cura": l'eval ti darà un numero concreto su cui decidere, non una scelta a priori.

## Non-goals

- Sparse retrieval / BM25 → task #7b se l'eval lo richiede
- Score fusion (RRF, DBSF) → task #7b
- Reranker → task #8
- Citation builder → task #9
- Endpoint HTTP `/retrieve` → task #10
- Query rewriting / espansione → fuori scope M2
- Cache layer → eventualmente M5
- Filtri ricchi (range, geo, full-text) → fuori scope M2

## Architecture

Singolo modulo `app/rag/retriever.py` con una classe e un singleton. Tutta la logica è orchestrazione di componenti che già esistono.

```
┌────────────────────────────────────────────────────────────┐
│  app/rag/retriever.py                                      │
│                                                            │
│   class Retriever:                                         │
│     __init__(store: VectorStore, openai_client)            │
│     async retrieve(query, collection, top_k=5, filter)     │
│         │                                                  │
│         ├─→ embed_texts([query]) (task #4) → [vector]      │
│         │                                                  │
│         └─→ store.search(collection, vector, top_k, filter)│
│             (task #5) → list[Match]                        │
│                                                            │
│   get_retriever() singleton modulo-livello                 │
└────────────────────────────────────────────────────────────┘
```

## Interfaces

```python
class Retriever:
    """Dense semantic retriever su Qdrant.

    Limiti noti — questo retriever fa SOLO dense search. Cattura bene la
    somiglianza semantica di query in linguaggio naturale, ma può sotto-
    performare su query lessicali esatte (codici errore, identifier,
    version numbers). Quando l'eval (task #11-12) misurerà precision@k
    su un golden dataset, potremo decidere se introdurre hybrid (sparse
    BM25 + RRF) come task #7b.
    """

    def __init__(
        self,
        store: VectorStore,
        openai_client: AsyncOpenAI,
    ) -> None: ...

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Match]:
        """Embed query → search → return top_k Match.

        Args:
            query: testo in linguaggio naturale. Stringa non vuota.
            collection: nome collection Qdrant da interrogare.
            top_k: numero massimo di risultati. Default 5.
            filter: shallow equality match sul payload (es. {"source": "X"}).

        Returns:
            list[Match] ordinata per score decrescente. Vuota se la
            collection è vuota o nessun risultato.

        Raises:
            ValueError: se la query è vuota.
            EmbedderError: se l'embedding fallisce dopo tutti i retry.
            Eccezioni vector_store (UnexpectedResponse, ecc.) propagate.
        """
        ...


def get_retriever() -> Retriever:
    """Singleton modulo-livello.

    Costruito da:
    - get_vector_store() (singleton già esistente)
    - AsyncOpenAI(api_key=settings.openai_api_key)

    Lazy: prima chiamata = inizializzazione. Solleva ValueError se
    OPENAI_API_KEY è vuota (pattern coerente con _run() di ingest.py).
    """
    ...
```

## Behavior contracts

### `retrieve`
- **Query vuota** (stringa vuota o solo whitespace) → `ValueError("query cannot be empty")` PRIMA di toccare l'API OpenAI (fail-fast, no waste).
- **Collection inesistente** → propaga eccezione di Qdrant. NON cattiamo: chi chiama deve sapere se ha sbagliato il nome collection (vs collection vuota che è situazione legittima).
- **Collection vuota** → ritorna `[]` (è il comportamento di `vector_store.search` su collection vuota).
- **Filter shallow** → pass-through a `vector_store.search`. La semantica AND fra chiavi è già implementata lato vector_store.
- **top_k > numero punti** → Qdrant ritorna tutto quello che ha; nessun errore.
- **Match.id** è l'id originale (sha256 del chunker), NON l'UUID interno di Qdrant. Garanzia ereditata dal mapping di task #5.

### `get_retriever`
- Singleton: due chiamate ritornano lo stesso oggetto.
- Lazy init: `AsyncOpenAI` viene istanziato solo alla prima chiamata.
- `OPENAI_API_KEY` mancante → `ValueError` chiaro al momento dell'istanziazione (NON al primo `retrieve()` — meglio fail-fast).

## Configuration

Nessun nuovo campo `Settings`. Riusiamo:
- `settings.openai_api_key` per istanziare AsyncOpenAI
- `settings.qdrant_url` / `qdrant_api_key` via `get_vector_store()`

## Dependencies

Nessuna nuova. Tutto esiste già nel `pyproject.toml`.

## Testing strategy

File: `apps/api/tests/test_rag/test_retriever.py`.

### Integration tests (Qdrant live + FakeOpenAIClient)

Tutti con `@pytest.mark.integration`. Setup: per ogni test, crea collection unica con un piccolo corpus pre-popolato, esegui retrieve, asserisci.

**Scenari:**

1. `test_retrieve_empty_collection` → collection vuota, retrieve qualsiasi query → `[]`.

2. `test_retrieve_returns_top_k_ordered` → upsert 5 chunk con vettori distinti; query close to chunk[2]; assert `len(result) == 3` con `top_k=3` e `result[0].id == chunk[2].id`.

3. `test_retrieve_respects_filter` → upsert 4 chunk con `payload["source"]` = `a.md` x2 e `b.md` x2; retrieve con `filter={"source": "a.md"}` → solo i 2 di `a.md`.

4. `test_retrieve_empty_query_raises` → `retrieve(query="")` → `ValueError` prima di chiamare OpenAI (assert: `fake_openai.calls == []`).

5. `test_retrieve_whitespace_only_query_raises` → `retrieve(query="   \n  ")` → `ValueError`.

6. `test_retrieve_top_k_larger_than_collection` → upsert 3 punti, retrieve top_k=10 → ritorna esattamente 3 senza errore.

7. `test_retrieve_match_id_is_original` → upsert un VectorPoint con id "my-sha256-like-id"; retrieve → match.id == "my-sha256-like-id" (regression del fix UUID di task #5).

### Singleton test (unit)

8. `test_get_retriever_is_singleton` → due chiamate stessa istanza.

9. `test_get_retriever_fails_without_openai_key` (monkeypatch `settings.openai_api_key = ""`) → `ValueError`.

### NO unit puri della logica interna

`Retriever.retrieve` è 3 righe di orchestrazione (`embed_texts` + `store.search`). Non c'è "logica interna" da testare unit; testare via mock sarebbe testare il mock. Tutti i test girano integration.

### Smoke manuale (parte del task, non automatizzato)

Ingest 3 file markdown + retrieve una query reale con OpenAI key valida; il top match deve essere coerente. Pattern già stabilito nel task #6.

## Risks & open questions

1. **Test `match_id_is_original` su collection già esistente**: l'ho già coperto via integration in task #5 ma è importante avere un regression test anche qui — un giorno potrei refactorare retriever e introdurre uno step di trasformazione che leak l'UUID.
2. **OpenAI rate limit nei test integration**: usiamo `FakeOpenAIClient` per tutti i test, niente call reali. Niente rate limit issue.
3. **Performance**: nessuna ottimizzazione adesso. Singolo round-trip OpenAI + singolo round-trip Qdrant è già il caso ottimale per dense-only.

## Acceptance criteria

- [ ] `apps/api/app/rag/retriever.py` con `Retriever` class + `get_retriever()` singleton.
- [ ] `apps/api/tests/test_rag/test_retriever.py` con 9 test (7 integration + 2 singleton unit-ish).
- [ ] `uv run pytest -q` su suite intera → tutto verde (no regressioni).
- [ ] `uv run ruff check . && uv run ruff format --check .` clean.
- [ ] Smoke manuale: 3 file ingestati → retrieve di una query reale → top match coerente con la query.
- [ ] `docs/ROADMAP.md` aggiornato: task #7 split in #7a (done) + #7b (backlog con condizione di attivazione "se eval mostra che serve").
