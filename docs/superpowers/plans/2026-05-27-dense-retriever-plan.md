# Dense Retriever — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementare `app/rag/retriever.py` — dense semantic retriever su Qdrant, orchestratore di `embed_texts` (task #4) + `VectorStore.search` (task #5).

**Architecture:** Singolo modulo Python con una classe `Retriever` (init: store + openai_client; metodo `retrieve(query, collection, top_k, filter)`) e una factory singleton `get_retriever()`. Niente nuove dipendenze. Tutta la logica è orchestrazione, testata via integration su Qdrant live + FakeOpenAIClient.

**Tech Stack:** Python 3.12, async OpenAI (text-embedding-3-small), AsyncQdrantClient via `app.rag.vector_store`, pytest-asyncio (mode=auto).

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `apps/api/app/rag/retriever.py` | **Create** | `Retriever` class + `get_retriever()` singleton |
| `apps/api/tests/test_rag/test_retriever.py` | **Create** | Integration test (Qdrant live + FakeOpenAIClient) + singleton unit |
| `docs/ROADMAP.md` | **Modify** | Task #7 split in #7a ✅ + #7b backlog con condizione di attivazione |

Niente dipendenze nuove, niente cambi a vector_store/ingest/config.

---

## Pre-flight check (NOT a task)

```powershell
git status                                                              # clean
docker ps --filter "name=agentic-rag-qdrant" --format "{{.Status}}"     # Up (healthy)
git branch --show-current                                               # main
```

Se Qdrant non gira: `docker compose up -d qdrant` dalla root del repo.

---

## Task 1: Retriever class skeleton + first integration test

**Files:**
- Create: `apps/api/app/rag/retriever.py`
- Create: `apps/api/tests/test_rag/test_retriever.py`

- [ ] **Step 1.1: Crea il modulo `retriever.py` con la classe (skeleton)**

Crea il file con:

```python
"""
app/rag/retriever.py
====================
Dense semantic retriever su Qdrant.

------------------------------------------------------------------------
Cosa fa
------------------------------------------------------------------------

Dato un testo in linguaggio naturale (la "query"), produce i top-k
chunk più simili dalla collection Qdrant. Workflow:

    embed(query) -> vector_store.search(...) -> list[Match]

I componenti sottostanti (embedder + vector_store) sono già stati
costruiti nei task #4 e #5; questo modulo è puro orchestratore.

------------------------------------------------------------------------
Limiti noti (importanti)
------------------------------------------------------------------------

Fa SOLO dense search. Cattura bene la somiglianza semantica di
query in linguaggio naturale, ma può sotto-performare su query
lessicali esatte (codici errore, identifier, version numbers). Quando
l'eval (task #11-12) misurerà precision@k su un golden dataset, potremo
decidere se introdurre hybrid (sparse BM25 + RRF fusion) come task #7b.

Vedi `docs/superpowers/specs/2026-05-27-dense-retriever-design.md`.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from app.config import settings
from app.rag.embedder import embed_texts
from app.rag.vector_store import Match, VectorStore, get_vector_store

logger = logging.getLogger(__name__)


class Retriever:
    """Dense semantic retriever su Qdrant.

    Costruttore prende store + openai_client espliciti — dependency
    injection-friendly per i test. In produzione `get_retriever()`
    fornisce un singleton configurato dalle settings.
    """

    def __init__(
        self,
        store: VectorStore,
        openai_client: AsyncOpenAI,
    ) -> None:
        self._store = store
        self._openai_client = openai_client

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Match]:
        """Embed query → search dense → return top-k Match.

        Args:
            query: testo in linguaggio naturale. Stringa non vuota
                (whitespace-only viene rifiutato).
            collection: nome collection Qdrant.
            top_k: numero max risultati. Default 5.
            filter: shallow equality match sul payload (es. {"source": "X"}).

        Returns:
            list[Match] ordinata per score decrescente. Vuota se la
            collection è vuota o nessun risultato matcha il filter.

        Raises:
            ValueError: se la query è vuota o whitespace-only.
            EmbedderError: se l'embedding fallisce dopo tutti i retry.
            Eccezioni vector_store propagate.
        """
        if not query.strip():
            raise ValueError("query cannot be empty")

        embeddings = await embed_texts(self._openai_client, [query])
        # embed_texts garantisce len(result) == len(input), quindi
        # l'indice [0] è sicuro.
        query_vector = embeddings[0].vector

        matches = await self._store.search(
            collection=collection,
            query=query_vector,
            top_k=top_k,
            filter=filter,
        )

        logger.info(
            "retrieve_done",
            extra={
                "collection": collection,
                "query_len": len(query),
                "top_k": top_k,
                "results": len(matches),
            },
        )
        return matches


# ---------------------------------------------------------------------------
# Factory singleton
# ---------------------------------------------------------------------------

_retriever_singleton: Retriever | None = None


def get_retriever() -> Retriever:
    """Singleton modulo-livello.

    Lazy init: prima chiamata costruisce store (via get_vector_store)
    + AsyncOpenAI (via settings.openai_api_key). Solleva ValueError
    se OPENAI_API_KEY è vuota.

    Pensato per `Depends(get_retriever)` da endpoint FastAPI (task #10).
    """
    global _retriever_singleton
    if _retriever_singleton is None:
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY non configurata. Imposta la variabile nel file .env."
            )
        _retriever_singleton = Retriever(
            store=get_vector_store(),
            openai_client=AsyncOpenAI(api_key=settings.openai_api_key),
        )
    return _retriever_singleton


__all__ = ["Retriever", "get_retriever"]
```

- [ ] **Step 1.2: Crea il test file con un primo smoke test integration**

Crea `apps/api/tests/test_rag/test_retriever.py`:

```python
"""
tests/test_rag/test_retriever.py
================================
Test del retriever dense (`app/rag/retriever.py`).

Tutti integration: il retriever è puro orchestratore (3 righe di logica
interna), quindi non c'è valore in test unit della logica — testiamo
end-to-end con Qdrant live + FakeOpenAIClient.

Eccezione: i 2 test del singleton `get_retriever()` sono unit puri
(non toccano né Qdrant né OpenAI).
"""

from __future__ import annotations

import uuid

import pytest

from app.rag.retriever import Retriever
from app.rag.vector_store import VectorPoint


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_retrieve_empty_collection_returns_empty(
    qdrant_store, fake_openai
) -> None:
    """Collection vuota → lista vuota, no errore."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")
        retriever = Retriever(store=qdrant_store, openai_client=fake_openai)

        result = await retriever.retrieve(
            query="qualsiasi cosa", collection=collection, top_k=5
        )
        assert result == []
    finally:
        await qdrant_store.delete_collection(collection)
```

- [ ] **Step 1.3: Run il test — deve passare**

```powershell
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack\apps\api
uv run pytest tests/test_rag/test_retriever.py -v
```

Expected: 1 passed.

- [ ] **Step 1.4: Ruff**

```powershell
uv run ruff format app/rag/retriever.py tests/test_rag/test_retriever.py
uv run ruff check --fix app/rag/retriever.py tests/test_rag/test_retriever.py
uv run ruff check app/rag/retriever.py tests/test_rag/test_retriever.py
```

Expected: `All checks passed!`.

- [ ] **Step 1.5: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/app/rag/retriever.py apps/api/tests/test_rag/test_retriever.py
git commit -m "feat(rag): add dense Retriever class + first integration test (M2 task #7a)"
```

---

## Task 2: Top-k ordering + filter tests

**Files:**
- Modify: `apps/api/tests/test_rag/test_retriever.py`

- [ ] **Step 2.1: Aggiungi i test di ranking + filter**

In [apps/api/tests/test_rag/test_retriever.py](../../../apps/api/tests/test_rag/test_retriever.py), aggiungi alla fine:

```python


@pytest.mark.integration
async def test_retrieve_returns_top_k_ordered(qdrant_store, fake_openai) -> None:
    """retrieve ritorna top_k ordinato per score decrescente.

    Strategia: usiamo FakeOpenAIClient.vector_for(text), che produce
    un vettore deterministico dipendente dalla lunghezza del testo
    (1536 float tutti uguali a (len(text) % 100) / 100). Costruiamo
    chunks con testi di lunghezza specifica + query con la stessa
    lunghezza del chunk "near" per garantire la similarity max.
    """
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")
        retriever = Retriever(store=qdrant_store, openai_client=fake_openai)

        # Tre chunk con testi di lunghezze diverse — il FakeOpenAIClient
        # darà vettori diversi proporzionali alla lunghezza.
        # Usiamo il fake.vector_for() per pre-calcolare i vettori
        # da upsertare (così upsert e query usano lo stesso schema).
        chunks_text = ["near", "midd" * 5, "farfarfar" * 4]
        for i, text in enumerate(chunks_text):
            await qdrant_store.upsert(
                collection,
                [
                    VectorPoint(
                        id=f"chunk-{i}",
                        vector=fake_openai.vector_for(text),
                        payload={"text": text},
                    )
                ],
            )

        # Query con la STESSA lunghezza di "near" → cosine = 1.0 per chunk-0.
        result = await retriever.retrieve(
            query="abcd",  # len=4 = len("near")
            collection=collection,
            top_k=3,
        )

        assert len(result) == 3
        assert result[0].id == "chunk-0"
        # Ordine decrescente:
        assert result[0].score >= result[1].score >= result[2].score
    finally:
        await qdrant_store.delete_collection(collection)


@pytest.mark.integration
async def test_retrieve_respects_filter(qdrant_store, fake_openai) -> None:
    """filter={'source': X} → solo punti con payload['source']==X."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")
        retriever = Retriever(store=qdrant_store, openai_client=fake_openai)

        # 4 chunk con vector identico (così il filter è l'unico
        # discriminante) ma source diverso.
        same_vec = fake_openai.vector_for("any")
        for i, source in enumerate(["a.md", "a.md", "b.md", "b.md"]):
            await qdrant_store.upsert(
                collection,
                [
                    VectorPoint(
                        id=f"chunk-{i}",
                        vector=same_vec,
                        payload={"source": source, "text": f"chunk {i}"},
                    )
                ],
            )

        result = await retriever.retrieve(
            query="any",
            collection=collection,
            top_k=10,
            filter={"source": "a.md"},
        )

        assert len(result) == 2
        assert all(m.payload["source"] == "a.md" for m in result)
    finally:
        await qdrant_store.delete_collection(collection)


@pytest.mark.integration
async def test_retrieve_top_k_larger_than_collection(qdrant_store, fake_openai) -> None:
    """top_k > numero punti → ritorna tutti i punti, no errore."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")
        retriever = Retriever(store=qdrant_store, openai_client=fake_openai)

        for i in range(3):
            await qdrant_store.upsert(
                collection,
                [
                    VectorPoint(
                        id=f"chunk-{i}",
                        vector=fake_openai.vector_for(f"text-{i}"),
                        payload={"i": i},
                    )
                ],
            )

        result = await retriever.retrieve(query="hi", collection=collection, top_k=10)
        assert len(result) == 3
    finally:
        await qdrant_store.delete_collection(collection)
```

- [ ] **Step 2.2: Run i test**

```powershell
uv run pytest tests/test_rag/test_retriever.py -v
```

Expected: 4 passed (1 da Task 1 + 3 nuovi).

- [ ] **Step 2.3: Ruff**

```powershell
uv run ruff format tests/test_rag/test_retriever.py
uv run ruff check --fix tests/test_rag/test_retriever.py
uv run ruff check tests/test_rag/test_retriever.py
```

- [ ] **Step 2.4: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/tests/test_rag/test_retriever.py
git commit -m "test(retriever): top-k ordering, filter shallow, top_k > N coverage"
```

---

## Task 3: Empty/whitespace query rejection

**Files:**
- Modify: `apps/api/tests/test_rag/test_retriever.py`

- [ ] **Step 3.1: Test che `retrieve(query="")` rifiuta ANTES di chiamare OpenAI**

Aggiungi in fondo a [apps/api/tests/test_rag/test_retriever.py](../../../apps/api/tests/test_rag/test_retriever.py):

```python


@pytest.mark.integration
async def test_retrieve_empty_query_raises_before_calling_openai(
    qdrant_store, fake_openai
) -> None:
    """Query vuota → ValueError PRIMA di toccare OpenAI (no waste)."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")
        retriever = Retriever(store=qdrant_store, openai_client=fake_openai)

        with pytest.raises(ValueError, match="query cannot be empty"):
            await retriever.retrieve(query="", collection=collection)

        # Garanzia di no-waste: niente call OpenAI.
        assert fake_openai.calls == []
    finally:
        await qdrant_store.delete_collection(collection)


@pytest.mark.integration
async def test_retrieve_whitespace_only_query_raises(qdrant_store, fake_openai) -> None:
    """Stringa solo whitespace è equivalente a vuota."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")
        retriever = Retriever(store=qdrant_store, openai_client=fake_openai)

        with pytest.raises(ValueError, match="query cannot be empty"):
            await retriever.retrieve(query="   \n  \t", collection=collection)

        assert fake_openai.calls == []
    finally:
        await qdrant_store.delete_collection(collection)
```

- [ ] **Step 3.2: Run i test**

```powershell
uv run pytest tests/test_rag/test_retriever.py -v
```

Expected: 6 passed.

- [ ] **Step 3.3: Ruff**

```powershell
uv run ruff format tests/test_rag/test_retriever.py
uv run ruff check tests/test_rag/test_retriever.py
```

- [ ] **Step 3.4: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/tests/test_rag/test_retriever.py
git commit -m "test(retriever): fail-fast on empty/whitespace query before OpenAI call"
```

---

## Task 4: Regression test on original ID preservation

**Files:**
- Modify: `apps/api/tests/test_rag/test_retriever.py`

- [ ] **Step 4.1: Test che `Match.id` ritorna l'id originale, non l'UUID Qdrant**

Aggiungi in fondo:

```python


@pytest.mark.integration
async def test_retrieve_match_id_is_original_not_uuid(qdrant_store, fake_openai) -> None:
    """Regression test del fix di task #5: Match.id deve essere l'id
    ORIGINALE passato a VectorPoint, NON l'UUID v5 derivato interno
    a Qdrant. E la chiave riservata __vp_id NON deve trapelare nel payload.
    """
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")
        retriever = Retriever(store=qdrant_store, openai_client=fake_openai)

        original_id = "sha256-mock-not-actually-a-hash-abc123"
        await qdrant_store.upsert(
            collection,
            [
                VectorPoint(
                    id=original_id,
                    vector=fake_openai.vector_for("hi"),
                    payload={"text": "hi"},
                )
            ],
        )

        result = await retriever.retrieve(query="hi", collection=collection, top_k=1)
        assert len(result) == 1
        assert result[0].id == original_id
        assert "__vp_id" not in result[0].payload
    finally:
        await qdrant_store.delete_collection(collection)
```

- [ ] **Step 4.2: Run i test**

```powershell
uv run pytest tests/test_rag/test_retriever.py -v
```

Expected: 7 passed.

- [ ] **Step 4.3: Ruff**

```powershell
uv run ruff format tests/test_rag/test_retriever.py
uv run ruff check tests/test_rag/test_retriever.py
```

- [ ] **Step 4.4: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/tests/test_rag/test_retriever.py
git commit -m "test(retriever): regression on original ID preservation through retrieve"
```

---

## Task 5: Singleton `get_retriever()` tests + smoke + ROADMAP

**Files:**
- Modify: `apps/api/tests/test_rag/test_retriever.py`
- Modify: `docs/ROADMAP.md`

- [ ] **Step 5.1: Test del singleton**

Aggiungi gli import in cima al file (se non già presenti):

```python
from app.rag.retriever import Retriever, get_retriever
```

E in fondo al file, aggiungi:

```python


# ---------------------------------------------------------------------------
# Singleton get_retriever()
# ---------------------------------------------------------------------------


def test_get_retriever_is_singleton(monkeypatch) -> None:
    """Due chiamate ritornano la stessa istanza."""
    # Reset del singleton globale (altri test potrebbero averlo già creato).
    monkeypatch.setattr("app.rag.retriever._retriever_singleton", None)
    # Assicura che la key sia presente (settings carica da .env).
    monkeypatch.setattr("app.config.settings.openai_api_key", "sk-test-fake")

    a = get_retriever()
    b = get_retriever()
    assert a is b
    assert isinstance(a, Retriever)


def test_get_retriever_fails_without_openai_key(monkeypatch) -> None:
    """OPENAI_API_KEY vuota → ValueError chiaro al primo get_retriever()."""
    monkeypatch.setattr("app.rag.retriever._retriever_singleton", None)
    monkeypatch.setattr("app.config.settings.openai_api_key", "")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_retriever()
```

- [ ] **Step 5.2: Run i test**

```powershell
uv run pytest tests/test_rag/test_retriever.py -v
```

Expected: 9 passed (7 integration + 2 singleton unit).

- [ ] **Step 5.3: Verifica suite completa (no regressioni)**

```powershell
uv run pytest -q
```

Expected: 97 + 9 = 106 passed (con il solito 1-2 warning Qdrant version).

- [ ] **Step 5.4: Smoke manuale con OpenAI reale**

```powershell
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack\apps\api

# Ingesta 3 docs nella collection smoke_retriever (re-using il pattern del task #6)
$tmp = Join-Path $env:TEMP "retriever-smoke-$(Get-Random)"
New-Item -ItemType Directory -Path $tmp | Out-Null
"# FastAPI Tutorial`n`nFastAPI is a modern web framework for building APIs with Python." | Out-File "$tmp\fastapi.md" -Encoding utf8
"# Pydantic Validation`n`nPydantic provides runtime type validation using Python type hints." | Out-File "$tmp\pydantic.md" -Encoding utf8
"# Async I/O`n`nasyncio is Python's library for writing concurrent code using async/await syntax." | Out-File "$tmp\asyncio.md" -Encoding utf8

uv run python -m app.ingest --source $tmp --collection smoke_retriever

# Ora prova il retriever con 3 query distinte. Here-string single-quoted
# (@'...'@) per evitare interpolazione PowerShell: dentro c'è codice Python
# con quote " che NON vanno escapate.
$pyScript = @'
import asyncio
from app.rag.retriever import get_retriever

async def main():
    r = get_retriever()
    queries = [
        "How do I build a web API in Python?",
        "data validation with type hints",
        "concurrency primitives",
    ]
    for q in queries:
        print("Q:", q)
        matches = await r.retrieve(query=q, collection="smoke_retriever", top_k=2)
        for m in matches:
            text_snippet = (m.payload.get("text") or "")[:60]
            print(f"  score={m.score:.3f}  source={m.payload.get('source')}  text={text_snippet}...")

asyncio.run(main())
'@
uv run python -c $pyScript

# Cleanup
Remove-Item -Recurse $tmp
uv run python -c "import asyncio; from app.rag.vector_store import get_vector_store; asyncio.run(get_vector_store().delete_collection('smoke_retriever'))"
```

Expected: per ogni query, top-2 match coerenti col tema (la query "build a web API in Python" deve mettere `fastapi.md` in cima, ecc.).

- [ ] **Step 5.5: Aggiorna `docs/ROADMAP.md`**

In [docs/ROADMAP.md](../../ROADMAP.md), sostituisci la riga del task #7:

```
7. **Retriever** (`app/rag/retriever.py`): hybrid search = vector + BM25
   (o full-text di Postgres), con metadata filters.
```

con:

```
7a. ✅ **Retriever (dense)** (`app/rag/retriever.py`): orchestratore
    `embed_texts` + `vector_store.search`. Metadata filters via dict
    shallow (AND fra chiavi). Singleton `get_retriever()`. 9 test verdi.
    Smoke con OpenAI reale verificato.
7b. ⚪ **Retriever (hybrid sparse+dense+RRF)** — backlog. Da attivare
    SOLO se l'eval (task #11-12) mostra che dense fallisce su query
    lessicali (codici, identifier). Richiederebbe: dep
    `qdrant-client[fastembed]`, collection multi-vector, re-ingest,
    breaking change vector_store. Vedi memo nello spec
    `docs/superpowers/specs/2026-05-27-dense-retriever-design.md`.
```

- [ ] **Step 5.6: Commit finale**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/tests/test_rag/test_retriever.py docs/ROADMAP.md
git commit -m "feat(rag): get_retriever singleton + smoke verified; mark M2 #7a done, #7b backlog"
```

- [ ] **Step 5.7: Verifica git log**

```powershell
git log --oneline -7
```

Expected: 5 commit nuovi su HEAD (uno per task).

---

## Spec coverage map

| Spec requirement | Task |
|---|---|
| `Retriever` class con `__init__(store, openai_client)` | Task 1 |
| `Retriever.retrieve(query, collection, top_k=5, filter=None)` | Task 1 |
| Embed query via `embed_texts` (riuso task #4) | Task 1 |
| Search via `vector_store.search` (riuso task #5) | Task 1 |
| Query vuota → ValueError prima di OpenAI | Task 3 |
| Query whitespace-only → ValueError | Task 3 |
| Filter pass-through con semantica AND | Task 2 |
| Top-k ordering by score decrescente | Task 2 |
| top_k > N punti → ritorna tutti | Task 2 |
| Collection vuota → `[]` | Task 1 |
| `Match.id` è id originale (non UUID interno) | Task 4 |
| `__vp_id` non trapela nel payload | Task 4 |
| `get_retriever()` singleton lazy | Task 5 |
| `get_retriever()` ValueError se OPENAI_API_KEY vuota | Task 5 |
| Smoke manuale con OpenAI reale | Task 5 |
| ROADMAP aggiornato con split 7a/7b | Task 5 |
| Limiti noti documentati nel docstring del Retriever | Task 1 |

---

## Final acceptance check

A fine Task 5, eseguire:

```powershell
# 1. Suite verde
uv run --directory apps/api pytest -q

# 2. Linter clean
uv run --directory apps/api ruff check .
uv run --directory apps/api ruff format --check .

# 3. Import + singleton smoke
uv run --directory apps/api python -c "from app.rag.retriever import Retriever, get_retriever; print('OK:', get_retriever().__class__.__name__)"
```

Expected:
- 106 test verdi
- 0 ruff errors
- `OK: Retriever`
