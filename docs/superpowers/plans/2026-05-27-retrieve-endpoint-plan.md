# POST /retrieve Endpoint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Esporre via `POST /retrieve` il `Retriever` dense costruito in task #7a, secondo il design [docs/superpowers/specs/2026-05-27-retrieve-endpoint-design.md](../specs/2026-05-27-retrieve-endpoint-design.md).

**Architecture:** Singolo endpoint nuovo in `app/main.py` accanto a `/classify`. Schemi Pydantic inline. Dependency injection del retriever via `Depends(get_retriever)`. Test unit-only via `TestClient` + `FakeRetriever` iniettato con `app.dependency_overrides` (pattern del repo).

**Tech Stack:** FastAPI, Pydantic v2, pytest + FastAPI TestClient.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `apps/api/app/main.py` | **Modify** | Aggiungi 3 schemi (RetrieveRequest, RetrievedChunk, RetrieveResponse) + endpoint `POST /retrieve` |
| `apps/api/tests/conftest.py` | **Modify** | Aggiungi `FakeRetriever` + fixture `fake_retriever` |
| `apps/api/tests/test_retrieve.py` | **Create** | 11 test (happy path, validation, errori upstream) |
| `docs/ROADMAP.md` | **Modify** | Split task #10 in #10a ✅ + #10b ⚪ |

---

## Pre-flight check (NOT a task)

```powershell
git status                                                              # clean
docker ps --filter "name=agentic-rag-qdrant" --format "{{.Status}}"     # Up (healthy) — serve solo per lo smoke
git branch --show-current                                               # main
```

---

## Task 1: Schemi Pydantic in main.py

**Files:**
- Modify: `apps/api/app/main.py`

- [ ] **Step 1.1: Aggiungi gli import necessari in `main.py`**

In [apps/api/app/main.py](../../../apps/api/app/main.py), nella sezione import (in alto, dopo i `from ...` esistenti), aggiungi:

```python
from typing import Any

from qdrant_client.http.exceptions import UnexpectedResponse

from app.rag.retriever import Retriever, get_retriever
```

Se `typing.Any` non c'è già, lo aggiungi. Se c'è già, non duplicarlo.

- [ ] **Step 1.2: Aggiungi i 3 schemi Pydantic in `main.py`**

Dopo gli schemi esistenti (`HealthResponse`, ecc.) e prima della definizione `app = FastAPI(...)`, aggiungi:

```python
# ============================================================================
# Schemi per /retrieve (M2 task #10a)
# ============================================================================
# RetrievedChunk è strutturalmente uguale a app.rag.vector_store.Match ma
# vive come schema HTTP separato per evitare leak di refactor interni nel
# response JSON pubblico.


class RetrieveRequest(BaseModel):
    """Request per POST /retrieve."""

    query: str = Field(
        min_length=1,
        description="Domanda in linguaggio naturale.",
    )
    collection: str | None = Field(
        default=None,
        description="Nome collection Qdrant. Default: settings.qdrant_collection_name.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Numero max di risultati.",
    )
    filter: dict[str, Any] | None = Field(
        default=None,
        description="Shallow equality match sul payload (AND fra chiavi).",
    )


class RetrievedChunk(BaseModel):
    """Singolo chunk recuperato (id originale + score + payload)."""

    id: str
    score: float
    payload: dict[str, Any]


class RetrieveResponse(BaseModel):
    """Response per POST /retrieve. chunks ordinati per score decrescente."""

    chunks: list[RetrievedChunk]
```

- [ ] **Step 1.3: Verifica che il modulo importi correttamente**

```powershell
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack\apps\api
uv run python -c "from app.main import RetrieveRequest, RetrievedChunk, RetrieveResponse; print('OK')"
```

Expected output: `OK`

- [ ] **Step 1.4: Ruff**

```powershell
uv run ruff format app/main.py
uv run ruff check --fix app/main.py
uv run ruff check app/main.py
```

Expected: `All checks passed!`

- [ ] **Step 1.5: Verifica suite ancora verde**

```powershell
uv run pytest -q
```

Expected: 106 passed (no regressioni).

- [ ] **Step 1.6: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/app/main.py
git commit -m "feat(api): add /retrieve request/response schemas (M2 task #10a prep)"
```

---

## Task 2: FakeRetriever fixture in conftest

**Files:**
- Modify: `apps/api/tests/conftest.py`

- [ ] **Step 2.1: Aggiungi `FakeRetriever` + fixture in fondo a `conftest.py`**

In [apps/api/tests/conftest.py](../../../apps/api/tests/conftest.py), aggiungi alla fine del file (dopo le altre fixture):

```python


# ============================================================================
# FakeRetriever (per i test dell'endpoint /retrieve)
# ============================================================================
# Stesso pattern di FakeAnthropicClient: si aggancia tramite
# app.dependency_overrides[get_retriever] = lambda: fake.


from app.rag.retriever import get_retriever
from app.rag.vector_store import Match


class FakeRetriever:
    """Test double per app.rag.retriever.Retriever.

    Configurabile dal test:
        fake.matches_to_return = [Match(id="x", score=0.9, payload={...})]
        fake.error_to_raise = UnexpectedResponse(...)

    Espone la stessa superficie usata dall'endpoint: retrieve(...).
    """

    def __init__(self) -> None:
        self.matches_to_return: list[Match] = []
        self.error_to_raise: Exception | None = None
        # Log delle chiamate, per asserzioni dai test.
        self.calls: list[dict[str, Any]] = []

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Match]:
        self.calls.append(
            {
                "query": query,
                "collection": collection,
                "top_k": top_k,
                "filter": filter,
            }
        )
        if self.error_to_raise is not None:
            raise self.error_to_raise
        return self.matches_to_return


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    """Crea FakeRetriever e lo aggancia alla dependency get_retriever.

    Yield-fixture: setup + cleanup di dependency_overrides.
    """
    fake = FakeRetriever()
    app.dependency_overrides[get_retriever] = lambda: fake
    yield fake
    app.dependency_overrides.clear()
```

Nota: gli import di `get_retriever` e `Match` sono inline qui (non in cima) per separare visivamente le due sezioni di fixture (anthropic vs retriever). Ruff potrebbe segnalare E402; se lo fa, sposta gli import in cima al file accanto agli altri.

- [ ] **Step 2.2: Verifica che la fixture sia importabile**

```powershell
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack\apps\api
uv run python -c "from tests.conftest import FakeRetriever; f = FakeRetriever(); print('OK', f.matches_to_return)"
```

Expected: `OK []`

- [ ] **Step 2.3: Ruff**

```powershell
uv run ruff format tests/conftest.py
uv run ruff check --fix tests/conftest.py
uv run ruff check tests/conftest.py
```

Se ruff segnala E402 sugli import in mezzo file, sposta `from app.rag.retriever import get_retriever` e `from app.rag.vector_store import Match` in cima al file accanto agli altri import.

- [ ] **Step 2.4: Verifica suite ancora verde**

```powershell
uv run pytest -q
```

Expected: 106 passed (no nuove regressioni).

- [ ] **Step 2.5: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/tests/conftest.py
git commit -m "test: add FakeRetriever fixture for /retrieve endpoint tests"
```

---

## Task 3: Endpoint happy path

**Files:**
- Modify: `apps/api/app/main.py`
- Create: `apps/api/tests/test_retrieve.py`

- [ ] **Step 3.1: Test happy path (rosso prima)**

Crea `apps/api/tests/test_retrieve.py`:

```python
"""
tests/test_retrieve.py
======================
Test dell'endpoint `POST /retrieve` (M2 task #10a).

Tutti unit: il retriever (componente interno) è già coperto da
integration in tests/test_rag/test_retriever.py. Qui testiamo solo
lo strato HTTP — request validation, error mapping, response shape.
Il retriever è sostituito da FakeRetriever via app.dependency_overrides.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.rag.vector_store import Match


def test_retrieve_happy_path(client: TestClient, fake_retriever) -> None:
    """Query valida + fake che ritorna 2 Match → response 200 con 2 chunks."""
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
```

- [ ] **Step 3.2: Run il test — deve FALLIRE (endpoint non esiste)**

```powershell
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack\apps\api
uv run pytest tests/test_retrieve.py::test_retrieve_happy_path -v
```

Expected: 404 Not Found (route inesistente) oppure assertion failure su 200.

- [ ] **Step 3.3: Implementa l'endpoint `POST /retrieve` in main.py**

In [apps/api/app/main.py](../../../apps/api/app/main.py), dopo l'endpoint `/classify` (cerca `@app.post("/classify"...)`), aggiungi:

```python
@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    request: RetrieveRequest,
    retriever: Retriever = Depends(get_retriever),
) -> RetrieveResponse:
    """Dense semantic retrieval da Qdrant.

    Vedi `docs/superpowers/specs/2026-05-27-retrieve-endpoint-design.md`.
    """
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
            detail=f"vector store error: {exc.reason_phrase or exc}",
        ) from exc

    return RetrieveResponse(
        chunks=[RetrievedChunk(**m.model_dump()) for m in matches]
    )
```

- [ ] **Step 3.4: Run il test — deve PASSARE**

```powershell
uv run pytest tests/test_retrieve.py::test_retrieve_happy_path -v
```

Expected: 1 passed.

- [ ] **Step 3.5: Ruff**

```powershell
uv run ruff format app/main.py tests/test_retrieve.py
uv run ruff check --fix app/main.py tests/test_retrieve.py
uv run ruff check app/main.py tests/test_retrieve.py
```

- [ ] **Step 3.6: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/app/main.py apps/api/tests/test_retrieve.py
git commit -m "feat(api): POST /retrieve endpoint with happy-path test (M2 task #10a)"
```

---

## Task 4: Collection default + provided + filter pass-through + top_k default

**Files:**
- Modify: `apps/api/tests/test_retrieve.py`

- [ ] **Step 4.1: 4 nuovi test che verificano i parametri**

Aggiungi alla fine di [apps/api/tests/test_retrieve.py](../../../apps/api/tests/test_retrieve.py):

```python


def test_retrieve_uses_default_collection(client: TestClient, fake_retriever) -> None:
    """Request senza 'collection' → fake riceve settings.qdrant_collection_name."""
    fake_retriever.matches_to_return = []
    response = client.post("/retrieve", json={"query": "x"})
    assert response.status_code == 200
    assert len(fake_retriever.calls) == 1
    assert fake_retriever.calls[0]["collection"] == settings.qdrant_collection_name


def test_retrieve_uses_provided_collection(client: TestClient, fake_retriever) -> None:
    """Request con 'collection': 'x' → fake riceve 'x'."""
    fake_retriever.matches_to_return = []
    response = client.post("/retrieve", json={"query": "x", "collection": "custom"})
    assert response.status_code == 200
    assert fake_retriever.calls[0]["collection"] == "custom"


def test_retrieve_passes_filter(client: TestClient, fake_retriever) -> None:
    """'filter': {...} → fake riceve filter."""
    fake_retriever.matches_to_return = []
    response = client.post(
        "/retrieve",
        json={"query": "x", "filter": {"source": "intro.md"}},
    )
    assert response.status_code == 200
    assert fake_retriever.calls[0]["filter"] == {"source": "intro.md"}


def test_retrieve_default_top_k_is_5(client: TestClient, fake_retriever) -> None:
    """Request senza 'top_k' → fake riceve top_k=5."""
    fake_retriever.matches_to_return = []
    response = client.post("/retrieve", json={"query": "x"})
    assert response.status_code == 200
    assert fake_retriever.calls[0]["top_k"] == 5
```

- [ ] **Step 4.2: Run i test**

```powershell
uv run pytest tests/test_retrieve.py -v
```

Expected: 5 passed (1 da Task 3 + 4 nuovi).

- [ ] **Step 4.3: Ruff**

```powershell
uv run ruff format tests/test_retrieve.py
uv run ruff check tests/test_retrieve.py
```

- [ ] **Step 4.4: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/tests/test_retrieve.py
git commit -m "test(retrieve): collection/filter/top_k parameter pass-through"
```

---

## Task 5: Validation 422 (empty query, top_k bounds)

**Files:**
- Modify: `apps/api/tests/test_retrieve.py`

- [ ] **Step 5.1: 3 test 422 (validation Pydantic)**

Aggiungi:

```python


def test_retrieve_empty_query_422(client: TestClient, fake_retriever) -> None:
    """query='' → 422 (Pydantic min_length=1). Fake NON viene chiamato."""
    response = client.post("/retrieve", json={"query": ""})
    assert response.status_code == 422
    assert fake_retriever.calls == []


def test_retrieve_top_k_zero_422(client: TestClient, fake_retriever) -> None:
    """top_k=0 → 422 (Pydantic ge=1)."""
    response = client.post("/retrieve", json={"query": "x", "top_k": 0})
    assert response.status_code == 422
    assert fake_retriever.calls == []


def test_retrieve_top_k_too_large_422(client: TestClient, fake_retriever) -> None:
    """top_k=1000 → 422 (Pydantic le=50)."""
    response = client.post("/retrieve", json={"query": "x", "top_k": 1000})
    assert response.status_code == 422
    assert fake_retriever.calls == []
```

- [ ] **Step 5.2: Run i test**

```powershell
uv run pytest tests/test_retrieve.py -v
```

Expected: 8 passed.

- [ ] **Step 5.3: Ruff + commit**

```powershell
uv run ruff format tests/test_retrieve.py
uv run ruff check tests/test_retrieve.py
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/tests/test_retrieve.py
git commit -m "test(retrieve): 422 on empty query and top_k out of [1,50]"
```

---

## Task 6: Upstream error mapping (404 + 502)

**Files:**
- Modify: `apps/api/tests/test_retrieve.py`

- [ ] **Step 6.1: 2 test sugli errori upstream**

Aggiungi:

```python


def _make_unexpected_response(status_code: int) -> UnexpectedResponse:
    """Helper: costruisce un UnexpectedResponse minimo per i test.

    Il costruttore richiede (status_code, reason_phrase, content, headers).
    """
    return UnexpectedResponse(
        status_code=status_code,
        reason_phrase="Mock Error",
        content=b"mock body",
        headers={},
    )


def test_retrieve_collection_not_found_404(client: TestClient, fake_retriever) -> None:
    """Qdrant ritorna 404 (collection inesistente) → endpoint risponde 404."""
    fake_retriever.error_to_raise = _make_unexpected_response(404)

    response = client.post(
        "/retrieve",
        json={"query": "x", "collection": "does-not-exist"},
    )
    assert response.status_code == 404
    body = response.json()
    assert "does-not-exist" in body["detail"]


def test_retrieve_qdrant_500_returns_502(client: TestClient, fake_retriever) -> None:
    """Qdrant ritorna 500 (errore generico upstream) → endpoint risponde 502."""
    fake_retriever.error_to_raise = _make_unexpected_response(500)

    response = client.post("/retrieve", json={"query": "x"})
    assert response.status_code == 502
    assert "vector store error" in response.json()["detail"]
```

- [ ] **Step 6.2: Run i test**

```powershell
uv run pytest tests/test_retrieve.py -v
```

Expected: 10 passed.

- [ ] **Step 6.3: Ruff + commit**

```powershell
uv run ruff format tests/test_retrieve.py
uv run ruff check tests/test_retrieve.py
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/tests/test_retrieve.py
git commit -m "test(retrieve): map Qdrant UnexpectedResponse 404 -> 404, others -> 502"
```

---

## Task 7: Match shape preservation + smoke manuale + ROADMAP

**Files:**
- Modify: `apps/api/tests/test_retrieve.py`
- Modify: `docs/ROADMAP.md`

- [ ] **Step 7.1: Test di preservazione shape (id originale + payload nested)**

Aggiungi:

```python


def test_retrieve_match_shape_preserved(client: TestClient, fake_retriever) -> None:
    """Il Match dal retriever viene serializzato field-by-field nel JSON
    di response, incluso un payload con chiavi nested arbitrarie.

    Verifica:
    - Match.id originale (anche se sha256-like, NON UUID) → response intatto.
    - payload nested preservato.
    - score float preservato.
    """
    nested_payload = {
        "text": "the quick brown fox",
        "source": "tutorial/animals.md",
        "heading": "Mammals > Foxes",
        "position": 7,
        "token_count": 42,
        "extra": {"nested_key": "nested_value", "list": [1, 2, 3]},
    }
    fake_retriever.matches_to_return = [
        Match(
            id="sha256-original-not-uuid-format-abc123",
            score=0.7654321,
            payload=nested_payload,
        ),
    ]

    response = client.post("/retrieve", json={"query": "fox"})
    assert response.status_code == 200
    chunk = response.json()["chunks"][0]
    assert chunk["id"] == "sha256-original-not-uuid-format-abc123"
    assert chunk["score"] == 0.7654321
    assert chunk["payload"] == nested_payload
```

- [ ] **Step 7.2: Run i test**

```powershell
uv run pytest tests/test_retrieve.py -v
```

Expected: 11 passed.

- [ ] **Step 7.3: Run suite intera**

```powershell
uv run pytest -q
```

Expected: 106 + 11 = 117 passed.

- [ ] **Step 7.4: Ruff su tutto il codice toccato**

```powershell
uv run ruff format app/ tests/
uv run ruff check app/ tests/
```

Expected: `All checks passed!`

- [ ] **Step 7.5: Smoke manuale con corpus reale + curl**

NOTA: serve il backend FastAPI running. Se non gira già, lancialo in un terminale a parte:

```powershell
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack\apps\api
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

In un altro terminale, dopo avere ingestato un piccolo corpus:

```powershell
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack\apps\api

# Crea corpus minimo (UTF-8 no BOM)
$tmp = Join-Path $env:TEMP "retrieve-endpoint-smoke-$(Get-Random)"
New-Item -ItemType Directory -Path $tmp | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText("$tmp\fastapi.md", "# FastAPI`n`nFastAPI is a modern Python web framework.", $utf8)
[System.IO.File]::WriteAllText("$tmp\pydantic.md", "# Pydantic`n`nPydantic provides type validation.", $utf8)

uv run python -m app.ingest --source $tmp --collection smoke_endpoint

# Chiamata curl al nuovo endpoint
$body = '{"query": "modern Python web framework", "collection": "smoke_endpoint", "top_k": 2}'
curl -X POST http://127.0.0.1:8000/retrieve -H "Content-Type: application/json" -d $body

# Cleanup
Remove-Item -Recurse $tmp
uv run python -c "import asyncio; from app.rag.vector_store import get_vector_store; asyncio.run(get_vector_store().delete_collection('smoke_endpoint'))" 2>&1 | Out-Null
```

Expected curl output: JSON con `chunks` non vuoto, top result è `fastapi.md`, score plausibile.

Verifica anche su browser: http://localhost:8000/docs deve mostrare `POST /retrieve` con il suo schema.

- [ ] **Step 7.6: Aggiorna `docs/ROADMAP.md`**

In [docs/ROADMAP.md](../../ROADMAP.md), cerca la riga del task #10:

```
10. **Endpoint** `POST /retrieve`: `{query, top_k, filters}` →
    `{chunks: [...], citations: [...]}`. Schema Pydantic.
```

Sostituisci con:

```
10a. ✅ **Endpoint `POST /retrieve` (dense-only)**: schemi Pydantic
     (RetrieveRequest/RetrievedChunk/RetrieveResponse) inline in
     app/main.py. Dependency injection del retriever via
     `Depends(get_retriever)`. Validazione: query non-vuota, top_k in
     [1, 50]. Error mapping: Qdrant 404 → HTTP 404 con collection
     name nel detail; altri Qdrant errors → HTTP 502. Default
     collection da settings se omesso. 11 test unit verdi (TestClient
     + FakeRetriever via dependency_overrides). Smoke con curl
     verificato.
10b. ⚪ **Endpoint `POST /retrieve` arricchito** — backlog. Quando
     #8 reranker e #9 citation builder saranno fatti, modificheranno
     questo endpoint per aggiungere reranker step + arricchire la
     response con `citations` strutturate.
```

- [ ] **Step 7.7: Commit finale**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/tests/test_retrieve.py docs/ROADMAP.md
git commit -m "feat(api): preserve full Match shape in /retrieve response; mark M2 #10a done"
```

- [ ] **Step 7.8: Verifica git log**

```powershell
git log --oneline -10
```

Expected: 7 commit nuovi (uno per task + lo spec di prima).

---

## Spec coverage map

| Spec requirement | Task |
|---|---|
| RetrieveRequest (query, collection, top_k, filter) | Task 1 |
| RetrievedChunk schema | Task 1 |
| RetrieveResponse schema | Task 1 |
| Endpoint POST /retrieve registrato | Task 3 |
| Dependency injection Retriever via Depends | Task 3 |
| collection default da settings | Task 4 (test) + Task 3 (impl) |
| collection custom pass-through | Task 4 |
| filter pass-through | Task 4 |
| top_k default 5 | Task 4 |
| 422 su query vuota | Task 5 |
| 422 su top_k=0 e top_k>50 | Task 5 |
| 404 su collection inesistente | Task 6 |
| 502 su altri errori Qdrant | Task 6 |
| Match.id originale preservato (non UUID) | Task 7 |
| payload nested preservato | Task 7 |
| FakeRetriever via app.dependency_overrides | Task 2 |
| Smoke manuale con corpus reale | Task 7 |
| ROADMAP split 10a/10b | Task 7 |

---

## Final acceptance check

A fine Task 7, eseguire:

```powershell
# 1. Suite verde
uv run --directory apps/api pytest -q

# 2. Linter clean
uv run --directory apps/api ruff check .
uv run --directory apps/api ruff format --check .

# 3. /retrieve esposto in OpenAPI
uv run --directory apps/api python -c "from app.main import app; print([r.path for r in app.routes if 'retrieve' in str(r.path)])"
```

Expected:
- 117 test verdi
- 0 ruff errors
- `['/retrieve']`
