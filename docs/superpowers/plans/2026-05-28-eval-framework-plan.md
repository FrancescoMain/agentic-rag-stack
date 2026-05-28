# RAG Eval Framework — Implementation Plan (M2 #11 + #12 + #13)

> **For agentic workers:** Implement task-by-task with checkpoints. Step syntax uses `- [ ]`. Design ref: [docs/superpowers/specs/2026-05-28-eval-framework-design.md](../specs/2026-05-28-eval-framework-design.md).

**Goal:** Costruire il framework di eval MVP della pipeline RAG. A fine plan: collection `fastapi_docs` popolata, dataset golden con ≥ 20 entries curate, evaluator + runner funzionanti, DoD M2 `precision@5 ≥ 0.7` verificato.

**Tech stack:** Python 3.12, Pydantic v2, typer, pytest. Niente nuove dipendenze (typer è già installato per `app.ingest`).

---

## File structure (riferimento)

| File | Action | Task |
|---|---|---|
| `apps/api/app/evals/__init__.py` | Create (empty) | T1 |
| `apps/api/app/evals/dataset.py` | Create | T2 |
| `apps/api/app/evals/golden_datasets/fastapi_docs.jsonl` | Create | T4 |
| `apps/api/app/evals/tools/__init__.py` | Create (empty) | T3 |
| `apps/api/app/evals/tools/explore.py` | Create | T3 |
| `apps/api/app/evals/evaluators/__init__.py` | Create (empty) | T5 |
| `apps/api/app/evals/evaluators/precision_at_k.py` | Create | T5 |
| `apps/api/app/evals/runners/__init__.py` | Create (empty) | T6 |
| `apps/api/app/evals/runners/run_regression.py` | Create | T6 |
| `apps/api/tests/test_evals/__init__.py` | Create (empty) | T2 |
| `apps/api/tests/test_evals/test_dataset.py` | Create | T2 |
| `apps/api/tests/test_evals/test_precision_at_k.py` | Create | T5 |
| `apps/api/tests/test_evals/test_run_regression.py` | Create | T6 |
| `apps/api/README.md` | Modify (eval setup section) | T1 |
| `docs/ROADMAP.md` | Modify (mark #11/#12/#13 done) | T7 |

---

## Pre-flight (NOT a task)

```bash
cd /Users/francesco.cesarano/agentic-rag-stack
git status                                                         # clean
docker ps --filter "name=qdrant" --format "{{.Status}}"            # Up (healthy)
git branch --show-current                                          # main
curl -s http://localhost:6333/collections | python3 -m json.tool   # {"collections": []}
```

---

## Task 1: Scaffolding + corpus FastAPI ingestion (#11.0)

**Goal:** preparare la struttura di cartelle `app/evals/...` e popolare la collection `fastapi_docs` con i markdown reali dei docs di FastAPI.

**Files:**
- Create: `apps/api/app/evals/__init__.py`
- Modify: `apps/api/README.md` (sezione "Eval setup")

- [ ] **Step 1.1: Verifica che la cartella `app/evals/` non esista o sia vuota**

```bash
ls /Users/francesco.cesarano/agentic-rag-stack/apps/api/app/evals/ 2>&1 || echo "non esiste"
```

Se esiste con contenuto, fermarsi e capire perché.

- [ ] **Step 1.2: Crea `app/evals/__init__.py`**

File vuoto (è solo per marcarlo come package Python).

- [ ] **Step 1.3: Clone shallow dei FastAPI docs**

```bash
cd ~
git clone --depth 1 https://github.com/fastapi/fastapi.git fastapi-docs-source 2>&1 | tail -5
ls fastapi-docs-source/docs/en/docs | head -10
```

Expected: clona ~50MB, vedi una lista di markdown (`index.md`, `tutorial/`, ecc.).

- [ ] **Step 1.4: Ingest del corpus reale**

```bash
cd /Users/francesco.cesarano/agentic-rag-stack/apps/api
uv run python -m app.ingest \
  --source ~/fastapi-docs-source/docs/en/docs \
  --collection fastapi_docs 2>&1 | tail -20
```

Expected: `OK: N files ingested (X skipped). Y chunks, Z tokens.` con N ~ 150, Y ~ 1500-2000.

⚠️ Tempi attesi: 30-60s per gli embedding via OpenAI API. Costo: ~$0.04 (text-embedding-3-small @ $0.02/1M tokens × ~2M tokens).

- [ ] **Step 1.5: Verifica collection popolata**

```bash
curl -s http://localhost:6333/collections/fastapi_docs | python3 -m json.tool
```

Expected: `status: green`, `points_count: ~1500-2000`.

- [ ] **Step 1.6: Aggiungi sezione "Eval setup" al `apps/api/README.md`**

In coda al README di `apps/api`, aggiungi sotto un nuovo h2:

```markdown
## Eval setup (M2 #11.0)

Per eseguire l'eval framework serve la collection `fastapi_docs` popolata con i docs reali di FastAPI:

\`\`\`bash
# 1. Clone shallow (~50MB)
cd ~
git clone --depth 1 https://github.com/fastapi/fastapi.git fastapi-docs-source

# 2. Avvia Qdrant
cd /Users/francesco.cesarano/agentic-rag-stack
docker-compose up -d qdrant   # o `docker compose up -d qdrant` se hai il plugin

# 3. Ingest (~30-60s, ~$0.04 di embeddings)
cd apps/api
uv run python -m app.ingest \
  --source ~/fastapi-docs-source/docs/en/docs \
  --collection fastapi_docs

# 4. Verifica
curl -s http://localhost:6333/collections/fastapi_docs | python3 -m json.tool
\`\`\`

La collection sopravvive ai docker restart (volume `qdrant_storage`). Va re-ingestata
solo se cambi chunker config o aggiorni i docs source.
```

- [ ] **Step 1.7: Commit (NO codice in src/, solo scaffolding + doc)**

```bash
cd /Users/francesco.cesarano/agentic-rag-stack
git add apps/api/app/evals/__init__.py apps/api/README.md
git commit -m "chore(evals): scaffold app/evals package + document #11.0 corpus ingest"
```

---

## Task 2: Golden dataset schema + loader

**Files:**
- Create: `apps/api/app/evals/dataset.py`
- Create: `apps/api/tests/test_evals/__init__.py`
- Create: `apps/api/tests/test_evals/test_dataset.py`

- [ ] **Step 2.1: Scrivi i test prima (TDD)**

Crea `apps/api/tests/test_evals/test_dataset.py`:

```python
"""
tests/test_evals/test_dataset.py
================================
Test dello schema + loader del golden dataset (M2 task #11).

Tutti unit: lavoriamo su tmp file .jsonl, niente Qdrant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evals.dataset import GoldenItem, load_golden_dataset


def test_load_happy_path(tmp_path: Path) -> None:
    """2 righe JSONL valide → 2 GoldenItem caricati."""
    p = tmp_path / "ds.jsonl"
    p.write_text(
        '{"query": "q1", "expected_chunk_id": "id1"}\n'
        '{"query": "q2", "expected_chunk_id": "id2"}\n'
    )
    items = load_golden_dataset(p)
    assert len(items) == 2
    assert items[0] == GoldenItem(query="q1", expected_chunk_id="id1")
    assert items[1].query == "q2"


def test_load_skips_blank_lines(tmp_path: Path) -> None:
    """Righe vuote o solo-spazi ignorate."""
    p = tmp_path / "ds.jsonl"
    p.write_text(
        '{"query": "q1", "expected_chunk_id": "id1"}\n'
        "\n"
        "   \n"
        '{"query": "q2", "expected_chunk_id": "id2"}\n'
    )
    items = load_golden_dataset(p)
    assert len(items) == 2


def test_load_raises_on_invalid_json(tmp_path: Path) -> None:
    """Linea non-JSON → ValueError con num riga."""
    p = tmp_path / "ds.jsonl"
    p.write_text(
        '{"query": "ok", "expected_chunk_id": "x"}\n'
        "not-a-json\n"
    )
    with pytest.raises(ValueError, match="line 2"):
        load_golden_dataset(p)


def test_load_raises_on_missing_field(tmp_path: Path) -> None:
    """Linea senza expected_chunk_id → ValueError con num riga."""
    p = tmp_path / "ds.jsonl"
    p.write_text('{"query": "missing-id"}\n')
    with pytest.raises(ValueError, match="line 1"):
        load_golden_dataset(p)
```

- [ ] **Step 2.2: Run test — devono FALLIRE (modulo non esiste)**

```bash
cd /Users/francesco.cesarano/agentic-rag-stack/apps/api
mkdir -p tests/test_evals
touch tests/test_evals/__init__.py
uv run pytest tests/test_evals/test_dataset.py -v 2>&1 | tail -10
```

Expected: `ImportError: No module named 'app.evals.dataset'`.

- [ ] **Step 2.3: Implementa `app/evals/dataset.py`**

```python
"""
app/evals/dataset.py
====================
Schema + loader del golden dataset RAG (M2 task #11).

------------------------------------------------------------------------
Cos'è un "golden dataset" in RAG eval
------------------------------------------------------------------------
Una collezione curata a mano di coppie (query, expected_chunk_id) che
funge da "verità rivelata" contro cui misurare il retriever. Per
ciascuna query un essere umano ha deciso quale chunk del corpus è il
miglior match. Il retriever è "buono" nella misura in cui ritrova
quel chunk fra i suoi top-k.

Formato file: JSONL (1 riga = 1 entry). Vedi design doc per le ragioni.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class GoldenItem(BaseModel):
    """Singola entry del golden dataset.

    `expected_chunk_id` è l'`id` SHA256-derived prodotto dal chunker
    (vedi `app/rag/chunker.py`). MVP: un solo expected per query;
    se serviranno casi multi-doc valutiamo `list[str]`.
    """

    query: str = Field(min_length=1)
    expected_chunk_id: str = Field(min_length=1)


def load_golden_dataset(path: Path) -> list[GoldenItem]:
    """Legge un golden dataset .jsonl.

    Una riga = un JSON oggetto = un GoldenItem.
    Righe vuote / solo-spazi → ignorate (utile per organizzazione
    visiva del file, raggruppando entries per topic con righe vuote).

    Raises:
        ValueError: se una riga non è JSON valido o non rispetta lo
            schema GoldenItem. Il messaggio include il numero di riga
            per facilitare il fix.
    """
    items: list[GoldenItem] = []
    with path.open() as f:
        for n, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                items.append(GoldenItem.model_validate_json(line))
            except (ValidationError, ValueError) as exc:
                raise ValueError(f"line {n}: {exc}") from exc
    return items
```

- [ ] **Step 2.4: Run test — devono PASSARE**

```bash
uv run pytest tests/test_evals/test_dataset.py -v 2>&1 | tail -10
```

Expected: 4 passed.

- [ ] **Step 2.5: Ruff**

```bash
uv run ruff format app/evals tests/test_evals
uv run ruff check app/evals tests/test_evals
```

- [ ] **Step 2.6: Commit**

```bash
cd /Users/francesco.cesarano/agentic-rag-stack
git add apps/api/app/evals/dataset.py apps/api/tests/test_evals/__init__.py apps/api/tests/test_evals/test_dataset.py
git commit -m "feat(evals): GoldenItem schema + load_golden_dataset (M2 task #11)"
```

---

## Task 3: Curation tool (`explore.py`)

**Files:**
- Create: `apps/api/app/evals/tools/__init__.py`
- Create: `apps/api/app/evals/tools/explore.py`

Helper CLI per facilitare la cura manuale: dato una query, mostra i top-N chunk del corpus con id+source+heading+snippet. Francesco copia l'id giusto nel dataset.

- [ ] **Step 3.1: Crea `app/evals/tools/__init__.py`** (vuoto)

- [ ] **Step 3.2: Implementa `app/evals/tools/explore.py`**

```python
"""
app/evals/tools/explore.py
==========================
CLI di curation: dato una query, mostra i top-N chunk del corpus.

Uso: Francesco scrive una query, il tool gli mostra cosa il retriever
recupera per quella query, lui copia l'id del chunk che ritiene
"il miglior match" nel golden dataset.

------------------------------------------------------------------------
Perché esiste questo strumento
------------------------------------------------------------------------
Senza, per ogni query del dataset bisognerebbe:
  - aprire la dashboard Qdrant a mano
  - oppure scrivere uno script ad hoc
  - oppure usare curl POST /retrieve (verboso)

Questo è il workflow "tight loop" per la curation.
"""

from __future__ import annotations

import asyncio

import typer

from app.config import settings
from app.rag.retriever import get_retriever

app = typer.Typer(name="evals-explore", add_completion=False)


async def _explore(query: str, collection: str, top_k: int) -> None:
    retriever = get_retriever()
    matches = await retriever.retrieve(
        query=query,
        collection=collection,
        top_k=top_k,
    )

    if not matches:
        typer.echo(f"No matches for query: {query!r}")
        return

    typer.echo(f"Query: {query!r}")
    typer.echo(f"Collection: {collection}  top_k: {top_k}\n")

    for i, m in enumerate(matches, start=1):
        payload = m.payload
        source = payload.get("source", "<no source>")
        heading = payload.get("heading", "<no heading>")
        text = payload.get("text", "")
        snippet = text[:120].replace("\n", " ")
        if len(text) > 120:
            snippet += "..."

        typer.echo(f"[{i}] id={m.id}")
        typer.echo(f"    score={m.score:.4f}")
        typer.echo(f"    source={source}")
        typer.echo(f"    heading={heading}")
        typer.echo(f"    snippet: {snippet}\n")


@app.command()
def main(
    query: str = typer.Option(..., "--query", "-q", help="Query in NL."),
    collection: str = typer.Option(
        None,
        "--collection",
        "-c",
        help=f"Nome collection (default: {settings.qdrant_collection_name}).",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="N risultati."),
) -> None:
    """Mostra i top-K chunk recuperati dal retriever per una query."""
    coll = collection or settings.qdrant_collection_name
    asyncio.run(_explore(query, coll, top_k))


if __name__ == "__main__":
    app()
```

- [ ] **Step 3.3: Verifica che si avvii e mostri help**

```bash
cd /Users/francesco.cesarano/agentic-rag-stack/apps/api
uv run python -m app.evals.tools.explore --help 2>&1 | head -15
```

Expected: typer help con `--query`, `--collection`, `--top-k`.

- [ ] **Step 3.4: Smoke con una query reale**

```bash
uv run python -m app.evals.tools.explore --query "how do I declare a path parameter" --collection fastapi_docs --top-k 3
```

Expected: 3 chunk con id, score, source, heading, snippet.

- [ ] **Step 3.5: Ruff**

```bash
uv run ruff format app/evals/tools
uv run ruff check app/evals/tools
```

- [ ] **Step 3.6: Commit**

```bash
cd /Users/francesco.cesarano/agentic-rag-stack
git add apps/api/app/evals/tools/
git commit -m "feat(evals): add explore CLI for golden dataset curation (M2 task #11)"
```

---

## Task 4: Cura del golden dataset (lavoro umano)

**Goal:** Francesco usa `explore` per costruire ≥ 20 entries in `fastapi_docs.jsonl`.

**File:**
- Create: `apps/api/app/evals/golden_datasets/fastapi_docs.jsonl`

⚠️ Questo task è **prevalentemente lavoro umano**: ~1-2h di cura manuale. Non automatizzabile dal worker; il worker può aiutare creando 2-3 entries di esempio per mostrare il pattern.

- [ ] **Step 4.1: Crea la cartella**

```bash
mkdir -p /Users/francesco.cesarano/agentic-rag-stack/apps/api/app/evals/golden_datasets
```

- [ ] **Step 4.2: Worker produce 3 entries SEED**

Strategia: il worker propone 3 query che corrispondono a topic noti dei FastAPI docs (path params, dependency injection, request body Pydantic). Per ciascuna esegue `app.evals.tools.explore` per ottenere l'id del top match e lo scrive nel file.

Esempio (i query+id reali emergeranno dall'esecuzione):

```bash
cd /Users/francesco.cesarano/agentic-rag-stack/apps/api
uv run python -m app.evals.tools.explore \
  -q "how do I declare a path parameter in FastAPI" \
  -c fastapi_docs -k 3
```

Worker copia il top-1 id (se sembra correlato), aggiunge una riga a `fastapi_docs.jsonl`.

Ripete per 2-3 query SEED diverse.

- [ ] **Step 4.3: Francesco completa a ≥ 20**

Worker non può fare i restanti 17+ a freddo (richiede giudizio umano sui contenuti). Francesco esegue `app.evals.tools.explore` con sue query, copia gli id corretti.

Suggerimenti di copertura (worker propone, Francesco rivede):
- Query "facili" (lessicali, parole-chiave esatte): 6-8
- Query "concettuali" (parafrasi, sinonimi): 6-8
- Query "rare" (terminologia avanzata: middleware, SQLAlchemy, OAuth2): 4-6

- [ ] **Step 4.4: Verifica formato con loader**

```bash
uv run python -c "
from pathlib import Path
from app.evals.dataset import load_golden_dataset
items = load_golden_dataset(Path('app/evals/golden_datasets/fastapi_docs.jsonl'))
print(f'OK {len(items)} entries')
"
```

Expected: `OK 20 entries` (o più).

- [ ] **Step 4.5: Commit**

```bash
cd /Users/francesco.cesarano/agentic-rag-stack
git add apps/api/app/evals/golden_datasets/fastapi_docs.jsonl
git commit -m "data(evals): curate first 20+ golden entries for FastAPI docs corpus"
```

---

## Task 5: Precision@k evaluator

**Files:**
- Create: `apps/api/app/evals/evaluators/__init__.py`
- Create: `apps/api/app/evals/evaluators/precision_at_k.py`
- Create: `apps/api/tests/test_evals/test_precision_at_k.py`

- [ ] **Step 5.1: Test prima (TDD)**

Crea `apps/api/tests/test_evals/test_precision_at_k.py`:

```python
"""
tests/test_evals/test_precision_at_k.py
=======================================
Unit test del calcolo precision@k (M2 task #12).

Usano un fake retriever inline (no FastAPI, no Qdrant): il punto è
testare la matematica + l'aggregazione misses, non il wiring.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.evals.dataset import GoldenItem
from app.evals.evaluators.precision_at_k import precision_at_k
from app.rag.vector_store import Match


class _StubRetriever:
    """Retriever-like minimo: ritorna ciò che il test imposta per query."""

    def __init__(self, mapping: dict[str, list[Match]]) -> None:
        # mapping: query → lista di Match che il retriever ritornerebbe
        self.mapping = mapping

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Match]:
        return self.mapping.get(query, [])[:top_k]


def _mk_match(id: str) -> Match:
    return Match(id=id, score=0.9, payload={"text": "..."})


@pytest.mark.asyncio
async def test_all_hits() -> None:
    """Ogni query ha l'expected nei top-k → precision = 1.0."""
    dataset = [
        GoldenItem(query="q1", expected_chunk_id="a"),
        GoldenItem(query="q2", expected_chunk_id="b"),
    ]
    retriever = _StubRetriever(
        {"q1": [_mk_match("a"), _mk_match("x")], "q2": [_mk_match("b")]}
    )
    result = await precision_at_k(dataset, retriever, collection="c", k=2)
    assert result.precision == 1.0
    assert result.hits == 2
    assert result.misses == []


@pytest.mark.asyncio
async def test_no_hits() -> None:
    """Nessuna query trova expected → precision = 0.0."""
    dataset = [
        GoldenItem(query="q1", expected_chunk_id="a"),
        GoldenItem(query="q2", expected_chunk_id="b"),
    ]
    retriever = _StubRetriever({"q1": [_mk_match("x")], "q2": [_mk_match("y")]})
    result = await precision_at_k(dataset, retriever, collection="c", k=2)
    assert result.precision == 0.0
    assert result.hits == 0
    assert len(result.misses) == 2


@pytest.mark.asyncio
async def test_partial_hits() -> None:
    """2 query su 4 trovano expected → precision = 0.5."""
    dataset = [
        GoldenItem(query=f"q{i}", expected_chunk_id=f"id{i}") for i in range(4)
    ]
    retriever = _StubRetriever(
        {
            "q0": [_mk_match("id0")],  # hit
            "q1": [_mk_match("xxx")],  # miss
            "q2": [_mk_match("id2"), _mk_match("yyy")],  # hit
            "q3": [_mk_match("zzz")],  # miss
        }
    )
    result = await precision_at_k(dataset, retriever, collection="c", k=2)
    assert result.precision == 0.5
    assert result.hits == 2
    assert {m.query for m in result.misses} == {"q1", "q3"}


@pytest.mark.asyncio
async def test_top_k_cutoff() -> None:
    """Se expected è al rank > k, è MISS (non basta essere nel corpus)."""
    dataset = [GoldenItem(query="q1", expected_chunk_id="target")]
    # Il retriever ha 'target' al rank 3 ma top_k=2 lo taglia fuori.
    retriever = _StubRetriever(
        {"q1": [_mk_match("a"), _mk_match("b"), _mk_match("target")]}
    )
    result = await precision_at_k(dataset, retriever, collection="c", k=2)
    assert result.precision == 0.0
    assert len(result.misses) == 1


@pytest.mark.asyncio
async def test_empty_dataset() -> None:
    """Dataset vuoto → precision = 0.0 (no division by zero)."""
    retriever = _StubRetriever({})
    result = await precision_at_k([], retriever, collection="c", k=5)
    assert result.precision == 0.0
    assert result.hits == 0
    assert result.total_queries == 0
```

- [ ] **Step 5.2: Run — fallisce (modulo non esiste)**

```bash
cd /Users/francesco.cesarano/agentic-rag-stack/apps/api
mkdir -p app/evals/evaluators
touch app/evals/evaluators/__init__.py
uv run pytest tests/test_evals/test_precision_at_k.py -v 2>&1 | tail -10
```

Expected: ImportError.

- [ ] **Step 5.3: Implementa l'evaluator**

`apps/api/app/evals/evaluators/precision_at_k.py`:

```python
"""
app/evals/evaluators/precision_at_k.py
======================================
Precision@k evaluator per il retriever (M2 task #12).

Definizione semantica adottata (vedi design doc):
  hit(query) = expected_chunk_id ∈ {id di top-k chunk recuperati}
  precision@k = sum(hits) / total_queries

Con 1 expected per query è in senso stretto "recall@k", ma il
ROADMAP e l'uso colloquiale industry chiamano questa metrica
"precision@k". Mantenuto per coerenza.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.evals.dataset import GoldenItem
from app.rag.vector_store import Match


class _RetrieverLike(Protocol):
    """Superficie minima usata dall'evaluator (per type-checking).

    Coincide con `app.rag.retriever.Retriever.retrieve`. La Protocol
    è qui per consentire al test di passare un fake che non importa
    il vero Retriever (no Qdrant / OpenAI nei test).
    """

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[Match]: ...


@dataclass
class PrecisionAtKResult:
    """Output strutturato di precision_at_k. Adatto al report runner."""

    k: int
    total_queries: int
    hits: int
    precision: float
    misses: list[GoldenItem] = field(default_factory=list)


async def precision_at_k(
    dataset: list[GoldenItem],
    retriever: _RetrieverLike,
    collection: str,
    k: int,
) -> PrecisionAtKResult:
    """Calcola precision@k su un golden dataset.

    Esecuzione **serial**: per dataset piccoli (~20-50) il costo è ~10s
    in totale. Parallelizzeremo con asyncio.gather + semaforo se diventerà
    rilevante (M5 / CI).

    Edge case: dataset vuoto → precision = 0.0 (non NaN), per evitare
    crash in report stdout. Documentato esplicitamente.
    """
    if not dataset:
        return PrecisionAtKResult(
            k=k, total_queries=0, hits=0, precision=0.0, misses=[]
        )

    hits = 0
    misses: list[GoldenItem] = []

    for item in dataset:
        matches = await retriever.retrieve(
            query=item.query,
            collection=collection,
            top_k=k,
        )
        retrieved_ids = {m.id for m in matches}
        if item.expected_chunk_id in retrieved_ids:
            hits += 1
        else:
            misses.append(item)

    return PrecisionAtKResult(
        k=k,
        total_queries=len(dataset),
        hits=hits,
        precision=hits / len(dataset),
        misses=misses,
    )
```

- [ ] **Step 5.4: Run — verde**

```bash
uv run pytest tests/test_evals/test_precision_at_k.py -v 2>&1 | tail -15
```

Expected: 5 passed.

- [ ] **Step 5.5: Ruff**

```bash
uv run ruff format app/evals/evaluators tests/test_evals
uv run ruff check app/evals/evaluators tests/test_evals
```

- [ ] **Step 5.6: Commit**

```bash
cd /Users/francesco.cesarano/agentic-rag-stack
git add apps/api/app/evals/evaluators/ apps/api/tests/test_evals/test_precision_at_k.py
git commit -m "feat(evals): precision_at_k evaluator + 5 unit tests (M2 task #12)"
```

---

## Task 6: Runner CLI

**Files:**
- Create: `apps/api/app/evals/runners/__init__.py`
- Create: `apps/api/app/evals/runners/run_regression.py`
- Create: `apps/api/tests/test_evals/test_run_regression.py`

- [ ] **Step 6.1: Implementa il runner**

`apps/api/app/evals/runners/run_regression.py`:

```python
"""
app/evals/runners/run_regression.py
===================================
CLI: gira l'eval framework sul golden dataset corrente.

Esempio:
    uv run python -m app.evals.runners.run_regression
    uv run python -m app.evals.runners.run_regression --k 1 --k 5 --show-misses

Stampa precision@k human-readable. Per uso CI (futuro M5) aggiungeremo
--output json.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from app.evals.dataset import load_golden_dataset
from app.evals.evaluators.precision_at_k import precision_at_k
from app.rag.retriever import get_retriever

app = typer.Typer(name="eval-runner", add_completion=False)


async def _run(
    dataset_path: Path,
    collection: str,
    k_values: list[int],
    show_misses: bool,
) -> None:
    dataset = load_golden_dataset(dataset_path)
    retriever = get_retriever()

    typer.echo(f"Dataset:    {dataset_path}")
    typer.echo(f"Collection: {collection}")
    typer.echo(f"Queries:    {len(dataset)}\n")

    results_by_k = {}
    for k in sorted(k_values):
        r = await precision_at_k(
            dataset=dataset, retriever=retriever, collection=collection, k=k
        )
        results_by_k[k] = r
        typer.echo(f"precision@{k:<3}: {r.precision:.3f}  ({r.hits}/{r.total_queries})")

    if show_misses:
        # Mostra le misses al k più alto (di solito il più informativo).
        max_k = max(k_values)
        misses = results_by_k[max_k].misses
        if misses:
            typer.echo(f"\nMisses at precision@{max_k}:")
            for m in misses:
                typer.echo(f"  - query:    {m.query!r}")
                typer.echo(f"    expected: {m.expected_chunk_id}")


@app.command()
def main(
    dataset: Path = typer.Option(
        Path("app/evals/golden_datasets/fastapi_docs.jsonl"),
        help="Path al file .jsonl del dataset.",
    ),
    collection: str = typer.Option(
        "fastapi_docs", help="Nome collection Qdrant da interrogare."
    ),
    k_values: list[int] = typer.Option(
        [1, 5, 10],
        "--k",
        help="Valori di k da misurare. Ripetibile: --k 1 --k 5.",
    ),
    show_misses: bool = typer.Option(
        False, "--show-misses", help="Stampa le query fallite a max(k)."
    ),
) -> None:
    """Run regression eval su un golden dataset."""
    asyncio.run(_run(dataset, collection, k_values, show_misses))


if __name__ == "__main__":
    app()
```

- [ ] **Step 6.2: Smoke test sulla CLI surface**

Crea `apps/api/tests/test_evals/test_run_regression.py`:

```python
"""
tests/test_evals/test_run_regression.py
=======================================
Smoke test della CLI eval runner (M2 task #13).

Non gira l'eval vero (richiederebbe Qdrant + OpenAI live, già coperto
dal smoke manuale del Task 7). Verifichiamo solo che la CLI surface
sia stabile: typer help non crasha, opzioni dichiarate.
"""

from __future__ import annotations

from typer.testing import CliRunner

from app.evals.runners.run_regression import app

runner = CliRunner()


def test_cli_help_does_not_crash() -> None:
    """`--help` ritorna 0 e mostra le opzioni principali."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.output
    assert "--collection" in result.output
    assert "--k" in result.output
    assert "--show-misses" in result.output
```

- [ ] **Step 6.3: Crea `__init__.py` + run test**

```bash
cd /Users/francesco.cesarano/agentic-rag-stack/apps/api
mkdir -p app/evals/runners
touch app/evals/runners/__init__.py
uv run pytest tests/test_evals/test_run_regression.py -v 2>&1 | tail -10
```

Expected: 1 passed.

- [ ] **Step 6.4: Ruff**

```bash
uv run ruff format app/evals/runners tests/test_evals
uv run ruff check app/evals/runners tests/test_evals
```

- [ ] **Step 6.5: Commit**

```bash
cd /Users/francesco.cesarano/agentic-rag-stack
git add apps/api/app/evals/runners/ apps/api/tests/test_evals/test_run_regression.py
git commit -m "feat(evals): run_regression CLI runner + smoke test (M2 task #13)"
```

---

## Task 7: Smoke end-to-end + DoD verification + ROADMAP

**Goal:** girare il runner sul dataset reale, verificare DoD `precision@5 ≥ 0.7`, chiudere la sequenza nel ROADMAP.

- [ ] **Step 7.1: Suite intera + ruff**

```bash
cd /Users/francesco.cesarano/agentic-rag-stack/apps/api
uv run pytest -q 2>&1 | tail -5
uv run ruff check . && uv run ruff format --check .
```

Expected: tutti i test verdi (117 + 4 + 5 + 1 = 127), ruff clean.

- [ ] **Step 7.2: Smoke run end-to-end**

```bash
uv run python -m app.evals.runners.run_regression --k 1 --k 5 --k 10
```

Expected output stile:
```
Dataset:    app/evals/golden_datasets/fastapi_docs.jsonl
Collection: fastapi_docs
Queries:    24

precision@1  : 0.625  (15/24)
precision@5  : 0.833  (20/24)
precision@10 : 0.917  (22/24)
```

**Pass/fail della milestone:** precision@5 ≥ 0.7. Se sotto:
1. Rerun con `--show-misses` per vedere quali query falliscono.
2. Indagare: chunker spezza male? Dataset ha query mal-poste? Pochi sample (statistico)?
3. Se davvero hybrid serve → attivare #7b. Se è solo cura del dataset → iterare a #11 e ri-eseguire.

- [ ] **Step 7.3: Aggiorna ROADMAP**

In `docs/ROADMAP.md`, sostituisci le righe 11, 12, 13 della M2 (le voci dei task) con versioni `✅` che includono i numeri reali ottenuti dallo smoke. Esempio:

```markdown
11. ✅ **Golden dataset** (`app/evals/golden_datasets/fastapi_docs.jsonl`):
    24 coppie (query, expected_chunk_id) curate manualmente sui FastAPI
    docs reali (~1900 chunks ingestati in collection `fastapi_docs`).
    Tool di curation: `app.evals.tools.explore`.
12. ✅ **Eval `precision_at_k`** (`app/evals/evaluators/precision_at_k.py`):
    semantica top-k hit/miss su 1 expected per query. 5 unit test verdi.
13. ✅ **Eval runner** (`app/evals/runners/run_regression.py`):
    CLI typer, stampa precision@k human-readable. Smoke verificato.
    **DoD M2**: precision@5 = 0.83 (DoD soglia 0.7 ✓).
```

- [ ] **Step 7.4: Aggiorna anche la tabella "Status d'insieme" (M2 → ✅)**

In testa al `docs/ROADMAP.md`, cerca la riga "M2 — Knowledge base & RAG pipeline" e cambia `⚪` → `✅` con sintesi aggiornata.

Inoltre, aggiungere alla milestone M2:
```
**Status:** ✅ Done (chiusa 2026-05-28)
```

- [ ] **Step 7.5: Tag git semver**

Per convenzione del progetto (vedi "Note generali" del ROADMAP):

```bash
cd /Users/francesco.cesarano/agentic-rag-stack
git tag -a v0.2.0 -m "Milestone M2 complete: knowledge base + RAG pipeline"
```

- [ ] **Step 7.6: Commit finale**

```bash
git add docs/ROADMAP.md
git commit -m "docs(roadmap): mark M2 done (precision@5 = <value> on fastapi_docs)"
```

- [ ] **Step 7.7: Verifica git log**

```bash
git log --oneline -12
```

Expected: 7 commit di questo plan + il commit precedente del design + i commit di #10a.

---

## Spec coverage map

| Spec requirement | Task |
|---|---|
| `GoldenItem` schema | T2 |
| `load_golden_dataset` (JSONL, skip blanks, error con line num) | T2 |
| `app/evals/tools/explore.py` CLI | T3 |
| Golden dataset curato (≥ 20 entries) | T4 |
| `precision_at_k` algoritmica (hits, misses, edge cases) | T5 |
| `PrecisionAtKResult` strutturato | T5 |
| Runner CLI typer (`--dataset`, `--collection`, `--k`, `--show-misses`) | T6 |
| Smoke CLI surface (`--help`) | T6 |
| **Corpus FastAPI persistente ingestato (#11.0)** | T1 |
| **README sezione "Eval setup"** | T1 |
| **DoD M2 precision@5 ≥ 0.7** | T7 |
| **ROADMAP M2 → ✅ + tag v0.2.0** | T7 |

---

## Final acceptance check

```bash
# 1. Suite verde
cd /Users/francesco.cesarano/agentic-rag-stack/apps/api
uv run pytest -q

# 2. Linter clean
uv run ruff check .
uv run ruff format --check .

# 3. Eval runner produce numeri sensati
uv run python -m app.evals.runners.run_regression --k 5

# 4. ROADMAP a posto
grep -A1 "^| M2 " ../../docs/ROADMAP.md
```

Expected:
- ≥ 127 test verdi
- 0 ruff errors
- precision@5 ≥ 0.7
- `M2 | ... | ✅ |` nella tabella di stato
