# Design — RAG eval framework MVP (M2 tasks #11 + #12 + #13)

**Status:** Proposed
**Date:** 2026-05-28
**Milestone:** M2 — Knowledge base & RAG pipeline
**Tasks:** #11 (golden dataset), #12 (precision@k evaluator), #13 (eval runner)
**Depends on:** #7a (retriever singleton), #6 (ingest CLI), #5 (vector store)

## Goal

Chiudere il loop di **misurazione** della pipeline RAG: poter dire con un numero quanto bene il retriever recupera i chunk attesi. È il prerequisito per qualunque decisione di "ha senso aggiungere reranker / hybrid / un altro embedder?" — senza un metro non c'è motivo di credere alle proprie ottimizzazioni.

A fine task:
- Una **collection Qdrant persistente** `fastapi_docs` con i FastAPI docs reali ingestati.
- Un **golden dataset** `app/evals/golden_datasets/fastapi_docs.jsonl` di ≥ 20 coppie `(query, expected_chunk_id)` curate a mano.
- Una funzione `precision_at_k(...)` che misura "quante query hanno il loro expected_chunk nei top-k recuperati".
- Una CLI `uv run python -m app.evals.runners.run_regression` che stampa un report.
- DoD della milestone M2: `precision@5 ≥ 0.7` raggiunto sul dataset.

## Decomposizione dei tasks

Il ROADMAP elenca #11/#12/#13 come task separati ma sono **strettamente accoppiati**:
- Lo schema del dataset (#11) determina la firma dell'evaluator (#12).
- L'evaluator determina cosa il runner stampa (#13).

Quindi un unico design doc che li copre tutti, ma 3 commit separati (uno per task) durante l'implementazione, per coerenza con la git history didattica.

In aggiunta, **#11 ha un sub-task implicito non listato nel ROADMAP**: ingestare il corpus FastAPI reale in una collection persistente. Lo chiamo qui **#11.0**: è la precondizione fisica per curare il dataset (servono chunk reali da cui prendere gli `expected_id`).

## Non-goals

- **Recall@k, MRR, NDCG**: solo precision@k per ora. Le altre arriveranno se l'eval mostrerà che precision@k da sola è insufficiente (es. quando inizieremo a distinguere fra "trovato 1 chunk corretto" e "trovato + ordinato bene").
- **Eval di faithfulness/hallucination**: richiede LLM-as-judge → fuori scope M2, arriva in M5.
- **CI gate**: il task è solo "runner CLI che stampa". L'integrazione con GitHub Actions e il gate "PR bloccata se precision cala" è M5 task #6/#7.
- **Multi-dataset support**: per ora un solo dataset (`fastapi_docs.jsonl`). Schema progettato per ammettere altri in futuro, ma non li costruiamo ora.
- **Confidence intervals, statistical significance**: il dataset sarà piccolo (20-50 query), nessun valore aggiunto nel costruire CI bootstrap. M5 magari.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Persistent setup (one-shot)                                      │
│                                                                  │
│   git clone fastapi/fastapi → docs/en/docs/*.md                  │
│            ↓                                                     │
│   uv run python -m app.ingest --source <path> \                  │
│                              --collection fastapi_docs           │
│            ↓                                                     │
│   Qdrant collection `fastapi_docs` (persistente, ~1500 chunks)   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Eval data (versionata in git)                                    │
│                                                                  │
│   app/evals/golden_datasets/fastapi_docs.jsonl                   │
│   (20+ righe {query, expected_chunk_id})                         │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Eval runtime                                                     │
│                                                                  │
│   load(dataset)                                                  │
│      ↓                                                           │
│   for each (query, expected_id):                                 │
│      matches = retriever.retrieve(query, collection, top_k)      │
│      hit = expected_id in [m.id for m in matches]                │
│      ↓                                                           │
│   precision@k = sum(hits) / len(dataset)                         │
│      ↓                                                           │
│   stdout report                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Golden dataset schema (task #11)

File: `apps/api/app/evals/golden_datasets/fastapi_docs.jsonl`

```jsonl
{"query": "how do I declare a path parameter in FastAPI?", "expected_chunk_id": "a3f9b21c..."}
{"query": "what is Depends() used for?", "expected_chunk_id": "b8e2d40f..."}
```

**Schema (Pydantic in `app/evals/dataset.py`):**

```python
class GoldenItem(BaseModel):
    """Singola entry del golden dataset.

    `expected_chunk_id` è l'`id` SHA256-derived del chunk atteso, cosi
    come prodotto dal `chunker` (vedi app/rag/chunker.py). Un solo
    expected per query: scelta MVP, tradeoff documentato in design.
    """

    query: str = Field(min_length=1)
    expected_chunk_id: str = Field(min_length=1)
```

**Loader:**

```python
def load_golden_dataset(path: Path) -> list[GoldenItem]:
    """Legge un .jsonl, una riga = una entry, parse Pydantic."""
    items: list[GoldenItem] = []
    with path.open() as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(GoldenItem.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(f"line {n}: {exc}") from exc
    return items
```

**Perché JSONL e non YAML/JSON?**
- 1 riga = 1 entry → diff git puliti quando aggiungiamo/togliamo una query.
- Append-friendly (curiamo iterativamente).
- Standard `de facto` in ML eval (HuggingFace, OpenAI evals).
- Niente "single point of failure": una riga corrotta non rompe il parsing delle altre (loader può saltarla).

**Perché 1 expected_id (top-1) e non lista (precision@k classica)?**
- **MVP**: per il primo dataset di 20-30 query, "esiste UN chunk che risponde a questa domanda meglio degli altri" è una semantica gestibile dal curatore umano.
- Definizione di precision@k qui: **`hit = expected_id ∈ retrieved_top_k`** (1 se dentro, 0 altrimenti). È in realtà **recall@k** in senso stretto (1 doc rilevante, lo trovo o no), ma per dataset top-1 le due metriche coincidono e l'uso colloquiale è "precision@k" in molte demo industriali.
- **Tradeoff**: se la query ha 2 chunk equivalentemente buoni, il dataset penalizza il retriever che ne sceglie uno "sbagliato per noi" ma corretto in assoluto. Accettabile per ora (la sceltedel curatore è informata); aggiorneremo la firma a `list[str]` se nei risultati vediamo casi-limite.

### 2. Precision@k evaluator (task #12)

File: `apps/api/app/evals/evaluators/precision_at_k.py`

```python
@dataclass
class PrecisionAtKResult:
    """Output di precision_at_k. Strutturato per facilitare il report."""

    k: int
    total_queries: int
    hits: int                # numero di query con expected in top-k
    precision: float         # hits / total_queries
    misses: list[GoldenItem] # query che hanno fallito (per il debugging)


async def precision_at_k(
    dataset: list[GoldenItem],
    retriever: Retriever,
    collection: str,
    k: int,
) -> PrecisionAtKResult:
    """Calcola precision@k. Async perché retriever.retrieve è async."""
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
        precision=hits / len(dataset) if dataset else 0.0,
        misses=misses,
    )
```

**Note:**
- **Serial, non parallel**: ~20-50 query × ~200ms (embed + search) = ~10s totali. Parallelizzare con `asyncio.gather` darebbe ~2-3x, ma rate-limit OpenAI lo rende fragile. YAGNI per ora.
- **Niente caching**: ogni run riembedda le query. Sono pochi token, ~$0.0001/run. Quando avremo CI a ogni PR (M5), valuteremo caching delle query embeddings.

### 3. Runner CLI (task #13)

File: `apps/api/app/evals/runners/run_regression.py`

```python
import typer

app = typer.Typer(name="eval-runner")


@app.command()
def main(
    dataset: Path = typer.Option(
        Path("app/evals/golden_datasets/fastapi_docs.jsonl"),
        help="Path al file .jsonl del dataset.",
    ),
    collection: str = typer.Option(
        "fastapi_docs",
        help="Nome collection Qdrant da interrogare.",
    ),
    k_values: list[int] = typer.Option(
        [1, 5, 10],
        "--k",
        help="Valori di k da misurare. Ripetibile: --k 1 --k 5 --k 10.",
    ),
    verbose_misses: bool = typer.Option(
        False, "--show-misses", help="Stampa le query che hanno fallito."
    ),
) -> None:
    """Run regression eval su un golden dataset.

    Esempio:
        uv run python -m app.evals.runners.run_regression
        uv run python -m app.evals.runners.run_regression --k 1 --k 5
    """
    asyncio.run(_run(dataset, collection, k_values, verbose_misses))
```

**Output (esempio):**

```
Dataset: app/evals/golden_datasets/fastapi_docs.jsonl
Collection: fastapi_docs
Queries: 24

precision@1 : 0.625  (15/24)
precision@5 : 0.833  (20/24) ← DoD M2 target
precision@10: 0.917  (22/24)
```

Con `--show-misses` aggiunge sotto le query fallite a precision@5.

**Perché stdout testuale e non JSON?**
- MVP human-readable. Quando M5 introdurrà il CI gate, aggiungeremo un flag `--output json` per pipe-friendly emission. YAGNI per ora.

### 4. Setup persistente del corpus (task #11.0)

**Operazione one-shot, non ricoperta da codice nuovo.** Esegue il comando già esistente (M2 #6):

```bash
# Clone shallow dei docs di FastAPI (non includiamo nel monorepo).
cd ~
git clone --depth 1 https://github.com/fastapi/fastapi.git fastapi-docs-source

# Ingest del subset /docs/en/docs/
cd /Users/francesco.cesarano/agentic-rag-stack/apps/api
uv run python -m app.ingest \
  --source ~/fastapi-docs-source/docs/en/docs \
  --collection fastapi_docs
```

Atteso: ~150 file markdown → ~1500-2000 chunk (a 500 token/chunk).

**Documentazione del setup**: una sezione nuova nel `README.md` di `apps/api` "Setup dataset per eval (#11)".

## Helper per la cura manuale (NON un task, ma raccomandato)

Curare 20+ query con expected_id richiede un workflow: leggi il corpus, scrivi una query plausibile, **trova** l'id del chunk che la risponde meglio.

**Proposta**: piccolo script di "explore" che dato una query stampa i top-N chunk con id+source+heading, per facilitare il copia/incolla del giusto `expected_chunk_id`.

```bash
uv run python -m app.evals.tools.explore \
  --query "how do I declare a path parameter" \
  --collection fastapi_docs \
  --top-k 5
```

Output:
```
[1] id=a3f9b21c  score=0.87  source=tutorial/path-params.md  heading=Path Parameters
    text snippet: "Declare a path parameter the same way..."
[2] id=...
```

Francesco copia l'id giusto nel dataset. È uno strumento di **curation**, non parte dell'eval runtime.

**Decisione**: includo questo nello scope di #11 come `app/evals/tools/explore.py`. Riusa direttamente il `Retriever` esistente, ~20 LOC.

## File touched

| File | Action | Task |
|---|---|---|
| `apps/api/app/evals/__init__.py` | **Create** (empty) | #11 |
| `apps/api/app/evals/dataset.py` | **Create** (`GoldenItem`, `load_golden_dataset`) | #11 |
| `apps/api/app/evals/golden_datasets/fastapi_docs.jsonl` | **Create** (≥ 20 entries, cura manuale) | #11 |
| `apps/api/app/evals/tools/__init__.py` | **Create** (empty) | #11 |
| `apps/api/app/evals/tools/explore.py` | **Create** (curation CLI) | #11 |
| `apps/api/app/evals/evaluators/__init__.py` | **Create** (empty) | #12 |
| `apps/api/app/evals/evaluators/precision_at_k.py` | **Create** (`PrecisionAtKResult`, `precision_at_k`) | #12 |
| `apps/api/app/evals/runners/__init__.py` | **Create** (empty) | #13 |
| `apps/api/app/evals/runners/run_regression.py` | **Create** (CLI typer) | #13 |
| `apps/api/tests/test_evals/test_dataset.py` | **Create** | #11 |
| `apps/api/tests/test_evals/test_precision_at_k.py` | **Create** | #12 |
| `apps/api/tests/test_evals/test_run_regression.py` | **Create** (smoke CLI surface) | #13 |
| `apps/api/README.md` | **Modify** (sezione "Eval setup") | #11.0 |
| `docs/ROADMAP.md` | **Modify** (mark #11/#12/#13 ✅) | finale |

Esisteva già `apps/api/app/evals/` come placeholder vuoto (dalla scaffolding di M0/M1)? Verificare; in caso negativo crearlo.

## Test plan

### Unit tests (no Qdrant, no OpenAI)

| File | Test | Verifica |
|---|---|---|
| `test_dataset.py` | `test_load_happy_path` | tmp .jsonl con 2 entries → load ritorna 2 GoldenItem |
| `test_dataset.py` | `test_load_skips_blank_lines` | righe vuote ignorate |
| `test_dataset.py` | `test_load_raises_on_invalid_line` | linea non-JSON → ValueError con num riga |
| `test_dataset.py` | `test_load_raises_on_missing_field` | linea senza expected_chunk_id → ValueError |
| `test_precision_at_k.py` | `test_all_hits` | fake retriever ritorna sempre expected → precision = 1.0 |
| `test_precision_at_k.py` | `test_no_hits` | fake retriever ritorna chunk diversi → precision = 0.0 |
| `test_precision_at_k.py` | `test_partial_hits` | 2/4 query hit → precision = 0.5 |
| `test_precision_at_k.py` | `test_misses_populated` | precisione 0.5 → result.misses ha le 2 query fallite |
| `test_precision_at_k.py` | `test_empty_dataset` | dataset vuoto → precision = 0.0 (no division by zero) |
| `test_run_regression.py` | `test_cli_help` | typer help non crasha (smoke surface CLI) |

### Integration (Qdrant + OpenAI live, marked `integration`)

| Test | Verifica |
|---|---|
| `test_precision_at_k_real_corpus` (skip if no Qdrant) | dataset di 2 entries dummy + collection unique → precision = 1.0 |

Niente eval end-to-end automatico sul vero dataset `fastapi_docs.jsonl` in test: è una verifica manuale durante il task #13.

## Smoke manuale (parte di #13)

```bash
cd apps/api
uv run python -m app.evals.runners.run_regression --k 1 --k 5 --k 10
```

Expected: stampa precision@1/5/10 con valori sensati. **DoD M2**: precision@5 ≥ 0.7.

Se precision@5 < 0.7:
- Indagare le `misses`: chunk attesi troppo specifici? Query mal formulate? Chunker che spezza male?
- Eventualmente raffinare il dataset (eliminando query mal-poste) o aggiungere reranker (#8).
- Se proprio dense fallisce su query lessicali → attivare #7b (hybrid).

## Risks & open questions

1. **Stabilità degli `expected_chunk_id` rispetto ai re-ingest.** Il chunker produce ID deterministici (`sha256(source + position + token_count + text)`). MA: se cambiamo chunker config (size, overlap), gli ID cambiano e il dataset diventa stale. **Mitigation**: pin la config del chunker nel `README` del dataset; aggiungiamo una header line al .jsonl `# chunker: size=500, overlap=50` (commento commentato dal loader).

2. **Quanti golden entries servono?** 20-50 è la finestra dichiarata nel ROADMAP. 20 è poco per essere statisticamente confidenti, ma sufficiente per "ho un segnale". Accettiamo questa imprecisione per il MVP; in M5 cresceremo a 100+.

3. **OpenAI cost per ogni run dell'eval**. Embed di 20-50 query × ~10 token l'una = ~500 token = $0.00001. Trascurabile.

4. **Tempo di esecuzione**. Serial: ~50 query × 250ms = ~12s. Accettabile. Se diventa problema, parallelizziamo con `asyncio.gather(limit=5)`.

5. **`expected_chunk_id` non più presente nel corpus dopo re-ingest del corpus aggiornato**. Se i docs FastAPI vengono aggiornati e re-ingestati, vecchi ID potrebbero non esistere più. **Mitigation**: il runner deve segnalare in modo distinto "expected_id mancante dalla collection corrente" (diverso da "expected_id non ranked nei top-k"). Aggiungiamo questo check.

## Acceptance criteria

- [ ] **#11.0**: collection `fastapi_docs` esistente in Qdrant locale, popolata con ≥ 1000 chunk.
- [ ] **#11**: `app/evals/dataset.py` con `GoldenItem` + `load_golden_dataset` + 4 unit test verdi.
- [ ] **#11**: `app/evals/golden_datasets/fastapi_docs.jsonl` con ≥ 20 entries curate manualmente.
- [ ] **#11**: `app/evals/tools/explore.py` CLI funzionante (verifica con almeno 1 invocazione manuale).
- [ ] **#12**: `app/evals/evaluators/precision_at_k.py` con `precision_at_k(...)` + 5 unit test verdi.
- [ ] **#13**: `app/evals/runners/run_regression.py` CLI typer + smoke test.
- [ ] **DoD M2**: `precision@5 ≥ 0.7` sul dataset `fastapi_docs.jsonl`.
- [ ] Suite intera verde, ruff clean.
- [ ] `docs/ROADMAP.md` aggiornato (#11, #12, #13 → ✅).
