# Design — Ingestion CLI (M2 task #6)

**Status:** Approved
**Date:** 2026-05-27
**Milestone:** M2 — Knowledge base & RAG pipeline
**Task:** #6 — `app/ingest.py`
**Related ADRs:** [ADR-0003](../../adr/0003-vector-db-choice-qdrant.md) (Qdrant), [ADR-0004](../../adr/0004-embedding-model-choice.md) (text-embedding-3-small)
**Depends on:** task #3 (chunker), #4 (embedder), #5 (vector_store)

## Goal

Esporre un CLI eseguibile via `uv run python -m app.ingest --source ... --collection ...` che orchestra l'ingestione end-to-end di un corpus markdown nel vector store Qdrant. Sostituisce il "ce lo facciamo a mano in REPL" — diventa l'azione riproducibile che popola la knowledge base.

## Non-goals

- Garbage collection di chunk vecchi (i chunk modificati lasciano gli old in giro fino a un cleanup task futuro)
- Re-ingest incrementale (solo file modificati) → potenzialmente M5
- Loader non-markdown (HTML, PDF, web pages) → fuori scope M2
- Endpoint HTTP `/admin/ingest` → M5
- Parallelizzazione del processing file → embedder e upsert già batchano internamente
- Progress bar token-by-token → embedder è async batched, impossibile granulare

## Architecture

Pipeline lineare in 4 step, orchestrata da una `main()` typer:

```
┌────────────────────────────────────────────────────────────────┐
│  app/ingest.py                                                 │
│                                                                │
│  1. resolve_source(--source)                                   │
│     ├─ path locale          → Path                             │
│     └─ URL git              → clone in TemporaryDirectory      │
│                                                                │
│  2. load_markdown_files(root)                                  │
│     → list[Path]  (glob **/*.md, ordinato)                     │
│                                                                │
│  3. ensure_collection(collection, vector_size=1536, "Cosine")  │
│                                                                │
│  4. for each file:                                             │
│       ingest_file(path, store, openai_client, collection)      │
│       ├─ chunk_markdown(text, source=relpath) -> list[Chunk]   │
│       ├─ embed_texts(client, [c.text for c in chunks])         │
│       │   -> list[EmbeddingResult]                             │
│       ├─ build list[VectorPoint] (id, vector, payload)         │
│       └─ store.upsert(collection, points)                      │
│                                                                │
│  5. summary report (stats per file, totale, skipped)           │
└────────────────────────────────────────────────────────────────┘
```

## Interfaces

### CLI signature

```
uv run python -m app.ingest \
    --source <PATH | URL>          [required]
    --collection <NAME>            [default: settings.qdrant_collection_name]
    --strict                       [flag, default false]
    --repo-subpath <PATH>          [opzionale; usato solo se --source è URL]
```

- `--source`: path locale (assoluto o relativo) OR URL git (`https://...`, `http://...`, `git@...`).
- `--collection`: nome della collection Qdrant. Override del default in `.env`.
- `--strict`: se passato, fail-fast al primo errore di chunking/embedding/upsert di un file. Default: resilient (log warning + skip).
- `--repo-subpath`: solo per source git. Limita lo scope al sub-path dentro al repo clonato (es. `docs/en/docs/` per il repo `fastapi/fastapi`). Se omesso, glob da repo root.

### DTO interni

```python
class IngestStats(BaseModel):
    file: str                     # path relativo al source root
    chunks: int                   # numero chunk prodotti
    tokens: int                   # somma token_count dei chunk
    skipped_reason: str | None    # None se ingestato; motivo se skippato
```

### Funzioni pubbliche del modulo

```python
def resolve_source(source: str) -> AbstractContextManager[Path]: ...
# Ritorna un context manager che yield-a la Path locale. Se source è git,
# il __exit__ cancella la TemporaryDirectory.

def load_markdown_files(root: Path) -> list[Path]: ...
# Glob `**/*.md`, ordinato deterministicamente per path stringa.

async def ingest_file(
    path: Path,
    source_root: Path,
    store: VectorStore,
    openai_client: AsyncOpenAI,
    collection: str,
) -> IngestStats: ...
# Processa un singolo file. Alza eccezione se il file ha errori
# (chunking vuoto, embedder failure, ecc.). Caller decide se skippare
# o re-raise (a seconda del flag --strict).

# entry point typer (decoratore @app.command)
async def main(source: str, collection: str | None, strict: bool,
               repo_subpath: str | None) -> None: ...
```

## Behavior contracts

### `resolve_source`
- Detection regex: se `re.match(r"^(https?://|git@)", source)` → trattalo come URL git.
- Per URL: shallow clone (`git clone --depth 1 <url> <tmpdir>`). Se `--repo-subpath` passato, ritorna `tmpdir / repo_subpath`. Validazione: se il subpath non esiste post-clone → `ValueError` chiaro.
- Per path locale: `Path(source).resolve()`. Se non esiste o non è directory → `ValueError`.
- Implementato come `contextlib.contextmanager`: `with resolve_source(s) as root: ...`. Il `TemporaryDirectory` viene cleanato anche su eccezione.

### `load_markdown_files`
- Glob `**/*.md` (lowercase). Esclude `.git/`, `node_modules/`, `__pycache__/`, `.venv/` (hardcoded set).
- Ritorna lista ordinata per `str(path)` ascendente — output deterministico, repeat-friendly.
- Lista vuota → ritorna `[]` senza errore (la `main` poi log warn "no files found").

### `ingest_file`
- Legge `path.read_text(encoding="utf-8")`. Se vuoto → alza `ValueError("file is empty")`.
- Chiama `chunk_markdown(text, source=str(path.relative_to(source_root)))`. Se ritorna `[]` → alza `ValueError("file produced no chunks after chunking")`.
- `embed_texts(openai_client, [c.text for c in chunks])` → propaga `EmbedderError` se i retry sono esauriti.
- Build `VectorPoint`: `id=chunk.id`, `vector=embedding.vector`, `payload={"text", "source", "heading", "position", "token_count"}`.
- `await store.upsert(collection, points)`.
- Ritorna `IngestStats(file=relpath, chunks=N, tokens=T, skipped_reason=None)`.

### `main` (typer command)
1. Setup logging.
2. `with resolve_source(source) as root:`
   1. `files = load_markdown_files(root)`.
   2. Se vuota: warning + return 0.
   3. `await store.ensure_collection(collection, vector_size=1536, distance="Cosine")`.
   4. Inizializza `AsyncOpenAI(api_key=settings.openai_api_key)`. Se key vuota → errore esplicito e exit 2.
   5. Per ogni file (con `typer.progressbar`):
      - try `await ingest_file(...)` → append `IngestStats`.
      - except → if `strict`: re-raise (typer mostra traceback, exit 1). Else: append `IngestStats(skipped_reason=str(e))` + log warning.
3. Stampa summary: N ingested / M skipped, totale chunks, totale token. Se skippati: elenco con motivi.

### Codici di uscita
- `0` — tutto ok (anche con skip in modalità non-strict).
- `1` — errore in `--strict` mode (un file ha fallito).
- `2` — errore di configurazione (OpenAI key mancante, source path invalido, git clone fallito).

## Concurrency model

Sequenziale tra file. Razionale:
- L'embedder già batcha 100 input per call OpenAI, con retry+backoff
- L'upsert già batcha 100 punti per call Qdrant
- Il chunking è CPU/IO trascurabile (~ms per file da 5KB)
- Parallelizzare i file complicherebbe rate-limit handling (OpenAI 429) per ~zero gain reale
- 152 file × ~1s/file = ~2.5 minuti totale — accettabile per uno script di setup

In M5, se serve ottimizzare il batch totale (es. corpus 10K file), valuteremo `asyncio.Semaphore(N)` per limitare concurrency.

## Configuration

Nessun nuovo campo `Settings` necessario: usiamo i già esistenti `openai_api_key`, `qdrant_url`, `qdrant_api_key`, `qdrant_collection_name`.

`qdrant_collection_name` viene usato come default per `--collection` se non passato.

## Dependencies

Aggiunta a [apps/api/pyproject.toml](../../../apps/api/pyproject.toml):

```toml
"typer>=0.12,<1",
```

Versione `>=0.12` per avere il supporto dei `Annotated[str, typer.Option(...)]` (sintassi moderna 2026).

`git` come binario di sistema: già richiesto per il versionamento del repo, non aggiungiamo nulla.

## Testing strategy

File: `apps/api/tests/test_ingest.py`. Tre gruppi.

### Unit puri (no Qdrant, no rete)

- `resolve_source("./relative/path")` su path esistente → context manager che yield-a Path resolto
- `resolve_source("/non/esiste")` → `ValueError` chiaro
- `resolve_source("https://github.com/x/y")` riconosciuto come git (mock subprocess: monkeypatch `subprocess.run`, asserisci args)
- `load_markdown_files(tmp_path)` con tmp_path popolato di .md + .txt + .py + subdir → ritorna solo .md, ordinati, esclusi i `.git/`

### Integration end-to-end (Qdrant live + FakeOpenAIClient)

Usa la fixture `qdrant_store` esistente e `fake_openai` esistente (definite in `tests/test_rag/conftest.py`; faremo `pytest.fixture` import condivisi o ridefinizione minima).

- `test_ingest_3_files_happy_path`: tmp dir con 3 markdown, ingest, verifica `_client.count(collection) > 0`.
- `test_ingest_idempotent_rerun`: stesso input due volte → count identico (overwrite).
- `test_ingest_resilient_skips_broken_file`: 3 file di cui uno vuoto → 2 ingestati, 1 skipped nello stats.
- `test_ingest_strict_raises_on_broken_file`: stesso scenario con `--strict` → eccezione propagata.

Marker: `@pytest.mark.integration`.

### CLI surface (typer CliRunner)

- `--help` non crasha, contiene "Ingest a markdown corpus"
- Missing `--source` → exit non-zero con messaggio chiaro
- `--source /non/esiste` → exit 2 con messaggio chiaro
- Mock di `AsyncOpenAI` (la key c'è ma non vogliamo chiamare API reale)

NON testiamo il git clone vero (richiederebbe rete + URL stabile). Il path "git" è coperto da unit con `subprocess.run` mockato.

## Risks & open questions

1. **`fake_openai` riuso**: la fixture vive oggi in `tests/test_rag/conftest.py`. Per usarla da `tests/test_ingest.py` la sposteremo in `tests/conftest.py` (top-level) — pulita perché non è specifica al RAG e ne beneficeranno anche altri test futuri. La classe `FakeOpenAIClient` non cambia, solo dove vive la fixture.
2. **Git binario assente**: in CI futuri (M5) servirà ubuntu image con `git`. Per ora il dev (Francesco) ha git locale.
3. **OpenAI quota / rate limit**: 152 file × ~10 chunk/file = ~1500 chunk = 15 batch da 100 = 15 API call. Sotto le rate limits del tier free OpenAI. Se in futuro il corpus cresce, valuteremo.
4. **Granularità del progress bar**: `typer.progressbar` itera sui file, non sui chunk. Per file grandi (es. 50 chunk) la barra "si ferma" per ~5s. Accettabile per un comando one-shot.
5. **Collisione windows path con regex git detection**: `re.match(r"^(https?://|git@)", "C:\\foo")` ritorna None — non collide. ✓

## Acceptance criteria

- [ ] `apps/api/app/ingest.py` esiste con `resolve_source`, `load_markdown_files`, `ingest_file`, `main` typer command.
- [ ] `typer` aggiunto a `pyproject.toml`.
- [ ] `uv run python -m app.ingest --help` mostra la signature documentata.
- [ ] `uv run python -m app.ingest --source ./test-corpus --collection test_demo` su una piccola dir locale popola Qdrant e stampa summary.
- [ ] `uv run pytest apps/api/tests/test_ingest.py -v` → tutti verdi (incl. integration con marker).
- [ ] `uv run pytest apps/api/` → suite intera verde (no regressioni).
- [ ] `uv run ruff check apps/api/ && uv run ruff format --check apps/api/` → clean.
- [ ] Re-run dello stesso ingest produce stesso count (idempotent).
