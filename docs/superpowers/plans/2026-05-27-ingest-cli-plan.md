# Ingestion CLI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementare il CLI `app/ingest.py` che orchestra load → chunk → embed → upsert end-to-end, secondo il design [docs/superpowers/specs/2026-05-27-ingest-cli-design.md](../specs/2026-05-27-ingest-cli-design.md).

**Architecture:** Modulo singolo `app/ingest.py` con typer come framework CLI. Pipeline lineare di funzioni piccole e testabili: `resolve_source` (context manager, locale o git shallow clone), `load_markdown_files` (glob), `ingest_file` (per-file orchestrator), `_run` (orchestrazione async, dep injection-friendly per test), `main` (entry point typer).

**Tech Stack:** Python 3.12, typer ≥0.12, AsyncOpenAI (text-embedding-3-small), AsyncQdrantClient via `app.rag.vector_store`, pytest-asyncio.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `apps/api/pyproject.toml` | **Modify** | Aggiungi `typer>=0.12,<1` |
| `apps/api/tests/conftest.py` | **Modify** | Sposta `FakeOpenAIClient` + fixture `fake_openai` qui (top-level) |
| `apps/api/tests/test_rag/conftest.py` | **Modify** | Rimuovi `FakeOpenAIClient` + `fake_openai` (lasciano solo le fixture qdrant) |
| `apps/api/app/ingest.py` | **Create** | Modulo + CLI typer |
| `apps/api/tests/test_ingest.py` | **Create** | Test suite (unit + integration + CLI surface) |
| `docs/ROADMAP.md` | **Modify** | M2 task #6 da ⚪ a ✅ |

---

## Pre-flight check (NOT a task)

```powershell
git status              # clean
docker ps --filter "name=agentic-rag-qdrant" --format "{{.Status}}"  # Up (healthy)
git branch --show-current  # main
```

---

## Task 1: Dependency `typer` + fixture relocation

**Files:**
- Modify: `apps/api/pyproject.toml`
- Modify: `apps/api/tests/conftest.py`
- Modify: `apps/api/tests/test_rag/conftest.py`

- [ ] **Step 1.1: Aggiungi `typer` come dependency**

In [apps/api/pyproject.toml](../../../apps/api/pyproject.toml), dentro `[project] dependencies`, prima della chiusura `]`:

```toml
  # typer: CLI framework built on click + type hints, stesso autore di FastAPI.
  # Lo usiamo per `app/ingest.py` (M2 task #6). Versione 0.12+ supporta la
  # sintassi `Annotated[T, typer.Option(...)]` che è il pattern 2026.
  "typer>=0.12,<1",
```

- [ ] **Step 1.2: `uv sync`**

```powershell
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack\apps\api
uv sync
```

Expected: `Installed N packages` (incluso `typer`, `click`, `rich`).

- [ ] **Step 1.3: Sposta `FakeOpenAIClient` in `tests/conftest.py` top-level**

Leggi `apps/api/tests/test_rag/conftest.py` e copia INTEGRALMENTE il blocco `_FakeEmbeddingsResource`, `FakeOpenAIClient`, e la fixture `fake_openai` in `apps/api/tests/conftest.py` (appendi alla fine se non vuoto, altrimenti crealo).

Il contenuto da aggiungere a `apps/api/tests/conftest.py` (alla fine, dopo eventuali contenuti esistenti):

```python


# ---------------------------------------------------------------------------
# Test double per OpenAI (condiviso da test_rag/ e test_ingest/)
# ---------------------------------------------------------------------------
# Spostato qui da test_rag/conftest.py per essere visibile a TUTTI i test
# che hanno bisogno di un AsyncOpenAI fittizio (embedder + ingest CLI).
# Implementato a mano (niente `unittest.mock`) per due ragioni:
#   1. Si VEDE chiaramente quale superficie del SDK stiamo usando.
#   2. È facile estenderlo (errori, out-of-order, ecc.).

from types import SimpleNamespace
from typing import Any


class _FakeEmbeddingsResource:
    """Simula `AsyncOpenAI.embeddings`: ha un metodo async `.create()`."""

    def __init__(self, parent: FakeOpenAIClient) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        """Restituisce una response Embeddings-like.

        Per default produce un embedding "tutto 0.1" per ogni input (1536-dim
        come `text-embedding-3-small`). I test che vogliono comportamenti
        diversi (errori, out-of-order, ...) configurano il fake.
        """
        self._parent.calls.append(kwargs)

        if self._parent.errors_queue:
            err = self._parent.errors_queue.pop(0)
            raise err

        inputs: list[str] = kwargs["input"]
        order = self._parent.next_response_order or list(range(len(inputs)))

        data = [
            SimpleNamespace(
                embedding=self._parent.vector_for(inputs[i]),
                index=i,
            )
            for i in order
        ]

        usage = SimpleNamespace(
            prompt_tokens=self._parent.tokens_per_call,
            total_tokens=self._parent.tokens_per_call,
        )

        self._parent.next_response_order = None

        return SimpleNamespace(data=data, usage=usage)


class FakeOpenAIClient:
    """Test double per `openai.AsyncOpenAI`.

    Configurabile dal test:
        fake.tokens_per_call = 42
        fake.errors_queue = [RateLimitError(...), None]
        fake.next_response_order = [2,0,1]
        fake.vector_dim = 1536
    """

    def __init__(self) -> None:
        self.vector_dim = 1536
        self.tokens_per_call = 10
        self.calls: list[dict[str, Any]] = []
        self.errors_queue: list[Exception] = []
        self.next_response_order: list[int] | None = None
        self.embeddings = _FakeEmbeddingsResource(self)

    def vector_for(self, text: str) -> list[float]:
        """Vettore deterministico per un dato testo."""
        seed = (len(text) % 100) / 100.0
        return [seed] * self.vector_dim

    async def close(self) -> None:
        """No-op: il fake non ha connessioni reali da chiudere."""
        return None


@pytest.fixture
def fake_openai() -> FakeOpenAIClient:
    """Istanzia un FakeOpenAIClient pronto all'uso.

    NON tocca `app.dependency_overrides`: embedder e ingest accettano
    il client direttamente, quindi nei test lo passiamo come argomento.
    """
    return FakeOpenAIClient()
```

Assicurati che in cima al file `apps/api/tests/conftest.py` ci sia:
```python
from __future__ import annotations

import pytest
```

Se non c'è già, aggiungilo.

- [ ] **Step 1.4: Rimuovi `FakeOpenAIClient` + `fake_openai` da `tests/test_rag/conftest.py`**

In [apps/api/tests/test_rag/conftest.py](../../../apps/api/tests/test_rag/conftest.py), cancella il blocco `_FakeEmbeddingsResource`, `FakeOpenAIClient`, e `fake_openai` fixture (sono diventati duplicati).

Lascia intatti: il modulo docstring iniziale, gli import in cima, le fixtures `qdrant_store` e `unique_collection`.

Il file post-modifica dovrà:
- iniziare con il docstring + `from __future__ import annotations` + import + `logger = logging.getLogger(__name__)`
- contenere SOLO le fixture `qdrant_store` (session) e `unique_collection` (function).

Note: pytest cerca conftest.py in TUTTE le directory ancestor — fixture in `tests/conftest.py` sono visibili a `tests/test_rag/`, quindi i test esistenti continueranno a funzionare.

- [ ] **Step 1.5: Verifica suite ancora verde**

```powershell
uv run pytest -q
```

Expected: 73 passed (nessun regressione dalla relocation della fixture).

- [ ] **Step 1.6: Smoke test typer**

```powershell
uv run python -c "import typer; print('typer', typer.__version__)"
```

Expected: `typer 0.12.x` (o successiva).

- [ ] **Step 1.7: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/pyproject.toml apps/api/uv.lock apps/api/tests/conftest.py apps/api/tests/test_rag/conftest.py
git commit -m "chore(api): add typer dep + relocate FakeOpenAIClient to top-level conftest"
```

---

## Task 2: `resolve_source` + unit tests

**Files:**
- Create: `apps/api/app/ingest.py`
- Create: `apps/api/tests/test_ingest.py`

- [ ] **Step 2.1: Crea `app/ingest.py` con header docstring + skeleton**

Crea il file con:

```python
"""
app/ingest.py
=============
CLI di ingestione del corpus markdown nel vector store.

Si usa così:
    uv run python -m app.ingest --source ./docs --collection my-kb
    uv run python -m app.ingest --source https://github.com/x/y --repo-subpath docs

Orchestra i 4 componenti già esistenti (M2 task #3-5):
    load -> chunk -> embed -> upsert

Vedi il design doc per le scelte:
    docs/superpowers/specs/2026-05-27-ingest-cli-design.md
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


# Pattern di detection per identificare una URL git. Conservativo: matcha
# solo http(s)://... e git@host:... — i path Windows con drive letter
# (es. C:\foo) non collidono perché non iniziano con questi prefix.
_GIT_URL_RE = re.compile(r"^(https?://|git@)", re.IGNORECASE)


@contextmanager
def resolve_source(source: str, repo_subpath: str | None = None) -> Iterator[Path]:
    """Risolve `source` a una directory locale, gestita come context manager.

    Args:
        source: path locale OPPURE URL git (https://..., git@...).
        repo_subpath: se source è URL, restringi lo scope a questo sotto-path
            dentro al repo clonato (es. "docs/en/docs" per fastapi/fastapi).

    Yields:
        Path della directory radice da cui caricare i markdown.

    Raises:
        ValueError: se path locale non esiste o non è directory, o se il
            repo_subpath non esiste post-clone.
        subprocess.CalledProcessError: se git clone fallisce.
    """
    if _GIT_URL_RE.match(source):
        with tempfile.TemporaryDirectory(prefix="agentic-rag-ingest-") as tmpdir:
            tmp_path = Path(tmpdir)
            logger.info("git_clone_start", extra={"url": source, "dest": str(tmp_path)})
            subprocess.run(
                ["git", "clone", "--depth", "1", source, str(tmp_path)],
                check=True,
                capture_output=True,
            )
            root = tmp_path / repo_subpath if repo_subpath else tmp_path
            if not root.is_dir():
                raise ValueError(
                    f"repo_subpath '{repo_subpath}' non trovato in {source}"
                )
            yield root
    else:
        path = Path(source).resolve()
        if not path.is_dir():
            raise ValueError(f"source '{source}' non è una directory esistente")
        yield path
```

- [ ] **Step 2.2: Crea `tests/test_ingest.py` con unit test per `resolve_source`**

```python
"""
tests/test_ingest.py
====================
Test del CLI di ingestione `app/ingest.py`.

Tre gruppi:
- Unit puri (no Qdrant, no rete): test di funzioni come resolve_source,
  load_markdown_files.
- Integration end-to-end: pipeline completa su Qdrant live con FakeOpenAIClient.
- CLI surface: typer CliRunner per --help / errori di argomento.

Marker:
- (default) test unit, sempre on.
- @pytest.mark.integration: richiede Qdrant live (skip altrimenti).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingest import resolve_source


# ---------------------------------------------------------------------------
# Unit: resolve_source
# ---------------------------------------------------------------------------


def test_resolve_source_local_path_ok(tmp_path: Path) -> None:
    """Path locale esistente → yield la stessa path resolto."""
    with resolve_source(str(tmp_path)) as root:
        assert root == tmp_path.resolve()


def test_resolve_source_local_path_missing(tmp_path: Path) -> None:
    """Path locale inesistente → ValueError chiaro."""
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="non è una directory"):
        with resolve_source(str(missing)):
            pass  # pragma: no cover


def test_resolve_source_local_file_not_dir(tmp_path: Path) -> None:
    """Path che esiste ma è un file (non directory) → ValueError."""
    f = tmp_path / "regular_file.txt"
    f.write_text("ciao")
    with pytest.raises(ValueError, match="non è una directory"):
        with resolve_source(str(f)):
            pass  # pragma: no cover


def test_resolve_source_https_url_triggers_clone(tmp_path: Path) -> None:
    """URL https → invoca git clone shallow."""
    # Mock subprocess.run per evitare network call vera.
    with patch("app.ingest.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        # Anche TemporaryDirectory crea una dir vera; il "clone" mockato
        # non popola nulla, quindi yieldiamo una dir vuota → ok per il test.
        with resolve_source("https://github.com/fake/repo") as root:
            assert root.is_dir()
            # Verifica che subprocess sia stato chiamato con "git clone --depth 1".
            args = mock_run.call_args[0][0]
            assert args[0] == "git"
            assert args[1] == "clone"
            assert "--depth" in args and "1" in args
            assert args[-2] == "https://github.com/fake/repo"


def test_resolve_source_git_at_url_triggers_clone() -> None:
    """URL git@ ssh-style → trattato come git."""
    with patch("app.ingest.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with resolve_source("git@github.com:fake/repo.git") as root:
            assert root.is_dir()
            assert mock_run.called


def test_resolve_source_repo_subpath_missing_raises() -> None:
    """Se repo_subpath non esiste post-clone → ValueError."""
    with patch("app.ingest.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with pytest.raises(ValueError, match="repo_subpath"):
            with resolve_source(
                "https://github.com/fake/repo", repo_subpath="not-a-real-subpath"
            ):
                pass  # pragma: no cover


def test_resolve_source_windows_path_not_url() -> None:
    """Path Windows-style con drive letter non collide col regex git.

    `C:\\foo` non matcha `^(https?://|git@)`. Va trattato come path locale.
    """
    # Usiamo una path locale fittizia che non esiste — l'importante è il branching.
    with pytest.raises(ValueError, match="non è una directory"):
        with resolve_source("C:\\non-esiste-davvero"):
            pass  # pragma: no cover
```

- [ ] **Step 2.3: Run i test — devono passare tutti**

```powershell
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack\apps\api
uv run pytest tests/test_ingest.py -v
```

Expected: 6 passed.

- [ ] **Step 2.4: Ruff**

```powershell
uv run ruff format app/ingest.py tests/test_ingest.py
uv run ruff check --fix app/ingest.py tests/test_ingest.py
uv run ruff check app/ingest.py tests/test_ingest.py
```

Expected: `All checks passed!`.

- [ ] **Step 2.5: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/app/ingest.py apps/api/tests/test_ingest.py
git commit -m "feat(ingest): add resolve_source context manager (local path | git URL)"
```

---

## Task 3: `load_markdown_files` + unit tests

**Files:**
- Modify: `apps/api/app/ingest.py`
- Modify: `apps/api/tests/test_ingest.py`

- [ ] **Step 3.1: Aggiungi i test (rosso prima)**

Aggiungi alla FINE di [apps/api/tests/test_ingest.py](../../../apps/api/tests/test_ingest.py):

```python


# ---------------------------------------------------------------------------
# Unit: load_markdown_files
# ---------------------------------------------------------------------------

from app.ingest import load_markdown_files


def test_load_markdown_files_glob_recursive(tmp_path: Path) -> None:
    """Glob ricorsivo su **/*.md."""
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("# B")
    (tmp_path / "sub" / "c.md").write_text("# C")

    found = load_markdown_files(tmp_path)
    names = sorted(p.name for p in found)
    assert names == ["a.md", "b.md", "c.md"]


def test_load_markdown_files_ignores_non_md(tmp_path: Path) -> None:
    """File non-.md vengono ignorati."""
    (tmp_path / "doc.md").write_text("# md")
    (tmp_path / "readme.txt").write_text("txt")
    (tmp_path / "code.py").write_text("py")

    found = load_markdown_files(tmp_path)
    assert len(found) == 1
    assert found[0].name == "doc.md"


def test_load_markdown_files_excludes_hidden_and_build_dirs(tmp_path: Path) -> None:
    """Skip .git/, node_modules/, __pycache__/, .venv/."""
    (tmp_path / "kept.md").write_text("ok")

    for excluded in (".git", "node_modules", "__pycache__", ".venv"):
        (tmp_path / excluded).mkdir()
        (tmp_path / excluded / "inside.md").write_text("excluded")

    found = load_markdown_files(tmp_path)
    names = [p.name for p in found]
    assert names == ["kept.md"]


def test_load_markdown_files_sorted_deterministic(tmp_path: Path) -> None:
    """Risultato ordinato per str(path)."""
    for name in ["zzz.md", "aaa.md", "mmm.md"]:
        (tmp_path / name).write_text("x")

    found = load_markdown_files(tmp_path)
    names = [p.name for p in found]
    assert names == sorted(names)


def test_load_markdown_files_empty_dir_returns_empty(tmp_path: Path) -> None:
    """Directory senza .md → lista vuota, no errore."""
    assert load_markdown_files(tmp_path) == []
```

- [ ] **Step 3.2: Verifica fail (ImportError)**

```powershell
uv run pytest tests/test_ingest.py::test_load_markdown_files_glob_recursive -v
```

Expected: `ImportError: cannot import name 'load_markdown_files' from 'app.ingest'`.

- [ ] **Step 3.3: Implementa `load_markdown_files`**

In [apps/api/app/ingest.py](../../../apps/api/app/ingest.py), aggiungi dopo la funzione `resolve_source`:

```python


# Directory che escludiamo dal glob: convenzionalmente "rumore" sotto un
# repo (controllo versione, dipendenze, cache, virtualenv). Set per
# lookup O(1) durante il filtraggio.
_EXCLUDED_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv"})


def load_markdown_files(root: Path) -> list[Path]:
    """Glob `**/*.md` ricorsivo su `root`, ordinato per path stringa.

    Esclude file dentro a `.git/`, `node_modules/`, `__pycache__/`, `.venv/`
    (rumore tipico in un repo clonato).

    Args:
        root: directory radice da cui partire.

    Returns:
        Lista ordinata di Path ai file .md. Vuota se nessun match.
    """
    all_md = root.rglob("*.md")
    filtered = [
        p for p in all_md if not any(part in _EXCLUDED_DIRS for part in p.parts)
    ]
    return sorted(filtered, key=str)
```

- [ ] **Step 3.4: Run i test — devono passare**

```powershell
uv run pytest tests/test_ingest.py -v
```

Expected: 11 passed (6 da Task 2 + 5 nuovi).

- [ ] **Step 3.5: Ruff**

```powershell
uv run ruff format app/ingest.py tests/test_ingest.py
uv run ruff check --fix app/ingest.py tests/test_ingest.py
uv run ruff check app/ingest.py tests/test_ingest.py
```

- [ ] **Step 3.6: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/app/ingest.py apps/api/tests/test_ingest.py
git commit -m "feat(ingest): add load_markdown_files (recursive glob with exclusions)"
```

---

## Task 4: `IngestStats` DTO + `ingest_file` orchestrator

**Files:**
- Modify: `apps/api/app/ingest.py`
- Modify: `apps/api/tests/test_ingest.py`

- [ ] **Step 4.1: Aggiungi i test (rosso prima)**

In [apps/api/tests/test_ingest.py](../../../apps/api/tests/test_ingest.py), aggiungi in cima (sotto gli import esistenti) gli import necessari:

```python
import uuid

from app.ingest import IngestStats, ingest_file
```

E in fondo al file, aggiungi:

```python


# ---------------------------------------------------------------------------
# Integration: ingest_file (Qdrant live + FakeOpenAIClient)
# ---------------------------------------------------------------------------


def test_ingest_stats_basic() -> None:
    """IngestStats Pydantic costruzione minima."""
    s = IngestStats(file="a.md", chunks=3, tokens=120)
    assert s.file == "a.md"
    assert s.chunks == 3
    assert s.tokens == 120
    assert s.skipped_reason is None


def test_ingest_stats_skipped() -> None:
    """IngestStats con skipped_reason."""
    s = IngestStats(file="b.md", chunks=0, tokens=0, skipped_reason="empty")
    assert s.skipped_reason == "empty"


@pytest.mark.integration
async def test_ingest_file_happy_path(qdrant_store, fake_openai, tmp_path) -> None:
    """Un singolo file markdown viene chunkato, embeddato e upsertato."""
    # Prepara collection.
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")

        # Prepara un file markdown.
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\nThis is a test paragraph.\n")

        stats = await ingest_file(
            path=md,
            source_root=tmp_path,
            store=qdrant_store,
            openai_client=fake_openai,
            collection=collection,
        )

        assert stats.file == "doc.md"
        assert stats.chunks >= 1
        assert stats.tokens > 0
        assert stats.skipped_reason is None

        # Il file ha generato punti nel vector store.
        count_resp = await qdrant_store._client.count(collection, exact=True)
        assert count_resp.count == stats.chunks
    finally:
        await qdrant_store.delete_collection(collection)


@pytest.mark.integration
async def test_ingest_file_empty_raises(qdrant_store, fake_openai, tmp_path) -> None:
    """File vuoto → ValueError (resilient/strict lo decide il caller)."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")
        md = tmp_path / "empty.md"
        md.write_text("")
        with pytest.raises(ValueError, match="empty"):
            await ingest_file(
                path=md,
                source_root=tmp_path,
                store=qdrant_store,
                openai_client=fake_openai,
                collection=collection,
            )
    finally:
        await qdrant_store.delete_collection(collection)
```

- [ ] **Step 4.2: Verifica fail (ImportError)**

```powershell
uv run pytest tests/test_ingest.py::test_ingest_stats_basic -v
```

Expected: ImportError per `IngestStats` / `ingest_file`.

- [ ] **Step 4.3: Implementa `IngestStats` + `ingest_file`**

In [apps/api/app/ingest.py](../../../apps/api/app/ingest.py), aggiungi gli import in cima (sotto gli import esistenti):

```python
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.rag.chunker import chunk_markdown
from app.rag.embedder import embed_texts
from app.rag.vector_store import VectorPoint, VectorStore
```

Poi, dopo `load_markdown_files`, aggiungi:

```python


# ---------------------------------------------------------------------------
# DTO interni
# ---------------------------------------------------------------------------


class IngestStats(BaseModel):
    """Statistiche di ingestione per un singolo file.

    Attributes:
        file: path relativo al source root (es. "tutorial/intro.md").
        chunks: numero di chunk prodotti (0 se skipped).
        tokens: somma dei token_count dei chunk (0 se skipped).
        skipped_reason: None se ingestato; messaggio del motivo se skippato
            (in modalità resilient). In modalità --strict il file fallito
            propaga eccezione e non genera IngestStats.
    """

    file: str
    chunks: int = Field(ge=0)
    tokens: int = Field(ge=0)
    skipped_reason: str | None = None


# ---------------------------------------------------------------------------
# Orchestratore per-file
# ---------------------------------------------------------------------------


async def ingest_file(
    path: Path,
    source_root: Path,
    store: VectorStore,
    openai_client: AsyncOpenAI,
    collection: str,
) -> IngestStats:
    """Processa un singolo file markdown: chunk + embed + upsert.

    Args:
        path: path assoluto al file .md.
        source_root: radice del corpus (usata per calcolare il `source`
            relativo che finisce nel payload di Qdrant).
        store: implementazione di VectorStore (es. QdrantVectorStore).
        openai_client: client OpenAI async (reale o fake nei test).
        collection: nome della collection Qdrant.

    Returns:
        IngestStats con counters. `skipped_reason` è sempre None qui — i
        casi di skip sono propagati come eccezione e gestiti dal caller.

    Raises:
        ValueError: se il file è vuoto o non produce chunk dopo il chunking.
        EmbedderError: se l'embedding fallisce dopo tutti i retry.
        Eccezioni del vector_store: propagate.
    """
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"file is empty: {path}")

    rel = str(path.relative_to(source_root)).replace("\\", "/")
    chunks = chunk_markdown(text, source=rel)
    if not chunks:
        raise ValueError(f"file produced no chunks after chunking: {rel}")

    embeddings = await embed_texts(openai_client, [c.text for c in chunks])

    points = [
        VectorPoint(
            id=chunk.id,
            vector=emb.vector,
            payload={
                "text": chunk.text,
                "source": chunk.source,
                "heading": chunk.heading,
                "position": chunk.position,
                "token_count": chunk.token_count,
            },
        )
        for chunk, emb in zip(chunks, embeddings, strict=True)
    ]

    await store.upsert(collection, points)

    logger.info(
        "ingest_file_done",
        extra={
            "file": rel,
            "chunks": len(chunks),
            "tokens": sum(c.token_count for c in chunks),
        },
    )

    return IngestStats(
        file=rel,
        chunks=len(chunks),
        tokens=sum(c.token_count for c in chunks),
    )
```

- [ ] **Step 4.4: Run i test — devono passare**

```powershell
uv run pytest tests/test_ingest.py -v
```

Expected: 15 passed (11 esistenti + 4 nuovi, di cui 2 integration).

- [ ] **Step 4.5: Ruff**

```powershell
uv run ruff format app/ingest.py tests/test_ingest.py
uv run ruff check --fix app/ingest.py tests/test_ingest.py
uv run ruff check app/ingest.py tests/test_ingest.py
```

- [ ] **Step 4.6: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/app/ingest.py apps/api/tests/test_ingest.py
git commit -m "feat(ingest): IngestStats + ingest_file per-file orchestrator"
```

---

## Task 5: `_run` orchestrator + typer command `main`

**Files:**
- Modify: `apps/api/app/ingest.py`
- Modify: `apps/api/tests/test_ingest.py`

- [ ] **Step 5.1: Implementa `_run` + `main` (typer)**

In [apps/api/app/ingest.py](../../../apps/api/app/ingest.py), aggiungi agli import esistenti:

```python
import asyncio
from typing import Annotated

import typer

from app.config import settings
from app.rag.vector_store import get_vector_store
```

Poi alla fine del file (dopo `ingest_file`):

```python


# ---------------------------------------------------------------------------
# Orchestratore async (dep-injection friendly per i test)
# ---------------------------------------------------------------------------


async def _run(
    source: str,
    collection: str,
    strict: bool,
    repo_subpath: str | None,
    store: VectorStore | None = None,
    openai_client: AsyncOpenAI | None = None,
) -> list[IngestStats]:
    """Pipeline di ingestione completa. Restituisce le stats per file.

    Args:
        source: path locale o URL git.
        collection: nome collection Qdrant.
        strict: se True, fail-fast al primo errore. Se False, log warn + skip.
        repo_subpath: solo per source URL.
        store: opzionale, default `get_vector_store()`. Iniettabile per test.
        openai_client: opzionale, default `AsyncOpenAI(api_key=...)`.

    Returns:
        Lista di IngestStats (uno per file processato).

    Raises:
        ValueError: se OpenAI key mancante, source path invalido, ecc.
        Exception: in modalità strict, re-raise del primo errore per-file.
    """
    if store is None:
        store = get_vector_store()
    if openai_client is None:
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY non configurata. Imposta la variabile nel file .env."
            )
        openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    all_stats: list[IngestStats] = []
    with resolve_source(source, repo_subpath=repo_subpath) as root:
        files = load_markdown_files(root)
        if not files:
            logger.warning("ingest_no_files_found", extra={"root": str(root)})
            return []

        logger.info(
            "ingest_start",
            extra={"files": len(files), "collection": collection, "root": str(root)},
        )

        # Crea la collection (idempotente).
        await store.ensure_collection(collection, vector_size=1536, distance="Cosine")

        for path in files:
            try:
                stats = await ingest_file(
                    path=path,
                    source_root=root,
                    store=store,
                    openai_client=openai_client,
                    collection=collection,
                )
                all_stats.append(stats)
            except Exception as exc:
                if strict:
                    raise
                rel = str(path.relative_to(root)).replace("\\", "/")
                logger.warning(
                    "ingest_file_skipped",
                    extra={"file": rel, "reason": str(exc)},
                )
                all_stats.append(
                    IngestStats(file=rel, chunks=0, tokens=0, skipped_reason=str(exc))
                )

    return all_stats


# ---------------------------------------------------------------------------
# typer entry point
# ---------------------------------------------------------------------------

app = typer.Typer(
    add_completion=False,
    help="Ingest a markdown corpus into the Qdrant vector store.",
)


@app.command()
def main(
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help="Path locale o URL git del corpus markdown.",
        ),
    ],
    collection: Annotated[
        str | None,
        typer.Option(
            "--collection",
            help="Nome della collection Qdrant. Default: settings.qdrant_collection_name.",
        ),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Fail-fast al primo errore (default: skip + summary).",
        ),
    ] = False,
    repo_subpath: Annotated[
        str | None,
        typer.Option(
            "--repo-subpath",
            help="Solo per --source URL: sub-path dentro al repo (es. docs/en/docs).",
        ),
    ] = None,
) -> None:
    """Ingest a markdown corpus into Qdrant."""
    coll = collection or settings.qdrant_collection_name

    try:
        stats = asyncio.run(_run(source, coll, strict, repo_subpath))
    except ValueError as exc:
        # Configurazione invalida (source path / OpenAI key) → exit 2.
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    # Summary
    ingested = [s for s in stats if s.skipped_reason is None]
    skipped = [s for s in stats if s.skipped_reason is not None]

    typer.echo("─" * 50)
    typer.echo(
        f"✓ {len(ingested)} files ingested ({len(skipped)} skipped)"
    )
    typer.echo(
        f"  {sum(s.chunks for s in ingested)} chunks, "
        f"{sum(s.tokens for s in ingested):,} tokens"
    )
    typer.echo("─" * 50)
    if skipped:
        typer.echo("Skipped:")
        for s in skipped:
            typer.echo(f"  - {s.file}: {s.skipped_reason}")


if __name__ == "__main__":
    app()
```

Nota: gli import di `_run`/`main` sono già completi sopra (`asyncio`, `Annotated`, `typer`, `settings`, `get_vector_store`); non serve altro.

- [ ] **Step 5.2: Aggiungi i CLI surface test**

In [apps/api/tests/test_ingest.py](../../../apps/api/tests/test_ingest.py), aggiungi in fondo:

```python


# ---------------------------------------------------------------------------
# CLI surface (typer CliRunner)
# ---------------------------------------------------------------------------

from typer.testing import CliRunner

from app.ingest import app as ingest_app


def test_cli_help_does_not_crash() -> None:
    """--help mostra la signature documentata."""
    runner = CliRunner()
    result = runner.invoke(ingest_app, ["--help"])
    assert result.exit_code == 0
    assert "Ingest a markdown corpus" in result.stdout


def test_cli_missing_source_fails(tmp_path: Path) -> None:
    """Senza --source typer alza errore di argomento."""
    runner = CliRunner()
    result = runner.invoke(ingest_app, [])
    assert result.exit_code != 0


def test_cli_invalid_source_exits_2(tmp_path: Path) -> None:
    """--source verso path inesistente → exit 2 con messaggio chiaro."""
    runner = CliRunner()
    missing = tmp_path / "does-not-exist"
    result = runner.invoke(ingest_app, ["--source", str(missing)])
    assert result.exit_code == 2
    assert "Configuration error" in result.stderr or "Configuration error" in result.stdout
```

Nota: typer.testing usa Click's CliRunner internamente; `stderr` può finire in `stdout` a seconda della versione. Il check con `or` copre entrambi.

- [ ] **Step 5.3: Run i test**

```powershell
uv run pytest tests/test_ingest.py -v
```

Expected: 18 passed (15 esistenti + 3 CLI).

- [ ] **Step 5.4: Smoke manuale del CLI**

```powershell
uv run python -m app.ingest --help
```

Expected output: `Usage: ... [OPTIONS]` con le 4 opzioni documentate.

- [ ] **Step 5.5: Ruff**

```powershell
uv run ruff format app/ingest.py tests/test_ingest.py
uv run ruff check --fix app/ingest.py tests/test_ingest.py
uv run ruff check app/ingest.py tests/test_ingest.py
```

- [ ] **Step 5.6: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/app/ingest.py apps/api/tests/test_ingest.py
git commit -m "feat(ingest): typer CLI command + _run async orchestrator"
```

---

## Task 6: Integration end-to-end + idempotency + resilient/strict

**Files:**
- Modify: `apps/api/tests/test_ingest.py`

- [ ] **Step 6.1: Aggiungi i test integration end-to-end**

In [apps/api/tests/test_ingest.py](../../../apps/api/tests/test_ingest.py), aggiungi in fondo:

```python


# ---------------------------------------------------------------------------
# Integration end-to-end (_run con Qdrant live + FakeOpenAIClient)
# ---------------------------------------------------------------------------

from app.ingest import _run


@pytest.mark.integration
async def test_run_happy_path_3_files(qdrant_store, fake_openai, tmp_path) -> None:
    """3 file markdown vengono tutti ingestati; Qdrant contiene N chunks."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        for i, content in enumerate(
            [
                "# Doc A\n\nFirst paragraph.\n",
                "# Doc B\n\nSecond paragraph.\n",
                "# Doc C\n\nThird paragraph.\n",
            ]
        ):
            (tmp_path / f"doc{i}.md").write_text(content)

        stats = await _run(
            source=str(tmp_path),
            collection=collection,
            strict=False,
            repo_subpath=None,
            store=qdrant_store,
            openai_client=fake_openai,
        )

        assert len(stats) == 3
        assert all(s.skipped_reason is None for s in stats)

        total_chunks = sum(s.chunks for s in stats)
        count_resp = await qdrant_store._client.count(collection, exact=True)
        assert count_resp.count == total_chunks
    finally:
        await qdrant_store.delete_collection(collection)


@pytest.mark.integration
async def test_run_idempotent_rerun(qdrant_store, fake_openai, tmp_path) -> None:
    """Re-run dello stesso ingest produce stesso count (overwrite per id)."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        (tmp_path / "a.md").write_text("# A\n\npara a\n")
        (tmp_path / "b.md").write_text("# B\n\npara b\n")

        await _run(
            source=str(tmp_path),
            collection=collection,
            strict=False,
            repo_subpath=None,
            store=qdrant_store,
            openai_client=fake_openai,
        )
        count1 = (await qdrant_store._client.count(collection, exact=True)).count

        await _run(
            source=str(tmp_path),
            collection=collection,
            strict=False,
            repo_subpath=None,
            store=qdrant_store,
            openai_client=fake_openai,
        )
        count2 = (await qdrant_store._client.count(collection, exact=True)).count

        assert count1 == count2
        assert count1 > 0
    finally:
        await qdrant_store.delete_collection(collection)


@pytest.mark.integration
async def test_run_resilient_skips_empty(qdrant_store, fake_openai, tmp_path) -> None:
    """Resilient default: file vuoto skippato, gli altri ingestati."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        (tmp_path / "ok.md").write_text("# ok\n\npara\n")
        (tmp_path / "empty.md").write_text("")

        stats = await _run(
            source=str(tmp_path),
            collection=collection,
            strict=False,
            repo_subpath=None,
            store=qdrant_store,
            openai_client=fake_openai,
        )

        ingested = [s for s in stats if s.skipped_reason is None]
        skipped = [s for s in stats if s.skipped_reason is not None]
        assert len(ingested) == 1
        assert len(skipped) == 1
        assert skipped[0].file == "empty.md"
        assert "empty" in skipped[0].skipped_reason
    finally:
        await qdrant_store.delete_collection(collection)


@pytest.mark.integration
async def test_run_strict_raises_on_empty(qdrant_store, fake_openai, tmp_path) -> None:
    """In --strict, il primo errore propaga."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        (tmp_path / "empty.md").write_text("")
        (tmp_path / "ok.md").write_text("# ok\n\npara\n")

        with pytest.raises(ValueError, match="empty"):
            await _run(
                source=str(tmp_path),
                collection=collection,
                strict=True,
                repo_subpath=None,
                store=qdrant_store,
                openai_client=fake_openai,
            )
    finally:
        await qdrant_store.delete_collection(collection)


@pytest.mark.integration
async def test_run_empty_dir_returns_empty(qdrant_store, fake_openai, tmp_path) -> None:
    """Directory senza .md → ritorna [] senza errore (no collection created)."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        stats = await _run(
            source=str(tmp_path),
            collection=collection,
            strict=False,
            repo_subpath=None,
            store=qdrant_store,
            openai_client=fake_openai,
        )
        assert stats == []
    finally:
        # collection potrebbe non esistere — delete_collection è no-op.
        await qdrant_store.delete_collection(collection)
```

- [ ] **Step 6.2: Run i test**

```powershell
uv run pytest tests/test_ingest.py -v
```

Expected: 23 passed (18 esistenti + 5 nuovi end-to-end).

- [ ] **Step 6.3: Verifica suite completa**

```powershell
uv run pytest -q
```

Expected: 73 + 23 = 96 passed (con magari il solito 1 warning Qdrant version).

- [ ] **Step 6.4: Ruff**

```powershell
uv run ruff format app/ingest.py tests/test_ingest.py
uv run ruff check --fix app/ingest.py tests/test_ingest.py
uv run ruff check app/ingest.py tests/test_ingest.py
```

- [ ] **Step 6.5: Commit**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add apps/api/tests/test_ingest.py
git commit -m "test(ingest): end-to-end integration (happy path, idempotency, strict)"
```

---

## Task 7: Smoke manuale con corpus reale + ROADMAP + final verification

**Files:**
- Modify: `docs/ROADMAP.md`

- [ ] **Step 7.1: Smoke test manuale con corpus reale**

Crea un piccolo corpus di prova e ingestalo per verificare end-to-end con OpenAI reale:

```powershell
$env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack\apps\api

# Crea 3 markdown di prova
$tmp = New-TemporaryFile | ForEach-Object { Remove-Item $_; New-Item -ItemType Directory -Path $_.FullName }
"# First`n`nFirst document about FastAPI." | Out-File "$($tmp.FullName)\a.md" -Encoding utf8
"# Second`n`nSecond document about Pydantic." | Out-File "$($tmp.FullName)\b.md" -Encoding utf8
"# Third`n`nThird document about ASGI." | Out-File "$($tmp.FullName)\c.md" -Encoding utf8

# Run l'ingest reale
uv run python -m app.ingest --source $tmp.FullName --collection smoke_test
```

Expected output (esempio):
```
─────────────────────────────────────────────
✓ 3 files ingested (0 skipped)
  3 chunks, 30 tokens
─────────────────────────────────────────────
```

E la dashboard Qdrant su http://localhost:6333/dashboard mostrerà la collection `smoke_test` con 3 punti.

Cleanup:

```powershell
# (Opzionale) elimina la collection di prova
uv run python -c "import asyncio; from app.rag.vector_store import get_vector_store; asyncio.run(get_vector_store().delete_collection('smoke_test'))"
Remove-Item -Recurse $tmp.FullName
```

- [ ] **Step 7.2: Aggiorna `docs/ROADMAP.md`**

In [docs/ROADMAP.md](../../ROADMAP.md), sezione M2 task #6, sostituisci:

```
6. **Ingestion CLI** (`app/ingest.py`): `uv run python -m app.ingest --source ./sample_docs --collection demo`
   — orchestra load → chunk → embed → upsert.
```

con:

```
6. ✅ **Ingestion CLI** (`app/ingest.py`): typer command che orchestra
   load → chunk → embed → upsert end-to-end. Supporta `--source` locale
   OR URL git (shallow clone), `--collection`, `--repo-subpath`, `--strict`.
   Resilient di default (skip + summary), `--strict` per fail-fast.
   Idempotente (overwrite per chunk id). 23 test verdi (unit + integration
   + CLI surface).
```

- [ ] **Step 7.3: Verifica finale**

```powershell
# Suite intera
uv run pytest -q

# Linter clean
uv run ruff check app/ tests/
uv run ruff format --check app/ tests/
```

Expected: 96 passed, all checks passed.

- [ ] **Step 7.4: Commit finale**

```powershell
Set-Location c:\Users\cesar\development\repos\agentic-rag-stack
git add docs/ROADMAP.md
git commit -m "docs(roadmap): mark M2 task #6 (ingestion CLI) as done"
```

- [ ] **Step 7.5: Verifica git log**

```powershell
git log --oneline -10
```

Expected: 7 commit nuovi su HEAD relativi a task #6 + 2 task pre-esistenti.

---

## Spec coverage map

| Spec requirement | Task |
|---|---|
| typer come CLI framework | Task 1 |
| `FakeOpenAIClient` riusabile (relocate) | Task 1 |
| `resolve_source` con local path | Task 2 |
| `resolve_source` con git URL + shallow clone | Task 2 |
| `resolve_source` con `--repo-subpath` | Task 2 |
| Detection regex `^(https?://|git@)` no collisioni Windows | Task 2 |
| `load_markdown_files` glob ricorsivo + esclusioni | Task 3 |
| `IngestStats` DTO | Task 4 |
| `ingest_file` per-file orchestrator | Task 4 |
| Payload Qdrant con `text, source, heading, position, token_count` | Task 4 |
| `_run` async orchestrator dep-injection friendly | Task 5 |
| `ensure_collection(1536, "Cosine")` chiamata da `_run` | Task 5 |
| CLI signature typer (`--source`, `--collection`, `--strict`, `--repo-subpath`) | Task 5 |
| Codici di uscita (0 ok / 1 strict-fail / 2 config-error) | Task 5 |
| Output summary con count + skip elenco | Task 5 |
| Idempotency via chunker sha256 | Task 6 |
| Resilient default + `--strict` flag | Task 5, Task 6 |
| Test unit puri | Task 2, Task 3 |
| Test integration end-to-end | Task 4, Task 6 |
| Test CLI surface (typer CliRunner) | Task 5 |
| OpenAI key mancante → exit 2 | Task 5 (`_run` ValueError + main catch) |
| Smoke manuale con corpus reale | Task 7 |

---

## Final acceptance check

A fine Task 7, eseguire:

```powershell
# 1. Suite verde
uv run --directory apps/api pytest -q

# 2. Linter clean
uv run --directory apps/api ruff check .
uv run --directory apps/api ruff format --check .

# 3. CLI smoke
uv run --directory apps/api python -m app.ingest --help

# 4. Smoke manuale (opzionale, richiede OPENAI_API_KEY valida + Qdrant up)
# Vedi step 7.1.
```

Expected:
- 96 test verdi
- 0 ruff errors
- `--help` mostra la signature documentata
- Smoke manuale popola Qdrant senza errori
