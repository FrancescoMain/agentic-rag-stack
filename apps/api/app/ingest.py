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

import asyncio
import logging
import re
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.rag.chunker import chunk_markdown
from app.rag.embedder import embed_texts
from app.rag.vector_store import VectorPoint, VectorStore, get_vector_store

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
                raise ValueError(f"repo_subpath '{repo_subpath}' non trovato in {source}")
            yield root
    else:
        path = Path(source).resolve()
        if not path.is_dir():
            raise ValueError(f"source '{source}' non è una directory esistente")
        yield path


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
    filtered = [p for p in all_md if not any(part in _EXCLUDED_DIRS for part in p.parts)]
    return sorted(filtered, key=str)


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
            raise ValueError("OPENAI_API_KEY non configurata. Imposta la variabile nel file .env.")
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
                all_stats.append(IngestStats(file=rel, chunks=0, tokens=0, skipped_reason=str(exc)))

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

    # Summary. ASCII-only per portabilità Windows (cp1252) — i caratteri
    # box-drawing Unicode farebbero crashare typer.echo sul PowerShell default.
    ingested = [s for s in stats if s.skipped_reason is None]
    skipped = [s for s in stats if s.skipped_reason is not None]

    typer.echo("-" * 50)
    typer.echo(f"OK: {len(ingested)} files ingested ({len(skipped)} skipped)")
    typer.echo(
        f"  {sum(s.chunks for s in ingested)} chunks, {sum(s.tokens for s in ingested):,} tokens"
    )
    typer.echo("-" * 50)
    if skipped:
        typer.echo("Skipped:")
        for s in skipped:
            typer.echo(f"  - {s.file}: {s.skipped_reason}")


if __name__ == "__main__":
    app()
