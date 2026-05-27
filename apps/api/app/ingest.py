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
                raise ValueError(f"repo_subpath '{repo_subpath}' non trovato in {source}")
            yield root
    else:
        path = Path(source).resolve()
        if not path.is_dir():
            raise ValueError(f"source '{source}' non è una directory esistente")
        yield path
