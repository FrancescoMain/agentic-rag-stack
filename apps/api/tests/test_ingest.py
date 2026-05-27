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
        with resolve_source("https://github.com/fake/repo") as root:
            assert root.is_dir()
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
            with resolve_source("https://github.com/fake/repo", repo_subpath="not-a-real-subpath"):
                pass  # pragma: no cover


def test_resolve_source_windows_path_not_url() -> None:
    """Path Windows-style con drive letter non collide col regex git.

    `C:\\foo` non matcha `^(https?://|git@)`. Va trattato come path locale.
    """
    with pytest.raises(ValueError, match="non è una directory"):
        with resolve_source("C:\\non-esiste-davvero"):
            pass  # pragma: no cover
