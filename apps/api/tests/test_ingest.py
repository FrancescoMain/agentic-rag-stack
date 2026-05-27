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

from app.ingest import load_markdown_files, resolve_source

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


# ---------------------------------------------------------------------------
# Unit: load_markdown_files
# ---------------------------------------------------------------------------


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
