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

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from app.ingest import (
    IngestStats,
    _run,
    ingest_file,
    load_markdown_files,
    resolve_source,
)
from app.ingest import (
    app as ingest_app,
)

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
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")

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


# ---------------------------------------------------------------------------
# CLI surface (typer CliRunner)
# ---------------------------------------------------------------------------


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
    combined = (result.stderr or "") + (result.stdout or "")
    assert "Configuration error" in combined


# ---------------------------------------------------------------------------
# Integration end-to-end (_run con Qdrant live + FakeOpenAIClient)
# ---------------------------------------------------------------------------


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
