"""
tests/test_rag/test_chunker.py
==============================
Test sul chunker markdown (`app/rag/chunker.py`).

Strategia: testiamo l'API pubblica (`chunk_markdown`) sui casi rilevanti:
  - Edge cases: vuoto, whitespace, validazione config.
  - Single chunk (doc piccolo).
  - Multi chunk con heading (heading-aware).
  - Sliding window (doc grande senza heading).
  - Idempotenza degli id (re-chunkare = stessi id).
  - Overlap reale (chunk consecutivi condividono token).
  - Heading-path (gerarchia rispettata).

Tutto in-process: il chunker non fa I/O né chiamate di rete. Veloce.
"""

from __future__ import annotations

import pytest

from app.rag import ChunkerConfig, chunk_markdown

# ----------------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------------


def test_chunk_markdown_empty_returns_empty() -> None:
    """Documento vuoto → lista vuota, nessun errore."""
    assert chunk_markdown("", source="empty.md") == []


def test_chunk_markdown_only_whitespace_returns_empty() -> None:
    """Solo spazi/newline → lista vuota."""
    assert chunk_markdown("   \n\n  \t  \n", source="ws.md") == []


def test_chunk_markdown_invalid_config_raises() -> None:
    """Overlap >= chunk_size deve alzare ValueError esplicito al chiamante."""
    bad_config = ChunkerConfig(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_markdown("ciao", source="x.md", config=bad_config)


# ----------------------------------------------------------------------------
# Single chunk path (sezione che entra nel chunk_size)
# ----------------------------------------------------------------------------


def test_short_document_produces_single_chunk() -> None:
    """Un documento corto produce esattamente 1 chunk."""
    text = "Just a short paragraph about FastAPI dependency injection."
    chunks = chunk_markdown(text, source="short.md")
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].source == "short.md"
    assert chunks[0].position == 0
    assert chunks[0].heading is None  # nessuna heading nel doc
    assert chunks[0].token_count > 0


def test_chunk_has_deterministic_id() -> None:
    """Stesso input → stesso id, indipendentemente dall'esecuzione."""
    text = "FastAPI is a modern web framework for building APIs with Python."
    chunks_run1 = chunk_markdown(text, source="doc.md")
    chunks_run2 = chunk_markdown(text, source="doc.md")
    assert chunks_run1[0].id == chunks_run2[0].id
    # Id formato: 16 hex chars (vedi _deterministic_id).
    assert len(chunks_run1[0].id) == 16
    int(chunks_run1[0].id, 16)  # parsing hex non deve esplodere


def test_different_sources_produce_different_ids() -> None:
    """Stesso testo, source diverso → id diverso (perché source è nella key)."""
    text = "Same content."
    c1 = chunk_markdown(text, source="a.md")[0]
    c2 = chunk_markdown(text, source="b.md")[0]
    assert c1.id != c2.id


# ----------------------------------------------------------------------------
# Heading-aware splitting
# ----------------------------------------------------------------------------


def test_single_heading_propagates_to_chunk() -> None:
    """Una heading di primo livello deve apparire come heading-path del chunk."""
    text = "# Introduction\n\nFastAPI is a modern framework."
    chunks = chunk_markdown(text, source="intro.md")
    assert len(chunks) == 1
    assert chunks[0].heading == "Introduction"


def test_nested_headings_produce_path() -> None:
    """Heading nidificate → heading-path concatenata con ' > '."""
    text = (
        "# Tutorial\n"
        "\n"
        "Intro to the tutorial.\n"
        "\n"
        "## Dependencies\n"
        "\n"
        "How to use Depends().\n"
        "\n"
        "### Sub-dependencies\n"
        "\n"
        "You can have dependencies of dependencies.\n"
    )
    chunks = chunk_markdown(text, source="tut.md")
    headings = [c.heading for c in chunks]
    assert "Tutorial" in headings
    assert any(h and h.endswith("Tutorial > Dependencies") for h in headings)
    assert any(h and h.endswith("Tutorial > Dependencies > Sub-dependencies") for h in headings)


def test_sibling_heading_pops_deeper_level() -> None:
    """Dopo `### Foo` un nuovo `## Bar` deve azzerare il livello 3."""
    text = "## Section A\n\nBody A.\n\n### Subsection A1\n\nBody A1.\n\n## Section B\n\nBody B.\n"
    chunks = chunk_markdown(text, source="sib.md")
    headings = {c.heading for c in chunks}
    # "Section B" non deve trascinarsi dietro "Subsection A1".
    assert "Section B" in headings
    assert "Section B > Subsection A1" not in headings


def test_text_before_first_heading_has_none_heading() -> None:
    """Testo prima della prima heading → chunk con heading=None."""
    text = "Some preamble text.\n\n# First Heading\n\nSection content.\n"
    chunks = chunk_markdown(text, source="preamble.md")
    assert chunks[0].heading is None
    assert chunks[0].text.startswith("Some preamble")


# ----------------------------------------------------------------------------
# Sliding window (sezioni più grandi di chunk_size)
# ----------------------------------------------------------------------------


def test_large_section_produces_multiple_chunks_with_overlap() -> None:
    """Una sezione molto più grande di chunk_size produce N chunk con overlap.

    Costruiamo una sezione di ~3000 token. Con chunk_size=300, overlap=50
    ci aspettiamo circa 3000 / (300-50) ≈ 12 chunk (più o meno).
    """
    # Una frase ripetuta tante volte → produce un testo lungo deterministico.
    big_section = ("FastAPI uses Pydantic for data validation. " * 400).strip()
    config = ChunkerConfig(chunk_size=300, chunk_overlap=50)
    chunks = chunk_markdown(big_section, source="big.md", config=config)

    assert len(chunks) > 5, "Doc grande dovrebbe produrre >5 chunk"
    # Ogni chunk (tranne forse l'ultimo) deve essere vicino a chunk_size.
    for chunk in chunks[:-1]:
        assert chunk.token_count <= config.chunk_size
        assert chunk.token_count >= config.chunk_size - 10  # tolleranza
    # L'ultimo può essere più corto.
    assert chunks[-1].token_count <= config.chunk_size

    # Position deve essere progressiva e contigua.
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_sliding_window_chunks_share_overlapping_text() -> None:
    """Due chunk consecutivi devono condividere del testo (overlap)."""
    big_section = ("Hello FastAPI dependency injection example. " * 200).strip()
    config = ChunkerConfig(chunk_size=200, chunk_overlap=50)
    chunks = chunk_markdown(big_section, source="ov.md", config=config)

    assert len(chunks) >= 2, "Servono almeno 2 chunk per testare overlap"
    # L'ultima parte del chunk N e l'inizio del chunk N+1 devono
    # contenere parole comuni. Test euristico ma robusto: prendiamo
    # le ultime 20 parole del primo chunk e controlliamo che almeno
    # alcune appaiano all'inizio del secondo chunk.
    tail_words = set(chunks[0].text.split()[-20:])
    head_words = set(chunks[1].text.split()[:20])
    common = tail_words & head_words
    assert common, "Chunk consecutivi devono condividere parole (overlap)"


# ----------------------------------------------------------------------------
# Realistic markdown (mini-snippet stile FastAPI docs)
# ----------------------------------------------------------------------------


_FASTAPI_LIKE_DOC = """\
# Tutorial - User Guide

This section shows you how to use FastAPI step by step.

## First Steps

The simplest FastAPI file looks like this:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

### Run it

Run the server with:

```bash
fastapi dev main.py
```

## Path Parameters

You can declare path parameters with the same syntax used by Python format strings.
"""


def test_fastapi_like_doc_produces_well_structured_chunks() -> None:
    """Su markdown realistico: chunk multipli con heading-path corrette."""
    chunks = chunk_markdown(_FASTAPI_LIKE_DOC, source="tut.md")

    # Ci aspettiamo almeno 3-4 chunk (uno per heading principale).
    assert len(chunks) >= 3

    # Tutte le heading-path attese devono essere presenti.
    headings = [c.heading for c in chunks]
    assert any(h == "Tutorial - User Guide" for h in headings)
    assert any(h and h.endswith("First Steps") for h in headings), (
        f"First Steps non trovata in {headings}"
    )
    assert any(h and h.endswith("First Steps > Run it") for h in headings)
    assert any(h and h.endswith("Path Parameters") for h in headings)

    # Verifica che i chunk contengano effettivamente il testo atteso.
    full_text = "\n".join(c.text for c in chunks)
    assert "Hello World" in full_text
    assert "fastapi dev" in full_text
    assert "path parameters" in full_text.lower()
