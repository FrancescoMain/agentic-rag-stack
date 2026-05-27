"""
tests/test_rag/test_vector_store.py
===================================
Test del modulo `app/rag/vector_store.py`.

Convenzioni:
- Test unit "puri" (mapping helpers, DTO): nessun marker, girano sempre.
- Test integration che richiedono Qdrant live: marker `@pytest.mark.integration`,
  skippati se Qdrant non risponde su /readyz.

Per girare solo unit:    `uv run pytest -m "not integration" tests/test_rag/test_vector_store.py`
Per girare integration:  `uv run pytest -m integration       tests/test_rag/test_vector_store.py`
"""

from __future__ import annotations

import pytest

from app.rag.vector_store import Match, VectorPoint, VectorStore

# ---------------------------------------------------------------------------
# DTO smoke tests (unit, no Qdrant)
# ---------------------------------------------------------------------------


def test_vector_point_minimal() -> None:
    """VectorPoint accetta id non vuoto + vector non vuoto, payload default {}."""
    point = VectorPoint(id="abc123", vector=[0.1, 0.2, 0.3])
    assert point.id == "abc123"
    assert point.vector == [0.1, 0.2, 0.3]
    assert point.payload == {}


def test_vector_point_with_payload() -> None:
    """VectorPoint accetta dict arbitrario come payload."""
    point = VectorPoint(
        id="x",
        vector=[1.0],
        payload={"text": "hello", "source": "intro.md", "heading": "H1"},
    )
    assert point.payload["text"] == "hello"
    assert point.payload["source"] == "intro.md"


def test_vector_point_empty_id_rejected() -> None:
    """id stringa vuota viene rifiutata da Pydantic."""
    with pytest.raises(ValueError):
        VectorPoint(id="", vector=[0.1])


def test_vector_point_empty_vector_rejected() -> None:
    """vector vuoto viene rifiutato (un punto senza embedding non ha senso)."""
    with pytest.raises(ValueError):
        VectorPoint(id="x", vector=[])


def test_match_basic() -> None:
    """Match si costruisce con id + score + payload opzionale."""
    m = Match(id="x", score=0.95, payload={"text": "hi"})
    assert m.id == "x"
    assert m.score == 0.95
    assert m.payload == {"text": "hi"}


def test_match_default_payload() -> None:
    """Match permette payload omesso → {} default."""
    m = Match(id="x", score=0.5)
    assert m.payload == {}


# ---------------------------------------------------------------------------
# Protocol smoke test (unit)
# ---------------------------------------------------------------------------


def test_protocol_is_importable() -> None:
    """Il Protocol è importabile e definisce i 4 metodi del contratto.

    Non istanziamo Protocol direttamente (è una "interfaccia"); verifichiamo
    solo che esista come simbolo e che abbia i metodi attesi.
    """
    methods = {"ensure_collection", "upsert", "search", "delete_collection"}
    assert methods.issubset(set(dir(VectorStore)))
