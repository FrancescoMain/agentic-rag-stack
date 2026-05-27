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
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, ScoredPoint

from app.rag.vector_store import (
    _RESERVED_PAYLOAD_KEY,
    Match,
    VectorPoint,
    VectorStore,
    _filter_to_qdrant,
    _from_qdrant_scored_point,
    _to_qdrant_point,
)

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


# ---------------------------------------------------------------------------
# Mapping helpers (unit, no Qdrant)
# ---------------------------------------------------------------------------


def test_to_qdrant_point_normalizes_id_to_uuid() -> None:
    """Un id stringa arbitraria viene normalizzato a UUID v5 deterministico."""
    vp = VectorPoint(id="not-a-uuid", vector=[0.1, 0.2], payload={"text": "hi"})
    ps = _to_qdrant_point(vp)
    assert isinstance(ps, PointStruct)
    # Lo UUID prodotto è una stringa nel formato xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.
    assert isinstance(ps.id, str)
    assert len(ps.id) == 36
    assert ps.id.count("-") == 4


def test_to_qdrant_point_id_is_deterministic() -> None:
    """Stesso id originale → stesso UUID Qdrant (upsert idempotente)."""
    vp1 = VectorPoint(id="stable-id", vector=[0.1])
    vp2 = VectorPoint(id="stable-id", vector=[0.9])  # vector diverso, id uguale
    assert _to_qdrant_point(vp1).id == _to_qdrant_point(vp2).id


def test_to_qdrant_point_preserves_original_id_in_payload() -> None:
    """L'id originale viene preservato nel payload sotto chiave riservata."""
    vp = VectorPoint(id="sha256-abc", vector=[0.1], payload={"text": "hi"})
    ps = _to_qdrant_point(vp)
    assert ps.payload[_RESERVED_PAYLOAD_KEY] == "sha256-abc"
    # Il resto del payload utente è preservato:
    assert ps.payload["text"] == "hi"


def test_from_qdrant_scored_point_restores_original_id() -> None:
    """Il Match.id è l'id originale, non l'UUID Qdrant. La chiave riservata
    viene rimossa dal payload pubblico."""
    sp = ScoredPoint(
        id="00000000-0000-0000-0000-000000000001",
        version=0,
        score=0.87,
        payload={_RESERVED_PAYLOAD_KEY: "original-sha256-xyz", "text": "hi"},
    )
    m = _from_qdrant_scored_point(sp)
    assert m.id == "original-sha256-xyz"
    assert m.score == 0.87
    assert m.payload == {"text": "hi"}
    # La chiave riservata NON deve trapelare al consumer:
    assert _RESERVED_PAYLOAD_KEY not in m.payload


def test_from_qdrant_scored_point_handles_missing_reserved_key() -> None:
    """Fallback robusto: se per qualche motivo manca la chiave riservata
    (es. punto inserito direttamente bypassando il nostro upsert),
    il Match.id ricade sull'UUID Qdrant come stringa."""
    sp = ScoredPoint(
        id="00000000-0000-0000-0000-000000000002",
        version=0,
        score=0.5,
        payload={"text": "hi"},  # niente __vp_id
    )
    m = _from_qdrant_scored_point(sp)
    assert m.id == "00000000-0000-0000-0000-000000000002"
    assert m.payload == {"text": "hi"}


def test_filter_to_qdrant_none_returns_none() -> None:
    """filter=None → None (Qdrant accetta None per 'nessun filtro')."""
    assert _filter_to_qdrant(None) is None


def test_filter_to_qdrant_empty_dict_returns_none() -> None:
    """filter={} → None (nessuna condizione = nessun filtro)."""
    assert _filter_to_qdrant({}) is None


def test_filter_to_qdrant_single_key_produces_field_condition() -> None:
    """{key: value} → Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])."""
    f = _filter_to_qdrant({"source": "intro.md"})
    assert isinstance(f, Filter)
    assert f.must is not None
    assert len(f.must) == 1
    cond = f.must[0]
    assert isinstance(cond, FieldCondition)
    assert cond.key == "source"
    assert isinstance(cond.match, MatchValue)
    assert cond.match.value == "intro.md"


def test_filter_to_qdrant_multiple_keys_produces_and() -> None:
    """{k1: v1, k2: v2} → tutti come `must` (AND logico)."""
    f = _filter_to_qdrant({"source": "a.md", "heading": "H1"})
    assert isinstance(f, Filter)
    assert f.must is not None
    assert len(f.must) == 2
    keys = sorted(cond.key for cond in f.must)
    assert keys == ["heading", "source"]


# ---------------------------------------------------------------------------
# Integration tests — Qdrant live (skip se /readyz down)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_ensure_collection_creates_new(qdrant_store, unique_collection) -> None:
    """Una collection nuova viene creata correttamente."""
    # Pre: la collection NON esiste.
    assert not await qdrant_store._client.collection_exists(unique_collection)
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    # Post: la collection esiste, con la dimensione richiesta.
    assert await qdrant_store._client.collection_exists(unique_collection)
    info = await qdrant_store._client.get_collection(unique_collection)
    existing_params = info.config.params.vectors
    size = (
        next(iter(existing_params.values())).size
        if isinstance(existing_params, dict)
        else existing_params.size
    )
    assert size == 4


@pytest.mark.integration
async def test_ensure_collection_idempotent_same_config(qdrant_store, unique_collection) -> None:
    """Chiamare ensure_collection due volte con stessa config → no-op."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    # Seconda chiamata non deve sollevare nulla.
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)


@pytest.mark.integration
async def test_ensure_collection_size_mismatch_raises(qdrant_store, unique_collection) -> None:
    """Stessa collection con vector_size diverso → ValueError."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    with pytest.raises(ValueError, match="vector_size"):
        await qdrant_store.ensure_collection(unique_collection, vector_size=8)


@pytest.mark.integration
async def test_delete_collection_existing(qdrant_store, unique_collection) -> None:
    """delete_collection rimuove una collection esistente."""
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)
    await qdrant_store.delete_collection(unique_collection)
    # Dopo delete, ricreiamo la collection pulita così il teardown della
    # fixture ha qualcosa da pulire (anche se delete su missing è no-op).
    await qdrant_store.ensure_collection(unique_collection, vector_size=4)


@pytest.mark.integration
async def test_delete_collection_missing_is_noop(qdrant_store) -> None:
    """delete_collection su una collection inesistente → no-op (no raise)."""
    import uuid as _uuid

    fake = f"_test_does_not_exist_{_uuid.uuid4().hex[:8]}"
    # Non sollevare. Se solleva, il test fallisce naturalmente.
    await qdrant_store.delete_collection(fake)


@pytest.mark.integration
async def test_ensure_collection_invalid_distance_raises(qdrant_store, unique_collection) -> None:
    """Distance non supportata → ValueError prima di toccare Qdrant."""
    with pytest.raises(ValueError, match="Distance"):
        await qdrant_store.ensure_collection(unique_collection, vector_size=4, distance="Manhattan")
