"""
tests/test_rag/test_retriever.py
================================
Test del retriever dense (`app/rag/retriever.py`).

Tutti integration: il retriever è puro orchestratore (3 righe di logica
interna), quindi non c'è valore in test unit della logica — testiamo
end-to-end con Qdrant live + FakeOpenAIClient.

Eccezione: i 2 test del singleton `get_retriever()` sono unit puri
(non toccano né Qdrant né OpenAI).
"""

from __future__ import annotations

import uuid

import pytest

from app.rag.retriever import Retriever
from app.rag.vector_store import VectorPoint

# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_retrieve_empty_collection_returns_empty(qdrant_store, fake_openai) -> None:
    """Collection vuota → lista vuota, no errore."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")
        retriever = Retriever(store=qdrant_store, openai_client=fake_openai)

        result = await retriever.retrieve(query="qualsiasi cosa", collection=collection, top_k=5)
        assert result == []
    finally:
        await qdrant_store.delete_collection(collection)


def _padded_vec(values: list[float]) -> list[float]:
    """Helper: crea un vettore 1536-dim con i `values` in cima e 0 nel resto."""
    return values + [0.0] * (1536 - len(values))


@pytest.mark.integration
async def test_retrieve_returns_top_k_ordered(qdrant_store, fake_openai) -> None:
    """retrieve ritorna top_k ordinato per score decrescente.

    Strategia: i vettori dei chunk sono costruiti a mano (one-hot di
    posizioni diverse) per avere cosine similarities distinti con il
    vettore della query. La query embedda via fake_openai dopo aver
    override-ato vector_for per ritornare un vettore "puntato" sulla
    posizione 0 (così "chunk-near" con vector=[1,0,...] avrà cosine=1).
    """
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")

        # Override del vector_for SOLO per questo test: la query "q"
        # produce un vettore [1, 0, 0, ..., 0].
        fake_openai.vector_for = lambda text: _padded_vec([1.0])

        # Tre chunk con vettori distintivi:
        # - chunk-near: identico al vettore query → cosine = 1.0
        # - chunk-mid: parzialmente sovrapposto → cosine ~0.707
        # - chunk-far: ortogonale → cosine ~0
        await qdrant_store.upsert(
            collection,
            [
                VectorPoint(id="chunk-near", vector=_padded_vec([1.0]), payload={}),
                VectorPoint(id="chunk-mid", vector=_padded_vec([0.5, 0.5]), payload={}),
                VectorPoint(id="chunk-far", vector=_padded_vec([0.0, 0.0, 1.0]), payload={}),
            ],
        )

        retriever = Retriever(store=qdrant_store, openai_client=fake_openai)
        result = await retriever.retrieve(query="q", collection=collection, top_k=3)

        assert len(result) == 3
        assert result[0].id == "chunk-near"
        assert result[1].id == "chunk-mid"
        assert result[2].id == "chunk-far"
        # Ordine decrescente:
        assert result[0].score >= result[1].score >= result[2].score
    finally:
        await qdrant_store.delete_collection(collection)


@pytest.mark.integration
async def test_retrieve_respects_filter(qdrant_store, fake_openai) -> None:
    """filter={'source': X} → solo punti con payload['source']==X."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")
        retriever = Retriever(store=qdrant_store, openai_client=fake_openai)

        # 4 chunk con vector identico (così il filter è l'unico
        # discriminante) ma source diverso.
        same_vec = fake_openai.vector_for("any")
        for i, source in enumerate(["a.md", "a.md", "b.md", "b.md"]):
            await qdrant_store.upsert(
                collection,
                [
                    VectorPoint(
                        id=f"chunk-{i}",
                        vector=same_vec,
                        payload={"source": source, "text": f"chunk {i}"},
                    )
                ],
            )

        result = await retriever.retrieve(
            query="any",
            collection=collection,
            top_k=10,
            filter={"source": "a.md"},
        )

        assert len(result) == 2
        assert all(m.payload["source"] == "a.md" for m in result)
    finally:
        await qdrant_store.delete_collection(collection)


@pytest.mark.integration
async def test_retrieve_top_k_larger_than_collection(qdrant_store, fake_openai) -> None:
    """top_k > numero punti → ritorna tutti i punti, no errore."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")
        retriever = Retriever(store=qdrant_store, openai_client=fake_openai)

        for i in range(3):
            await qdrant_store.upsert(
                collection,
                [
                    VectorPoint(
                        id=f"chunk-{i}",
                        vector=fake_openai.vector_for(f"text-{i}"),
                        payload={"i": i},
                    )
                ],
            )

        result = await retriever.retrieve(query="hi", collection=collection, top_k=10)
        assert len(result) == 3
    finally:
        await qdrant_store.delete_collection(collection)


@pytest.mark.integration
async def test_retrieve_empty_query_raises_before_calling_openai(qdrant_store, fake_openai) -> None:
    """Query vuota → ValueError PRIMA di toccare OpenAI (no waste)."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")
        retriever = Retriever(store=qdrant_store, openai_client=fake_openai)

        with pytest.raises(ValueError, match="query cannot be empty"):
            await retriever.retrieve(query="", collection=collection)

        # Garanzia di no-waste: niente call OpenAI.
        assert fake_openai.calls == []
    finally:
        await qdrant_store.delete_collection(collection)


@pytest.mark.integration
async def test_retrieve_whitespace_only_query_raises(qdrant_store, fake_openai) -> None:
    """Stringa solo whitespace è equivalente a vuota."""
    collection = f"_test_{uuid.uuid4().hex[:12]}"
    try:
        await qdrant_store.ensure_collection(collection, vector_size=1536, distance="Cosine")
        retriever = Retriever(store=qdrant_store, openai_client=fake_openai)

        with pytest.raises(ValueError, match="query cannot be empty"):
            await retriever.retrieve(query="   \n  \t", collection=collection)

        assert fake_openai.calls == []
    finally:
        await qdrant_store.delete_collection(collection)
