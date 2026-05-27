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
