"""
app/rag/retriever.py
====================
Dense semantic retriever su Qdrant.

------------------------------------------------------------------------
Cosa fa
------------------------------------------------------------------------

Dato un testo in linguaggio naturale (la "query"), produce i top-k
chunk più simili dalla collection Qdrant. Workflow:

    embed(query) -> vector_store.search(...) -> list[Match]

I componenti sottostanti (embedder + vector_store) sono già stati
costruiti nei task #4 e #5; questo modulo è puro orchestratore.

------------------------------------------------------------------------
Limiti noti (importanti)
------------------------------------------------------------------------

Fa SOLO dense search. Cattura bene la somiglianza semantica di
query in linguaggio naturale, ma può sotto-performare su query
lessicali esatte (codici errore, identifier, version numbers). Quando
l'eval (task #11-12) misurerà precision@k su un golden dataset, potremo
decidere se introdurre hybrid (sparse BM25 + RRF fusion) come task #7b.

Vedi `docs/superpowers/specs/2026-05-27-dense-retriever-design.md`.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from app.config import settings
from app.rag.embedder import embed_texts
from app.rag.vector_store import Match, VectorStore, get_vector_store

logger = logging.getLogger(__name__)


class Retriever:
    """Dense semantic retriever su Qdrant.

    Costruttore prende store + openai_client espliciti — dependency
    injection-friendly per i test. In produzione `get_retriever()`
    fornisce un singleton configurato dalle settings.
    """

    def __init__(
        self,
        store: VectorStore,
        openai_client: AsyncOpenAI,
    ) -> None:
        self._store = store
        self._openai_client = openai_client

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Match]:
        """Embed query → search dense → return top-k Match.

        Args:
            query: testo in linguaggio naturale. Stringa non vuota
                (whitespace-only viene rifiutato).
            collection: nome collection Qdrant.
            top_k: numero max risultati. Default 5.
            filter: shallow equality match sul payload (es. {"source": "X"}).

        Returns:
            list[Match] ordinata per score decrescente. Vuota se la
            collection è vuota o nessun risultato matcha il filter.

        Raises:
            ValueError: se la query è vuota o whitespace-only.
            EmbedderError: se l'embedding fallisce dopo tutti i retry.
            Eccezioni vector_store propagate.
        """
        if not query.strip():
            raise ValueError("query cannot be empty")

        embeddings = await embed_texts(self._openai_client, [query])
        # embed_texts garantisce len(result) == len(input), quindi
        # l'indice [0] è sicuro.
        query_vector = embeddings[0].vector

        matches = await self._store.search(
            collection=collection,
            query=query_vector,
            top_k=top_k,
            filter=filter,
        )

        logger.info(
            "retrieve_done",
            extra={
                "collection": collection,
                "query_len": len(query),
                "top_k": top_k,
                "results": len(matches),
            },
        )
        return matches


# ---------------------------------------------------------------------------
# Factory singleton
# ---------------------------------------------------------------------------

_retriever_singleton: Retriever | None = None


def get_retriever() -> Retriever:
    """Singleton modulo-livello.

    Lazy init: prima chiamata costruisce store (via get_vector_store)
    + AsyncOpenAI (via settings.openai_api_key). Solleva ValueError
    se OPENAI_API_KEY è vuota.

    Pensato per `Depends(get_retriever)` da endpoint FastAPI (task #10).
    """
    global _retriever_singleton
    if _retriever_singleton is None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY non configurata. Imposta la variabile nel file .env.")
        _retriever_singleton = Retriever(
            store=get_vector_store(),
            openai_client=AsyncOpenAI(api_key=settings.openai_api_key),
        )
    return _retriever_singleton


__all__ = ["Retriever", "get_retriever"]
