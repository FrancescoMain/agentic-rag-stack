"""
app/rag/vector_store.py
=======================
Astrazione di accesso al vector database.

Espone:
- `VectorPoint` (Pydantic): input di `upsert`. id stringa stabile +
  vettore float + payload arbitrario (testo + metadata).
- `Match` (Pydantic): output di `search`. id + score + payload.
- `VectorStore` (typing.Protocol): l'interfaccia astratta — 4 metodi async.
- `QdrantVectorStore`: implementazione concreta su `AsyncQdrantClient`.
- `get_vector_store()`: factory singleton modulo-livello.

------------------------------------------------------------------------
Perché un Protocol e non una `abc.ABC`?
------------------------------------------------------------------------

Python 3.8+ ha `typing.Protocol` (PEP 544): definisce "duck typing
strutturale" — una classe soddisfa il Protocol se ha i metodi giusti,
SENZA bisogno di ereditare esplicitamente. Vantaggi rispetto ad ABC:

- Niente accoppiamento di ereditarietà: domani posso fare un
  `InMemoryVectorStore` per test senza modificare nessun import.
- Più Pythonic 2026: lo usano LangChain, LlamaIndex, fastapi.
- Più test-friendly: nei test posso passare un duck-typed object.

Per i lettori Java/C#: pensa a "interface implicita".

------------------------------------------------------------------------
Mapping ID interno (dettaglio implementativo)
------------------------------------------------------------------------

Qdrant accetta come ID di un punto SOLO uint64 o UUID. Il chunker
produce `Chunk.id` come SHA-256 hex (64 char), non valido. Soluzione:
internamente `QdrantVectorStore` mappa qualsiasi stringa stabile a un
UUID v5 deterministico (namespace fisso), e preserva l'id originale
nel payload sotto chiave `__vp_id`. Il `Match.id` ritorna sempre
l'id originale, mai l'UUID Qdrant — il consumer non si accorge.

Vedi ADR-0003 per la motivazione di Qdrant e ADR-0004 per gli
embedding (text-embedding-3-small, 1536-dim).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DTO
# ---------------------------------------------------------------------------


class VectorPoint(BaseModel):
    """Punto vettoriale da upsertare.

    Attributes:
        id: identità STABILE (idempotenza). Una stringa qualunque; il
            QdrantVectorStore la normalizza internamente a UUID v5 e
            preserva l'originale nel payload. Il chunker produce sha256.
        vector: embedding, lista di float. La dimensione DEVE coincidere
            con il `vector_size` con cui è stata creata la collection
            (es. 1536 per text-embedding-3-small). Mismatch → errore
            Qdrant a upsert.
        payload: metadata + dati. Convenzione (non enforced):
            `{text, source, heading, position, token_count}`. La chiave
            `__vp_id` è RISERVATA all'uso interno — non usarla.
    """

    id: str = Field(min_length=1)
    vector: list[float] = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class Match(BaseModel):
    """Risultato di una ricerca.

    Attributes:
        id: id ORIGINALE (quello passato all'upsert come `VectorPoint.id`),
            non l'UUID Qdrant interno. La chiave `__vp_id` è già stata
            rimossa dal payload.
        score: similarity score. Per Cosine distance, range tipico
            [0, 1] su testi normalizzati; range teorico [-1, 1].
            Maggiore = più simile.
        payload: payload originale dell'upsert (depurato della chiave
            riservata interna).
    """

    id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class VectorStore(Protocol):
    """Interfaccia astratta per un vector database.

    Ogni metodo è async per coerenza con il resto dello stack (FastAPI,
    embedder) e per non bloccare l'event loop sulle chiamate di rete.

    Convenzione: gli errori "di configurazione" (vector_size mismatch su
    ensure_collection) sono `ValueError`. Gli errori di rete o del servizio
    propagano come eccezioni del client sottostante (es. `qdrant_client`
    alza `UnexpectedResponse` per gli HTTP error).
    """

    async def ensure_collection(
        self,
        name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        """Crea la collection se non esiste; no-op se esiste con stessa config.

        Args:
            name: nome della collection.
            vector_size: dimensione del vettore (es. 1536).
            distance: "Cosine" | "Dot" | "Euclid". Default "Cosine".

        Raises:
            ValueError: se la collection esiste ma con vector_size diverso.
        """
        ...

    async def upsert(
        self,
        collection: str,
        points: Sequence[VectorPoint],
    ) -> int:
        """Inserisce o aggiorna i punti. Idempotente per id.

        Args:
            collection: nome collection.
            points: sequenza di VectorPoint. Se >100, internamente
                batchato in chiamate di 100 per chiamata.

        Returns:
            Numero di punti inviati (= len(points)).
        """
        ...

    async def search(
        self,
        collection: str,
        query: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Match]:
        """Ricerca i top-k punti più simili a `query`.

        Args:
            collection: nome collection.
            query: vettore di query (stessa dimensione della collection).
            top_k: numero massimo di risultati. Default 5.
            filter: shallow equality match sul payload, es.
                `{"source": "intro.md"}` → solo punti con
                `payload["source"] == "intro.md"`. None → nessun filtro.
                Multiple chiavi → AND logico.

        Returns:
            Lista di Match ordinata per score decrescente.
            Lista vuota se collection vuota o nessun match.
        """
        ...

    async def delete_collection(self, name: str) -> None:
        """Elimina la collection. No-op se non esiste."""
        ...


# ---------------------------------------------------------------------------
# Implementazione concreta: Qdrant
# ---------------------------------------------------------------------------
# (Implementata nelle task successive.)


# ---------------------------------------------------------------------------
# Factory singleton
# ---------------------------------------------------------------------------
# (Implementato nelle task successive.)


__all__ = [
    "Match",
    "VectorPoint",
    "VectorStore",
]
