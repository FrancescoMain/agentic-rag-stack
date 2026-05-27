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
import uuid
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

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
# Internals: ID normalization & mapping helpers
# ---------------------------------------------------------------------------

# Namespace UUID fisso per la derivazione deterministica dell'id Qdrant
# da una stringa arbitraria (es. sha256 hex dal chunker). Usiamo un namespace
# da RFC 4122 § Appendix C — è un valore arbitrario ma stabile e
# riconoscibile; l'unica cosa che conta è che NON cambi mai.
_ID_NAMESPACE = uuid.UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")

# Chiave riservata nel payload Qdrant: contiene l'id originale del
# VectorPoint, in modo che `Match` possa restituirlo invece dell'UUID
# Qdrant interno. Prefisso "__" per minimizzare collisioni con metadata
# user-defined.
_RESERVED_PAYLOAD_KEY = "__vp_id"


def _normalize_id(raw: str) -> str:
    """Mappa una stringa stabile a un UUID v5 deterministico.

    Qdrant accetta come point id SOLO uint64 o UUID. Stringhe arbitrarie
    (sha256 hex, slug, ecc.) vanno tradotte. UUID v5 è perfetto: stesso
    input → stesso UUID, sempre, ovunque.
    """
    return str(uuid.uuid5(_ID_NAMESPACE, raw))


def _to_qdrant_point(point: VectorPoint) -> PointStruct:
    """VectorPoint → PointStruct di qdrant-client.

    Side-effect: arricchisce il payload con la chiave riservata
    `__vp_id` per preservare l'id originale.
    """
    qdrant_id = _normalize_id(point.id)
    payload = {**point.payload, _RESERVED_PAYLOAD_KEY: point.id}
    return PointStruct(id=qdrant_id, vector=point.vector, payload=payload)


def _from_qdrant_scored_point(sp: ScoredPoint) -> Match:
    """ScoredPoint di qdrant-client → Match nostro.

    Side-effect: rimuove la chiave riservata `__vp_id` dal payload prima
    di esporlo al consumer, e la usa come `Match.id`. Se manca, fallback
    sull'UUID Qdrant (caso "rotto" — punto inserito senza il nostro upsert).
    """
    payload = dict(sp.payload or {})
    original_id = payload.pop(_RESERVED_PAYLOAD_KEY, str(sp.id))
    return Match(id=original_id, score=sp.score, payload=payload)


def _filter_to_qdrant(filter: dict[str, Any] | None) -> Filter | None:
    """Dict shallow equality → Filter Qdrant con AND logico (clausola `must`).

    None o dict vuoto → None (Qdrant accetta None per "nessun filtro").
    Esempio: {"source": "a.md", "heading": "H1"} → Filter(must=[
        FieldCondition(key="source", match=MatchValue(value="a.md")),
        FieldCondition(key="heading", match=MatchValue(value="H1")),
    ])
    """
    if not filter:
        return None
    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value)) for key, value in filter.items()
    ]
    return Filter(must=conditions)


# ---------------------------------------------------------------------------
# Implementazione concreta: Qdrant
# ---------------------------------------------------------------------------


class QdrantVectorStore:
    """Implementazione concreta di `VectorStore` su Qdrant.

    Configurata con un URL HTTP e una API key opzionale. Mantiene
    internamente un `AsyncQdrantClient` long-lived.
    """

    def __init__(self, url: str, api_key: str | None = None) -> None:
        self._client = AsyncQdrantClient(url=url, api_key=api_key)

    async def ensure_collection(
        self,
        name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        """Crea la collection se non esiste; no-op se esiste con stessa config.

        Sui mismatch di `vector_size` alziamo ValueError invece di tentare
        un drop+recreate silenzioso (che cancellerebbe dati). Cambiare
        embedding model è una decisione esplicita: il chiamante chiama
        `delete_collection` prima.
        """
        distance_enum = self._parse_distance(distance)

        if await self._client.collection_exists(name):
            info = await self._client.get_collection(name)
            # `vectors` può essere un VectorParams (single unnamed vector,
            # nostro caso) o un dict[str, VectorParams] (multi-vector).
            # Estraiamo la size in modo robusto.
            existing_params = info.config.params.vectors
            if isinstance(existing_params, dict):
                # Per now ci aspettiamo single-vector; se è dict, prendi il
                # primo entry e confronta. Se servirà multi-vector in M2+,
                # estenderemo l'API di ensure_collection.
                existing_size = next(iter(existing_params.values())).size
            else:
                existing_size = existing_params.size
            if existing_size != vector_size:
                raise ValueError(
                    f"Collection '{name}' esiste con vector_size={existing_size}, "
                    f"richiesto={vector_size}. Drop+recreate esplicito necessario."
                )
            logger.info(
                "ensure_collection_noop",
                extra={"collection": name, "vector_size": vector_size},
            )
            return

        await self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=distance_enum),
        )
        logger.info(
            "ensure_collection_created",
            extra={"collection": name, "vector_size": vector_size, "distance": distance},
        )

    async def delete_collection(self, name: str) -> None:
        """Elimina la collection. No-op se non esiste."""
        if not await self._client.collection_exists(name):
            return
        await self._client.delete_collection(name)
        logger.info("delete_collection_done", extra={"collection": name})

    # Soglia di batching interna. Coerente con l'embedder (batch_size=100).
    # Qdrant accetta upsert più grandi, ma 100 è uno sweet spot fra throughput
    # (meno round-trip) e robustezza (payload più piccolo → timeout più rari).
    _UPSERT_BATCH_SIZE = 100

    async def upsert(
        self,
        collection: str,
        points: Sequence[VectorPoint],
    ) -> int:
        """Insert/update di punti. Idempotente per id (overwrite su collisione).

        Batching interno a chunk di 100 per chiamata; il consumer non se ne
        accorge — vede una singola operazione.
        """
        if not points:
            return 0

        total = 0
        for start in range(0, len(points), self._UPSERT_BATCH_SIZE):
            batch = points[start : start + self._UPSERT_BATCH_SIZE]
            qdrant_points = [_to_qdrant_point(p) for p in batch]
            await self._client.upsert(collection_name=collection, points=qdrant_points)
            total += len(batch)

        logger.info(
            "upsert_done",
            extra={
                "collection": collection,
                "total_points": total,
                "batches": (len(points) + self._UPSERT_BATCH_SIZE - 1) // self._UPSERT_BATCH_SIZE,
            },
        )
        return total

    async def search(
        self,
        collection: str,
        query: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[Match]:
        """Top-k semantic search. Filter shallow equality opzionale.

        Usa `query_points` (endpoint unificato di Qdrant 1.10+) invece del
        vecchio `search`. La response è un `QueryResponse` con `.points`.
        """
        qdrant_filter = _filter_to_qdrant(filter)
        response = await self._client.query_points(
            collection_name=collection,
            query=query,
            limit=top_k,
            query_filter=qdrant_filter,
        )
        return [_from_qdrant_scored_point(sp) for sp in response.points]

    # -----------------------------------------------------------------------
    # Helpers privati
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_distance(distance: str) -> Distance:
        """Parse del nome distance ("Cosine"/"Dot"/"Euclid") al tipo enum."""
        mapping = {
            "Cosine": Distance.COSINE,
            "Dot": Distance.DOT,
            "Euclid": Distance.EUCLID,
        }
        if distance not in mapping:
            raise ValueError(
                f"Distance '{distance}' non supportata. Valori validi: {sorted(mapping)}"
            )
        return mapping[distance]


# ---------------------------------------------------------------------------
# Factory singleton
# ---------------------------------------------------------------------------
# Pattern coerente con `app.config.settings`: istanza modulo-livello creata
# alla prima chiamata e riusata. AsyncQdrantClient è progettato per essere
# long-lived (gestisce internamente un pool httpx); ricrearlo per ogni
# request sarebbe wasteful.

_vector_store_singleton: QdrantVectorStore | None = None


def get_vector_store() -> QdrantVectorStore:
    """Ritorna l'istanza singleton di QdrantVectorStore.

    Usa `app.config.settings` per URL e API key. Pensato per essere usato
    con `Depends(get_vector_store)` da endpoint FastAPI (dal task #10 in poi).
    """
    global _vector_store_singleton
    if _vector_store_singleton is None:
        from app.config import settings  # import lazy: evita ciclo all'avvio

        _vector_store_singleton = QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return _vector_store_singleton


__all__ = [
    "Match",
    "QdrantVectorStore",
    "VectorPoint",
    "VectorStore",
    "get_vector_store",
]
