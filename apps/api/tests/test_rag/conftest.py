"""
tests/test_rag/conftest.py
==========================
Fixture locali alla cartella `tests/test_rag/`.

Le fixture comuni (`fake_openai`, `FakeOpenAIClient`) vivono nel
conftest top-level `tests/conftest.py` — pytest le rende automaticamente
visibili anche a questa sotto-cartella.

Cosa c'è qui:
- `qdrant_store` (session-scoped): client Qdrant configurato; skippa
  i test se Qdrant non risponde.
- `unique_collection` (function-scoped): nome collection unico per
  isolare i test, con cleanup automatico al teardown.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixture per i test integration di vector_store.py
# ---------------------------------------------------------------------------
# Pattern: session-scoped per il client (riusare la connessione),
# function-scoped per il nome collection (isolare i test fra loro).
# Se Qdrant non risponde, i test marcati @pytest.mark.integration vengono
# SKIPPATI con motivo chiaro — non falliscono.


@pytest.fixture(scope="session")
def qdrant_store() -> Any:
    """Restituisce un QdrantVectorStore configurato.

    Verifica connettività con un GET /readyz; se Qdrant è down, skippa
    tutti i test che dipendono da questa fixture.

    `session` scope: riusiamo lo stesso client per tutta la sessione
    pytest. AsyncQdrantClient mantiene un pool httpx internamente.
    """
    import httpx

    from app.config import settings
    from app.rag.vector_store import QdrantVectorStore

    try:
        r = httpx.get(f"{settings.qdrant_url}/readyz", timeout=2.0)
        if r.status_code != 200:
            pytest.skip(
                f"Qdrant non pronto su {settings.qdrant_url}/readyz "
                f"(status {r.status_code}). Avvia: docker compose up -d qdrant"
            )
    except httpx.RequestError as exc:
        pytest.skip(
            f"Qdrant non raggiungibile su {settings.qdrant_url}: {exc}. "
            f"Avvia: docker compose up -d qdrant"
        )

    return QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


@pytest.fixture
async def unique_collection(qdrant_store: Any) -> Any:
    """Yield di un nome di collection unico; cleanup automatico al teardown.

    Prefisso `_test_` per identificare a colpo d'occhio le collection
    create dai test (utili a `delete_collection` manuale di pulizia).
    """
    import uuid as _uuid

    name = f"_test_{_uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        # Best-effort cleanup. Se delete fallisce (es. collection mai
        # creata, o Qdrant nel frattempo down), non fallire il teardown.
        try:
            await qdrant_store.delete_collection(name)
        except Exception as exc:  # noqa: BLE001 — è un teardown best-effort
            logger.warning("Cleanup collection %s fallito: %s", name, exc)
