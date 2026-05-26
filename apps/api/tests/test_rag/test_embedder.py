"""
tests/test_rag/test_embedder.py
===============================
Test sull'embedder (`app/rag/embedder.py`).

Strategia: testiamo `embed_texts` con un `FakeOpenAIClient` configurato
caso per caso. Niente chiamate reali a OpenAI → veloce, deterministico,
non consuma budget.

Casi coperti:
  - Edge: input vuoto, batch < batch_size, batch > batch_size.
  - Robustezza: retry su 429/timeout, fail-fast su auth, ordering.
  - Output: shape del risultato (vector_dim, token_count > 0, ordine).
"""

from __future__ import annotations

from typing import cast

import httpx
import pytest
from openai import (
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from app.rag.embedder import (
    EmbedderConfig,
    EmbedderError,
    EmbeddingResult,
    embed_texts,
)
from tests.test_rag.conftest import FakeOpenAIClient

# Le eccezioni del SDK OpenAI 2.x richiedono una `httpx.Response`
# reale (per accedere a `.request` internamente). Costruiamo un
# Request/Response "finto ma valido" da riusare nei test.

_FAKE_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/embeddings")


def _make_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_FAKE_REQUEST)


def _make_rate_limit_error() -> RateLimitError:
    return RateLimitError(
        message="Rate limit exceeded",
        response=_make_response(429),
        body=None,
    )


def _make_auth_error() -> AuthenticationError:
    return AuthenticationError(
        message="Invalid API key",
        response=_make_response(401),
        body=None,
    )


def _make_bad_request_error() -> BadRequestError:
    return BadRequestError(
        message="Input too long",
        response=_make_response(400),
        body=None,
    )


def _make_timeout_error() -> APITimeoutError:
    return APITimeoutError(request=_FAKE_REQUEST)


# ============================================================================
# Edge cases
# ============================================================================


async def test_embed_empty_input_returns_empty(
    fake_openai: FakeOpenAIClient,
) -> None:
    """Lista vuota → ritorno vuoto, NESSUNA chiamata API."""
    result = await embed_texts(cast(AsyncOpenAI, fake_openai), [])
    assert result == []
    assert fake_openai.calls == []  # nessuna chiamata


# ============================================================================
# Happy path
# ============================================================================


async def test_embed_single_batch_preserves_order(
    fake_openai: FakeOpenAIClient,
) -> None:
    """Input piccolo (<batch_size) → 1 chiamata, risultati nell'ordine di input."""
    texts = ["alpha", "beta", "gamma"]
    results = await embed_texts(cast(AsyncOpenAI, fake_openai), texts)

    assert len(results) == 3
    assert [r.text for r in results] == texts
    # 1 sola chiamata, con tutto il batch.
    assert len(fake_openai.calls) == 1
    assert fake_openai.calls[0]["input"] == texts
    # Default model
    assert fake_openai.calls[0]["model"] == "text-embedding-3-small"


async def test_embed_result_has_expected_shape(
    fake_openai: FakeOpenAIClient,
) -> None:
    """Ogni EmbeddingResult ha vector di dim corretta e token_count > 0."""
    fake_openai.tokens_per_call = 50
    results = await embed_texts(cast(AsyncOpenAI, fake_openai), ["hello"])

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, EmbeddingResult)
    assert r.text == "hello"
    assert len(r.vector) == 1536  # text-embedding-3-small
    assert all(isinstance(v, float) for v in r.vector)
    assert r.token_count > 0


async def test_embed_splits_into_multiple_batches(
    fake_openai: FakeOpenAIClient,
) -> None:
    """Input grande → spezza in più chiamate di al massimo batch_size."""
    texts = [f"text {i}" for i in range(250)]
    config = EmbedderConfig(batch_size=100)
    results = await embed_texts(cast(AsyncOpenAI, fake_openai), texts, config)

    assert len(results) == 250
    # 250 / 100 = 3 batch (100, 100, 50).
    assert len(fake_openai.calls) == 3
    assert len(fake_openai.calls[0]["input"]) == 100
    assert len(fake_openai.calls[1]["input"]) == 100
    assert len(fake_openai.calls[2]["input"]) == 50

    # Tutti i testi originali devono apparire nei risultati nel giusto ordine.
    assert [r.text for r in results] == texts


async def test_embed_handles_out_of_order_response(
    fake_openai: FakeOpenAIClient,
) -> None:
    """Se OpenAI ritorna i `data` con index permutato, l'output va riordinato."""
    texts = ["A", "B", "C"]
    # Forza l'order: l'API ci dà C(idx=2), A(idx=0), B(idx=1).
    fake_openai.next_response_order = [2, 0, 1]
    results = await embed_texts(cast(AsyncOpenAI, fake_openai), texts)

    # Il riordinamento deve restituire l'ordine originale.
    assert [r.text for r in results] == ["A", "B", "C"]


# ============================================================================
# Retry & error handling
# ============================================================================


async def test_embed_retries_on_rate_limit_then_succeeds(
    fake_openai: FakeOpenAIClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RateLimitError al primo tentativo → retry → success al secondo.

    Monkey-patchiamo asyncio.sleep per non aspettare davvero i secondi
    del backoff (1s, 2s, ...) — i test devono essere veloci.
    """
    # asyncio.sleep dentro app.rag.embedder è quello a cui ci interessa.
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("app.rag.embedder.asyncio.sleep", fake_sleep)

    # Programma: primo call solleva RateLimit, secondo va a buon fine.
    fake_openai.errors_queue = [_make_rate_limit_error()]

    config = EmbedderConfig(max_retries=3)
    results = await embed_texts(cast(AsyncOpenAI, fake_openai), ["hello"], config)

    assert len(results) == 1
    # 2 chiamate totali: 1 fallita + 1 riuscita.
    assert len(fake_openai.calls) == 2
    # Backoff atteso: 2 ** 0 = 1s
    assert sleeps == [1]


async def test_embed_retries_with_exponential_backoff(
    fake_openai: FakeOpenAIClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3 rate-limit consecutivi → 3 retry → success al quarto."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("app.rag.embedder.asyncio.sleep", fake_sleep)

    fake_openai.errors_queue = [
        _make_rate_limit_error(),
        _make_timeout_error(),
        _make_rate_limit_error(),
    ]

    config = EmbedderConfig(max_retries=3)
    results = await embed_texts(cast(AsyncOpenAI, fake_openai), ["x"], config)

    assert len(results) == 1
    assert len(fake_openai.calls) == 4  # 3 fallite + 1 OK
    # Exponential backoff: 1, 2, 4
    assert sleeps == [1, 2, 4]


async def test_embed_raises_after_max_retries_exhausted(
    fake_openai: FakeOpenAIClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Se i retry sono esauriti → EmbedderError (con cause RateLimitError)."""
    monkeypatch.setattr(
        "app.rag.embedder.asyncio.sleep",
        lambda _s: _noop(),
    )

    # 5 errori in coda, max_retries=2 → 3 tentativi totali, tutti falliti.
    fake_openai.errors_queue = [_make_rate_limit_error() for _ in range(5)]

    config = EmbedderConfig(max_retries=2)
    with pytest.raises(EmbedderError, match="dopo 3 tentativi"):
        await embed_texts(cast(AsyncOpenAI, fake_openai), ["x"], config)


async def test_embed_fails_fast_on_authentication_error(
    fake_openai: FakeOpenAIClient,
) -> None:
    """AuthenticationError → EmbedderError SUBITO, niente retry."""
    fake_openai.errors_queue = [_make_auth_error()]

    config = EmbedderConfig(max_retries=5)  # alto, ma non deve usarli
    with pytest.raises(EmbedderError, match="AuthenticationError"):
        await embed_texts(cast(AsyncOpenAI, fake_openai), ["x"], config)

    # Una sola chiamata (no retry).
    assert len(fake_openai.calls) == 1


async def test_embed_fails_fast_on_bad_request(
    fake_openai: FakeOpenAIClient,
) -> None:
    """BadRequestError → niente retry (è un errore di payload, non transient)."""
    fake_openai.errors_queue = [_make_bad_request_error()]

    with pytest.raises(EmbedderError, match="BadRequestError"):
        await embed_texts(cast(AsyncOpenAI, fake_openai), ["x"])

    assert len(fake_openai.calls) == 1


# Helper async no-op: serve come `lambda` async per monkeypatch di
# asyncio.sleep nei test che non vogliono ispezionare i tempi.
async def _noop() -> None:
    return None
