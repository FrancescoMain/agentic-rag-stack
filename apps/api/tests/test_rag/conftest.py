"""
tests/test_rag/conftest.py
==========================
Fixture locali alla cartella `tests/test_rag/`. Definite qui (e non nel
conftest globale) perché servono solo ai test del RAG. Pattern di
"località delle fixture": una fixture vive il più vicino possibile a
chi la usa.

Cosa c'è qui:
- `FakeOpenAIClient`: test double minimale per `openai.AsyncOpenAI`.
  Implementato a mano (niente `unittest.mock`) per due ragioni:
    1. Si VEDE chiaramente quale superficie del SDK stiamo usando
       (`client.embeddings.create(model=..., input=...)`).
    2. È facile estenderlo (es. configurare scenari di rate-limit
       o di out-of-order).
- `fake_openai`: fixture che istanzia un FakeOpenAIClient pronto all'uso.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


class _FakeEmbeddingsResource:
    """Simula `AsyncOpenAI.embeddings`: ha un metodo async `.create()`."""

    def __init__(self, parent: FakeOpenAIClient) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        """Restituisce una response Embeddings-like.

        Per default produce un embedding "tutto 0.1" per ogni input (1536-dim
        come `text-embedding-3-small`). I test che vogliono comportamenti
        diversi (errori, out-of-order, ...) configurano il fake.
        """
        self._parent.calls.append(kwargs)

        # Se il test ha programmato una sequenza di errori, alziamoli
        # nell'ordine. Utile per testare il retry.
        if self._parent.errors_queue:
            err = self._parent.errors_queue.pop(0)
            raise err

        inputs: list[str] = kwargs["input"]
        # Per simulare risposta "out-of-order" servono test specifici;
        # default: order = input order.
        order = self._parent.next_response_order or list(range(len(inputs)))

        # Costruiamo i `data` items: ognuno con `embedding` (vector) e
        # `index` (posizione originale).
        data = [
            SimpleNamespace(
                embedding=self._parent.vector_for(inputs[i]),
                index=i,
            )
            for i in order
        ]

        usage = SimpleNamespace(
            prompt_tokens=self._parent.tokens_per_call,
            total_tokens=self._parent.tokens_per_call,
        )

        # Pulisci order così non si propaga al batch successivo
        self._parent.next_response_order = None

        return SimpleNamespace(data=data, usage=usage)


class FakeOpenAIClient:
    """Test double per `openai.AsyncOpenAI`.

    Configurabile dal test:
        fake.tokens_per_call = 42         # token finti per la usage
        fake.errors_queue = [RateLimitError(...), None]
                                          # primo call rate-limited, secondo OK
        fake.next_response_order = [2,0,1]
                                          # forza un order specifico nella next
        fake.vector_dim = 1536            # default 1536 (text-embedding-3-small)

    Espone:
        fake.embeddings.create(model=..., input=[...]) -> response
        fake.calls: list[dict] dei kwargs di ogni chiamata
        fake.aclose() async no-op
    """

    def __init__(self) -> None:
        self.vector_dim = 1536
        self.tokens_per_call = 10
        self.calls: list[dict[str, Any]] = []
        # Coda di eccezioni da alzare alle prossime chiamate, in ordine.
        # Item = Exception istanza per "alza questo errore", oppure
        # nessun item per "rispondi normalmente".
        self.errors_queue: list[Exception] = []
        # Se settato, forza un ordine specifico dell'array `data` nella
        # prossima response (per testare il riordinamento via `.index`).
        self.next_response_order: list[int] | None = None

        # Sotto-resource come da SDK OpenAI: client.embeddings.create(...)
        self.embeddings = _FakeEmbeddingsResource(self)

    def vector_for(self, text: str) -> list[float]:
        """Vettore deterministico per un dato testo.

        Default: 1536 float, tutti uguali a una funzione della lunghezza
        del testo (così testi diversi → vettori diversi, utile per
        asserzioni). Non realistici come embedding ma sufficienti per
        verificare la pipeline.
        """
        seed = (len(text) % 100) / 100.0
        return [seed] * self.vector_dim

    async def close(self) -> None:
        """No-op: il fake non ha connessioni reali da chiudere."""
        return None


@pytest.fixture
def fake_openai() -> FakeOpenAIClient:
    """Istanzia un FakeOpenAIClient pronto all'uso.

    A differenza di `fake_anthropic`, questo NON tocca
    `app.dependency_overrides`: l'embedder è una funzione pura che
    accetta direttamente il client, quindi nei test lo passiamo come
    argomento. Più semplice e più diretto.
    """
    return FakeOpenAIClient()
