"""
tests/conftest.py
=================
Fixture pytest condivise da tutti i file di test della cartella.

`conftest.py` è il meccanismo "magico" di pytest per condividere setup:
qualunque fixture qui dichiarata è disponibile in TUTTI i test di questa
cartella (e sotto-cartelle), senza import esplicito. Mental model:
"l'`@beforeEach` globale di Jest, ma per fixture nominate".

------------------------------------------------------------------------
Cosa c'è qui dentro
------------------------------------------------------------------------

1. `FakeAnthropicClient` — un "test double" che si comporta come
   `AsyncAnthropic` ma ritorna risposte predefinite. Sostituisce il client
   reale nei test.

2. `fake_anthropic` — fixture che istanzia un FakeAnthropicClient e lo
   collega all'app FastAPI tramite `app.dependency_overrides`. Il test
   ha accesso al fake per configurarlo (response_text / error_to_raise);
   l'app, quando un endpoint chiama `Depends(get_anthropic_client)`,
   riceve il fake.

3. `test_client` — il TestClient di FastAPI, riusabile.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_anthropic_client

# ============================================================================
# Fake Anthropic client
# ============================================================================
# Implementiamo a mano (niente unittest.mock) per due motivi didattici:
#   1. Si VEDE chiaramente quale superficie del SDK Anthropic stiamo usando
#      (`messages.create(...)` → un oggetto con `.content` e `.usage`).
#   2. Estensione facile: domani aggiungiamo `tools=...` al servizio?
#      Estendiamo il fake e amen.
#
# `SimpleNamespace` (stdlib) crea oggetti con attributi al volo, senza
# definire una classe. Comodo per simulare struct-like data come la
# response Anthropic. Equivalente JS: `{ foo: 1, bar: 2 }` ma con `.foo`.


class _FakeMessagesResource:
    """Simula `AsyncAnthropic.messages`: ha un metodo async `.create()`."""

    def __init__(self, parent: FakeAnthropicClient) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        # Registriamo la chiamata: i test possono ispezionare `fake.calls`
        # per asserire che il servizio abbia chiamato con i parametri attesi
        # (es. il giusto modello, il giusto max_tokens).
        self._parent.calls.append(kwargs)

        # Se il test ha configurato un errore, lo alziamo. Utile per
        # simulare timeout / 5xx di Anthropic.
        if self._parent.error_to_raise is not None:
            raise self._parent.error_to_raise

        # Altrimenti, ritorniamo una "Message" Anthropic-like.
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._parent.response_text)],
            usage=SimpleNamespace(
                input_tokens=self._parent.input_tokens,
                output_tokens=self._parent.output_tokens,
            ),
        )


class FakeAnthropicClient:
    """Test double per `anthropic.AsyncAnthropic`.

    Configurabile dal test che lo usa:
        fake.response_text = '{"category":"bug",...}'
        fake.error_to_raise = APIError(...)
        fake.input_tokens = 42

    Espone la stessa superficie usata dal servizio: `client.messages.create(...)`.
    """

    def __init__(self) -> None:
        # Default: stringa vuota. Ogni test la imposta esplicitamente.
        self.response_text: str = ""
        self.error_to_raise: Exception | None = None
        # Contatori di token "finti", giusto per popolare i log.
        self.input_tokens: int = 10
        self.output_tokens: int = 5
        # Log di tutte le chiamate effettuate (per asserzioni).
        self.calls: list[dict[str, Any]] = []
        # `client.messages` è una sotto-resource; il SDK lo struttura così.
        self.messages = _FakeMessagesResource(self)


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def fake_anthropic() -> FakeAnthropicClient:
    """Crea un FakeAnthropicClient e lo aggancia alla dependency `get_anthropic_client`.

    Yield-fixture: prima del `yield` succede il setup, dopo il cleanup.
    Il cleanup (svuotare `dependency_overrides`) è FONDAMENTALE: senza,
    il fake del primo test inquinerebbe il secondo.

    Mental model JS: `beforeEach` + `afterEach` in un'unica funzione.
    """
    fake = FakeAnthropicClient()
    # `dependency_overrides[fn] = replacement_fn` dice a FastAPI: ogni volta
    # che un endpoint dichiara `Depends(fn)`, usa invece `replacement_fn`.
    # `replacement_fn` può ritornare qualunque cosa che soddisfi l'interfaccia
    # attesa — nel nostro caso, un AsyncAnthropic-like.
    app.dependency_overrides[get_anthropic_client] = lambda: fake
    yield fake
    # Cleanup: torna allo stato originale per non inquinare altri test.
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient per chiamate HTTP in-process."""
    return TestClient(app)
