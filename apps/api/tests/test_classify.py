"""
tests/test_classify.py
======================
Test sul servizio classificatore e sull'endpoint POST /classify.

------------------------------------------------------------------------
Strategia
------------------------------------------------------------------------

Due livelli di test, complementari:

1. **Unit test del servizio (`classify_text`)** — chiamiamo direttamente
   la funzione `async def classify_text(client, text)` con un
   FakeAnthropicClient. Verifichiamo che il parsing JSON, lo stripping
   dei code fence, la validation Pydantic e la propagazione errori
   funzionino. Niente FastAPI, niente HTTP — è puro Python.

2. **Test di integrazione dell'endpoint (`POST /classify`)** — chiamiamo
   l'endpoint via FastAPI TestClient. La dependency `get_anthropic_client`
   è sostituita dal nostro fake (vedi `conftest.py`). Verifichiamo che
   gli errori del servizio diventino i giusti status HTTP (422, 502, 503).

------------------------------------------------------------------------
Convenzione di naming
------------------------------------------------------------------------

`test_<oggetto>_<situazione>_<comportamento atteso>`

Esempi:
    test_classify_text_strips_markdown_fences_before_parsing
    test_post_classify_returns_422_on_empty_text

Lunghezza? Va benissimo. Quando un test fallisce in CI, il nome è
TUTTO quello che vedi nel report: meglio una riga lunga ma chiara.
"""

from __future__ import annotations

import httpx
import pytest
from anthropic import APIConnectionError

from app.services.classifier import ClassifyResult, classify_text
from tests.conftest import FakeAnthropicClient

# ============================================================================
# UNIT TESTS: classify_text() direttamente
# ============================================================================
# Funzioni async, gestite automaticamente da pytest-asyncio in modalità
# "auto" (vedi pyproject.toml). Niente decorator extra necessario.


async def test_classify_text_returns_parsed_result_on_valid_json(
    fake_anthropic: FakeAnthropicClient,
) -> None:
    """Happy path: Claude restituisce JSON valido → parsing + validation ok."""
    fake_anthropic.response_text = (
        '{"category":"bug","confidence":0.92,"reasoning":"L\'utente segnala un errore 500."}'
    )

    result = await classify_text(fake_anthropic, "Save bottone rotto")

    assert isinstance(result, ClassifyResult)
    assert result.category == "bug"
    assert result.confidence == 0.92
    assert "errore 500" in result.reasoning


async def test_classify_text_strips_markdown_code_fences(
    fake_anthropic: FakeAnthropicClient,
) -> None:
    """Anche se il modello incarta il JSON in ```json ... ```, parsing ok.

    Questo test "blinda" il workaround che abbiamo introdotto in task #9
    quando Haiku 4.5 ha aggiunto i fence Markdown nonostante il prompt.
    """
    fake_anthropic.response_text = (
        '```json\n{"category":"feature","confidence":0.8,"reasoning":"richiesta dark mode"}\n```'
    )

    result = await classify_text(fake_anthropic, "Vorrei dark mode")

    assert result.category == "feature"
    assert result.confidence == 0.8


async def test_classify_text_raises_value_error_on_invalid_json(
    fake_anthropic: FakeAnthropicClient,
) -> None:
    """Se Claude restituisce qualcosa che non è JSON, alziamo ValueError."""
    fake_anthropic.response_text = "Mi spiace, non posso classificare questo testo."

    # pytest.raises è il "try/except" idiomatico nei test: fail-fast se
    # il blocco NON alza l'eccezione attesa.
    with pytest.raises(ValueError, match="JSON parsabile"):
        await classify_text(fake_anthropic, "qualcosa")


async def test_classify_text_raises_validation_error_on_unknown_category(
    fake_anthropic: FakeAnthropicClient,
) -> None:
    """Se Claude inventa una categoria fuori dal Literal[...], fail-fast.

    `ValidationError` è una sottoclasse di `ValueError`, quindi un singolo
    `except ValueError` nell'endpoint cattura entrambi i casi.
    """
    fake_anthropic.response_text = (
        '{"category":"complaint","confidence":0.9,"reasoning":"non valido"}'
    )

    with pytest.raises(ValueError):
        await classify_text(fake_anthropic, "x")


async def test_classify_text_raises_validation_error_on_confidence_out_of_range(
    fake_anthropic: FakeAnthropicClient,
) -> None:
    """Confidence fuori [0, 1] viene rifiutata dal Pydantic Field constraint."""
    fake_anthropic.response_text = '{"category":"bug","confidence":1.5,"reasoning":"fuori range"}'

    with pytest.raises(ValueError):
        await classify_text(fake_anthropic, "x")


async def test_classify_text_propagates_anthropic_api_error(
    fake_anthropic: FakeAnthropicClient,
) -> None:
    """Un APIError dell'SDK viene rilanciato così com'è, non swallowed.

    Il chiamante (route handler) decide come tradurlo in status HTTP.
    """
    # `APIConnectionError` ha bisogno di un argomento `request` non-None
    # nel suo costruttore (è una scelta del SDK Anthropic).
    fake_anthropic.error_to_raise = APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )

    with pytest.raises(APIConnectionError):
        await classify_text(fake_anthropic, "x")


async def test_classify_text_passes_correct_args_to_anthropic(
    fake_anthropic: FakeAnthropicClient,
) -> None:
    """Verifichiamo che chiamiamo Anthropic con i parametri attesi.

    Asserzione su `fake.calls` — possibile proprio perché il nostro fake
    registra ogni chiamata.
    """
    fake_anthropic.response_text = (
        '{"category":"question","confidence":0.7,"reasoning":"chiede info"}'
    )

    await classify_text(fake_anthropic, "Come si esporta in Excel?")

    assert len(fake_anthropic.calls) == 1
    call = fake_anthropic.calls[0]
    assert call["model"] == "claude-haiku-4-5-20251001"
    assert call["max_tokens"] == 200
    # Il messaggio user contiene il prompt + il testo originale.
    messages = call["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Come si esporta in Excel?" in messages[0]["content"]


# ============================================================================
# INTEGRATION TESTS: POST /classify via TestClient
# ============================================================================
# Test SINCRONI (def, non async): TestClient gira l'event loop internamente.
# Il fake è già agganciato all'app via la fixture `fake_anthropic`.


def test_post_classify_returns_200_with_parsed_result(
    fake_anthropic: FakeAnthropicClient,
    client,
) -> None:
    """Happy path completo: HTTP request → 200 con body parsato."""
    fake_anthropic.response_text = (
        '{"category":"bug","confidence":0.95,"reasoning":"errore 500 esplicito"}'
    )

    response = client.post(
        "/classify",
        json={"text": "Il pulsante Save non funziona, errore 500"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "category": "bug",
        "confidence": 0.95,
        "reasoning": "errore 500 esplicito",
    }


def test_post_classify_returns_422_on_empty_text(
    fake_anthropic: FakeAnthropicClient,  # noqa: ARG001 — agganciato per attivare l'override DI
    client,
) -> None:
    """Pydantic constraint `min_length=1` su ClassifyRequest.text.

    Nota sul `fake_anthropic` unused: in FastAPI le dependencies vengono
    risolte PRIMA della validation del body. Senza il fake, la dependency
    `get_anthropic_client` alzerebbe 503 (no API key in test env) e non
    arriveremmo mai a vedere il 422. Il fake è qui solo per "sbloccare"
    la pipeline; non viene mai effettivamente chiamato in questo test.
    """
    response = client.post("/classify", json={"text": ""})

    assert response.status_code == 422
    body = response.json()
    # FastAPI mette i dettagli della validation error in body["detail"].
    # È una lista con almeno un errore che menziona il campo "text".
    assert any("text" in err["loc"] for err in body["detail"])


def test_post_classify_returns_422_on_missing_text(
    fake_anthropic: FakeAnthropicClient,  # noqa: ARG001
    client,
) -> None:
    """Body senza il campo text → validation error."""
    response = client.post("/classify", json={})

    assert response.status_code == 422


def test_post_classify_returns_422_on_text_too_long(
    fake_anthropic: FakeAnthropicClient,  # noqa: ARG001
    client,
) -> None:
    """Pydantic constraint `max_length=4000`: oltre soglia → 422."""
    long_text = "x" * 4001
    response = client.post("/classify", json={"text": long_text})

    assert response.status_code == 422


def test_post_classify_returns_502_on_anthropic_api_error(
    fake_anthropic: FakeAnthropicClient,
    client,
) -> None:
    """Errore upstream da Anthropic → 502 Bad Gateway con messaggio chiaro."""
    fake_anthropic.error_to_raise = APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )

    response = client.post("/classify", json={"text": "qualsiasi cosa"})

    assert response.status_code == 502
    assert "upstream" in response.json()["detail"].lower()


def test_post_classify_returns_502_on_invalid_model_output(
    fake_anthropic: FakeAnthropicClient,
    client,
) -> None:
    """Se il modello sbrocca (JSON malformato), endpoint risponde 502."""
    fake_anthropic.response_text = "Non riesco a classificare questo testo."

    response = client.post("/classify", json={"text": "qualsiasi cosa"})

    assert response.status_code == 502


def test_post_classify_returns_502_on_invalid_category_from_model(
    fake_anthropic: FakeAnthropicClient,
    client,
) -> None:
    """Modello inventa una categoria → ValidationError → 502."""
    fake_anthropic.response_text = '{"category":"complaint","confidence":0.9,"reasoning":"x"}'

    response = client.post("/classify", json={"text": "x"})

    assert response.status_code == 502


def test_post_classify_echoes_request_id_in_response_header(
    fake_anthropic: FakeAnthropicClient,
    client,
) -> None:
    """Il middleware request_id deve propagare l'header X-Request-ID.

    Questo è un test del wiring trasversale: assicura che il middleware
    di logging (task #8) continui a funzionare sull'endpoint nuovo.
    """
    fake_anthropic.response_text = '{"category":"spam","confidence":0.99,"reasoning":"x"}'

    response = client.post(
        "/classify",
        json={"text": "COMPRA SUBITO!!!"},
        headers={"X-Request-ID": "test-trace-id-42"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "test-trace-id-42"
