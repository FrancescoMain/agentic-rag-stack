"""
app/services/classifier.py
==========================
Servizio di classificazione testo via Claude.

Riceve un testo, ritorna una `ClassifyResult` con la categoria predetta,
la confidence dichiarata dal modello, e una breve motivazione.

------------------------------------------------------------------------
Punti chiave dell'architettura
------------------------------------------------------------------------

1. **Funzione pura, niente stato globale.**
   `classify_text()` accetta un `AsyncAnthropic` come *argomento*. Non
   importa da dove viene: in produzione viene dal `lifespan` di FastAPI,
   nei test (task #10) viene da un fake mockato. Questa è la *dependency
   injection* che rende il codice testabile senza `mock.patch`.

2. **Schema di output tipizzato e validato.**
   `ClassifyResult` usa `Literal[...]` per le categorie ammesse: pydantic
   rifiuta qualunque altra stringa che il modello potesse inventarsi.
   Se Claude inventasse `"complaint"`, partirebbe un `ValidationError`
   e l'endpoint risponderebbe 502 (errore upstream).

3. **JSON enforcement nel prompt.**
   Chiediamo a Claude di rispondere SOLO con JSON, senza Markdown né
   testo extra. Se sbagliasse, il `json.loads()` solleva `ValueError`
   che il chiamante deve gestire. È una scelta minimalista — in M4
   useremo i *tool use* di Anthropic per strutturazione garantita.

4. **Token usage loggato per cost tracking futuro.**
   In M5 lo aggregheremo in una dashboard di Langfuse.
"""

import json
import logging
import re
from typing import Literal

# `Anthropic` (sync) e `AsyncAnthropic` (async) sono i due client del SDK.
# Usiamo AsyncAnthropic per non bloccare l'event loop ASGI mentre
# aspettiamo la risposta di Claude (che può durare 1-5 secondi).
from anthropic import APIError, AsyncAnthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Schemi pubblici (input/output del servizio)
# ============================================================================
# `Literal["a", "b", ...]` è un tipo che ammette SOLO i valori elencati.
# Pydantic lo usa per validare: una `category` con valore "complaint"
# scatena un ValidationError. Mental model TS: union type stretto come
# `type Category = "bug" | "feature" | "question" | "spam"`.
Category = Literal["bug", "feature", "question", "spam"]


class ClassifyResult(BaseModel):
    """Esito della classificazione."""

    category: Category
    # `Field(ge=..., le=...)` aggiunge constraint numerici al campo.
    # Se Claude restituisse 1.5 come confidence, fail-fast.
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ============================================================================
# Prompt di sistema
# ============================================================================
# Tenuto come costante module-level: facile da rivedere, facile da iterare
# (basta git-diff per vedere chi ha cambiato il prompt e perché).
#
# Pattern: "structured output via prompt engineering". Diciamo esplicitamente
# il formato JSON atteso. In M4 valuteremo l'alternativa più robusta dei
# `tools` di Anthropic, che vincolano il modello a chiamare una "funzione"
# con argomenti tipizzati — niente parsing manuale del JSON.
_CLASSIFY_PROMPT_TEMPLATE = """Sei un classificatore di feedback utenti.

Categorie disponibili (scegli ESATTAMENTE una):
- bug: l'utente segnala un problema, un errore o un malfunzionamento.
- feature: l'utente richiede una nuova funzionalità o un miglioramento.
- question: l'utente fa una domanda o chiede informazioni.
- spam: contenuto irrilevante, pubblicitario, offensivo o automatizzato.

OUTPUT: rispondi con un SINGOLO oggetto JSON, niente altro. Non avvolgerlo
in code fence Markdown (`` ``` ``), non aggiungere testo prima o dopo.
Il primo carattere della tua risposta DEVE essere `{{`.

Schema atteso:
{{
  "category": "bug" | "feature" | "question" | "spam",
  "confidence": float tra 0.0 e 1.0,
  "reasoning": "breve motivazione (max una frase)"
}}

Testo da classificare:
\"\"\"
{text}
\"\"\""""


# Regex per ripulire risposte che, nonostante il prompt, vengono incartate
# in un blocco di code-fence Markdown (es. ```json\n{...}\n```). Cattura sia
# ``` che ```json. Strippiamo prima di parsare. In M4 questo workaround sparisce
# perché useremo il "tool use" di Anthropic che vincola l'output a uno schema.
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    """Rimuove un eventuale wrapping `` ``` `` o `` ```json `` dal testo.

    Esempi di input → output:
        '```json\\n{"a": 1}\\n```'  →  '{"a": 1}'
        '```{"a": 1}```'           →  '{"a": 1}'
        '{"a": 1}'                 →  '{"a": 1}'  (no-op se già pulito)
    """
    cleaned = _CODE_FENCE_RE.sub("", text.strip())
    return cleaned.strip()


# Modello di default. Hardcoded qui in linea con le scelte documentate nel
# README/ROADMAP (Haiku 4.5 per task semplici): facile da cambiare per test
# o per provare un modello più grande.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


# ============================================================================
# Funzione pubblica
# ============================================================================
async def classify_text(
    client: AsyncAnthropic,
    text: str,
    *,
    model: str = DEFAULT_MODEL,
) -> ClassifyResult:
    """Classifica `text` chiamando Claude.

    Args:
        client: client Anthropic già istanziato. Iniettato per testabilità.
        text: testo da classificare. Non deve essere vuoto (lo verifica
              FastAPI a livello di endpoint via `min_length`).
        model: ID del modello Claude da usare. Default Haiku 4.5.

    Returns:
        `ClassifyResult` validato.

    Raises:
        anthropic.APIError: chiamata Anthropic fallita (timeout, 5xx,
                            rate limit, ecc.). Il chiamante (endpoint)
                            la traduce in 502/503.
        ValueError: il modello ha restituito qualcosa che non è JSON
                    valido, o il JSON non rispetta `ClassifyResult`.
    """
    prompt = _CLASSIFY_PROMPT_TEMPLATE.format(text=text)

    logger.info(
        "classify_start",
        extra={"text_length": len(text), "model": model},
    )

    try:
        # `await` perché AsyncAnthropic.messages.create() è una coroutine.
        # max_tokens limita la lunghezza dell'output → costo prevedibile.
        response = await client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
    except APIError as exc:
        # Loghiamo e rilanciamo: chi gestisce l'endpoint decide come
        # tradurre l'errore in status HTTP.
        logger.error(
            "classify_anthropic_error",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise

    # `response.content` è una lista di "content blocks". Per un prompt
    # senza tool calling, di solito ce n'è uno solo di tipo "text".
    # Concateniamo eventuali blocchi multipli (defensive).
    raw_text = "".join(block.text for block in response.content if block.type == "text").strip()

    # Difesa contro Haiku/Sonnet che a volte avvolgono il JSON in code fence
    # Markdown nonostante il prompt dica di non farlo. Lo strippiamo prima
    # di parsare. NO-OP se il testo è già pulito.
    json_text = _strip_code_fences(raw_text)

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        # Tronchiamo il raw per non riempire i log se il modello sbrocca.
        # Logghiamo BOTH raw (originale) e cleaned (dopo stripping fences),
        # così se il problema è il nostro stripping ce ne accorgiamo.
        logger.error(
            "classify_invalid_json",
            extra={"raw_preview": raw_text[:200], "cleaned_preview": json_text[:200]},
        )
        raise ValueError(f"Claude non ha restituito JSON parsabile: {raw_text[:120]!r}") from exc

    # `model_validate` di pydantic alza `ValidationError` se la struttura
    # non corrisponde (categoria sconosciuta, confidence fuori range, ecc).
    # `ValidationError` è una sottoclasse di `ValueError`, quindi il
    # chiamante può catturare entrambi con `except ValueError`.
    result = ClassifyResult.model_validate(parsed)

    logger.info(
        "classify_success",
        extra={
            "category": result.category,
            "confidence": result.confidence,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    )
    return result
