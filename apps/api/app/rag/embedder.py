"""
app/rag/embedder.py
===================
Wrapper async per chiamare il modello di embedding di OpenAI.

------------------------------------------------------------------------
Cos'è un embedding (in 3 righe)
------------------------------------------------------------------------

Un *embedding* è una funzione (deterministica) che prende un testo e
restituisce un vettore di numeri float ad alta dimensione (qui 1536).
Testi semanticamente simili → vettori vicini per cosine similarity.
È il "ponte" che traduce significato in geometria, abilitando il
retrieval semantico.

Vedi [ADR-0004](../../../../docs/adr/0004-embedding-model-choice.md)
per la motivazione della scelta del modello.

------------------------------------------------------------------------
Cosa fa questo modulo
------------------------------------------------------------------------

Espone una funzione pubblica:

    await embed_texts(client, texts, config=None) -> list[EmbeddingResult]

che gestisce per noi:

1. **Batching automatico.** L'API OpenAI accetta fino a 2048 input per
   call, ma per stabilità batch più piccoli (~100) sono meglio (timeout
   minori, errori più contenuti). Se passi 1000 testi e batch_size=100,
   il wrapper fa 10 chiamate sequenziali.

2. **Retry con exponential backoff** su errori transitori:
   - `RateLimitError` (429 — abbiamo saturato il quota burst).
   - `APITimeoutError` (la chiamata è andata in timeout).
   - 5xx (errore del server OpenAI).
   Backoff: 1s → 2s → 4s → 8s, con un cap a `max_retries`.

3. **Errori "hard" → fail subito.** `AuthenticationError` (401) o
   `BadRequestError` (400) sono problemi di config/codice e non
   passano col retry. Re-raise.

4. **Ordine preservato.** L'API OpenAI ritorna ogni embedding con un
   `index` che corrisponde alla posizione nell'input. Lo usiamo per
   ri-ordinare in caso di out-of-order (raro ma possibile) e per
   essere robusti.

5. **Telemetria via logger.** Loggiamo ogni batch con n. testi e token
   consumati (utile per cost tracking; in M5 lo aggregheremo in
   Langfuse).

------------------------------------------------------------------------
Perché non una classe Embedder?
------------------------------------------------------------------------

Stesso pattern di `services/classifier.py`: una funzione che riceve il
client OpenAI già configurato (via dependency injection di FastAPI).
Vantaggi:
- Più testabile: nei test, il client è sostituito da un fake.
- Più Pythonic: niente OOP gratuita.
- Più componibile: ogni chunk diventa funzione pura `(input → output)`.

L'astrazione "scambia il provider in futuro" è già garantita dalla
firma: chi chiama l'embedder accetta una `list[EmbeddingResult]`,
non sa né gli importa che dentro ci sia OpenAI. Domani metti
FastEmbed → cambi questa funzione, non i call site.
"""

from __future__ import annotations

import asyncio
import logging

from openai import (
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemi (Pydantic)
# ---------------------------------------------------------------------------


class EmbedderConfig(BaseModel):
    """Parametri configurabili dell'embedder.

    Default 2026-friendly:
    - model: `text-embedding-3-small`, 1536-dim. Vedi ADR-0004.
    - batch_size: 100. L'API accetta fino a 2048; 100 è sweet spot fra
      throughput (meno round-trip) e robustezza (timeout/errori più
      contenuti).
    - max_retries: 4 → 1s, 2s, 4s, 8s = 15s cumulative al peggio.
      Sufficiente per superare rate-limit spike senza far stallare
      eccessivamente l'ingestion.
    - timeout_s: 30s. Per text-embedding-3-small su batch di 100 le
      response sono in ~1-2s; 30s è abbondante per coprire latenze
      anomale.
    """

    model: str = "text-embedding-3-small"
    batch_size: int = Field(default=100, gt=0, le=2048)
    max_retries: int = Field(default=4, ge=0)
    timeout_s: float = Field(default=30.0, gt=0)


class EmbeddingResult(BaseModel):
    """Un singolo embedding prodotto dal modello.

    Attributes:
        text: il testo originale (utile per log/debug e per associare
            risultato all'input quando la chiamante non vuole tenere
            un altro list).
        vector: 1536 float (per text-embedding-3-small). Sarà
            inserito direttamente come payload Qdrant nel task M2 #5.
        token_count: token consumati per generare questo embedding.
            Aggregato dal batch e diviso proporzionalmente (l'API
            OpenAI dà solo il totale del batch, vedi nota in
            `_embed_one_batch`).
    """

    text: str
    vector: list[float]
    token_count: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Eccezione semantica (più chiara di una APIError generica al chiamante).
# ---------------------------------------------------------------------------
class EmbedderError(RuntimeError):
    """Errore durante la generazione di embeddings.

    Sollevata quando i retry sono esauriti o quando l'errore upstream
    non è recuperabile (auth, bad request). Il chiamante può catturarla
    per logging/metrics senza dover importare tutti i tipi di errore
    dell'SDK OpenAI.
    """


# ===========================================================================
# API pubblica
# ===========================================================================


async def embed_texts(
    client: AsyncOpenAI,
    texts: list[str],
    config: EmbedderConfig | None = None,
) -> list[EmbeddingResult]:
    """Genera embeddings per una lista di testi.

    Args:
        client: client `AsyncOpenAI` già autenticato (tipicamente
            creato in `app/main.py` durante il lifespan).
        texts: testi da embeddare. Lista vuota → ritorna [] senza
            chiamare l'API.
        config: parametri di batching/retry. None → default sani.

    Returns:
        Lista di `EmbeddingResult` nello stesso ordine di `texts`.
        `len(result) == len(texts)`.

    Raises:
        EmbedderError: se i retry sono esauriti o l'errore upstream è
            non recuperabile (auth, bad request).
    """
    config = config or EmbedderConfig()
    if not texts:
        return []

    results: list[EmbeddingResult] = []
    # Splittiamo l'input in batch di al massimo `batch_size` elementi.
    # Range step-based + slice è il pattern Pythonic standard per fare
    # questo. Stesso pattern di `for (let i=0; i<n; i+=size)` in JS.
    for start in range(0, len(texts), config.batch_size):
        batch = texts[start : start + config.batch_size]
        batch_results = await _embed_one_batch(client, batch, config)
        results.extend(batch_results)

    logger.info(
        "embedder_done",
        extra={
            "total_texts": len(texts),
            "total_tokens": sum(r.token_count for r in results),
            "batches": (len(texts) + config.batch_size - 1) // config.batch_size,
            "model": config.model,
        },
    )
    return results


# ===========================================================================
# Implementazione interna
# ===========================================================================


async def _embed_one_batch(
    client: AsyncOpenAI,
    batch: list[str],
    config: EmbedderConfig,
) -> list[EmbeddingResult]:
    """Embedda un singolo batch con retry su errori transitori.

    Strategia:
    - Loop fino a `max_retries + 1` tentativi.
    - Su `RateLimitError` / `APITimeoutError` / 5xx → sleep
      `2 ** attempt` secondi (1, 2, 4, 8, ...) e riprova.
    - Su `AuthenticationError` / `BadRequestError` → raise subito
      (non sono recuperabili con retry).
    """
    last_exc: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            response = await client.embeddings.create(
                model=config.model,
                input=batch,
                timeout=config.timeout_s,
            )
            # `response.data` può ARRIVARE in ordine diverso dall'input.
            # È un comportamento documentato: ordina per `.index`.
            sorted_data = sorted(response.data, key=lambda d: d.index)

            # L'API ritorna `usage.total_tokens` per il batch intero
            # (non per item). Distribuiamo i token proporzionalmente
            # alla lunghezza dei testi: approssimazione ragionevole
            # per il cost-tracking; per un conteggio esatto bisognerebbe
            # passare item-by-item.
            total_tokens = response.usage.total_tokens or 0
            total_chars = sum(len(t) for t in batch) or 1  # evita /0
            token_per_char = total_tokens / total_chars

            return [
                EmbeddingResult(
                    text=batch[item.index],
                    vector=item.embedding,
                    token_count=max(1, round(token_per_char * len(batch[item.index]))),
                )
                for item in sorted_data
            ]

        except (AuthenticationError, BadRequestError) as exc:
            # Errori "hard": non recuperabili. Esempio: API key sbagliata,
            # input troppo lungo, modello sconosciuto. Re-raise come
            # `EmbedderError` semantica per non far filtrare il dettaglio
            # SDK nel resto del codice.
            logger.error(
                "embedder_hard_error",
                extra={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise EmbedderError(f"Errore non recuperabile da OpenAI: {type(exc).__name__}") from exc

        except (RateLimitError, APITimeoutError, APIError) as exc:
            # Errori "soft": riprova con backoff esponenziale.
            # APIError è la superclasse di vari errori che includono
            # i 5xx — il retry è sensato.
            last_exc = exc
            if attempt < config.max_retries:
                # 2 ** 0 = 1s, 2 ** 1 = 2s, 2 ** 2 = 4s, 2 ** 3 = 8s
                backoff_s = 2**attempt
                logger.warning(
                    "embedder_retrying",
                    extra={
                        "attempt": attempt + 1,
                        "max_retries": config.max_retries,
                        "backoff_s": backoff_s,
                        "error_type": type(exc).__name__,
                    },
                )
                await asyncio.sleep(backoff_s)
                continue
            # Esaurito i retry.
            break

    # Se siamo qui, abbiamo esaurito i retry su un errore soft.
    logger.error(
        "embedder_retries_exhausted",
        extra={
            "max_retries": config.max_retries,
            "error_type": type(last_exc).__name__ if last_exc else None,
        },
    )
    raise EmbedderError(f"Embedding fallito dopo {config.max_retries + 1} tentativi") from last_exc


__all__ = [
    "EmbedderConfig",
    "EmbedderError",
    "EmbeddingResult",
    "embed_texts",
]
