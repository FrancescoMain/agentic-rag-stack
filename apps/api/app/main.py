"""
app/main.py
===========
Entrypoint del backend FastAPI.

In questo file creiamo l'**ASGI application** (variabile `app`) che
`uvicorn` esegue. È il punto di partenza dell'intero backend: ogni
richiesta HTTP entra da qui e viene smistata a una funzione registrata.

------------------------------------------------------------------------
Concetti chiave (per chi viene da Node/Express)
------------------------------------------------------------------------

ASGI (Asynchronous Server Gateway Interface)
    Specifica Python che descrive come un server (uvicorn) parla con
    un'applicazione (FastAPI). È la versione async di WSGI, lo standard
    sincrono storico.

    Analogia: WSGI sta a Flask come Express-senza-async sta a Node;
    ASGI sta a FastAPI come l'event loop di Node sta a Express moderno.

FastAPI
    Framework web costruito sopra Starlette (motore HTTP/routing) +
    Pydantic (validazione I/O). Tre superpoteri:
      1. Async by default.
      2. Validazione automatica dai type hints Python.
      3. Documentazione OpenAPI generata in automatico (/docs).

Pydantic
    Libreria di validazione dati. Definisci una classe `BaseModel`,
    Pydantic la usa per validare input, serializzare output, e
    generare JSON schema. Equivalente concettuale di Zod in TypeScript.

------------------------------------------------------------------------
Come si esegue questo file
------------------------------------------------------------------------

Dalla root di `apps/api/`:

    uv sync                                    # installa dipendenze
    uv run uvicorn app.main:app --reload       # avvia il server

`app.main:app` significa: "nel modulo `app.main`, prendi la variabile
chiamata `app`". `--reload` riavvia il server a ogni modifica di file
(solo per dev).
"""

# stdlib: per leggere la versione dal pyproject.toml senza ripeterla a mano.
from importlib.metadata import PackageNotFoundError, version

# FastAPI: la classe principale, usata per creare l'ASGI app.
from fastapi import FastAPI

# Pydantic BaseModel: superclasse per definire schemi I/O tipizzati.
from pydantic import BaseModel


def _read_version() -> str:
    """Legge la versione del pacchetto installato.

    Se per qualche motivo il pacchetto non è installato (es. lo stiamo
    eseguendo prima di `uv sync`), ritorniamo `0.0.0+unknown`.

    Perché farlo così invece di hardcodare la versione?
    → Single source of truth: la versione è definita una sola volta in
      `pyproject.toml`, e da lì la leggiamo dovunque serva.
    """
    try:
        # "agentic-rag-api" è il nome dichiarato in `pyproject.toml`.
        return version("agentic-rag-api")
    except PackageNotFoundError:
        return "0.0.0+unknown"


# ---------------------------------------------------------------------------
# Creazione dell'ASGI application.
# ---------------------------------------------------------------------------
# `app` è la variabile che uvicorn cerca quando lo lanci con
# `uvicorn app.main:app`. I parametri qui sotto popolano la pagina
# interattiva OpenAPI servita su /docs (esiste GRATIS, è una delle killer
# feature di FastAPI).
app = FastAPI(
    title="agentic-rag-api",
    description="Backend del progetto agentic-rag-stack.",
    version=_read_version(),
)


# ---------------------------------------------------------------------------
# Schema della risposta di /health.
# ---------------------------------------------------------------------------
# Una classe Pydantic che descrive la *forma* della risposta JSON.
# FastAPI la usa per due cose:
#   1. Validare run-time: se l'endpoint ritorna qualcosa che non rispetta
#      lo schema, FastAPI alza un errore esplicito invece di rispondere
#      con dati corrotti.
#   2. Generare lo schema OpenAPI: la documentazione su /docs mostrerà
#      esattamente questi campi con i loro tipi.
class HealthResponse(BaseModel):
    """Risposta dell'endpoint /health."""

    status: str
    version: str


# ---------------------------------------------------------------------------
# Endpoint: GET /health
# ---------------------------------------------------------------------------
# Il decorator `@app.get("/health", ...)` registra la funzione come handler
# della route HTTP `GET /health`. In Express sarebbe:
#     app.get('/health', (req, res) => { ... })
#
# Differenze rispetto a Express:
#   - La funzione è `async`: il return value viene atteso come una Promise.
#   - `response_model=HealthResponse` dice a FastAPI di validare il return
#     contro lo schema. Se non corrisponde → errore 500 esplicito.
#   - Il type hint `-> HealthResponse` è ridondante a runtime, ma utile per
#     l'IDE e per i tipi statici (mypy, pyright).
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health-check endpoint.

    Usato da:
      - Il frontend per verificare che il backend sia raggiungibile.
      - Docker/Kubernetes per le liveness/readiness probes (M5).
      - Lo sviluppatore (`curl localhost:8000/health`) per smoke-test
        durante lo sviluppo.

    Convenzione: ritorna 200 OK se il processo è vivo. Più avanti (M5)
    estenderemo questo endpoint con check di dipendenze esterne
    (vector DB raggiungibile? LLM provider risponde? ecc.).
    """
    return HealthResponse(status="ok", version=app.version)
