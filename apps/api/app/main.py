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

# stdlib: per misurare la durata di una request, per generare ID univoci,
# e per leggere la versione del pacchetto.
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

# FastAPI: la classe principale, usata per creare l'ASGI app.
from fastapi import FastAPI, Request, Response

# CORSMiddleware: middleware ASGI per gestire la Cross-Origin Resource
# Sharing. È il "permesso esplicito" che il backend dà al browser per
# consentire fetch da un'origine diversa (es. frontend su :3000 chiama
# backend su :8000). Senza questo, il browser bloccherebbe le risposte.
# In termini Express: equivalente del package `cors`.
from fastapi.middleware.cors import CORSMiddleware

# Pydantic BaseModel: superclasse per definire schemi I/O tipizzati.
from pydantic import BaseModel

# Settings: il singleton di configurazione (vedi app/config.py).
from app.config import settings

# Logging: setup centralizzato + helper per il request_id.
from app.log import set_request_id, setup_logging

# Logger di modulo. Convenzione Python: `logging.getLogger(__name__)` usa
# il dotted path del modulo (qui "app.main"), così nei log puoi capire
# da dove arriva ogni riga. Tutte le librerie Python fanno così.
logger = logging.getLogger(__name__)


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
# Lifespan: codice che gira all'avvio e allo shutdown dell'app.
# ---------------------------------------------------------------------------
# Il pattern moderno di FastAPI per fare setup/teardown è il `lifespan`:
# un async context manager che parte all'avvio del processo e termina
# alla chiusura. Sostituisce i vecchi decorator `@app.on_event("startup")`
# e `@app.on_event("shutdown")`, che sono deprecati.
#
# Qui dentro inizializziamo il logging UNA volta: configurare il root
# logger è un'operazione globale, va fatta una volta sola e idealmente
# il prima possibile, in modo che tutti i log successivi (compresi quelli
# emessi durante il request handling) usino la nostra config.
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging(level=settings.log_level, fmt=settings.log_format)
    logger.info(
        "backend_starting",
        extra={
            "env": settings.app_env,
            "version": app.version,
            "log_format": settings.log_format,
        },
    )
    yield  # <-- l'app è "viva" mentre siamo bloccati qui
    logger.info("backend_shutdown")


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
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware: Request ID + structured request logging.
# ---------------------------------------------------------------------------
# FastAPI (Starlette) supporta i middleware via `@app.middleware("http")`.
# Sono funzioni `async` che ricevono `(request, call_next)`: come Express,
# decidi cosa fare prima e dopo che la prossima funzione gestisce la request.
#
# Cosa fa il nostro middleware:
#   1. Genera (o eredita dall'header X-Request-ID) un identificativo univoco.
#   2. Lo mette nel ContextVar (vedi app/log.py) così ogni log emesso
#      durante il request handling lo include automaticamente.
#   3. Logga `request_start` e poi `request_end` con duration_ms e status.
#   4. Restituisce l'header `X-Request-ID` al client (utile per il debug:
#      se l'utente segnala un errore, può copiare l'ID e cercarlo nei log).
@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    # Se il client manda un X-Request-ID (es. per propagare un trace ID
    # da un servizio upstream), lo riusiamo; altrimenti generiamo un UUID4.
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    set_request_id(request_id)

    logger.info(
        "request_start",
        extra={"path": request.url.path, "method": request.method},
    )

    # time.perf_counter() è il timer monotono ad alta risoluzione; ideale
    # per misurare durate, non orari assoluti (per quelli serve time.time()).
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # `exc_info=True` allega il traceback al log. Senza, vedresti solo
        # il messaggio e dovresti indovinare cosa è esploso.
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request_error",
            extra={
                "path": request.url.path,
                "method": request.method,
                "duration_ms": round(duration_ms, 2),
            },
        )
        raise  # rilancia: lasciamo che FastAPI risponda con 500.

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request_end",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status": response.status_code,
            "duration_ms": round(duration_ms, 2),
        },
    )
    response.headers["x-request-id"] = request_id
    return response


# ---------------------------------------------------------------------------
# Middleware CORS — consente al frontend di chiamare il backend dal browser.
# ---------------------------------------------------------------------------
# Quando il browser su `localhost:3000` fa `fetch('http://localhost:8000/...')`,
# scatta la same-origin policy: il browser BLOCCA la risposta a meno che il
# server non includa l'header `Access-Control-Allow-Origin` con l'origine
# del frontend.
#
# `CORSMiddleware` aggiunge quegli header automaticamente.
#
# Configurazione minimale (M1):
#   - allow_origins: solo l'origine del frontend dichiarato nel .env.
#                    NON usare ["*"] in produzione: significa "qualunque
#                    sito può chiamarti da browser".
#   - allow_credentials=True: permette al browser di mandare cookies /
#                    Authorization headers. Sarà utile in M3.
#   - allow_methods / allow_headers: per ora autorizziamo tutto, in
#                    futuro restringeremo (es. solo POST per /chat).
#
# Curiosità: CORS NON è una feature di sicurezza del server. Da `curl`
# (che non è un browser) il backend risponde sempre. CORS protegge gli
# utenti del browser contro siti malevoli che proverebbero a chiamare
# API su cui sono loggati altrove.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
