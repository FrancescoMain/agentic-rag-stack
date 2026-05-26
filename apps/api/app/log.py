"""
app/log.py
==========
Logging strutturato per il backend.

> Nota sul nome del file: si chiama `log.py` (non `logging.py`) per
> evitare collisione col modulo `logging` della stdlib. Python si
> regola con import assoluti, ma rinominare elimina l'ambiguità per
> chi legge.

------------------------------------------------------------------------
Concetti chiave
------------------------------------------------------------------------

stdlib `logging`
    Modulo standard Python. Quasi ogni libreria Python lo usa
    internamente (`anthropic`, `httpx`, `uvicorn`, ecc.). Configurando
    UNA volta il root logger, influenziamo i log di TUTTE le librerie
    a valle.

`LogRecord`
    L'oggetto che stdlib costruisce per ogni log line: contiene il
    livello, il messaggio, il timestamp, e ogni campo extra passato
    via `logger.info("msg", extra={...})`. Il Formatter prende un
    LogRecord e lo trasforma in stringa (testo o JSON).

`logging.Filter`
    A dispetto del nome, i Filter NON servono per filtrare in senso
    proprio (anche se possono). Il loro uso idiomatico in Python è
    ARRICCHIRE i LogRecord con campi extra prima della formattazione.
    Lo usiamo qui per iniettare il `request_id`.

`ContextVar`
    Variabile della stdlib `contextvars` che propaga il proprio valore
    attraverso async boundaries. In un server async/await, è l'unico
    modo affidabile per avere "una variabile per richiesta": ogni
    coroutine ha il suo valore, senza interferenze.

    Mental model JS: l'equivalente di `AsyncLocalStorage` di Node.
"""

import json
import logging
import sys
from contextvars import ContextVar

# ----------------------------------------------------------------------------
# Request ID context: propagato attraverso async tramite ContextVar.
# ----------------------------------------------------------------------------
# default=None → log emessi FUORI da una request HTTP (es. startup)
#                  non avranno request_id, ed è semanticamente corretto.
_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str) -> None:
    """Imposta il request_id per la coroutine corrente."""
    _request_id_ctx.set(request_id)


def get_request_id() -> str | None:
    """Restituisce il request_id corrente, se impostato."""
    return _request_id_ctx.get()


# ----------------------------------------------------------------------------
# Filter: inietta request_id come campo del LogRecord.
# ----------------------------------------------------------------------------
class _RequestIdFilter(logging.Filter):
    """Aggiunge `record.request_id` a ogni LogRecord, prendendolo dal
    ContextVar. Se non c'è un request_id corrente, il campo vale None.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Il "filter" qui usato per ARRICCHIRE, non per filtrare via.
        # Tornare True significa "lascia passare questo record" (sempre).
        record.request_id = get_request_id()
        return True


# ----------------------------------------------------------------------------
# Formatter umano per dev — output colorato leggibile a occhio.
# ----------------------------------------------------------------------------
class _DevFormatter(logging.Formatter):
    """Formato compatto colorato. Esempio di output:

    INFO     14:23:01 app.main          [req=a3f9be12]  request_start path=/health
    """

    # ANSI escape codes per colorare il livello nel terminale.
    # Su un file/pipe questi byte sono ignorati o visibili — è dev only.
    _COLORS = {
        "DEBUG": "\x1b[90m",  # grigio
        "INFO": "\x1b[36m",  # ciano
        "WARNING": "\x1b[33m",  # giallo
        "ERROR": "\x1b[31m",  # rosso
        "CRITICAL": "\x1b[35m",  # magenta
    }
    _RESET = "\x1b[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelname, "")
        ts = self.formatTime(record, "%H:%M:%S")
        level = f"{color}{record.levelname:<8}{self._RESET}"
        line = f"{level} {ts} {record.name:<22}"

        # request_id è iniettato dal _RequestIdFilter. Mostriamolo
        # accorciato (primi 8 char) per ridurre il rumore visivo.
        req_id = getattr(record, "request_id", None)
        if req_id:
            line += f" [req={req_id[:8]}]"

        line += f"  {record.getMessage()}"

        # Aggancia i campi extra (passati via logger.info(..., extra={...}))
        # come `key=value` in fondo alla riga.
        extras = _extract_extras(record)
        if extras:
            kv = " ".join(f"{k}={v}" for k, v in extras.items())
            line += f"  {kv}"

        # Se il log è dentro un except con exc_info=True, appendi il traceback.
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


# ----------------------------------------------------------------------------
# Formatter JSON — output strutturato per prod / aggregatori esterni.
# ----------------------------------------------------------------------------
class _JsonFormatter(logging.Formatter):
    """Serializza il LogRecord come singola riga JSON.

    Output esempio (riformattato):
        {
          "ts": "2026-05-26T14:23:01",
          "level": "info",
          "logger": "app.main",
          "message": "request_start",
          "request_id": "a3f9be12-...",
          "path": "/health",
          "method": "GET"
        }
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        # request_id sempre presente (può essere None), così a valle
        # è facile filtrare logs "fuori request" (es. startup).
        payload["request_id"] = getattr(record, "request_id", None)

        # Campi extra (passati via logger.info(..., extra={...}))
        # diventano chiavi top-level del JSON.
        payload.update(_extract_extras(record))

        # Traceback se c'è.
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # `default=str` serializza oggetti non-JSON-nativi (es. datetime)
        # via str(). `ensure_ascii=False` per non scappare lettere
        # accentate / unicode in output (più leggibili in caso umano).
        return json.dumps(payload, default=str, ensure_ascii=False)


# ----------------------------------------------------------------------------
# Helper: estrae i campi "extra" dal LogRecord.
# ----------------------------------------------------------------------------
# Quando chiami logger.info("msg", extra={"foo": 1}), Python attacca "foo"
# al LogRecord come attributo. Ma il LogRecord ha anche TANTI attributi
# standard (asctime, levelname, args, ecc.). Per recuperare SOLO gli extra
# che ho passato io, calcoliamo la differenza con un denylist dei campi
# standard.
_STANDARD_LOGRECORD_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
    "request_id",  # nostro campo iniettato, lo gestiamo separatamente
}


def _extract_extras(record: logging.LogRecord) -> dict[str, object]:
    """Ritorna SOLO i campi extra passati a logger.info(..., extra={...})."""
    return {
        key: value for key, value in record.__dict__.items() if key not in _STANDARD_LOGRECORD_ATTRS
    }


# ----------------------------------------------------------------------------
# Setup pubblico — chiamato all'avvio dell'app (vedi app/main.py).
# ----------------------------------------------------------------------------
def setup_logging(level: str = "INFO", fmt: str = "dev") -> None:
    """Configura il root logger di Python.

    Args:
        level: livello minimo di log (DEBUG / INFO / WARNING / ERROR / CRITICAL).
        fmt: "dev" per output umano colorato, "json" per output strutturato.

    Note pratiche:
    - Va chiamata una sola volta all'avvio dell'app.
    - Configurare la ROOT influenza anche uvicorn, httpx, anthropic, ecc.
    - Rimuoviamo eventuali handler precedenti perché in dev uvicorn
      reload importa il modulo più volte: senza pulizia ogni reload
      DUPLICA i messaggi (log emessi 2x, poi 3x, ecc.).
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Cleanup di handler precedenti (uvicorn reload safety).
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_RequestIdFilter())

    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_DevFormatter())

    root.addHandler(handler)

    # Uvicorn ha un suo `uvicorn.access` logger che logga ogni richiesta in
    # formato proprio (e *non* via root). Lo silenziamo, perché la nostra
    # `RequestIdMiddleware` (vedi app/main.py) emette già log strutturati
    # per request_start / request_end con i nostri campi.
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False
