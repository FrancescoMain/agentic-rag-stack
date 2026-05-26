"""
tests/test_health.py
====================
Smoke test del backend: verifica che l'endpoint /health risponda
correttamente. Se questo test si rompe, qualcosa di fondamentale nel
backend non funziona.

------------------------------------------------------------------------
Concetti chiave
------------------------------------------------------------------------

pytest
    Framework di testing standard nel mondo Python. Convenzioni:
    - I file di test si chiamano `test_*.py`.
    - Le funzioni di test si chiamano `test_*()`.
    - Per asserire si usa la keyword `assert` di Python (NON serve
      `expect().toBe()` come in Vitest).
    - Esegui con `uv run pytest`.

FastAPI TestClient
    Client HTTP che parla con l'app FastAPI **in-process**, senza
    bisogno di avviare uvicorn. Costruito sopra httpx. Permette di
    scrivere test di integrazione molto veloci.

    Mental model: come `supertest` in Node.
"""

from fastapi.testclient import TestClient

# Importiamo l'ASGI app dal modulo principale.
# Questo richiede che `app/` sia un package (vedi app/__init__.py) e che
# il progetto sia installato nel venv (`uv sync` lo fa automaticamente
# grazie a `[tool.hatch.build.targets.wheel] packages = ["app"]`).
from app.main import app

# TestClient prende l'ASGI app e fornisce un'interfaccia "stile requests"
# per fare chiamate sincrone all'app stessa.
client = TestClient(app)


def test_health_returns_ok() -> None:
    """GET /health → 200 OK con il body atteso."""
    # Esegue una richiesta HTTP "finta" verso l'app (in-process).
    response = client.get("/health")

    # Status code: 200 OK è l'indicazione minima che il backend è vivo.
    assert response.status_code == 200

    # Body: validiamo struttura e tipi.
    data = response.json()
    assert data["status"] == "ok"
    # La versione viene letta da pyproject.toml a runtime: ci basta sapere
    # che il campo c'è ed è una stringa non vuota.
    assert isinstance(data["version"], str)
    assert len(data["version"]) > 0


def test_health_response_matches_schema() -> None:
    """Assicura che la risposta abbia ESATTAMENTE i campi previsti dallo
    schema (niente campi extra, niente campi mancanti).

    Perché lo facciamo:
      Se in futuro qualcuno aggiunge un campo a HealthResponse senza
      pensarci, questo test fallisce e ci forza ad aggiornare anche il
      test (e quindi il contratto API documentato).
    """
    response = client.get("/health")
    data = response.json()

    # Set di chiavi attese. Usiamo set per ignorare l'ordine.
    assert set(data.keys()) == {"status", "version"}
