# `apps/api/` — Backend Python (FastAPI + LangGraph)

Il cervello del sistema. Espone un'API HTTP/streaming che il frontend
consuma, orchestrando agenti LangGraph che a loro volta usano una pipeline
RAG, tool e modelli LLM.

> **Stato attuale (fine M1).** Sono funzionanti:
> - `GET /health` — liveness/readiness probe
> - `POST /classify` — *Invisible AI*: un endpoint REST che dietro le quinte
>   chiama Claude Haiku 4.5 per classificare un testo (`bug | feature | question | spam`).
>
> Le cartelle `agents/`, `rag/`, `tools/`, `evals/` esistono ma sono ancora
> stub: saranno popolate dalle milestone successive (vedi
> [docs/ROADMAP.md](../../docs/ROADMAP.md)).

---

## Mappa della cartella

```
apps/api/
├── pyproject.toml          ← dipendenze + config Python (PEP 621/735)
├── uv.lock                 ← lockfile generato da `uv sync`
├── app/                    ← package principale (importabile come `app.*`)
│   ├── main.py             ← entrypoint FastAPI: lifespan, middleware, route
│   ├── config.py           ← Settings via pydantic-settings (.env → typed)
│   ├── log.py              ← structured logging + request_id (ContextVar)
│   ├── services/
│   │   └── classifier.py   ← logica /classify (prompt + parsing + retry)
│   ├── agents/             ← (M4) state machines LangGraph
│   ├── rag/                ← (M2) chunking, embedding, retrieval, rerank
│   ├── tools/              ← (M4) tool definitions per gli agenti
│   └── evals/              ← (M2/M5) golden datasets + evaluator
└── tests/
    ├── conftest.py         ← fixture pytest condivise (TestClient, fake LLM)
    ├── test_health.py      ← test su GET /health
    └── test_classify.py    ← test su POST /classify (Anthropic mockato)
```

---

## Come si avvia (dev)

### 1. Prerequisiti (una volta sola)

```bash
# Python 3.12 + uv installati a livello di sistema.
uv --version          # deve esistere
uv run python --version  # deve stampare 3.12.x
```

Se manca `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
(vedi [docs.astral.sh/uv](https://docs.astral.sh/uv/) per Windows).

### 2. Variabili d'ambiente

Le variabili stanno in `.env` **nella root del repo** (non in `apps/api/`).
La prima volta:

```bash
cd ../..              # vai alla root del repo
cp .env.example .env
```

Per M1 servono solo:

| Variabile           | Default                  | Note                                   |
| ------------------- | ------------------------ | -------------------------------------- |
| `APP_ENV`           | `development`            | "production" disattiva il reload       |
| `API_PORT`          | `8000`                   | porta su cui uvicorn ascolta           |
| `FRONTEND_ORIGIN`   | `http://localhost:3000`  | CORS allowlist                         |
| `LOG_LEVEL`         | `INFO`                   | DEBUG/INFO/WARNING/ERROR/CRITICAL      |
| `LOG_FORMAT`        | `dev`                    | `dev` (umano colorato) o `json`        |
| `ANTHROPIC_API_KEY` | — (richiesto da /classify) | senza chiave, `/classify` → 503       |

> **Senza `ANTHROPIC_API_KEY`** il backend parte lo stesso: `/health`
> risponde normalmente, `/classify` ritorna `503 Service Unavailable` con
> messaggio esplicito. Comodo per CI/CD e demo offline.

### 3. Installare le dipendenze e avviare il server

Dalla cartella `apps/api/`:

```bash
uv sync                                          # installa runtime + dev deps
uv run uvicorn app.main:app --reload             # avvia il server in dev
```

Il server è ora su `http://localhost:8000`. Tre cose da provare subito:

```bash
curl http://localhost:8000/health
# → {"status":"ok","version":"0.0.1"}

# /classify richiede ANTHROPIC_API_KEY configurata.
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"Il pulsante login non funziona su mobile, scompare al tap"}'
# → {"category":"bug","confidence":0.92,"reasoning":"..."}

# OpenAPI interattivo (Swagger UI) auto-generato da FastAPI:
open http://localhost:8000/docs
```

### 4. Cosa fa `uv run` sotto il cofano

- `uv sync` crea/aggiorna un virtualenv in `apps/api/.venv/` (gitignored)
  e installa esattamente ciò che dice `uv.lock`. Equivalente di
  `pnpm install` con il `pnpm-lock.yaml`.
- `uv run <cmd>` esegue `<cmd>` dentro quel virtualenv senza richiedere
  `source .venv/bin/activate`. È il pattern moderno per non sporcare la
  shell, comodissimo soprattutto in CI.

---

## Test, lint, format

```bash
uv run pytest                          # tutti i test (default: -v)
uv run pytest tests/test_classify.py   # solo un file
uv run pytest -k classify              # solo test che matchano la parola

uv run ruff check .                    # lint
uv run ruff format .                   # formatter (modifica i file)
uv run ruff format --check .           # solo verifica (usato in CI)
```

I test di `/classify` **non chiamano Anthropic per davvero**: il client
async viene sostituito via `app.dependency_overrides` con un fake che
ritorna risposte controllate (vedi `tests/conftest.py`). Quindi `pytest`
è veloce, deterministico e non consuma budget API.

---

## Endpoint disponibili oggi

### `GET /health`

Liveness check. Usato dal frontend per il pinger, e dai container
orchestrator (M5).

```jsonc
// 200 OK
{ "status": "ok", "version": "0.0.1" }
```

### `POST /classify`

*Invisible AI*: input testo libero, output categoria + confidence +
reasoning. Sotto il cofano chiama Claude Haiku 4.5 con un prompt che
forza la risposta in JSON e ne valida lo schema con Pydantic.

```jsonc
// Request
{ "text": "string (1..4000 chars)" }

// 200 OK
{
  "category": "bug" | "feature" | "question" | "spam",
  "confidence": 0.0..1.0,
  "reasoning": "string"
}
```

Mapping errori:

| Status | Quando                                                            |
| ------ | ----------------------------------------------------------------- |
| 400    | `text` mancante / troppo lungo / vuoto (validato da Pydantic).    |
| 422    | Body non JSON (gestito da FastAPI).                               |
| 502    | Anthropic ha risposto, ma con JSON malformato o schema sbagliato. |
| 503    | `ANTHROPIC_API_KEY` non configurata.                              |

---

## Concetti chiave (per chi viene dal frontend)

Cappello rapido per orientarsi nel codice. Per il dettaglio, ogni file
ha docstring lunghe in stile didattico.

### Toolchain

| Python                  | Equivalente JS/Node                          |
| ----------------------- | -------------------------------------------- |
| `pyproject.toml`        | `package.json` (PEP 621 + PEP 735)           |
| `uv.lock`               | `pnpm-lock.yaml`                             |
| `.venv/`                | `node_modules/`                              |
| `uv sync`               | `pnpm install`                               |
| `uv run <cmd>`          | `pnpm <cmd>` / `npx <cmd>`                   |
| `ruff check`/`format`   | `eslint` + `prettier` in un solo binario     |
| `pytest`                | `vitest` / `jest`                            |

### Runtime e framework

- **ASGI vs WSGI** — WSGI è lo standard Python *sincrono* storico
  (Flask, Django classico). ASGI è la versione *async*: il server
  (`uvicorn`) parla con l'app via un'interfaccia che supporta
  `async/await`. FastAPI è ASGI-native. Per il frontend engineer:
  WSGI ≈ Node *bloccante*, ASGI ≈ Node con event loop.

- **FastAPI** — framework basato su Starlette (router + middleware) +
  Pydantic (validazione). I superpoteri: async by default, schema OpenAPI
  generato automaticamente da `/docs`, validazione dei tipi a runtime
  partendo dai type hint Python.

- **Pydantic v2** — libreria di validazione. Definisci `class Foo(BaseModel)`,
  Pydantic la usa per validare input HTTP, serializzare output, generare
  JSON schema. *Mental model: Zod per Python, ma con type hint nativi del
  linguaggio invece di builder API.*

- **`pydantic-settings`** — estensione che popola un `BaseSettings` da
  variabili d'ambiente + `.env`. Validazione tipata gratis: se metti
  `API_PORT=banana`, il server non parte e l'errore dice esattamente
  cosa è sbagliato. *Mental model: Zod + dotenv in un solo pacchetto.*

### Pattern adottati nel codice

- **Lifespan** (`@asynccontextmanager`) — il pattern moderno di FastAPI
  per setup/teardown a livello di processo. Crea il client Anthropic
  una sola volta all'avvio e lo chiude correttamente allo shutdown.
  Sostituisce i deprecati `@app.on_event("startup"/"shutdown")`.

- **Dependency injection** (`Depends(...)`) — funzioni-fabbrica che
  FastAPI chiama prima dell'handler. Vantaggi: testabilità (test
  override via `app.dependency_overrides[...]`), errori coerenti
  (es. 503 se manca la API key è centralizzato in un solo posto).

- **Structured logging con `request_id`** — ogni log riga ha un campo
  `request_id` che identifica univocamente la richiesta HTTP. È
  propagato via `ContextVar`, l'equivalente Python del *AsyncLocalStorage*
  di Node: visibile in qualunque funzione async chiamata durante quella
  request, senza passarlo esplicitamente. Formato configurabile:
  `dev` (umano colorato) o `json` (per aggregatori esterni in prod).

- **CORS** — middleware che aggiunge gli header
  `Access-Control-Allow-*`. Senza, il browser blocca le risposte fra
  origini diverse (`:3000` → `:8000`). NB: CORS protegge gli utenti
  del browser, non il server; da `curl` il backend risponde sempre.

---

## Troubleshooting

| Sintomo                                      | Causa probabile / Fix                                          |
| -------------------------------------------- | -------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'app'` | Hai lanciato `uvicorn` fuori da `apps/api/`. Fai `cd apps/api`. |
| `503` su `/classify`                         | `ANTHROPIC_API_KEY` mancante nel `.env` della root.            |
| `CORS error` nel browser                     | `FRONTEND_ORIGIN` nel `.env` non combacia con la URL del FE.   |
| `uv: command not found`                      | Installa uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh`. |
| Test rallentati / chiamate vere ad Anthropic | Controlla `tests/conftest.py`: l'override deve essere attivo.  |

---

## Riferimenti

- **Roadmap del progetto**: [`docs/ROADMAP.md`](../../docs/ROADMAP.md)
- **ADR**: [`docs/adr/`](../../docs/adr/) (decisioni di architettura)
- **README dei sub-package**: [`app/agents/`](app/agents/README.md),
  [`app/rag/`](app/rag/README.md), [`app/tools/`](app/tools/README.md),
  [`app/evals/`](app/evals/README.md), [`tests/`](tests/README.md)
