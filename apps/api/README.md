# `apps/api/` — Backend Python (FastAPI + LangGraph)

Il cervello del sistema. Espone un'API HTTP/streaming che il frontend
consuma, orchestrando agenti LangGraph che a loro volta usano una pipeline
RAG, tool e modelli LLM.

> **Stato attuale:** scheletro vuoto. Il contenuto sarà creato in **M1**
> (FastAPI hello-world + struttura) e cresciuto nelle milestone successive.

## Mappa della cartella

```
apps/api/
├── pyproject.toml          ← (M1) dipendenze + config Python
├── uv.lock                 ← (M1) lockfile generato da uv
├── app/                    ← package principale (importabile come `app.*`)
│   ├── main.py             ← (M1) entrypoint FastAPI (crea l'ASGI app)
│   ├── agents/             ← (M4) state machines LangGraph
│   ├── rag/                ← (M2) chunking, embedding, retrieval, rerank
│   ├── tools/              ← (M4) tool definitions per gli agenti
│   └── evals/              ← (M2/M5) suite di valutazione + golden datasets
└── tests/                  ← test (pytest)
```

## Come si esegue (anteprima)

Dalla root del repo:

```bash
cd apps/api
uv sync                                  # installa le dipendenze
uv run uvicorn app.main:app --reload     # avvia il server in dev
```

`uv sync` crea/aggiorna un virtualenv in `.venv/` (gitignored) basandosi
su `pyproject.toml`. È l'equivalente di `pnpm install` per il mondo Python.

`uv run` esegue un comando dentro quel virtualenv senza bisogno di
`activate`. Comodo soprattutto in CI/CD.

## Concetti chiave (per chi viene dal frontend)

| Concetto Python/backend | Equivalente JS/Node            |
| ----------------------- | ------------------------------ |
| `pyproject.toml`        | `package.json`                 |
| `uv.lock`               | `pnpm-lock.yaml`               |
| `.venv/`                | `node_modules/`                |
| Virtualenv              | Ambiente isolato per progetto  |
| FastAPI                 | Express / Fastify              |
| Uvicorn (ASGI server)   | Node HTTP server               |
| Pydantic                | Zod                            |
| pytest                  | Vitest / Jest                  |
