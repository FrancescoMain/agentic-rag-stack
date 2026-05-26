# ROADMAP — agentic-rag-stack

> **Cos'è questo documento.** È sia il **piano di lavoro** del progetto
> sia la **mappa di studio** per chi (come Francesco) lo costruisce
> imparando. Ogni milestone è pensata per essere *shippable on its own*:
> a fine milestone hai un artefatto autonomo dimostrabile, non una
> "feature a metà".
>
> Per la decisione su come strutturare il repo che sottende tutta la
> roadmap, vedi [ADR-0001](adr/0001-monorepo-structure.md).

## Come leggere

Ogni milestone ha 5 sezioni standard:

1. **Focus** — la promessa della milestone in 1 frase.
2. **Tasks** — sotto-task concreti, numerati. Sono lo scope.
3. **Definition of Done (DoD)** — checklist verificabile. La milestone è
   chiusa quando tutti i criteri sono verdi.
4. **Concetti chiave da studiare** — la mappa didattica. Non leggere
   tutto in anticipo: studia il concetto quando arrivi al task che lo usa.
5. **Output (cosa esiste a fine milestone)** — gli artefatti tangibili
   (endpoint, comandi CLI, dashboard, ecc.).

### Legenda status

| Simbolo | Significato                     |
| ------- | ------------------------------- |
| ⚪      | Planned (non iniziata)          |
| 🟡      | In progress                     |
| ✅      | Done (tutti i DoD verdi)        |
| ⏸️      | Paused / blocked                |

---

## Status d'insieme

| #  | Milestone                                | Status | Sintesi                                                       |
| -- | ---------------------------------------- | ------ | ------------------------------------------------------------- |
| M0 | Setup & scaffolding                      | ✅     | Tooling, struttura monorepo, primo ADR, ROADMAP               |
| M1 | Backend foundations & Invisible AI       | ✅     | FastAPI con /health + /classify (Haiku 4.5), FE pinger + playground |
| M2 | Knowledge base & RAG pipeline            | ⚪     | Ingestion, embedding, hybrid search, reranking, citazioni     |
| M3 | Streaming AI-first frontend              | ⚪     | Next.js + AI SDK, streaming SSE, citazioni inline, Gen UI     |
| M4 | Agentic workflows & Human-in-the-Loop    | ⚪     | LangGraph, tool calling, evaluator loop, approval gates       |
| M5 | Production, MLOps & tracing              | ⚪     | Docker, CI/CD con eval gate, Langfuse, runbook                |

> M0 non era nel README originale: l'ho introdotto qui perché lo scaffolding
> ha valore narrativo proprio (la storia del repo inizia da come è organizzato).

---

# M0 — Setup & scaffolding

**Status:** ✅ Done (chiusa 2026-05-26)

## Focus

Tooling locale + struttura monorepo + primo ADR + ROADMAP dettagliato.
Da qui in poi possiamo scrivere codice senza più discutere *dove* metterlo.

## Tasks

1. ✅ Installare Python 3.12 via `uv` e `pnpm` globale.
2. ✅ Scaffold della monorepo (`apps/`, `docs/`, file di root) con README
   didattici per ogni cartella concettuale.
3. ✅ Scrivere `docs/adr/README.md` + `TEMPLATE.md` + `ADR-0001`
   (struttura monorepo).
4. ✅ Espandere `docs/ROADMAP.md` con dettaglio per milestone.

## Definition of Done

- [x] `uv run python --version` ritorna 3.12.x
- [x] `pnpm --version` esiste
- [x] `apps/api/`, `apps/web/`, `docs/adr/` esistono con i README di guida
- [x] `.gitignore` copre Python + Node + Claude + env files
- [x] `.env.example` documenta le variabili per milestone
- [x] `docs/adr/0001-monorepo-structure.md` esiste con status Accepted
- [x] `docs/ROADMAP.md` espande ogni milestone con Tasks, DoD, concetti,
      output

## Concetti chiave da studiare

- **`uv` come gestore unificato Python** — installa interpreti, virtualenv,
  dipendenze. Mental model: "npm + nvm + venv in un unico binario".
- **Convenzione monorepo `apps/` + `packages/`** — usata da Turborepo,
  Nx, Vercel templates.
- **ADR (Architecture Decision Record)** — vedi `docs/adr/README.md`.
- **Conventional Commits** — `<type>(<scope>): subject` per messaggi
  di commit standardizzati.

## Output

- Repo pronto a ricevere codice. Nessun endpoint/servizio gira ancora.

---

# M1 — Backend foundations & Invisible AI

**Status:** ✅ Done (chiusa 2026-05-26)

## Focus

Mettere su un backend Python solido (FastAPI + Pydantic + structured
logging) e dimostrare il primo caso d'uso AI **senza chat**: un endpoint
che fa una cosa AI utile (es. classificazione testo) in modo *invisibile*
al frontend, che lo consuma come una normale REST API.

> **Perché partire dall'"invisible AI"?**
> Perché è il caso più semplice ma anche il più sottovalutato: prima di
> costruire una chat con streaming, conviene avere un singolo endpoint
> `POST /classify` che funziona. Dimostra che sai integrare un LLM in
> un'architettura HTTP classica — competenza spesso più richiesta sul
> lavoro che lo streaming.

## Tasks

1. ✅ **Init Python project**: `uv init`, `pyproject.toml` con dipendenze
   minime, linting con `ruff`.
2. ✅ **App skeleton**: `app/main.py` con FastAPI app, `app/config.py` con
   settings caricati da `.env` via `pydantic-settings`.
3. ✅ **Health endpoint**: `GET /health` → `{"status": "ok", "version": "..."}`.
4. ✅ **CORS**: configurato per accettare `FRONTEND_ORIGIN` dal `.env`.
5. ✅ **Structured JSON logging**: configurazione che produce log
   leggibili in dev e parsabili in prod, con `request_id` propagato.
6. ✅ **Primo endpoint AI**: `POST /classify` → riceve `{text: string}`,
   chiama Anthropic Claude (Haiku 4.5) con un prompt che chiede una
   categoria (es. `bug | feature | question | spam`), ritorna
   `{category, confidence, reasoning}`. Tipizzato con schemi Pydantic.
7. ✅ **Test suite**: pytest + `TestClient` per testare `/health`
   e `/classify` (con mock del client Anthropic — 15 test totali).
8. ✅ **Frontend pinger**: `apps/web/` scaffolded con Next.js, home page
   che mostra lo stato live di `/health` del backend.
9. ✅ **README di `apps/api/`**: aggiornare con istruzioni "come si avvia"
   complete + concetti chiave.
10. ✅ **Frontend playground per `/classify`**: nella home page Next.js,
    seconda card con textarea + bottone "Classify" + visualizzazione
    risultato. Stesso pattern di `HealthStatus` (Client Component con
    stati loading/ok/error).

## Definition of Done

- [x] `cd apps/api && uv sync && uv run uvicorn app.main:app --reload`
      avvia il server su `localhost:8000`.
- [x] `curl localhost:8000/health` → 200 con `{"status":"ok",...}`.
- [x] `curl -X POST localhost:8000/classify -H "Content-Type: application/json" -d '{"text":"...."}'`
      → 200 con `{category, confidence, reasoning}` plausibile.
- [x] `uv run pytest` → tutti verdi (incluso test su `/classify`).
- [x] `uv run ruff check . && uv run ruff format --check .` → clean.
- [x] `localhost:8000/docs` mostra l'OpenAPI generato automaticamente.
- [x] `cd apps/web && pnpm dev` avvia Next.js su `localhost:3000`, la home
      mostra lo stato `/health` del backend.
- [x] La home page del frontend ha anche un playground per `/classify`
      (textarea + bottone + risultato).
- [x] Log strutturati JSON in apps/api visibili a stdout in formato
      umano in dev, JSON puro in prod.

## Concetti chiave da studiare

- **ASGI vs WSGI** — perché Python "asincrono" richiede ASGI.
- **FastAPI** — async-by-default, schemi Pydantic come contratto, OpenAPI
  generato automaticamente.
- **Pydantic v2** — validazione dati, mental model: "Zod per Python".
- **`pydantic-settings`** — caricamento type-safe di config da `.env`.
- **CORS** — perché esiste e come configurarlo correttamente.
- **Anthropic SDK Python** — prima chiamata `client.messages.create(...)`.
- **`pytest`** + fixtures + `monkeypatch` per mockare client esterni.
- **`ruff`** — linter + formatter unificato (sostituisce flake8 + black + isort).
- **`uv run`** — esecuzione comandi dentro venv senza `activate`.

## Output

- Backend FastAPI funzionante con 2 endpoint (`/health`, `/classify`).
- Frontend Next.js minimale che pinga il backend.
- Test suite pytest verde.
- ADR-0002 (probabile): scelta di `uv` come tooling Python (vs poetry/pip).

---

# M2 — Knowledge base & RAG pipeline

**Status:** ⚪ Planned

## Focus

Costruire la pipeline RAG end-to-end: prendere documenti, trasformarli
in vettori, recuperarli con accuratezza, ed esporre un endpoint
`/retrieve` che ritorna chunk con citazioni. Niente chat ancora — solo
"data → ricerca semantica con citazioni".

## Tasks

1. ✅ **ADR sul vector DB** (Pinecone managed vs pgvector self-hosted
   vs Qdrant self-hosted) — driver iniziali: rapidità di setup vs
   costo zero. Driver decisivo emerso: familiarità del decisore.
   → [ADR-0002](adr/0002-vector-db-choice.md) (Superseded) →
   [ADR-0003](adr/0003-vector-db-choice-qdrant.md): scelto Qdrant.
2. ✅ **Sample corpus**: scelto **i docs ufficiali di FastAPI**
   (repo [`fastapi/fastapi`](https://github.com/fastapi/fastapi),
   path `docs/en/docs/`, ~150 markdown). Vantaggi: stack coerente
   col backend (demo "meta"), lessico tecnico denso ottimo per
   stressare l'hybrid search, eval facile (il decisore conosce
   il dominio). Licenza MIT. **Freshness strategy:** re-ingest
   manuale on-demand via CLI (`uv run python -m app.ingest`);
   un'automazione via CI/CD potrà essere valutata in M5.
3. **Chunker** (`app/rag/chunker.py`): strategia token-based con overlap
   configurabile. Tipo `Chunk = {id, text, source, metadata}`.
4. **Embedder** (`app/rag/embedder.py`): wrapper attorno a OpenAI
   `text-embedding-3-small` con batch + retry.
5. **Vector store client** (`app/rag/vector_store.py`): abstract base
   class + implementazione concreta (Pinecone o pgvector secondo ADR).
6. **Ingestion CLI** (`app/ingest.py`): `uv run python -m app.ingest --source ./sample_docs --collection demo`
   — orchestra load → chunk → embed → upsert.
7. **Retriever** (`app/rag/retriever.py`): hybrid search = vector + BM25
   (o full-text di Postgres), con metadata filters.
8. **Reranker** (`app/rag/reranker.py`): wrapper Cohere `rerank-3` con
   fallback "no rerank" se non c'è chiave.
9. **Citation builder** (`app/rag/citations.py`): trasforma chunk
   recuperati in oggetti `{chunk_id, source, score, snippet}`.
10. **Endpoint** `POST /retrieve`: `{query, top_k, filters}` →
    `{chunks: [...], citations: [...]}`. Schema Pydantic.
11. **Golden dataset** (`app/evals/golden_datasets/`): 20-50 coppie
    `(query, expected_chunk_ids)` curate a mano.
12. **Eval `precision_at_k`** (`app/evals/evaluators/`): misura quanti
    chunk attesi sono nei top-k recuperati.
13. **Eval runner** (`app/evals/runners/run_regression.py`): esegue gli
    eval e stampa il report.

## Definition of Done

- [ ] `uv run python -m app.ingest --source ./sample_docs --collection demo`
      gira senza errori e popola il vector store.
- [ ] `POST /retrieve {query, top_k: 5}` ritorna 5 chunk con citazioni e score.
- [ ] `uv run python -m app.evals.runners.run_regression --collection demo`
      stampa precision@5, con valore ≥ 0.7 sul golden dataset.
- [ ] ADR-00NN documenta la scelta del vector DB e le sue conseguenze.
- [ ] Reranker on/off → differenza di precision@5 osservabile (anche
      piccola).
- [ ] Test pytest sui pezzi puri (chunker, citation builder) verdi.

## Concetti chiave da studiare

- **Embeddings** — vettori densi di alta dimensione (1536 per `text-embedding-3-small`).
  Cosine similarity come metrica di vicinanza semantica.
- **Vector DB internals** — HNSW (Hierarchical Navigable Small Worlds),
  IVF (Inverted File). Capire le costanti, non la matematica completa.
- **Chunking strategies** — fixed-size + overlap, semantic chunking,
  structure-aware (Markdown/HTML), e quando ognuna conviene.
- **Hybrid search** — perché solo-vettoriale fallisce su nomi propri /
  identificatori, e come BM25 lo compensa.
- **Reranking** — modello "cross-encoder" che vede `(query, chunk)` insieme
  e dà uno score più preciso ma più caro.
- **RAG evaluation** — precision@k, recall@k, MRR, NDCG. Quando usare quale.
- **Faithfulness vs relevance** — due dimensioni diverse della qualità
  RAG.

## Output

- Knowledge base demo popolata in un vector DB.
- Endpoint `/retrieve` funzionante con citazioni.
- Prima suite di eval con metriche numeriche.
- ADR-00NN su scelta vector DB.

---

# M3 — Streaming AI-first frontend

**Status:** ⚪ Planned

## Focus

Trasformare il backend in una chat con **streaming token-by-token** e
**citazioni inline**. Frontend "AI-first": il design segue le esigenze
di un'interfaccia LLM (latenza variabile, generative UI, content che
appare progressivamente), non il pattern "form + submit + risposta".

## Tasks

1. **Endpoint backend** `POST /chat`: stream SSE che produce token + eventi
   strutturati (start, token, citation, end). Schema documentato.
2. **Vercel AI SDK** installato in `apps/web/`, hook `useChat` configurato
   per consumare `/chat`.
3. **Chat UI base**: lista messaggi + input field, design Shadcn UI.
4. **Streaming rendering**: i token appaiono progressivamente con
   visual feedback (cursore lampeggiante).
5. **Citation Viewer component**: citazioni numerate `[1]`, `[2]` inline,
   click espande pannello con snippet + link alla fonte.
6. **Generative UI pattern**: il backend può emettere `{type: "structured", widget: "table", data: {...}}`
   e il frontend lo renderizza come componente React, non come testo.
7. **Optimistic UX**: messaggio utente appare istantaneamente, indicator
   "AI sta pensando" prima del primo token.
8. **Error handling**: rete persa, stream interrotto, retry.
9. **Mobile responsive** + accessibility (focus management, ARIA live regions).
10. **Test E2E con Playwright** (opzionale): "manda domanda → vedi
    risposta con almeno 1 citazione".

## Definition of Done

- [ ] Chat su `localhost:3000` produce risposte token-by-token consumando
      `/chat` del backend.
- [ ] Almeno 1 citazione `[1]` appare inline e si espande al click.
- [ ] Almeno un esempio di Generative UI funziona (es. domanda che produce
      una tabella renderizzata come componente).
- [ ] Interruzione di rete → messaggio d'errore + bottone "retry".
- [ ] Lighthouse mobile ≥ 90 per accessibility.

## Concetti chiave da studiare

- **Server-Sent Events (SSE)** vs WebSocket vs HTTP polling. Quando SSE
  vince (unidirezionale server→client, riconnessione automatica).
- **Vercel AI SDK**: `useChat`, `useCompletion`, `streamText` lato server.
- **Generative UI pattern** — perché "tutto come testo" è limitante e
  come superarlo.
- **React 19 / Next.js App Router**: Server Components, Server Actions
  (sai già queste, ma vediamo come si combinano con streaming AI).
- **ARIA live regions** per accessibility del contenuto streaming.

## Output

- Frontend chat completo, fluido, con citazioni.
- Pattern Generative UI dimostrato su almeno 1 caso d'uso.
- ADR sulla scelta SSE vs WebSocket.

---

# M4 — Agentic workflows & Human-in-the-Loop

**Status:** ⚪ Planned

## Focus

Trasformare il sistema da "chat che fa retrieval" a **agente che ragiona
in più passi**: decide quali tool chiamare, valuta i propri output, e si
ferma per chiedere approvazione umana quando sta per fare qualcosa di
irreversibile.

## Tasks

1. **ADR-00NN** su LangGraph vs state machine custom.
2. **Agent state** (`app/agents/state.py`): TypedDict che descrive lo
   stato condiviso fra i nodi (messages, retrieved_chunks, tool_calls,
   next_step, ecc).
3. **Tool: `search_documents`** (`app/tools/search.py`): wrapper attorno
   al retriever di M2.
4. **Tool: `web_search`** (`app/tools/web.py`): chiamata a Tavily o simile.
5. **Tool: `send_email`** (`app/tools/email.py`): **mock**, ma marcato come
   side-effect → richiede HITL.
6. **Agente baseline** (`app/agents/qa_agent.py`): grafo lineare
   `retrieve → answer → END`.
7. **Aggiunta tool calling**: il nodo "answer" può decidere di chiamare
   `search_documents` o `web_search`, e il grafo ha un loop.
8. **Evaluator loop**: nodo "critic" che valuta la risposta; se score < soglia
   torna al retrieve con query riformulata. Max N iterazioni.
9. **HITL checkpoint**: nodo "approval" prima di `send_email` → pausa il
   grafo e ritorna `{type: "awaiting_approval", action: ..., context: ...}`
   al frontend.
10. **Frontend UI di approvazione**: pannello che mostra l'azione proposta
    + bottoni Accept/Reject/Edit. Lo stato del grafo persistito (DB o file)
    per resume dopo refresh.
11. **Endpoint** `POST /agent/resume`: dato un `agent_run_id` e una decisione,
    rilancia il grafo.
12. **Test pytest sulle transizioni**: dato uno stato iniziale, verifica
    che il grafo finisca nel nodo atteso.

## Definition of Done

- [ ] Una query del tipo *"Trova X nei nostri documenti e poi cerca su Internet
      conferme"* fa chiamare 2 tool diversi e produce risposta unificata.
- [ ] Una query che chiede *"manda email a Y"* si ferma sul nodo HITL,
      mostra l'UI di approvazione, e resume correttamente dopo accept.
- [ ] Refresh del browser durante uno stato HITL non perde lo stato.
- [ ] Evaluator catturato in almeno un caso a fare retry (visibile nei trace).

## Concetti chiave da studiare

- **State machines per agenti** — perché un agente è una state machine
  (nodi = step, archi = transizioni, stato = "memoria").
- **LangGraph**: `StateGraph`, `add_node`, `add_edge`, `add_conditional_edges`,
  `interrupt_before` (per HITL).
- **Checkpointing**: come salvare/riprendere lo stato. LangGraph offre
  `MemorySaver`, `SqliteSaver`, `PostgresSaver`.
- **Tool calling API** — schema dei tool, "function calling" di OpenAI
  e Anthropic.
- **Evaluator pattern**: critic loops, "reflexion".
- **HITL design patterns**: quando interrompere, quanto contesto mostrare
  all'umano, come gestire timeout di approvazione.

## Output

- Sistema che orchestra >1 tool in un singolo turno.
- Almeno 1 HITL checkpoint funzionante end-to-end (BE→FE→BE).
- ADR su LangGraph.

---

# M5 — Production, MLOps & tracing

**Status:** ⚪ Planned

## Focus

Portare il sistema da "gira sul mio laptop" a "ha *almeno* gli ingredienti
di un sistema production": tracing su ogni chiamata, dashboard di costi,
container, CI/CD con eval gate, runbook. Non significa per forza deployarlo
da qualche parte — significa che *potresti*.

## Tasks

1. **Langfuse SDK** integrato in `apps/api/`: traccia ogni `client.messages.create`,
   ogni tool call, ogni retrieval, con tag per feature.
2. **Cost tracking**: token usage estratto dai response e logged in
   Langfuse + endpoint `/admin/costs` per riassunto.
3. **Dockerfile** per `apps/api/` (multi-stage, dependencies cache layer).
4. **Dockerfile** per `apps/web/` (Next.js standalone output).
5. **`docker-compose.yml`** orchestra: api + web + langfuse + postgres.
6. **GitHub Actions CI**:
   - `lint` (ruff + biome o eslint)
   - `test` (pytest + vitest se aggiunto)
   - `eval_regression` su un sub-set del golden dataset
7. **Eval gate**: PR bloccata se `precision@5` cala di > X% rispetto a `main`.
8. **Manual approval gate** per deploy "prod" (anche se prod non esiste,
   simuliamo con un environment GitHub Actions protetto).
9. **Runbook** (`docs/runbook.md`): deploy, rollback, rotazione chiavi,
   incident response template, cost alerting.
10. **ADR-00NN** su Langfuse (vs Datadog, vs Helicone, vs roll-our-own).
11. **Secret rotation procedure** documentata.

## Definition of Done

- [ ] `docker compose up` porta su l'intero stack senza errori, con un
      breve script di smoke-test.
- [ ] Langfuse dashboard mostra trace di una query, con cost breakdown
      sui sotto-step.
- [ ] Una PR su `main` esegue lint + test + eval_regression in GitHub Actions.
- [ ] Eval regression > soglia → check rosso, PR bloccata.
- [ ] `docs/runbook.md` copre almeno: deploy, rollback, rotazione chiavi,
      template di incident report.
- [ ] Almeno 1 alerta di costo (anche manuale, es. "se mensile > $X manda
      email") documentata.

## Concetti chiave da studiare

- **Tracing vs logging vs metrics** (i "3 pilastri" dell'observability).
- **LLM observability specificity**: perché un trace LLM ha bisogno di
  vedere prompt + completion + tool calls insieme, non basta uno span HTTP.
- **Langfuse data model**: traces, observations, scores.
- **Multi-stage Docker builds**: separare build deps (heavy) da runtime
  deps (light).
- **Next.js standalone output**: container "minimale" senza `node_modules` completo.
- **CI/CD con eval gate**: idea poco mainstream ma fondamentale per
  AI products (regressioni non si vedono nei test classici).
- **LLM cost monitoring**: token-level cost attribution, budget alerting.

## Output

- Sistema completo con tracing, container, CI/CD, runbook.
- Storia git che racconta passo-passo come ci siamo arrivati.
- README di root aggiornato con quick-start aggiornato e badge CI verde.

---

## Note generali

### Quando si chiude una milestone

Una milestone è chiusa quando **tutti** i DoD sono ✅ e i relativi commit
sono su `main`. A quel punto:

1. Aggiornare la tabella "Status d'insieme" in cima a questo file.
2. Aggiornare il README di root (badge "M_n: ✅").
3. Tag git: `git tag -a v0.X.0 -m "Milestone M_n complete"` (semantic
   versioning: 0.1.0 a fine M1, 0.2.0 a fine M2, ecc.).

### Quando si introduce un nuovo task

Se durante una milestone scopri un task **dentro lo scope** non previsto
→ aggiungilo direttamente qui. Se è **fuori scope** ma utile → backlog
mentale (o issue GitHub), non sporcare la roadmap.

### Quando si scrive un nuovo ADR

Vedi [`docs/adr/README.md`](adr/README.md). Regola pratica: se hai dovuto
considerare almeno un'alternativa seria, scrivi l'ADR *prima* di
implementare.
