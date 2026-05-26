# ADR-0001: Struttura del repo come monorepo `apps/` + `docs/`

- **Status:** Accepted
- **Date:** 2026-05-26
- **Deciders:** Francesco
- **Tags:** `repo`, `tooling`, `dx`

---

## Context

`agentic-rag-stack` è composto da due artefatti deployabili:

1. **Un backend Python** (`apps/api/`) che orchestra agenti LangGraph,
   pipeline RAG, tool calls e integrazioni LLM. Tooling: `uv`,
   `FastAPI`, `pytest`.
2. **Un frontend Next.js** (`apps/web/`) che consuma il backend in
   streaming, con componenti AI-first (citazioni inline, generative UI).
   Tooling: `pnpm`, `Next.js`, `Tailwind`, `Shadcn UI`.

I due artefatti condividono:

- Contratti API (idealmente tipati end-to-end in futuro).
- Documentazione e ADR.
- Configurazione di deploy (Docker, CI/CD da M5).
- Variabili d'ambiente (un singolo `.env` per la dev locale).

Vincoli aggiuntivi:

- Progetto **didattico + portfolio**: deve essere leggibile da chi apre
  il repo per la prima volta. Una struttura "esoterica" alza la barriera.
- **Sviluppatore singolo** (almeno inizialmente): non abbiamo i problemi
  di scala di Google/Meta — quello che funziona deve essere semplice.

## Considered Options

### Option 1: Due repo separati (`agentic-rag-api` + `agentic-rag-web`)

- **Descrizione:** un repo per il backend, un repo per il frontend,
  collegati da contratti API documentati a mano.
- **Pro:**
  - Build/CI completamente indipendenti.
  - Permessi e visibilità separabili (es. open-source uno, privato l'altro).
  - Niente conflitti tra tooling Python e Node.
- **Contro:**
  - Doppia gestione: due `README.md`, due `.gitignore`, due CI config.
  - Cambi cross-cutting (es. modifica schema API → modifica componenti FE)
    richiedono *due PR coordinate*, perdita di atomicità.
  - Per un valutatore di portfolio è meno comodo: deve clonare due repo
    per capire il quadro.

### Option 2: Monorepo "flat" (codice mescolato alla root)

- **Descrizione:** tutto sotto la root, distinguendo solo per estensione
  file o convenzione (es. `server/` + `client/` o `python/` + `typescript/`).
- **Pro:**
  - Massima semplicità per progetti molto piccoli.
- **Contro:**
  - Non scala oltre 2 servizi: aggiungere un worker, un cron, una libreria
    condivisa diventa subito disordinato.
  - Tooling deve "indovinare" cosa esegue — non c'è una convenzione
    riconoscibile.

### Option 3: Monorepo con convenzione `apps/` + `packages/` *(scelta)*

- **Descrizione:** convenzione standard nell'ecosistema JS moderno (Turborepo,
  Nx, Vercel templates): `apps/` per cose che si *eseguono*, `packages/`
  per cose che si *importano*. Adattata anche a un mix Python + Node.
- **Pro:**
  - **Riconoscibile**: chi ha mai visto un monorepo Turborepo capisce in
    10 secondi cosa va dove.
  - **Atomico**: un singolo PR può toccare BE+FE+docs (es. aggiunta di un
    nuovo endpoint con citazioni → schema, retriever, componente).
  - **Scala bene**: quando aggiungeremo (eventualmente) `packages/shared`
    per tipi TS condivisi, la convenzione c'è già.
  - **Una sola fonte di verità** per `.env.example`, `.gitignore`, ADR,
    ROADMAP, Docker compose.
- **Contro:**
  - Tooling Python e Node coabitano: `uv` gestisce un venv in
    `apps/api/.venv/`, `pnpm` gestisce `apps/web/node_modules/`. Servono
    due comandi diversi per installare.
  - CI deve gestire due build stack diversi (gestibile con job paralleli).
  - Più rumore nel diff di `git log` se non si filtra per path.

## Decision

Abbiamo scelto **Option 3: monorepo con convenzione `apps/` + (futuro)
`packages/`**.

Il driver principale è la **leggibilità per chi guarda il repo dall'esterno**
(valutatori di portfolio, reviewer, te-stesso-fra-6-mesi): la convenzione
è quella più diffusa nell'ecosistema fullstack moderno e non richiede
spiegazioni.

Il secondo driver è l'**atomicità dei cambi**: gran parte delle feature
che faremo (es. "aggiungere citazioni inline") toccheranno BE+FE+docs
nello stesso commit. Forzarci a fare 2-3 PR coordinate avrebbe rallentato
tutto.

Lo **svantaggio reale** (tooling Python + Node che coabitano) è gestibile:
`uv` e `pnpm` non si pestano i piedi perché lavorano in cartelle diverse;
nel `README.md` di root mostriamo i due comandi distinti senza fronzoli.

## Consequences

### Positive

- Una sola installazione (`git clone` + `cp .env.example .env`) ti dà
  tutto.
- I cross-cutting changes restano atomici (un solo commit, un solo PR).
- Pronti a scalare con `packages/shared` se in M3+ avremo tipi
  TypeScript condivisi BE↔FE.
- Convenzione riconoscibile: nessuno deve "imparare" la struttura.

### Negative

- Due tooling installati in locale (uv + pnpm). Mitigato: entrambi sono
  veloci, moderni, ben documentati.
- CI dovrà avere due job (lint+test Python, lint+test Node). Non è un
  problema, è il caso comune.
- Il `.gitignore` di root deve coprire entrambi i mondi (fatto, vedi
  sezioni "Python" e "Node" del file).

### Neutral / Follow-ups

- Per ora non usiamo Turborepo / Nx: gestire 2 app a mano è banale e ci
  fa capire cosa stiamo facendo. Se in futuro la build cross-app diventa
  complessa (es. typegen automatico FE da schema BE), un nuovo ADR
  potrà introdurre Turborepo.
- Da valutare in M5: se aggiungiamo `packages/shared` per tipi condivisi,
  servirà probabilmente un `pnpm workspace` config alla root.

## References

- [Turborepo monorepo conventions](https://turborepo.com/docs/crafting-your-repository/structuring-a-repository)
- [Vercel monorepo guide](https://vercel.com/docs/monorepos)
- Repo di esempio: `vercel/ai-chatbot` (struttura simile).
