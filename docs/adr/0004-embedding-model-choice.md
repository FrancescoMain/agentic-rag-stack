# ADR-0004: OpenAI `text-embedding-3-small` come embedding model

- **Status:** Accepted
- **Date:** 2026-05-26
- **Deciders:** Francesco
- **Tags:** `rag`, `m2`, `llm-provider`

---

## Context

M2 (RAG pipeline) richiede di trasformare ogni chunk di testo in un
**vettore denso** ad alta dimensione. È il prerequisito per il
*retrieval* semantico: la query dell'utente diventa anch'essa un
vettore, e Qdrant cerca i chunk più "vicini" per cosine similarity.

Vincoli del progetto rilevanti:

- **Anthropic non offre embeddings.** Claude è un LLM, non un
  embedding model. Le doc ufficiali Anthropic rimandano a fornitori
  terzi (principalmente Voyage AI) per gli embedding.
- **Sviluppatore singolo, budget basso ma >0.** Per gli LLM siamo
  disposti a spendere centesimi/dollaro al mese; per gli embedding
  il volume è una tantum (ingestion) + qualche centinaio di query al
  giorno in dev → costi trascurabili.
- **Didattico + portfolio.** "Ho usato `text-embedding-3-small` di
  OpenAI" è una riga che chi assume riconosce immediatamente come
  scelta industry-standard 2024-2026.
- **Coerenza con [ADR-0003](0003-vector-db-choice-qdrant.md)**: Qdrant
  accetta qualunque vettore, ma la dimensione (384 / 768 / 1536 / ...)
  va fissata alla creazione della collezione.

## Considered Options

### Option 1: OpenAI `text-embedding-3-small` *(scelta)*

- **Descrizione:** modello embedding di OpenAI rilasciato a gennaio
  2024. Output 1536-dim (riducibile via parametro `dimensions=` per
  trade-off qualità/costo). Disponibile via API REST con SDK Python
  ufficiale.
- **Pro:**
  - **Riconoscibilità industry-standard**: usato da migliaia di
    progetti production, ben documentato, ben capito.
  - **Qualità competitiva**: in top 10 globale del benchmark MTEB
    (Massive Text Embedding Benchmark), spesso sopra alternative
    più costose.
  - **Costo molto basso**: $0.020 per 1M token (input). Stima per
    M2: ingestare ~150 doc FastAPI (~500k token) costa ~$0.01.
    Margine enorme anche per N ri-ingest.
  - **SDK Python maturo** (`openai>=1.50`): async client, batch
    nativo, retry built-in.
  - **Default dichiarato nella roadmap originale** del progetto.
- **Contro:**
  - **Dipendenza esterna**: serve API key, rete, OpenAI deve
    funzionare. Niente sviluppo offline.
  - **Privacy**: i chunk passano dai server OpenAI per essere
    embeddati. Per docs pubblici (il nostro caso: FastAPI docs) è
    irrilevante; per dati privati richiederebbe valutazione.
  - **Vendor lock-in lieve**: i vettori 1536-dim di OpenAI non sono
    direttamente compatibili con vettori di altri provider (cambio
    provider = ri-embeddare tutto).

### Option 2: OpenAI `text-embedding-3-large`

- **Descrizione:** versione "grande" di Option 1. 3072-dim.
- **Pro:**
  - Qualità marginalmente superiore (~5-10% MTEB).
  - Stesso SDK e operativa di 3-small.
- **Contro:**
  - **Costo 6.5x più alto** ($0.130 per 1M token).
  - **Vettori 2x più grandi** → Qdrant lavora più lento, occupa più
    storage. Per il nostro corpus piccolo è irrilevante, per corpus
    grandi conta.
  - **Ritorno marginale**: la differenza qualitativa è visibile
    sotto eval rigorosi, non in una demo. Spendere 6.5x per il 5%
    non si giustifica didatticamente.
  - Da rivalutare se i nostri eval (M2 task #12) mostrano
    `precision@5 < 0.7` con 3-small.

### Option 3: Voyage AI `voyage-3-lite` o `voyage-3`

- **Descrizione:** servizio embedding di Voyage AI, l'unico provider
  esplicitamente raccomandato da Anthropic nella loro documentazione
  ufficiale. `voyage-3-lite` 512-dim, `voyage-3` 1024-dim.
- **Pro:**
  - **Free tier generoso**: 200M token gratis (basterebbe per anni
    del nostro caso).
  - **Ottimizzato per RAG**: training data specificamente curato
    per la similarity sui pattern "query → passage".
  - **Qualità top**: spesso vincono benchmark RAG-specifici.
  - **Raccomandato da Anthropic** → coerenza concettuale col fatto
    che usiamo Claude per altre cose.
- **Contro:**
  - **Riconoscibilità inferiore**: in colloquio devi spiegare cos'è
    Voyage, mentre OpenAI embeddings non richiedono presentazione.
  - **Provider giovane** (fondato 2023, acquisito da MongoDB nel
    2024). Più rischio di pricing/feature changes vs OpenAI.
  - **Account separato** da gestire.

### Option 4: Cohere `embed-v3` (`embed-english-v3.0`)

- **Descrizione:** servizio embedding di Cohere. 1024-dim, ottimizzato
  per inglese (esiste anche `embed-multilingual-v3`).
- **Pro:**
  - **Free tier abbondante**.
  - **Coerenza con il reranker**: la roadmap (M2 task #8) prevede
    Cohere `rerank-3` per il reranking. Usare lo stesso provider
    semplifica auth/billing/SDK.
  - Qualità competitiva.
- **Contro:**
  - **Vendor lock-in più "stretto"** se uniamo embed+rerank entrambi
    su Cohere.
  - **Quote del free tier strette** sul throughput (rate limit più
    aggressivi vs OpenAI).

### Option 5: FastEmbed (BGE-small-en-v1.5) — locale

- **Descrizione:** libreria Python sviluppata dal team Qdrant. Gira
  modelli open source (BGE, MiniLM, ...) localmente via ONNX
  Runtime. Niente API, niente rete.
- **Pro:**
  - **Costo zero**, **nessuna API key**, **offline**, **privacy**
    totale.
  - **Coerenza con Qdrant**: stesso team, integrazione nativa.
  - **384-dim**: vettori 4x più piccoli → Qdrant più veloce, meno
    storage.
  - **Qualità decente**: BGE-small è top 10-15 del MTEB tra i
    modelli open.
- **Contro:**
  - **Download modello al primo run** (~100 MB).
  - **Velocità su CPU**: ~50 chunk/secondo (più lento di un'API
    managed che gira su GPU server-side).
  - **Riconoscibilità sul CV inferiore** rispetto a "ho usato
    OpenAI". Subjective ma rilevante per un portfolio.

## Decision

Abbiamo scelto **Option 1: OpenAI `text-embedding-3-small`**.

Driver in ordine di peso:

1. **Riconoscibilità sul portfolio.** Per un progetto il cui scopo
   esplicito è dimostrare competenze fullstack-AI in cerca di lavoro,
   "ho usato `text-embedding-3-small`" è la frase più leggibile
   possibile per un recruiter o un hiring manager senior. Il
   risparmio di Voyage/FastEmbed (~$0.01) è irrilevante; il segnale
   di pattern-match con la maggior parte dei sistemi RAG production
   è molto rilevante.

2. **Qualità più che sufficiente per i nostri eval.** Su docs
   tecnici come FastAPI, `text-embedding-3-small` mantiene un
   `precision@5` ben sopra la soglia 0.7 fissata in roadmap. Non
   abbiamo bisogno di alternative più potenti finora.

3. **Coerenza con la roadmap originale.** Il README di progetto e la
   ROADMAP M2 già citavano `text-embedding-3-small` come default.
   Confermare la scelta evita "riprogettazioni implicite" del
   progetto a metà strada.

4. **Costi trascurabili.** Stima M2 completa: pochi centesimi.
   Anche con 100 ri-ingest si resta sotto $5.

5. **SDK Python maturo e async-friendly** → integrazione naturale
   con FastAPI senza glue code.

**Trade-off accettati:**

- Dipendenza esterna (rete + API key). Mitigato dal fatto che
  l'embedder sarà dietro un'astrazione (`Embedder` protocol) che
  permette di swappare in FastEmbed locale come backup futuro
  senza riscrivere il resto.
- Vendor lock-in lieve. Se OpenAI alza i prezzi 10x domani, ri-embeddare
  con un'alternativa costa solo il tempo di una ri-ingest.

**Non scelte (per memoria storica):**

- `text-embedding-3-large` resta in considerazione come **upgrade
  path** se gli eval di M2 task #12 mostrano regressioni
  significative su `precision@5`.
- FastEmbed locale resta in considerazione come **fallback no-key**
  per CI/CD o per dev offline. Implementeremo l'astrazione
  `Embedder` in modo che entrambi possano essere selezionati via
  config — magari in M5.

## Consequences

### Positive

- Setup immediato: l'utente esegue `cp .env.example .env`, mette
  `OPENAI_API_KEY=sk-proj-...`, e tutto funziona.
- Modello stabile, ben supportato, prevedibile.
- I costi sono talmente bassi che possiamo permetterci ri-ingest
  liberamente in dev senza preoccuparci.

### Negative

- **Niente dev offline.** Se OpenAI è giù o sei senza rete, M2
  ingest e retrieval non funzionano. Mitigato dalla rarità di
  questi eventi e dalla disponibilità del fallback locale futuro.
- **Necessario un account OpenAI con metodo di pagamento attivo.**
  Non c'è più free tier per nuovi account dal 2024. Friction
  iniziale per chiunque cloni il repo. Mitigato dal cap: anche
  $5 caricati durano tutto il progetto.
- **Privacy:** ogni chunk indicizzato passa dai server OpenAI.
  Acceptable per il nostro corpus pubblico (docs FastAPI);
  bloccante in scenari enterprise privati. Da menzionare nel
  README come limitation.

### Neutral / Follow-ups

- Aggiungere `openai>=1.50` come runtime dep in
  `apps/api/pyproject.toml`.
- L'`Embedder` sarà un'astrazione con metodo
  `async def embed_batch(texts: list[str]) -> list[list[float]]`,
  così la sostituzione del provider richiede solo cambiare la
  classe concreta.
- Qdrant collection sarà creata con `vector_size=1536`,
  `distance=Cosine`.
- Se gli eval mostrano necessità di qualità superiore, un nuovo
  ADR-NNNN passerà a `text-embedding-3-large` o a Voyage.
- In M5 / produzione: valutare se mantenere OpenAI o passare a un
  modello locale per ridurre dipendenze esterne e costi a scala
  (anche se a quel punto i costi saranno comunque bassi).

## References

- [OpenAI Embeddings docs](https://platform.openai.com/docs/guides/embeddings)
- [Pricing OpenAI 2026](https://openai.com/api/pricing/) — sezione "Embeddings"
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — benchmark
- [Voyage AI](https://voyageai.com/) — opzione 3 considerata
- [FastEmbed (Qdrant)](https://qdrant.github.io/fastembed/) — opzione 5
- [ADR-0003](0003-vector-db-choice-qdrant.md) — la scelta vector DB compatibile
- [`docs/adr/README.md`](README.md) — regole sugli ADR
