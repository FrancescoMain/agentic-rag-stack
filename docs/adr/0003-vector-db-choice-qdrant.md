# ADR-0003: Qdrant self-hosted come vector database (supersedes ADR-0002)

- **Status:** Accepted
- **Date:** 2026-05-26
- **Deciders:** Francesco
- **Supersedes:** [ADR-0002](0002-vector-db-choice.md)
- **Tags:** `rag`, `infra`, `m2`

---

## Context

Poche ore dopo aver chiuso [ADR-0002](0002-vector-db-choice.md) (che
sceglieva pgvector come vector database del progetto) è emerso un
fattore che lì non era stato pesato: il **decisore ha già usato Qdrant
in un progetto precedente**. Questo trasforma l'analisi costi/benefici
in modo non banale e merita un nuovo ADR — non una modifica del
precedente, in linea con la regola di immutabilità descritta in
[`docs/adr/README.md`](README.md).

Il contesto strutturale del progetto resta quello di ADR-0002:
- Progetto didattico + portfolio, sviluppatore singolo, budget zero.
- Corpus demo dell'ordine di 10k–100k chunk a 1536-dim.
- M2 richiede *hybrid search* (vettoriale + lessicale) obbligatoria.
- M5 prevede `docker-compose` con Postgres per Langfuse e cost tracking.

Cosa è cambiato rispetto a ADR-0002:
1. **Nuovo driver dominante: familiarità.** Il decisore ha già usato
   Qdrant. La curva di apprendimento per M2 si riduce significativamente.
2. **Rivalutazione del driver "hybrid in una query"** di ADR-0002:
   Qdrant dalla v1.10 supporta *sparse vectors* nativi (BM25 / SPLADE)
   con fusion built-in via *Reciprocal Rank Fusion*. Il vantaggio
   "tutto in una query SQL" di pgvector si riduce a una preferenza
   stilistica fra "SQL con CTE" e "Qdrant API con `prefetch` +
   `query` su due named vectors". Compattezza equivalente.
3. **Nessun fattore tecnico nuovo a sfavore di Qdrant** è emerso —
   la rivalutazione è interna ai driver già noti.

## Considered Options

> Stesse opzioni di ADR-0002, qui rivalutate con il nuovo peso dei driver.
> Per le descrizioni estese vedi l'ADR superato.

### Option 1: Pinecone (managed SaaS)

Esclusa per gli stessi motivi di ADR-0002: vendor lock-in, free tier
limitante, costo a scala, niente hybrid nativa pari a quella di
Qdrant/pgvector.

### Option 2: pgvector self-hosted

- **Pro che restano validi:**
  - Costo zero, coerenza con il Postgres che useremo comunque a M5
    per Langfuse, vendor lock-in nullo.
  - Filtering ricco via SQL, transazionalità nell'ingestion.
- **Pro che si annullano in luce dei nuovi pesi:**
  - "Hybrid in una query" → Qdrant offre lo stesso risultato con
    una sola chiamata API (sparse + dense + fusion).
- **Contro che pesano di più ora:**
  - Curva di apprendimento per chi non ha mai usato pgvector
    (operatori `<=>` `<->` `<#>`, trade-off HNSW vs IVFFlat,
    `tsvector` + `to_tsquery` per il lessicale).
  - Tempo che il decisore investirebbe a imparare la sintassi è
    tempo che NON investe sui pattern RAG (chunking strategies,
    eval, reranking).

### Option 3: Qdrant self-hosted *(scelta)*

- **Descrizione:** vector DB dedicato in Rust, accessibile via gRPC
  o REST, con SDK Python ufficiale. Gira in docker come single-node.
  Supporta hybrid search nativa (sparse + dense vectors con fusion
  RRF), filtering rich con DSL JSON, payload arbitrari (JSON), e
  snapshot per backup.
- **Pro:**
  - **Familiarità del decisore**: già usato in un progetto precedente.
    Riduce rischio di blocchi durante M2 e accelera l'esecuzione.
  - **Skill display reale (non aspirazionale)**: in colloquio si può
    parlare di Qdrant con esempi concreti da più di un progetto.
  - **Hybrid search nativa di prima classe**: sparse + dense
    vectors named, query in un'unica chiamata con fusion automatica.
  - **Specializzato**: filtering, payload, snapshot, replicazione
    pensati per il caso d'uso vettoriale. Niente "adatta-Postgres-
    a-vettore" da gestire.
  - **Self-hostable in docker** con `qdrant/qdrant:v1.x`: setup
    immediato, niente vendor lock-in (è OSS Apache 2.0).
  - **Ottimo dashboard locale** (built-in su porta 6333): utile per
    debug visivo dell'indice durante lo sviluppo dei task M2.
  - **SDK Python maturo** (`qdrant-client`): tipato, async, ben
    documentato. Il path "scrivere il primo `upsert` + `search`"
    è di una decina di righe.
- **Contro (trade-off accettati):**
  - **Un servizio in più** nel `docker-compose.yml`: oltre a Postgres
    (necessario per Langfuse a M5) avremo anche Qdrant. Costo
    cognitivo e operativo lieve.
  - **Niente transazioni ACID cross-collezione**: l'ingestion non è
    transazionale come potrebbe essere in Postgres. Mitigato con
    batch upsert idempotente + idempotency key sul chunk ID.
  - **Metadata in payload JSON, non colonne**: filtri ricchi ci sono
    ma sono espressi in DSL JSON, non SQL. Diverso ma non peggio.
  - **No SQL diretto** per query ad-hoc esplorative: per "vediamo
    cosa c'è nel DB" usiamo il dashboard Qdrant o l'SDK, non `psql`.

## Decision

Abbiamo scelto **Option 3: Qdrant self-hosted**.

Il driver decisivo è la **familiarità pregressa del decisore** con
Qdrant. Per un progetto didattico + portfolio con scadenza implicita
(16 mesi di studio, di cui M2 è una frazione), il valore di "usare
uno strumento che già conosci per esplorare *altri* pattern" è
superiore al valore di "usare uno strumento nuovo che ti costringe
a imparare prima il tool e poi i pattern".

Il driver tecnico "hybrid search nativa di prima classe" è
diventato pari fra pgvector e Qdrant dopo che pgvector ha perso il
vantaggio "tutto in SQL" (Qdrant 1.10+ ha sparse+dense fusion in una
chiamata). I restanti driver tecnici (filtering, dashboard,
specializzazione) leggermente favorivano Qdrant — non al punto da
giustificare un cambio da soli, ma sufficienti a confermare la
decisione una volta che la familiarità l'aveva spostata.

Il **trade-off accettato** è avere un servizio in più nel
`docker-compose.yml` (Qdrant accanto a Postgres). È un costo
operativo lieve, compensato dal beneficio architetturale di una
*separation of concerns* pulita: Postgres per OLTP+tracing, Qdrant
per il retrieval. Pattern comune in produzione.

## Consequences

### Positive

- Velocità di esecuzione di M2 più alta: niente "scuola pgvector"
  prima di iniziare il lavoro vero (chunker, retriever, reranker).
- Skill display sul CV/portfolio basato su esperienza reale, non
  su un tutorial.
- Hybrid search nativa con fusion built-in, niente glue code di
  Reciprocal Rank Fusion da scrivere.
- Dashboard Qdrant locale per ispezione visiva durante lo sviluppo.
- Architettura "Postgres per dati transazionali / Qdrant per
  retrieval" — pattern riconoscibile in produzione.

### Negative

- Un servizio in più in `docker-compose.yml` (lieve overhead
  operativo e cognitivo).
- Ingestion non transazionale ACID cross-store. Mitigato con
  upsert idempotente.
- Filtering espresso in JSON DSL invece di SQL. Più verboso per
  query molto strutturate (ma raramente serviranno qui).
- L'ADR-0002 resta nella storia del repo come decisione superata.
  Visibile in colloquio, da raccontare con onestà: "ho cambiato
  idea quando ho considerato un fattore che inizialmente non avevo
  pesato". Non è un negativo *tecnico*, ma richiede di saper
  raccontare bene la storia.

### Neutral / Follow-ups

- L'ingestion di M2 (task #6) creerà collezione Qdrant con due
  named vectors: `dense` (1536-dim, cosine) e `sparse` (per BM25
  via FastEmbed o SPLADE). Schema documentato nel task #5.
- `docker-compose.yml` di M5 includerà il servizio `qdrant` con
  volume persistente per `/qdrant/storage`.
- Lo strato di astrazione `VectorStore` (task #5) resta sensato:
  permette di sostituire Qdrant con un altro provider in futuro.
- A M5 / produzione: Qdrant offre **Qdrant Cloud** (free tier
  generoso) come alternativa managed. Probabile ADR-NNNN quando ci
  arriviamo, valutando self-host vs managed in base a operatività
  realistica.

## References

- [Qdrant docs](https://qdrant.tech/documentation/)
- [Qdrant hybrid search (sparse + dense)](https://qdrant.tech/documentation/concepts/hybrid-queries/)
- [Qdrant Python client](https://github.com/qdrant/qdrant-client)
- [Reciprocal Rank Fusion (RRF) explained](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)
- [ADR-0002](0002-vector-db-choice.md) — la decisione superata da questo.
- [`docs/adr/README.md`](README.md) — regole di immutabilità e superseding.
