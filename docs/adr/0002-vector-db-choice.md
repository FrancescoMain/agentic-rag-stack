# ADR-0002: pgvector self-hosted come vector database

- **Status:** Accepted
- **Date:** 2026-05-26
- **Deciders:** Francesco
- **Tags:** `rag`, `infra`, `m2`

---

## Context

M2 introduce la pipeline RAG: documenti → chunk → embeddings (vettori
densi ad alta dimensione) → archiviazione → retrieval con
*hybrid search* (vettoriale + lessicale) e *citazioni* sul chunk
originale. Per fare tutto questo serve un **vector database**: un
sistema che indicizzi vettori (qui 1536-dim, generati da
`text-embedding-3-small` di OpenAI, vedi [ADR-0001](0001-monorepo-structure.md)
e ROADMAP M2) e risponda a query del tipo "trovami i top-K vettori più
vicini a Q per cosine similarity, opzionalmente filtrando per
metadata".

Vincoli del progetto rilevanti per questa scelta:

- **Didattico + portfolio**: il sistema deve essere leggibile, dimostrabile
  in colloquio, e ogni componente deve avere un valore narrativo.
  "Black box managed" pesa meno di "ho capito gli indici HNSW".
- **Budget zero**: nessuna spesa ricorrente accettabile per la demo.
- **Sviluppatore singolo**: niente team operations dedicato.
- **M5 prevede `docker-compose` con Postgres** (per Langfuse self-hosted
  e per cost tracking). Quindi una dipendenza Postgres in locale c'è
  già a prescindere da questa decisione.
- **Hybrid search obbligatoria**: il retriever di M2 (task #7) deve
  combinare similarità vettoriale e ricerca lessicale tipo BM25 — è
  ciò che fa la differenza fra un RAG giocattolo e uno usabile su
  nomi propri, identificatori, codice.

## Considered Options

### Option 1: Pinecone (managed SaaS)

- **Descrizione:** vector DB cloud, accessibile via API HTTP. Il
  provider gestisce indici (HNSW), replication, scaling, backup.
- **Pro:**
  - Setup molto rapido: API key + creazione index via dashboard = ~5 min.
  - Free tier disponibile (un *starter index* con limiti su numero
    di vettori e namespace).
  - Production-grade di default: SLA, monitoring incluso.
  - Niente Postgres da gestire (se non ce ne fosse uno già nel compose).
- **Contro:**
  - **Vendor lock-in alto**: l'API non è uno standard, migrare via
    significa riscrivere il client e re-indicizzare tutto.
  - **Free tier limitante** per un corpus serio: ~100k vettori, 1 index,
    nessun controllo fine su parametri HNSW.
  - **Hybrid search non nativa**: Pinecone ha aggiunto un "sparse vector"
    pattern (BM25-like), ma è meno integrato di una FTS Postgres e
    richiede un secondo passaggio applicativo.
  - **Costo a scala**: a partire da ~$70/mo per il primo tier serio.
  - **Skill display sul CV**: "ho chiamato un'API gestita" è una storia
    più povera per un colloquio AI-engineering che "ho fatto hybrid
    search in una sola query SQL".
  - **Aggiunge un servizio esterno** alle dipendenze del progetto:
    senza chiave Pinecone, l'ambiente di sviluppo locale non funziona.

### Option 2: pgvector self-hosted *(scelta)*

- **Descrizione:** estensione Postgres che aggiunge il tipo `vector` e
  operatori per cosine similarity / L2 / inner product. Dalla v0.7
  supporta indici **HNSW** (default raccomandato) oltre a IVFFlat.
  Postgres gestisce il resto (transazioni, full-text search, joins).
- **Pro:**
  - **Costo zero** in dev (container Docker `pgvector/pgvector:pg16`)
    e bassissimo in prod (qualunque managed Postgres con extension
    abilitata, es. Supabase, Neon, RDS).
  - **Hybrid search nativa in una query**: combinare cosine similarity
    (`<=>` di pgvector) con full-text Postgres (`tsvector` + `ts_rank`)
    via CTE o Reciprocal Rank Fusion → niente codice applicativo che
    coordina due sistemi.
  - **Coerenza architetturale**: useremo Postgres comunque a M5 per
    Langfuse e cost tracking. Aggiungere `CREATE EXTENSION vector;` è
    gratis in termini di dipendenze nuove.
  - **Vendor lock-in nullo**: è Postgres standard. Migrazione futura
    a Qdrant/Weaviate/Pinecone resta possibile senza riscrivere il resto.
  - **Skill display alto**: spiegare HNSW, distance ops, ANN-vs-exact,
    quando IVFFlat è meglio di HNSW, è esattamente il tipo di
    conversazione che si fa in colloquio AI-engineering.
  - **Filtering ricco**: i metadata sono colonne Postgres → SQL completo
    per filtri (`WHERE source = 'docs.pyhton.org' AND created_at > ...`)
    senza la fatica delle "metadata filter expression" proprietarie.
  - **Transazionalità**: ingestion atomica (chunk + embedding insertati
    in transazione). Pinecone non offre questa garanzia.
- **Contro:**
  - **Setup iniziale più lungo** (~30 min): docker-compose con Postgres,
    `init.sql` con `CREATE EXTENSION`, schema della tabella documenti +
    indice HNSW. Mitigato: lo scriviamo una volta sola e diventa
    template per ogni prossimo progetto.
  - **Ops responsibilities**: vacuum, backup, monitoring sono nostri.
    Mitigato: in dev non importano; in prod si usa un managed Postgres
    (Supabase/Neon) che gestisce tutto questo.
  - **Performance a scala**: per >10M vettori pgvector è meno performante
    di Pinecone/Qdrant specializzati. Mitigato: irrilevante per un
    progetto demo con corpus dell'ordine di 10k-100k chunk.
  - **HNSW build time** alto su grandi dataset (è un trade-off di HNSW
    in generale). Mitigato: si può ingestare prima senza indice e
    crearlo dopo.

### Option 3: Qdrant self-hosted *(menzionata, non scelta)*

- **Descrizione:** vector DB dedicato in Rust, con hybrid search nativa
  e API moderne. Gira in docker.
- **Pro:**
  - Specializzato (potenzialmente più performante di pgvector su carichi
    pesanti).
  - Hybrid search nativa.
  - API più "moderna" di SQL per use-case vettoriali puri.
- **Contro:**
  - **Aggiunge un servizio**: avremmo Postgres (per Langfuse/altro) *e*
    Qdrant. Due sistemi da capire, due backup da fare, due fonti di
    verità per metadata.
  - **Lock-in mid**: meno di Pinecone (è open source), più di pgvector
    (API non-standard).
  - **Beneficio marginale** per il nostro carico: i punti dove Qdrant
    batte pgvector (>1M vettori, query >100/s) non sono nello scope.

Non considerate seriamente (per memoria storica):

- **Weaviate**: simile a Qdrant come trade-off, scelto Qdrant come
  rappresentante della categoria "vector DB dedicato self-hosted".
- **Chroma**: ottimo per prototipi locali, non production-grade.
- **Elasticsearch / OpenSearch con kNN plugin**: troppo pesante da
  gestire per un singolo dev.
- **Faiss + storage custom**: too low-level, riscriviamo cose che
  pgvector ha già.

## Decision

Abbiamo scelto **Option 2: pgvector self-hosted**.

I driver decisivi, in ordine:

1. **Hybrid search in una query SQL**. Il valore differenziale di un
   RAG serio è il retrieval ibrido. Farlo in Postgres significa una
   `WITH vector_results AS (...) SELECT ... JOIN text_results ...`
   che chiunque sappia SQL può leggere. In Pinecone/Qdrant lo stesso
   risultato richiede 2 round-trip + codice applicativo di rank fusion.

2. **Coerenza con M5**. Postgres entra comunque nel
   `docker-compose.yml` di produzione (per Langfuse e/o cost tracking).
   Riusarlo per i vettori non aggiunge nulla; cambiare vector DB
   aggiungerebbe un servizio in più.

3. **Valore didattico/portfolio**. "Spiego come ho implementato
   l'hybrid retrieval con pgvector + tsvector" è una storia che si
   può raccontare 30 secondi in colloquio dimostrando vere
   competenze. "Ho chiamato Pinecone" no.

4. **Costo zero**. Sia in dev che in prod (su un Postgres managed con
   pgvector abilitato — es. Supabase free tier copre l'ordine di
   grandezza di un portfolio).

Il **trade-off accettato**: setup iniziale di ~30 min vs i ~5 min di
Pinecone. È un costo *one-shot* contro un beneficio strutturale
sull'intero ciclo di vita del progetto.

## Consequences

### Positive

- Un solo servizio dati nel docker-compose (Postgres 16 + extension
  pgvector). Onboarding semplice: `docker compose up postgres`.
- Hybrid search senza glue code applicativo.
- Filtering arbitrario via SQL (no DSL proprietaria).
- Migrabile: il codice del retriever parlerà a un'astrazione
  `VectorStore` (vedi M2 task #5); l'implementazione concreta è
  sostituibile.
- Transazionalità nella ingestion (insert atomico chunk + embedding).

### Negative

- Costo cognitivo di prima volta: chi non ha mai usato pgvector deve
  imparare `vector(1536)`, operatori `<=>` (cosine) `<->` (L2)
  `<#>` (negative inner product), trade-off HNSW vs IVFFlat.
  Mitigato: scriveremo un README didattico in `app/rag/` che spiega
  questi punti come si fa con tutto il resto del progetto.
- In dev locale serve Docker (per il container Postgres). Non era un
  prerequisito a M1, lo diventa a M2.
- La build dell'indice HNSW su `>1M` vettori può richiedere minuti.
  Non un problema per la demo, da tenere a mente se in futuro
  qualcuno volesse scalare.

### Neutral / Follow-ups

- L'ingestion CLI di M2 (task #6) farà `CREATE EXTENSION IF NOT EXISTS
  vector;` come prima cosa, in modo idempotente.
- Lo schema iniziale (tabella `chunks` con colonne `id`, `content`,
  `embedding vector(1536)`, `source`, `metadata jsonb`, `tsv tsvector`)
  sarà documentato come migrazione SQL nel task #5 (`vector_store.py`)
  o in `app/rag/migrations/`.
- Se in futuro vorremo testare reali differenze di latency con un
  vector DB dedicato, l'astrazione `VectorStore` permetterà di
  affiancare un'implementazione Qdrant per benchmark — nuovo ADR
  necessario per cambiare default.
- A M5, quando portiamo Postgres "in prod", andrà valutato un
  managed con `pgvector` abilitato: Supabase (free tier abbondante),
  Neon, Crunchy Bridge, RDS PG 16+. Probabilmente nuovo ADR.

## References

- [pgvector — repo ufficiale](https://github.com/pgvector/pgvector)
- [pgvector docker image](https://hub.docker.com/r/pgvector/pgvector)
- [HNSW vs IVFFlat in pgvector (docs)](https://github.com/pgvector/pgvector#indexing)
- [Hybrid search with pgvector + tsvector (Supabase tutorial)](https://supabase.com/docs/guides/ai/hybrid-search)
- [Reciprocal Rank Fusion (RRF) explained](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)
- [Pinecone pricing 2026](https://www.pinecone.io/pricing/) (per contesto del trade-off)
