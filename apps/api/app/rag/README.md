# `app/rag/` — Retrieval-Augmented Generation pipeline

Tutto ciò che riguarda **trasformare documenti in conoscenza
recuperabile** e poi **recuperarla in modo affidabile** al momento della
query.

> **Stato attuale (inizio M2):** cartella vuota a livello di codice, ma
> l'infrastruttura sottostante è in piedi.
>
> - **Vector DB**: **Qdrant** self-hosted in docker
>   (vedi [ADR-0003](../../../../docs/adr/0003-vector-db-choice-qdrant.md)
>   per il perché, e `docker-compose.yml` alla root per il setup).
>   Lanciare con `docker compose up -d qdrant`, dashboard su
>   `http://localhost:6333/dashboard`.
> - **Embeddings**: `text-embedding-3-small` di OpenAI (1536-dim).
>   Vedi `.env.example` per le variabili.
> - **Reranker**: Cohere `rerank-3` (con fallback BGE self-hosted senza chiave).
>
> I file Python di questa cartella saranno creati nei prossimi task M2.

## Le 4 fasi di una pipeline RAG

```
1. INGESTION                  2. EMBEDDING
   load → chunk → clean          chunk → vector

3. RETRIEVAL                  4. RERANKING (opzionale ma raccomandato)
   query → top-k chunks          top-k → top-n (più rilevanti)
```

Ogni fase avrà il proprio file Python in questa cartella (`chunker.py`,
`embedder.py`, `retriever.py`, `reranker.py`), con la logica isolata e
testabile.

## Cosa va qui (e cosa NO)

✅ **Va qui:**
- Strategie di **chunking** (size-based, semantic, structure-aware).
- Wrapper attorno ai client di **embedding** (OpenAI, Cohere, ecc).
- Logica di **hybrid search** (vector + BM25 + metadata filters).
- **Reranker** (Cohere rerank-3 o BGE).
- **Citation builder** — costruisce gli oggetti `{chunk_id, source, score}`
  che il frontend renderizzerà come citazioni inline.

❌ **NON va qui:**
- Endpoint HTTP che espongono retrieval → quelli vanno in `app/main.py`
  o in un futuro `app/routers/`.
- State machine che *decidono quando* fare retrieval → quelle stanno in
  `app/agents/`.
- Schema Pydantic dell'output API → tipicamente in `app/schemas.py` o
  vicino all'endpoint.

## Concetti chiave da padroneggiare in M2

- **Embedding**: una funzione che mappa testo → vettore di float ad alta
  dimensione (es. 1536 per `text-embedding-3-small`). Testi simili
  hanno vettori "vicini" (cosine similarity).
- **Vector DB**: database ottimizzato per cercare i k vettori più vicini
  a un vettore query, in millisecondi anche su milioni di documenti.
- **Hybrid search**: combina vector search (cattura similarità semantica)
  + keyword search BM25 (cattura match letterali, es. nomi propri).
- **Reranking**: re-ordina i risultati top-k usando un modello più costoso
  ma più preciso. Spesso ottimo trade-off (recuperi 50, rerank a 5).
