"""Package `app.rag` — pipeline RAG (chunking, embedding, retrieval, rerank).

Re-export simbolico dei tipi/funzioni più usati così i moduli esterni
possono fare `from app.rag import Chunk, chunk_markdown` invece di
importare dal sotto-modulo specifico. Pattern equivalente al
`index.ts` di un package npm che ri-esporta i symbol pubblici.
"""

from app.rag.chunker import Chunk, ChunkerConfig, chunk_markdown

__all__ = ["Chunk", "ChunkerConfig", "chunk_markdown"]
