"""
tests/test_rag/conftest.py
==========================
Le fixture comuni vivono nel conftest top-level `tests/conftest.py`:
- `fake_openai` (FakeOpenAIClient)
- `qdrant_store` (QdrantVectorStore session-scoped)
- `unique_collection` (nome collection unico per test, con cleanup)

pytest le rende automaticamente visibili anche in questa sotto-cartella;
non serve nulla qui. Questo file resta come placeholder per fixture
RAG-specifiche future (es. fake reranker se serve).
"""
