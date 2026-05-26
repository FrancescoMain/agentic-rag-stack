<div align="center">

# agentic-rag-stack

**Production-ready RAG with LangGraph agents, streaming UI, and Langfuse tracing.**
A reference implementation of the Full-Stack AI Engineer pattern.

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-FF6B35.svg)](https://langchain-ai.github.io/langgraph/)
[![Langfuse](https://img.shields.io/badge/Langfuse-tracing-8B7355.svg)](https://langfuse.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Architecture](#architecture) · [Quick Start](#quick-start) · [Roadmap](#roadmap) · [Decisions](#architectural-decisions) · [About](#about)

</div>

---

## Why this exists

Most public RAG examples stop at the toy stage: load a PDF, embed it, return a string from an LLM. They skip everything that actually matters in production — citations, agent workflows, human approval gates, cost tracking, evaluation suites, tracing.

`agentic-rag-stack` is a reference implementation that takes RAG from notebook to production. It demonstrates the **Full-Stack AI Engineer** pattern: one codebase that owns the entire flow from embedding pipeline to React component, without the team handoffs that usually break AI products.

It is built to be **read, forked, and learned from**. Every architectural decision has a documented trade-off. Every milestone has a clear Definition of Done.

## What's inside

- **Production RAG pipeline** — chunking, embeddings, hybrid search with metadata filters, reranking, citations
- **Agentic workflows** — LangGraph state machines with tool calling, evaluator loops, and Human-in-the-Loop checkpoints
- **Streaming AI-first UI** — Next.js + Vercel AI SDK with Generative UI patterns (citations expand inline, structured outputs render as components)
- **End-to-end observability** — Langfuse tracing on every LLM call, tool call, and retrieval; cost tracking per feature
- **Evaluation suites** — golden datasets, regression evals on every PR, faithfulness and precision metrics
- **MLOps foundations** — Docker, CI/CD with eval gates, manual approval before production, runbook included

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Next.js Frontend                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │ useChat SDK │  │ AI Elements  │  │ Citation Viewer  │    │
│  └─────────────┘  └──────────────┘  └──────────────────┘    │
└────────────────────────────┬────────────────────────────────┘
                             │ SSE / Streaming
┌────────────────────────────▼────────────────────────────────┐
│                    FastAPI Gateway (Python)                 │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │  Auth & Rate     │  │      LangGraph Orchestrator      │ │
│  │   Limiting       │  │  (State Machine + Tool Calling)  │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
└──┬───────────────────────────┬───────────────────────────┬──┘
   │                           │                           │
┌──▼──────────┐         ┌──────▼──────┐            ┌───────▼────────┐
│  RAG Layer  │         │  LLM APIs   │            │   Tools / APIs │
│  Pinecone / │         │  Anthropic  │            │  SQL, Web,     │
│  pgvector + │         │  OpenAI     │            │  Internal APIs │
│  Reranker   │         │             │            │                │
└─────────────┘         └─────────────┘            └────────┬───────┘
                                                           │
                                ┌──────────────────────────┘
                                │
                       ┌────────▼─────────┐
                       │     Langfuse     │
                       │ Tracing + Evals  │
                       └──────────────────┘
```

## Tech stack

| Layer            | Choice                                                              |
| ---------------- | ------------------------------------------------------------------- |
| Frontend         | Next.js 14+ (App Router), Vercel AI SDK, Tailwind, Shadcn UI        |
| Backend          | Python 3.12+, FastAPI, Pydantic v2, asyncio                         |
| Orchestration    | LangGraph, Tool Calling API                                         |
| Vector DB        | Pinecone (managed) or Supabase pgvector (self-hosted)               |
| Embeddings       | OpenAI `text-embedding-3-small` + Cohere/BGE reranker               |
| LLM              | Claude (Anthropic) for reasoning · GPT-4o for cost-sensitive tasks  |
| Observability    | Langfuse (tracing + evals), structured JSON logging                 |
| Deploy           | Docker, GitHub Actions, multi-stage builds                          |

See [docs/adr/](docs/adr/) for the rationale behind each choice.

## Quick start

> Prerequisites: Python 3.12+, Node.js 20+, an Anthropic API key.
> *(Docker, OpenAI/Cohere keys, ingestion pipeline arrive in later milestones — see [Status](#roadmap).)*

### What works today (`v0.1.0`, end of M1)

```bash
# Clone and set up
git clone https://github.com/<your-username>/agentic-rag-stack.git
cd agentic-rag-stack
cp .env.example .env  # fill in ANTHROPIC_API_KEY at minimum

# Backend
cd apps/api
uv sync
uv run uvicorn app.main:app --reload   # → http://localhost:8000

# Frontend (separate terminal)
cd apps/web
pnpm install
pnpm dev                               # → http://localhost:3000
```

Visit `http://localhost:3000` for the health pinger and `/classify` playground.
Visit `http://localhost:8000/docs` for the auto-generated OpenAPI reference.

### What's coming (later milestones)

```bash
# M2 — ingest your first documents into the vector store
uv run python -m app.ingest --source ./sample_docs --collection demo

# M5 — full stack via docker-compose (api + web + langfuse + postgres)
docker compose up
```

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the per-milestone breakdown.

## Roadmap

The project is built in five milestones. Each milestone is shippable on its own.

| Milestone | Focus                                       | Status         |
| --------- | ------------------------------------------- | -------------- |
| **M1**    | Backend foundations & Invisible AI          | ✅ Done        |
| **M2**    | Knowledge base & RAG pipeline               | ⚪ Planned     |
| **M3**    | Streaming AI-first frontend                 | ⚪ Planned     |
| **M4**    | Agentic workflows & Human-in-the-Loop       | ⚪ Planned     |
| **M5**    | Production, MLOps & tracing                 | ⚪ Planned     |

Full breakdown with task-level detail is in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Architectural decisions

Every non-obvious choice is documented as an ADR (Architecture Decision Record) under [`docs/adr/`](docs/adr/). Examples:

- **Why Python on the backend** — the AI ecosystem is Python-first; fighting it slows you down
- **Why `text-embedding-3-small` by default** — 80% of the quality at 1/3 the cost; upgrade only when evals show it matters
- **Why LangGraph over a custom state machine** — checkpointing and HITL primitives come for free
- **Why Langfuse over generic APM** — agent trace visualization that actually shows the agent's reasoning, not just HTTP spans

If you disagree with a decision, open an issue. The point of ADRs is to make trade-offs debatable.

## Project structure

```
agentic-rag-stack/
├── apps/
│   ├── api/              # FastAPI backend, LangGraph agents, RAG pipeline
│   │   ├── app/
│   │   │   ├── agents/   # LangGraph state machines
│   │   │   ├── rag/      # Chunking, embedding, retrieval, reranking
│   │   │   ├── tools/    # Tool definitions for agents
│   │   │   └── evals/    # Eval suites and golden datasets
│   │   └── tests/
│   └── web/              # Next.js frontend, AI Elements, streaming chat
│       ├── app/
│       ├── components/
│       └── lib/
├── docs/
│   ├── ROADMAP.md        # Detailed milestone & task breakdown
│   ├── adr/              # Architecture Decision Records
│   └── runbook.md        # Operational runbook (M5)
└── docker-compose.yml
```

## Contributing

This is primarily a learning and portfolio project, but PRs are welcome — especially for:

- Additional tool implementations (the agent always needs more tools)
- Eval datasets for new domains
- Documentation improvements and translations
- Bug fixes

Before opening a PR, please run the eval suite locally and make sure the regression tests pass.

## About

Built by [Francesco](https://www.linkedin.com/in/<your-handle>/), a senior frontend engineer transitioning into Full-Stack AI Engineering. This project is the practical companion to a 16-month self-study plan covering LLM orchestration, RAG systems, and AI system design.

If you're on the same journey — or if you're hiring for AI engineering roles in Europe — feel free to reach out.

## License

MIT — see [LICENSE](LICENSE) for details.
