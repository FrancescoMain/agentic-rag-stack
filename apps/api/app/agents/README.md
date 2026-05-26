# `app/agents/` — Orchestratori LangGraph

Qui vivono gli **agenti**: i grafi di stato che decidono *cosa fare,
quando, e con quali tool* per rispondere a una richiesta utente.

> **Stato attuale:** vuota. Sarà popolata in **M4** (Agentic workflows
> & Human-in-the-Loop).

## Cosa va qui (e cosa NO)

✅ **Va qui:**
- Definizioni di `StateGraph` LangGraph (nodi, archi, condizioni).
- Tipi `TypedDict` / `Pydantic` che descrivono lo stato dell'agente.
- Logica di branching: "se la query è ambigua → chiedi chiarimento, altrimenti → retrieve".
- Checkpoint per Human-in-the-Loop (es. far approvare un'azione prima di eseguirla).

❌ **NON va qui:**
- Implementazione concreta dei tool → `app/tools/`.
- Codice di retrieval/embedding → `app/rag/`.
- Definizioni di endpoint HTTP → `app/main.py` (o `app/routers/` se cresceremo).

## Perché LangGraph e non un agente "fatto a mano"?

(Trade-off discusso in `docs/adr/` quando arriveremo a M4.)

In breve: un agente è una macchina a stati con loop, branching e
side effect (chiamate a tool). Scriversi una state machine da zero è
fattibile ma noioso; LangGraph dà *gratis*:

- **Checkpointing** (salva e riprendi lo stato → fondamentale per HITL).
- **Streaming** dei token e degli stati intermedi.
- **Trace strutturata** integrata con Langfuse.
- **Tool calling** uniforme su provider diversi (Anthropic / OpenAI).

Il costo è imparare una nuova astrazione, ma è una di quelle astrazioni
che paga rapidamente.
