# `app/tools/` — Tool definitions per gli agenti

Un **tool** è una funzione che l'agente può decidere di chiamare durante
il suo ragionamento. Tipici esempi:

- `search_documents(query: str)` → cerca nella knowledge base RAG.
- `run_sql(query: str)` → esegue SQL su un DB applicativo.
- `web_search(query: str)` → cerca su Internet.
- `send_email(to: str, body: str)` → effetto collaterale, va sempre con HITL.

> **Stato attuale:** vuota. Sarà popolata in **M4** (Agentic workflows).

## Anatomia di un tool

Un tool ben fatto ha tre parti:

```python
# (1) signature tipizzata — l'LLM legge le annotazioni e la docstring
def search_documents(query: str, top_k: int = 5) -> list[Citation]:
    """Cerca i top_k chunk più rilevanti per `query` nella knowledge base."""
    # (2) implementazione — può chiamare il modulo `app/rag/`
    ...

# (3) registrazione presso il framework di tool calling
# (di solito un decorator @tool o l'aggiunta a una lista)
```

L'LLM riceve **solo (1)** — la firma e la docstring. Da quelle deve capire
se e quando usare il tool. Quindi: docstring chiare > codice "intelligente".

## Cosa va qui (e cosa NO)

✅ **Va qui:**
- Definizioni di tool (la funzione + i suoi tipi Pydantic per input/output).
- Wrapper sottili attorno a sistemi esterni (API, DB, RAG).

❌ **NON va qui:**
- Logica di *quando* chiamare un tool → quello sta nell'agente
  (`app/agents/`).
- Implementazione della retrieval o dell'embedding → `app/rag/`. Il tool
  `search_documents` è un *consumatore* di `app/rag/`, non lo duplica.

## Tool e Human-in-the-Loop

Tool con **side effect irreversibili** (email, mutazioni DB, pagamenti)
DEVONO passare per un nodo di approvazione nell'agente. La regola
mnemonica:

> *"Se sbaglio, posso annullarlo con Ctrl+Z?"* — no → richiede HITL.
