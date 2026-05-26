# `apps/api/tests/` — Test suite

Test **unitari** e di **integrazione** del backend. Eseguiti con `pytest`.

> **Stato attuale:** vuota. I primi test arriveranno in **M1** (test sul
> `/health` endpoint) e cresceranno con ogni milestone.

## Differenza con `app/evals/`

| Cartella    | Domanda a cui risponde                         |
| ----------- | ---------------------------------------------- |
| `tests/`    | "Il codice fa quello che dice di fare?"        |
| `app/evals/`| "L'output AI è abbastanza buono per gli utenti?" |

`tests/` è binario (passa/fallisce), deterministico, veloce.
`app/evals/` è probabilistico, lento, soglie ("almeno 85% di faithfulness").

## Struttura prevista

```
tests/
├── conftest.py            ← fixture pytest condivise (es. test client)
├── test_health.py         ← (M1) GET /health → 200 OK
├── test_rag/              ← (M2) test su chunker, embedder, retriever
├── test_agents/           ← (M4) test su nodi LangGraph
└── test_e2e/              ← test end-to-end (richiedono API key)
```

## Eseguire i test

Dalla root di `apps/api/`:

```bash
uv run pytest                 # tutti i test
uv run pytest tests/test_health.py  # solo un file
uv run pytest -k health       # solo test che matchano "health"
uv run pytest -v              # output verboso
```

## Concetti chiave (per chi viene da Vitest/Jest)

| pytest                    | Vitest / Jest                   |
| ------------------------- | ------------------------------- |
| `def test_foo():`         | `test('foo', () => {})`         |
| `assert x == 1`           | `expect(x).toBe(1)`             |
| `conftest.py` (fixtures)  | `setup`/`beforeEach` files      |
| `@pytest.fixture`         | Custom fixture / setup hooks    |
| `monkeypatch`             | `vi.mock` / `jest.mock`         |

A differenza di Jest, pytest non ha `describe`/`it` annidati: i test sono
funzioni top-level, raggruppate in file. Più semplice, meno boilerplate.
