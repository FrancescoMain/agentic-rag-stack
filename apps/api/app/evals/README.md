# `app/evals/` — Valutazione e golden datasets

In un sistema AI **non basta che i test unitari passino**: bisogna
misurare quanto bene il sistema risponde su esempi realistici. Questa
cartella ospita le suite di valutazione.

> **Stato attuale:** vuota. I primi eval arriveranno in **M2** (precisione
> di retrieval) e cresceranno fino a M5 (eval gate in CI/CD).

## Anatomia di un eval

```
golden_datasets/
└── faq_v1.jsonl         ← coppie (input, output_atteso) curate a mano

evaluators/
├── faithfulness.py      ← la risposta è supportata dai chunk recuperati?
├── precision_at_k.py    ← i top-k chunk contengono la risposta corretta?
└── citation_accuracy.py ← le citazioni puntano davvero alle fonti giuste?

runners/
└── run_regression.py    ← esegue tutti gli eval su un golden dataset
```

## Cosa va qui (e cosa NO)

✅ **Va qui:**
- Golden datasets (JSONL preferito: facile da appendere, leggibile a riga).
- Evaluator (funzioni `evaluate(actual, expected) -> Score`).
- Runner / orchestratore degli eval.

❌ **NON va qui:**
- Test unitari classici (assert su valori fissi) → `tests/`.
- Logica applicativa → `app/agents/`, `app/rag/`, ecc.

## Tre tipi di eval che useremo

1. **Heuristic / rule-based** — controlli deterministici (es. "la risposta
   contiene un'URL?", "ha citato almeno una fonte?"). Veloci, gratis.
2. **LLM-as-judge** — un LLM più potente valuta l'output. Più flessibile
   ma costa soldi e rumorosità.
3. **Human eval** — un umano dà un punteggio. Lento ma è il gold standard.

In CI/CD (M5) useremo (1) sempre, (2) su un campione casuale, (3) come
spot check manuale prima dei rilasci grandi.
