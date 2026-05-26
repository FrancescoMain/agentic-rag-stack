# `docs/` — Documentazione del progetto

Cosa trovi qui:

| File / cartella | Contenuto                                                      |
| --------------- | -------------------------------------------------------------- |
| `ROADMAP.md`    | Dettaglio task-per-task delle 5 milestone, con DoD e concetti  |
| `adr/`          | Architecture Decision Records — una decisione = un file        |
| `runbook.md`    | Procedure operative (deploy, rollback, incidenti) — riempito in M5 |

## Filosofia di documentazione

Il README di root è la **vetrina** (cosa fa, perché esiste, quick start).
Questa cartella è **il motore**:

- **ROADMAP.md** è la *mappa di studio + piano di lavoro*. Si aggiorna a
  ogni milestone chiusa.
- **adr/** è la *memoria decisionale*. Ogni scelta non banale (libreria,
  pattern, trade-off) finisce in un file immutabile. Quando in futuro
  qualcuno chiederà *"perché abbiamo scelto Pinecone invece di Weaviate?"*
  la risposta è nel file ADR.
- **runbook.md** è il *libretto delle istruzioni operative*. Cosa fare
  quando un container va in OOM. Come ruotare le chiavi. Come fare
  rollback. Esiste perché alle 3 di notte non vuoi rileggerti il README.

## A chi è rivolta

- A **Francesco fra 6 mesi**, che non si ricorderà perché ha scelto X.
- A un **collega** che deve farsi un'idea senza leggere tutto il codice.
- A un **valutatore di portfolio**, che capisce la maturità del progetto
  dalla qualità della doc.
