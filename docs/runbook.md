# Runbook operativo

> **PLACEHOLDER.** Questo file sarà compilato in **M5** (Production /
> MLOps), quando avremo qualcosa che gira davvero in ambiente non-locale.

## Cosa andrà qui (anteprima)

Un **runbook** è il "manuale del pompiere": istruzioni passo-passo da
seguire in situazioni operative ricorrenti. Non sostituisce la
documentazione architetturale (quella sta nel `README.md` di root e
negli ADR), risponde a domande del tipo:

- **Deploy** — come faccio il deploy di una nuova versione? Come faccio
  rollback se va male?
- **Incidenti** — il modello restituisce errori 5xx, che faccio? Il vector
  DB è down, come degrade graceful?
- **Operazioni di routine** — come ruoto una API key? Come pulisco i
  trace Langfuse vecchi?
- **Costi** — dove monitoro lo speso giornaliero? Quando devo allarmarmi?

Lo scriviamo "in pieno" solo quando il sistema è in produzione: prima
sarebbero solo ipotesi.
