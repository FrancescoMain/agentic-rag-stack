# `docs/adr/` — Architecture Decision Records

Questa cartella contiene la **memoria decisionale** del progetto. Ogni
file documenta *una* scelta architetturale non banale: contesto,
alternative considerate, decisione, conseguenze.

## Indice degli ADR

| #    | Titolo                                                  | Status   |
| ---- | ------------------------------------------------------- | -------- |
| 0001 | [Struttura del repo: monorepo `apps/` + `docs/`](0001-monorepo-structure.md) | Accepted |

> Quando aggiungi un nuovo ADR ricordati di registrarlo qui sopra.

## Cos'è un ADR?

Un **Architecture Decision Record** è un file Markdown corto (~1 pagina)
che congela una decisione architetturale. È *l'unità minima* di
documentazione che ti permette di rispondere alla domanda *"perché lo
abbiamo fatto così?"* anche fra mesi.

### Cosa NON è

- ❌ **Non è documentazione di "come funziona"** — quella sta nel
  `README.md` di modulo e nei commenti del codice.
- ❌ **Non è un design doc** — un design doc è prospettico ("proponiamo
  di costruire X"); un ADR è retrospettivo ("abbiamo scelto X").
- ❌ **Non è un changelog** — un changelog dice *cosa* è cambiato; un ADR
  dice *perché* qualcosa è progettato in quel modo.

## Regole d'oro

### 1. Immutabilità

**Un ADR accettato non si modifica più.** Se cambi idea, scrivi un
nuovo ADR che dichiara `Supersedes: ADR-NNNN`. Il vecchio rimane lì,
con status aggiornato a `Superseded by ADR-MMMM`, come traccia storica.

Perché? Perché la decisione originale è *un fatto storico*: in quel
momento, con quel contesto, abbiamo deciso X. Cancellarla cancella la
storia. Sovrascriverla è come fare `git push --force` su `main`.

### 2. Granularità: una decisione = un ADR

Se stai per scrivere "Decisione A *e* Decisione B" → sono due ADR.
Permette di superare l'una senza l'altra in futuro.

### 3. Numerazione progressiva

Padding a 4 cifre: `0001-`, `0002-`, ... `0042-`, ... Mai riusare numeri
nemmeno se cancelli un ADR proposto (basta marcarlo `Rejected`).

### 4. Quando vale la pena scrivere un ADR?

Regola pratica:

> *"Ho dovuto considerare seriamente almeno un'alternativa?"*

- Sì → ADR.
- No (es. "usiamo Git", "facciamo type-checking") → niente ADR.

Alcuni trigger tipici nel nostro progetto:
- Scelta di una libreria/framework con alternative valide
  (LangGraph vs custom, Pinecone vs pgvector).
- Pattern architetturale (streaming SSE vs WebSocket, hybrid search
  vs solo vector).
- Trade-off costo/qualità (modello embedding piccolo vs grande,
  reranker o no).
- Politica di processo (eval gate in CI, manual approval prima del prod).

## Workflow per creare un nuovo ADR

1. **Copia** [`TEMPLATE.md`](TEMPLATE.md) in `NNNN-titolo-kebab.md`,
   dove `NNNN` è il prossimo numero libero.
2. **Riempi** le sezioni. Lascia `Status: Proposed` finché non è
   confermato.
3. **Aggiungi una riga** all'indice di questo README.
4. **Commit** con un messaggio del tipo `docs: ADR-NNNN — titolo`.
5. Quando la decisione è confermata, cambia `Status` a `Accepted` (in
   un commit separato).

## Risorse

- [adr.github.io](https://adr.github.io) — overview del formato MADR e
  varianti.
- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
  — il post originale di Michael Nygard che ha lanciato il concetto.
