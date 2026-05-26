"""
app/rag/chunker.py
==================
Strategia di **chunking** per documenti markdown.

------------------------------------------------------------------------
Cos'è un chunker, in breve
------------------------------------------------------------------------

Un *chunker* spezza un documento lungo (es. un file .md da 5000 token)
in pezzi piccoli e auto-contenuti chiamati **chunk** (~300-500 token
l'uno). Serve perché:

1. **Embedding più precisi.** I modelli di embedding (qui
   `text-embedding-3-small`) producono vettori che catturano "l'idea"
   del testo. Un chunk piccolo e focalizzato = un'idea netta = un
   vettore preciso. Un chunk troppo lungo = molte idee mescolate = un
   vettore "medio" che non rappresenta bene nulla.

2. **Limiti di contesto downstream.** Quando in M3 il modello LLM
   leggerà i risultati del retrieval, ha un limite di context window.
   Chunk piccoli permettono di mettere molti risultati nel prompt.

3. **Granularità delle citazioni.** Una citazione `[3]` deve poter
   puntare a un *singolo* paragrafo, non a un'intera pagina, altrimenti
   l'utente che ci clicca non trova subito il pezzo di testo rilevante.

------------------------------------------------------------------------
Strategia adottata: heading-aware con sliding window di fallback
------------------------------------------------------------------------

Approccio "naive": prendere il testo, contare 500 token, spezzare,
ripetere. Funziona ma taglia in mezzo alle frasi e mescola argomenti
diversi. Pessimo per la qualità del retrieval.

Approccio nostro: rispettiamo la **struttura del markdown** (heading
con `#`, `##`, `###`). Spezziamo il documento in "sezioni" ai confini
delle heading; ogni sezione diventa un chunk se entra in
`chunk_size`, altrimenti spezzata ulteriormente con una *sliding
window* token-based con overlap.

In più, propaghiamo la **heading path** (es. "Tutorial > Dependencies
> Sub-dependencies") nei metadata del chunk. Serve a due cose:
- Mostrarla nelle citazioni ("dal cap. Dependencies > Sub-dependencies")
- Migliorare il retrieval: spesso converrà prependere la heading al
  testo prima di embeddare (lo decideremo in `embedder.py`).

------------------------------------------------------------------------
Perché contare token con tiktoken e non caratteri
------------------------------------------------------------------------

I limiti dei modelli LLM sono espressi in token, non caratteri. 1 token
≠ 1 parola ≠ 1 carattere. "Hello world" è 2 token; "antidisestablishmentarianism"
è 6 token. Misurare in caratteri sbaglierebbe del ±30% e ci farebbe
sforare i limiti o sprecare contesto. `tiktoken` (libreria ufficiale
OpenAI, scritta in Rust) ci dà il conteggio esatto.

Usiamo `cl100k_base`, l'encoding di GPT-4 e di tutta la famiglia
`text-embedding-3-*`. Per Claude la tokenizzazione è leggermente
diversa, ma per dimensionare i chunk va benissimo (l'errore relativo
è <10%).

------------------------------------------------------------------------
Determinismo dell'ID e idempotenza dell'ingestion
------------------------------------------------------------------------

Ogni chunk ha un `id`: lo calcoliamo come hash SHA-256 di
`source + position + primi 100 char di testo`. Conseguenza importante:
re-ingestare lo stesso documento produce gli **stessi id**. Quindi
l'upsert su Qdrant (M2 task #6) diventa idempotente: rilanciare la
pipeline non duplica chunk, semplicemente li sovrascrive. Ottimo per
sviluppo iterativo.

Se in futuro il testo del documento cambia, l'hash cambia → il chunk
"nuovo" verrà aggiunto e il vecchio resterà fino al prossimo cleanup.
Faremo un task di garbage collection in M5 se serve.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterator
from functools import lru_cache

import tiktoken
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoder tiktoken — caricato lazy (è lento da inizializzare la prima volta).
# ---------------------------------------------------------------------------
# `@lru_cache` qui è un trucco idiomatico Python per fare un singleton:
# la funzione viene chiamata UNA volta, il risultato è memoizzato, le
# chiamate successive lo riusano. Vantaggio rispetto a una variabile
# globale `_encoder = None` + check `if _encoder is None`: niente
# stato mutabile a livello modulo, e niente race conditions con thread.
@lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    """Restituisce l'encoder tiktoken per `cl100k_base`."""
    # cl100k_base = encoding usato da GPT-4, GPT-4o, text-embedding-3-*.
    # È il default 2026 per quasi tutti i modelli OpenAI.
    return tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Configurazione del chunker.
# ---------------------------------------------------------------------------
# Pydantic invece di dataclass per consistency col resto del codice
# (gli endpoint usano già Pydantic) e per avere validazione gratis su
# i constraint (gt=0, ge=0).
class ChunkerConfig(BaseModel):
    """Parametri configurabili del chunker.

    I default sono pensati per documenti tecnici (docs di librerie OSS):
    - chunk_size=500 token ≈ ~2000 caratteri, una sezione media di un
      tutorial. Abbastanza grande da contenere un esempio di codice +
      spiegazione, abbastanza piccolo per restare focalizzato.
    - chunk_overlap=50 token ≈ 10% del chunk. Trasporta contesto fra
      chunk adiacenti (es. "in the previous example...") senza
      duplicare troppo testo.
    """

    chunk_size: int = Field(
        default=500,
        gt=0,
        description="Numero target di token per chunk.",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Token di overlap fra chunk adiacenti.",
    )


# ---------------------------------------------------------------------------
# Schema di un chunk.
# ---------------------------------------------------------------------------
# Pydantic BaseModel: same pattern di tutti gli altri schemi del progetto.
# Verrà serializzato in JSON / inserito come payload Qdrant.
class Chunk(BaseModel):
    """Un singolo pezzo di documento pronto per embedding + indicizzazione.

    Attributes:
        id: hash SHA-256 deterministico (vedi `_deterministic_id`).
            16 caratteri hex = 64 bit di entropia, sufficiente per
            evitare collisioni su un corpus di milioni di chunk.
        text: contenuto testuale del chunk (così come finirà nel prompt
            quando il retrieval lo restituirà).
        source: identificativo del documento di origine (tipicamente
            il path file, es. `docs/en/docs/tutorial/dependencies/index.md`).
        heading: heading-path del chunk, es. "Tutorial > Dependencies".
            None per i chunk prima della prima heading.
        position: indice 0-based del chunk nel documento sorgente,
            usato per ordine e per costruire l'id.
        token_count: numero di token misurato con tiktoken cl100k_base.
            Utile per debug, eval, e calcolo costi.
    """

    id: str
    text: str
    source: str
    heading: str | None = None
    position: int = Field(ge=0)
    token_count: int = Field(gt=0)


# ===========================================================================
# API pubblica
# ===========================================================================


def chunk_markdown(
    text: str,
    source: str,
    config: ChunkerConfig | None = None,
) -> list[Chunk]:
    """Spezza un documento markdown in chunk heading-aware.

    Strategia:
    1. Identifica le heading lines (`#`, `##`, `###`, ...) e usa la
       loro gerarchia per costruire la "heading path".
    2. Spezza il testo in sezioni ai confini delle heading.
    3. Per ogni sezione:
       - Se sta in `config.chunk_size` token → la sezione è un chunk.
       - Se è più grande → la sezione viene ulteriormente spezzata
         con sliding window (`chunk_size` con `chunk_overlap`).

    Args:
        text: contenuto del documento markdown.
        source: identificativo del documento (es. path relativo).
        config: parametri di chunking (default: 500/50).

    Returns:
        Lista di `Chunk` in ordine di lettura del documento. Lista
        vuota se `text` è vuoto o solo whitespace.

    Raises:
        ValueError: se `chunk_overlap >= chunk_size` (non avrebbe senso).
    """
    config = config or ChunkerConfig()
    if config.chunk_overlap >= config.chunk_size:
        # Catturato anche dentro `_sliding_window`, ma alziamo qui per
        # un errore esplicito al chiamante prima di iniziare il lavoro.
        raise ValueError(
            f"chunk_overlap ({config.chunk_overlap}) deve essere "
            f"strettamente minore di chunk_size ({config.chunk_size})."
        )

    if not text.strip():
        # Documento vuoto / solo whitespace → niente da chunkare.
        return []

    encoder = _get_encoder()
    sections = _split_by_headings(text)

    chunks: list[Chunk] = []
    position = 0

    for heading_path, section_text in sections:
        # Encode la sezione una volta sola e riusa i token: l'encoding
        # è O(n) ma non gratis, evitiamo di farlo per ogni finestra.
        section_tokens = encoder.encode(section_text)

        if not section_tokens:
            # Sezione vuota (è successo? può capitare se il markdown
            # ha heading senza body sotto). La saltiamo.
            continue

        if len(section_tokens) <= config.chunk_size:
            # Sezione "piccola": un chunk e via.
            chunks.append(
                _make_chunk(
                    text=section_text,
                    token_count=len(section_tokens),
                    source=source,
                    heading=heading_path,
                    position=position,
                )
            )
            position += 1
        else:
            # Sezione "grande": sliding window con overlap.
            for window_tokens in _sliding_window(
                section_tokens,
                window_size=config.chunk_size,
                overlap=config.chunk_overlap,
            ):
                # `decode` ricostruisce il testo a partire dai token.
                # Funziona perché cl100k_base è un BPE invertibile.
                window_text = encoder.decode(window_tokens)
                chunks.append(
                    _make_chunk(
                        text=window_text,
                        token_count=len(window_tokens),
                        source=source,
                        heading=heading_path,
                        position=position,
                    )
                )
                position += 1

    logger.debug(
        "chunker_done",
        extra={
            "source": source,
            "chunks_produced": len(chunks),
            "config": config.model_dump(),
        },
    )
    return chunks


# ===========================================================================
# Implementazione interna
# ===========================================================================
#
# Tutto da qui in giù è privato (prefisso `_`): non è API stabile, può
# cambiare. Lo separiamo per chiarezza didattica.


# Regex per le heading di markdown. Spiegazione:
#   ^         → inizio riga (con MULTILINE flag)
#   (#{1,6})  → da 1 a 6 caratteri `#` (gruppo 1, ci dice il livello)
#   \s+       → almeno uno spazio
#   (.+?)     → il titolo (gruppo 2), non-greedy
#   \s*$      → opzionali spazi prima del fine riga
#
# Limitazioni note (accettate):
# - Non gestiamo le heading "setext" (sottolineate con === o ---).
#   Sono rare nelle docs moderne; se servirà, le aggiungeremo.
# - Non gestiamo `#` dentro a code fences ``` ``` (li conta come heading).
#   Per i docs di FastAPI questo è quasi sempre OK; se i test su
#   esempi reali mostrano problemi, mettiamo uno state machine che
#   ignora le righe dentro fence.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _split_by_headings(text: str) -> list[tuple[str | None, str]]:
    """Spezza il testo in sezioni ai confini delle heading.

    Mantiene uno *stack* delle heading per livello, così che ogni
    sezione conosca la sua *heading-path* (es. "Tutorial > Dependencies").

    Returns:
        Lista di tuple `(heading_path, section_text)`. Il primo elemento
        può avere `heading_path = None` se il documento inizia con
        testo prima di qualsiasi heading.
    """
    sections: list[tuple[str | None, str]] = []

    # heading_stack: livello (int) → titolo (str) della heading più
    # recente a quel livello. Quando vediamo `## Foo`, rimuoviamo
    # tutte le heading di livello >2 (sono "scadute") e settiamo
    # heading_stack[2] = "Foo".
    heading_stack: dict[int, str] = {}

    current_lines: list[str] = []
    current_heading_path: str | None = None

    def flush() -> None:
        """Chiude la sezione in costruzione e la appende a `sections`."""
        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append((current_heading_path, section_text))

    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            # Nuova heading: chiudi la sezione corrente prima di iniziare.
            flush()
            current_lines = []

            level = len(match.group(1))
            title = match.group(2).strip()

            # Rimuovi le heading "più profonde" dello stack: una `##`
            # invalida tutte le `###`/`####` precedenti.
            heading_stack = {k: v for k, v in heading_stack.items() if k < level}
            heading_stack[level] = title

            # Costruisce la heading-path concatenando i titoli in ordine
            # di livello (1 → 2 → 3 → ...).
            current_heading_path = " > ".join(heading_stack[k] for k in sorted(heading_stack))

            # La riga della heading stessa fa parte della sezione
            # (così l'embedding "vede" anche il titolo, utile per il
            # match semantico).
            current_lines.append(line)
        else:
            current_lines.append(line)

    # Ultima sezione (dopo l'ultimo heading o tutto il doc se senza heading).
    flush()
    return sections


def _sliding_window(
    tokens: list[int],
    window_size: int,
    overlap: int,
) -> Iterator[list[int]]:
    """Sliding window su una lista di token.

    Produce finestre di `window_size` token con `overlap` token di
    sovrapposizione fra finestre adiacenti. L'ultima finestra può
    essere più corta di `window_size` se i token rimanenti non bastano.

    Esempio: window_size=4, overlap=1, tokens=[A,B,C,D,E,F,G]
        → [A,B,C,D]  (start=0)
        → [D,E,F,G]  (start=3, step=3)
    """
    if overlap >= window_size:
        raise ValueError(f"overlap ({overlap}) deve essere < window_size ({window_size}).")

    step = window_size - overlap
    start = 0
    n = len(tokens)
    while start < n:
        yield tokens[start : start + window_size]
        if start + window_size >= n:
            # Abbiamo già emesso l'ultima finestra che copre la fine
            # del documento. Stop.
            break
        start += step


def _make_chunk(
    *,
    text: str,
    token_count: int,
    source: str,
    heading: str | None,
    position: int,
) -> Chunk:
    """Costruisce un `Chunk` calcolandone l'id deterministico."""
    return Chunk(
        id=_deterministic_id(source=source, position=position, text=text),
        text=text,
        source=source,
        heading=heading,
        position=position,
        token_count=token_count,
    )


def _deterministic_id(*, source: str, position: int, text: str) -> str:
    """Hash deterministico per un chunk.

    Input: `source`, `position`, primi 100 caratteri del testo.
    Output: 16 caratteri esadecimali (64 bit) → entropia sufficiente
    per non avere collisioni anche con corpus di milioni di chunk
    (probabilità di collisione su 1M chunk: ~3e-8).

    Perché solo i primi 100 char del testo e non tutto?
    - Velocità (hash più corto da calcolare → trascurabile in pratica).
    - Stabilità: se cambiamo whitespace minore alla fine di un chunk,
      l'id non cambia. Trade-off: due chunk con stessi primi 100
      char ma resto diverso avranno lo stesso id (improbabile in
      pratica, e comunque `source + position` discrimina).
    """
    key = f"{source}:{position}:{text[:100]}".encode()
    return hashlib.sha256(key).hexdigest()[:16]


# Limita cosa viene "esportato" se qualcuno fa `from app.rag.chunker import *`.
# Pythonic equivalent della clausola `export { ... }` di un modulo ES.
__all__ = ["Chunk", "ChunkerConfig", "chunk_markdown"]
