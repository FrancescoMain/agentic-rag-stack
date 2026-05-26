"use client";

/**
 * components/classify-playground.tsx
 * ----------------------------------------------------------------------------
 * Playground per provare l'endpoint POST /classify del backend.
 *
 * Punto narrativo M1 — "Invisible AI":
 *   Questo componente è strutturalmente identico ad HealthStatus
 *   (Client Component, stati loading/ok/error, fetch a un endpoint REST).
 *   La differenza è che dietro le quinte il backend chiama Claude. Da qui
 *   non si vede: niente streaming, niente chat, niente "agente". È un
 *   normale endpoint che ritorna JSON tipizzato. È l'AI che si nasconde
 *   sotto l'API.
 */

import { useState } from "react";

import {
  classifyText,
  type ClassifyCategory,
  type ClassifyResult,
} from "@/lib/api";

// Discriminated union per gli stati possibili del componente.
// `idle` perché c'è un'azione esplicita dell'utente (non un mount-time fetch).
type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ok"; data: ClassifyResult }
  | { kind: "error"; message: string };

// Tabella colori per categoria. Usare un oggetto tipizzato come
// `Record<ClassifyCategory, ...>` ci protegge: se domani aggiungiamo una
// categoria al backend ed estendiamo il tipo, TypeScript ci OBBLIGA ad
// aggiornare anche questa mappa. Mancarne una è un type error.
const CATEGORY_STYLES: Record<ClassifyCategory, string> = {
  bug: "bg-red-100 text-red-900 border-red-300 dark:bg-red-950/40 dark:text-red-200 dark:border-red-900",
  feature:
    "bg-blue-100 text-blue-900 border-blue-300 dark:bg-blue-950/40 dark:text-blue-200 dark:border-blue-900",
  question:
    "bg-purple-100 text-purple-900 border-purple-300 dark:bg-purple-950/40 dark:text-purple-200 dark:border-purple-900",
  spam: "bg-zinc-100 text-zinc-900 border-zinc-300 dark:bg-zinc-900 dark:text-zinc-200 dark:border-zinc-700",
};

// Esempi precaricati: uno per categoria. Buona pratica per un playground —
// abbassa l'attrito per provare.
const EXAMPLES: { label: string; text: string }[] = [
  { label: "bug", text: "Il bottone Save non funziona da ieri, errore 500 nel browser." },
  { label: "feature", text: "Sarebbe bello avere il dark mode anche nella dashboard." },
  { label: "question", text: "Come si esporta il report in formato Excel?" },
  { label: "spam", text: "COMPRA SUBITO INTEGRATORE MIRACOLOSO!!! Sconto 50% solo oggi!" },
];

export function ClassifyPlayground() {
  const [text, setText] = useState("");
  const [state, setState] = useState<State>({ kind: "idle" });

  async function handleClassify() {
    // Guard: niente trip al server per stringhe vuote/whitespace.
    if (!text.trim()) return;

    setState({ kind: "loading" });
    try {
      const data = await classifyText(text);
      setState({ kind: "ok", data });
    } catch (err) {
      setState({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }

  return (
    <div className="w-full max-w-md rounded-lg border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
          Classify playground
        </h2>
        <code className="text-[10px] text-zinc-400 dark:text-zinc-600">
          POST /classify
        </code>
      </div>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Incolla un testo da classificare in bug | feature | question | spam"
        rows={4}
        maxLength={4000}
        className="w-full resize-none rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-400 focus:border-zinc-400 focus:outline-none dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-100 dark:placeholder:text-zinc-600"
      />

      {/* Esempi cliccabili: uno per categoria, per provare al volo. */}
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-zinc-400 dark:text-zinc-600">
        <span>Esempi:</span>
        {EXAMPLES.map((example) => (
          <button
            key={example.label}
            type="button"
            onClick={() => setText(example.text)}
            className="rounded border border-zinc-200 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider transition-colors hover:bg-zinc-100 hover:text-zinc-700 dark:border-zinc-800 dark:hover:bg-zinc-900 dark:hover:text-zinc-300"
          >
            {example.label}
          </button>
        ))}
      </div>

      <button
        type="button"
        onClick={handleClassify}
        disabled={!text.trim() || state.kind === "loading"}
        className="mt-3 w-full rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
      >
        {state.kind === "loading" ? "Classificando…" : "Classify"}
      </button>

      {state.kind === "ok" && <Result data={state.data} />}
      {state.kind === "error" && <ErrorBlock message={state.message} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components — estratti per leggibilità.
// ---------------------------------------------------------------------------

function Result({ data }: { data: ClassifyResult }) {
  const pct = Math.round(data.confidence * 100);
  return (
    <div className="mt-4 space-y-3">
      <div className="flex items-center justify-between">
        <span
          className={`rounded-md border px-2.5 py-1 font-mono text-[11px] font-medium uppercase tracking-wider ${CATEGORY_STYLES[data.category]}`}
        >
          {data.category}
        </span>
        <span className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
          confidence {pct}%
        </span>
      </div>
      <ConfidenceBar value={data.confidence} />
      <p className="text-sm leading-relaxed text-zinc-700 dark:text-zinc-300">
        {data.reasoning}
      </p>
    </div>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-900">
      <div
        className="h-full bg-emerald-500 transition-all duration-300"
        style={{ width: `${pct}%` }}
        aria-label={`Confidence ${pct} percento`}
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
      />
    </div>
  );
}

function ErrorBlock({ message }: { message: string }) {
  return (
    <div className="mt-4 space-y-1">
      <p className="text-xs font-medium text-red-700 dark:text-red-400">
        Errore durante la classificazione
      </p>
      <p className="break-words text-xs text-zinc-500 dark:text-zinc-400">
        {message}
      </p>
    </div>
  );
}
