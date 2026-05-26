"use client";

/**
 * components/health-status.tsx — Mostra lo stato del backend in tempo reale.
 * ----------------------------------------------------------------------------
 * Client Component (notare la direttiva `"use client"` in cima): gira nel
 * browser, può usare hook React e fare fetch da useEffect.
 *
 * Pattern didattico, M1:
 *   - Tre stati: loading | ok | error.
 *   - On mount fa il primo fetch. Bottone "Refresh" rifa la chiamata.
 *   - Nessuna libreria di data-fetching (SWR, React Query, ecc.): in M3
 *     valuteremo se introdurle. Per un solo endpoint useState basta.
 *
 * Questo componente esiste anche per dimostrare che la CORS è
 * configurata correttamente lato backend: se manca, il browser blocca
 * il fetch e finiamo nello stato `error`.
 */

import { useEffect, useState } from "react";
import { fetchHealth, type HealthResponse } from "@/lib/api";

// Stati possibili del componente. Discriminated union: TypeScript ci
// obbliga a gestire ogni caso nel rendering.
type State =
  | { kind: "loading" }
  | { kind: "ok"; data: HealthResponse }
  | { kind: "error"; message: string };

export function HealthStatus() {
  const [state, setState] = useState<State>({ kind: "loading" });

  // Funzione di fetch usata sia al mount sia dal click del bottone.
  async function load() {
    setState({ kind: "loading" });
    try {
      const data = await fetchHealth();
      setState({ kind: "ok", data });
    } catch (err) {
      setState({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }

  // Effetto: chiama load() una volta al mount del componente.
  // L'array vuoto come dipendenze significa "esegui solo al mount".
  useEffect(() => {
    load();
  }, []);

  return (
    <div className="w-full max-w-md rounded-lg border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
          Backend status
        </h2>
        <button
          onClick={load}
          disabled={state.kind === "loading"}
          className="rounded-md border border-zinc-200 px-3 py-1 text-xs font-medium text-zinc-700 transition-colors hover:bg-zinc-50 disabled:opacity-50 dark:border-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-900"
        >
          {state.kind === "loading" ? "Loading…" : "Refresh"}
        </button>
      </div>

      {state.kind === "loading" && (
        <div className="flex items-center gap-3">
          <Dot color="bg-zinc-300 dark:bg-zinc-700" pulse />
          <span className="text-sm text-zinc-500">Sto chiamando /health…</span>
        </div>
      )}

      {state.kind === "ok" && (
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <Dot color="bg-emerald-500" />
            <span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
              {state.data.status}
            </span>
          </div>
          <div className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
            version {state.data.version}
          </div>
        </div>
      )}

      {state.kind === "error" && (
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <Dot color="bg-red-500" />
            <span className="text-sm font-medium text-red-700 dark:text-red-400">
              Backend non raggiungibile
            </span>
          </div>
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            {state.message}
          </p>
          <p className="text-xs text-zinc-400 dark:text-zinc-500">
            Hai avviato il backend? Da{" "}
            <code className="rounded bg-zinc-100 px-1 dark:bg-zinc-800">
              apps/api
            </code>
            :{" "}
            <code className="rounded bg-zinc-100 px-1 dark:bg-zinc-800">
              uv run uvicorn app.main:app --reload
            </code>
          </p>
        </div>
      )}
    </div>
  );
}

// Piccolo "indicatore puntino" colorato. Tenuto inline perché è solo
// presentazione e non vale la pena estrarlo in un altro file finché
// non serve altrove.
function Dot({ color, pulse }: { color: string; pulse?: boolean }) {
  return (
    <span
      className={`inline-block h-2.5 w-2.5 rounded-full ${color} ${
        pulse ? "animate-pulse" : ""
      }`}
    />
  );
}
