/**
 * app/page.tsx — Home page del frontend.
 * ----------------------------------------------------------------------------
 * Questa è una **Server Component** (default in Next.js App Router):
 * viene renderizzata sul server, mai mandata al browser come JS. Non può
 * usare hook React né browser API.
 *
 * L'unico pezzo interattivo della pagina è `<HealthStatus />`, che è
 * marcato `"use client"`. Pattern raccomandato in Next.js 16:
 *   - Server Component per il guscio statico (header, layout, contenuto fisso).
 *   - Client Component solo dove serve interattività.
 *
 * Risultato: bundle JS più piccolo, FCP più veloce, e l'idratazione
 * client copre solo il quadratino che mostra lo stato del backend.
 */

import { HealthStatus } from "@/components/health-status";

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-8 bg-zinc-50 px-6 py-16 dark:bg-black">
      <header className="flex flex-col items-center gap-2 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-4xl">
          agentic-rag-stack
        </h1>
        <p className="max-w-xl text-balance text-sm text-zinc-600 dark:text-zinc-400">
          Reference implementation di un sistema RAG agentico —{" "}
          <span className="font-medium">milestone M1</span>: frontend Next.js
          comunica col backend FastAPI.
        </p>
      </header>

      <HealthStatus />

      <footer className="text-xs text-zinc-400 dark:text-zinc-600">
        Dev workflow: in un terminale{" "}
        <code className="rounded bg-zinc-100 px-1 dark:bg-zinc-900">
          cd apps/api && uv run uvicorn app.main:app --reload
        </code>
        , in un altro{" "}
        <code className="rounded bg-zinc-100 px-1 dark:bg-zinc-900">
          cd apps/web && pnpm dev
        </code>
        .
      </footer>
    </main>
  );
}
