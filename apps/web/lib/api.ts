/**
 * lib/api.ts — Wrapper tipizzato attorno alle chiamate al backend FastAPI.
 * ----------------------------------------------------------------------------
 * Tutte le chiamate al backend passano da qui. Vantaggi:
 *   1. Un solo posto dove leggere `NEXT_PUBLIC_API_URL` (no magic strings
 *      sparse nei componenti).
 *   2. Tipi condivisi: il `HealthResponse` è la mirror image dello schema
 *      Pydantic in `apps/api/app/main.py`. In M5 potremo generarli
 *      automaticamente da OpenAPI; per ora li manteniamo a mano.
 *   3. Error handling consistente: ogni funzione lancia con un messaggio
 *      strutturato, il chiamante decide come renderizzarlo.
 *
 * Pattern: nessuna classe, nessuna istanza globale. Solo funzioni pure.
 * Più semplice da testare e tree-shake-friendly per il bundler.
 */

// Tipo speculare dello schema Pydantic `HealthResponse` lato Python.
// Se cambia uno deve cambiare l'altro — finché non automatizziamo.
export type HealthResponse = {
  status: string;
  version: string;
};

/**
 * Ritorna l'URL base del backend. Errore esplicito se la env var manca:
 * preferiamo fallire forte a build/run-time piuttosto che inviare richieste
 * a "undefined/health" che producono errori confusi nel browser.
 */
function getApiUrl(): string {
  const url = process.env.NEXT_PUBLIC_API_URL;
  if (!url) {
    throw new Error(
      "NEXT_PUBLIC_API_URL non è definita. Crea apps/web/.env.local " +
        "copiando apps/web/.env.example.",
    );
  }
  return url;
}

/**
 * GET /health del backend.
 *
 * `cache: "no-store"` disabilita la cache di Next.js: vogliamo SEMPRE
 * lo stato attuale del backend, non un valore memorizzato. (In Next.js
 * 16 il default su `fetch` non è più cache automatica come in 13/14,
 * ma esplicitarlo rende l'intento chiaro a chi legge.)
 */
export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${getApiUrl()}/health`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `Backend ha risposto con status ${response.status} ${response.statusText}`,
    );
  }

  return (await response.json()) as HealthResponse;
}
