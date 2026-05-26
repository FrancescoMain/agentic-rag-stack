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

// Tipi speculari di `ClassifyResult` lato Python (apps/api/app/services/classifier.py).
// `ClassifyCategory` corrisponde al `Literal["bug","feature","question","spam"]`
// del backend. Tenendo l'union qui esplicita, TypeScript ci aiuta in tutti i
// componenti che la consumano (es. tabelle di colori, label).
export type ClassifyCategory = "bug" | "feature" | "question" | "spam";

export type ClassifyResult = {
  category: ClassifyCategory;
  confidence: number; // [0, 1]
  reasoning: string;
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

/**
 * Helper: estrae un messaggio leggibile dal body di un errore FastAPI/Pydantic.
 *
 * FastAPI mette gli errori in `{detail: ...}`:
 *   - 422 validation: `detail` è un array di oggetti con loc/msg/type
 *   - 4xx/5xx custom: `detail` è una stringa
 *
 * Tentiamo di estrarre la string più informativa. Cade gracefully sullo
 * `statusText` se il body non è parsabile (es. errore di rete prima di
 * arrivare al backend).
 */
async function extractErrorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      // Validation errors di Pydantic: prendiamo il primo messaggio.
      return body.detail.map((e: { msg?: string }) => e.msg ?? "").join("; ");
    }
    return JSON.stringify(body);
  } catch {
    return response.statusText;
  }
}

/**
 * POST /classify — classifica un testo nelle 4 categorie:
 * `bug | feature | question | spam`.
 *
 * Lancia con un messaggio HTTP-aware se la risposta non è 2xx.
 * Lascia volutamente all'errore di rete (es. backend spento) il suo
 * messaggio nativo: il chiamante lo distingue facilmente.
 */
export async function classifyText(text: string): Promise<ClassifyResult> {
  const response = await fetch(`${getApiUrl()}/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    cache: "no-store",
  });

  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(`HTTP ${response.status}: ${detail}`);
  }

  return (await response.json()) as ClassifyResult;
}
