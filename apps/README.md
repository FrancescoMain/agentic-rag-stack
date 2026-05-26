# `apps/` — Applicazioni deployabili

Questa cartella contiene **applicazioni indipendenti**: ognuna ha il proprio
ciclo di vita (build, deploy, dipendenze, runtime) e potrebbe in teoria
girare su una macchina diversa.

| Cartella | Tipo                       | Linguaggio | Tooling          |
| -------- | -------------------------- | ---------- | ---------------- |
| `api/`   | Backend HTTP + agenti AI   | Python 3.12 | uv, FastAPI     |
| `web/`   | Frontend chat streaming    | TypeScript  | pnpm, Next.js   |

## Convenzione monorepo

Stiamo usando la convenzione `apps/` + (in futuro) `packages/`:

- **`apps/`** → cose che si **eseguono**: server, frontend, worker, cron.
- **`packages/`** → cose che si **importano**: librerie condivise, tipi
  TypeScript, design system. (Non ne abbiamo ancora.)

È la stessa convenzione che usano Turborepo, Nx, e i monorepo di Vercel.

## Comunicazione tra `api/` e `web/`

In sviluppo: il frontend (porta 3000) chiama il backend (porta 8000) via
HTTP/SSE. CORS è configurata nel backend per accettare `localhost:3000`.

In produzione: ne riparleremo in M5. Probabilmente saranno due container
separati dietro un reverse proxy.
