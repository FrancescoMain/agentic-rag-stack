# `apps/web/` — Frontend Next.js

Il frontend del progetto. In M1 mostra solo lo stato del backend; cresce
nelle milestone successive fino a diventare una chat AI-first completa.

> **Stato attuale (M1):** scaffold Next.js 16 + React 19 + Tailwind 4 +
> TypeScript. Una sola pagina (home) che pinga il backend `/health` e
> mostra stato live.

## Stack

| Pezzo            | Versione             | Note                                              |
| ---------------- | -------------------- | ------------------------------------------------- |
| Next.js          | 16.x (App Router)    | Turbopack default                                 |
| React            | 19.x                 | RSC + Server Actions                              |
| TypeScript       | 5.x                  | Strict mode dal config generato                   |
| Tailwind CSS     | 4.x                  | Niente `tailwind.config.ts`: config inline in CSS |
| ESLint           | 9.x (flat config)    | `eslint-config-next` come base                    |
| Package manager  | pnpm                 | Lockfile committato                               |

## Mappa della cartella

```
apps/web/
├── app/                    ← App Router (Next 13+)
│   ├── layout.tsx          ← root layout (html, body, font)
│   ├── page.tsx            ← home page (Server Component)
│   └── globals.css         ← stili Tailwind + reset
├── components/             ← componenti riusabili
│   └── health-status.tsx   ← Client Component che pinga /health
├── lib/                    ← logica condivisa / client API
│   └── api.ts              ← wrapper tipizzato attorno al backend
├── public/                 ← asset statici serviti as-is
├── .env.example            ← template variabili d'ambiente
├── .env.local              ← valori reali (gitignored)
├── next.config.ts          ← config Next
├── postcss.config.mjs      ← per Tailwind
├── eslint.config.mjs       ← ESLint flat config
└── tsconfig.json
```

## Come si esegue

Dalla root di `apps/web/`:

```bash
pnpm install         # prima volta — scarica node_modules
pnpm dev             # avvia il dev server su http://localhost:3000
pnpm build           # build di produzione
pnpm start           # serve la build di produzione
pnpm lint            # ESLint
```

> Il frontend chiama il backend (`localhost:8000`). Assicurati che gira
> in parallelo: in un altro terminale, da `apps/api/`:
>
> ```bash
> uv run uvicorn app.main:app --reload
> ```

## Variabili d'ambiente

Copia `.env.example` in `.env.local` e popola i valori (solo
`NEXT_PUBLIC_API_URL` per ora).

**Regola d'oro:** solo le variabili prefissate `NEXT_PUBLIC_` finiscono
nel bundle browser. Tutte le altre sono visibili solo lato server
(Server Components, Route Handlers, Server Actions). Mai mettere
chiavi segrete in variabili `NEXT_PUBLIC_*`.

## Concetti chiave (per chi non viene da Next App Router)

- **Server Components di default**: ogni `.tsx` sotto `app/` è Server
  Component se non ha `"use client"` in cima. Niente hook, niente browser
  API; può fare `fetch` direttamente da database/API senza esporre nulla
  al client.
- **`"use client"` directive**: marca un componente (e tutto ciò che
  importa) come "deve girare nel browser". Necessario per `useState`,
  `useEffect`, `onClick`, ecc.
- **Pattern composizionale**: Server Component come "guscio", Client
  Component come "isole" interattive — minimizza il JS spedito.

## Note su Next.js 16

Questa versione (rilasciata fine 2025/inizio 2026) ha breaking changes
rispetto a 14/15. Le rilevanti per noi:

- **Turbopack default** per `next dev` e `next build`. Niente flag.
- **Async request APIs**: `params`, `searchParams`, `cookies()`, `headers()`
  sono ora sempre Promise. Conta solo quando useremo route dinamiche.
- **`middleware` → `proxy`**: il file `middleware.ts` è stato rinominato
  in `proxy.ts`. Non lo usiamo ancora.

Per la lista completa: `node_modules/next/dist/docs/01-app/02-guides/upgrading/version-16.md`.
