# Splittability Monitor

A self-hosted research platform that monitors stock pairs, reads company news and filings with
Claude, and tracks whether each pair is at risk of drifting apart. It computes two indices, writes
immutable daily predictions and scores them against what the market actually does.

It is a **research and paper-tracking tool**. It never places orders, and its scores are relative
risk rankings, not probabilities or investment advice.

## What it does

* **Watchlist** of pairs (initially KO/PEP, WMT/TGT, HD/LOW, V/MA, XOM/CVX, AMD/NVDA, JPM/BAC,
  MS/GS, CVS/WBA, MCD/YUM, FDX/UPS; the duplicate LOW/HD is merged). Add, disable, archive (history
  kept), group into watchlists, set thresholds and check frequency per pair.
* **SI-v1**: the original market-only index, (25C + 15D + 25R) × 100 / 65.
* **SI-v2**: news-enhanced, 35N + 25F + 20M + 10P + 10S with configurable, versioned weights.
  Missing components are marked unavailable, never treated as zero.
* For every pair: SI (0–100 relative risk), **directional bias** (which company the evidence favours)
  and **evidence confidence**, each with a plain-language explanation and an inspector showing every
  input, formula and source.
* **Claude news analyst**: SEC filings, company news (Finnhub, Alpha Vantage), RSS and
  investor-relations feeds are deduplicated, then interpreted into a validated schema. Every quote is
  checked against the source text; scores are computed by deterministic code.
* **Forward testing**: immutable daily snapshots for 30/90/180/252/504-session horizons, resolved
  automatically with D(t,H) on total-return prices; ROC-AUC, PR-AUC, Brier, calibration, quartiles,
  bootstrap intervals, by industry and against baselines. Walk-forward backtests and the original
  100-pair results are shown separately. CSV and PDF exports.
* **Bot**: a persistent worker runs on New York market time (DST and NYSE holidays aware), records
  every run, retries with backoff, never runs a job twice at once, recovers after restarts, and
  tracks Claude tokens and cost against a monthly budget.

The full method is in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Quick start (local or VPS)

```bash
cp .env.example .env      # set SITE_ADDRESS, POSTGRES_PASSWORD, SECRET_KEY, BOOTSTRAP_ADMIN_*
docker compose up -d --build
```

Open `https://SITE_ADDRESS` and sign in with the bootstrap admin. With `APP_MODE=demo` (the default)
the app runs end to end on clearly labelled **SIMULATED** data and needs no API keys. For real data,
add keys and switch to production as described in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md), which
also covers the Hostinger VPS setup step by step, HTTPS, health checks, logs, updates, backups and
restores.

## Repository layout

```
backend/          FastAPI API, scoring engine, Claude agent, data adapters, worker, Alembic migrations, tests
  app/scoring/    SI-v1, SI-v2 components, point-in-time guards (pure functions)
  app/agents/     Claude news analyst (structured output, evidence verification)
  app/data_sources/  Tiingo, Stooq, SEC EDGAR, Finnhub, Alpha Vantage, RSS adapters
  app/services/   ingestion, scoring, predictions, evaluation, backtests, jobs, backups
  worker/         scheduler + job runner process
frontend/         Next.js (App Router) + TypeScript + Tailwind + Recharts dashboard, Vitest and Playwright tests
infrastructure/   Caddyfile (reverse proxy, automatic HTTPS)
scripts/          deploy, backup, restore, create-admin, switch-to-production
docs/             methodology, deployment, testing
```

## API

REST under `/api/v1` (OpenAPI at `/api/docs`): pairs (list/filter/sort, create, update, archive,
scores, score detail, price analytics, news, predictions, events), news, performance, backtests,
exports, jobs (list, run, pause, resume, schedule edits, run history), settings, watchlists, usage,
alerts, auth. Paginated, validated, with structured errors
`{"error": {"code", "message", "details"}}`. Authentication is an HttpOnly session cookie; mutating
requests also need the header `X-Requested-With: splittability`.

## Development

```bash
# backend (Python 3.12, PostgreSQL running locally)
cd backend && python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e . --no-deps
createdb sm && createdb sm_test
DATABASE_URL=postgresql+psycopg://USER:PASS@localhost/sm alembic upgrade head
DATABASE_URL=... python -m app.cli init        # demo data by default
DATABASE_URL=... uvicorn app.main:app --reload # API on :8000
DATABASE_URL=... python -m worker.main         # worker
pytest                                          # uses TEST_DATABASE_URL (default .../sm_test)

# frontend (Node 22)
cd frontend && npm ci && npm run dev            # http://localhost:3000, proxies /api to :8000
npm run lint && npm test && npm run test:e2e
```

See [docs/TESTING.md](docs/TESTING.md) for what the test suites cover.

## Security notes

* Secrets come only from environment variables; none are stored in the database, logged, or sent to
  the browser. `.env` is git-ignored.
* Passwords are hashed with Argon2id; sessions are signed, HttpOnly, Secure and SameSite=Strict;
  CSRF header on mutations; login and API rate limits; admin vs read-only viewer roles.
* PostgreSQL, the API and the web server are reachable only on the internal Docker network; Caddy
  is the single public entry point with HTTPS and security headers.
* Article and filing text is treated as untrusted data: delimited in prompts, never followed as
  instructions, and every AI claim must be backed by a verbatim quote.
* Production mode never uses simulated data: the worker refuses to run if demo rows remain.
