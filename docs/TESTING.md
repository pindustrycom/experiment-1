# Testing

All fixtures are SIMULATED and labelled as such. No test calls a real data provider or the Claude
API; the Claude client is replaced by a fake that returns typed responses.

## Backend (`cd backend && pytest`)

Database tests need PostgreSQL. They use `TEST_DATABASE_URL` (default
`postgresql+psycopg://sm:sm@localhost/sm_test`), rebuild its schema with Alembic, and truncate
tables between tests, so point it at a throwaway database. Without a reachable database those tests
are skipped and the pure tests still run.

| File | Covers |
|---|---|
| `test_scoring.py` | SI-v1 formula and normalisations; every SI-v2 component; missing-data handling (unavailable vs zero, provisional vs unavailable, required M); weights normalisation and versioning; directional bias and conflicts; evidence confidence; **future-data leakage** (future-published and late-ingested events, price rows after the cutoff, fundamentals and restatements filed after the cutoff) |
| `test_evaluation.py` | D(t,H) as an absolute log-return gap (and invariance to dividend re-basing), thresholds per horizon and overrides, maturity in NYSE sessions, metrics (ROC-AUC against brute force, PR-AUC, Brier, confusion, quartiles, industry-clustered bootstrap, insufficient data) |
| `test_news.py` | URL canonicalisation and syndication clustering; delimiter-escape attempts in untrusted text; structured-output parsing, refusals, truncation; schema validation; verbatim evidence verification; persistence of events and usage; AI budget enforced before calls; failures recorded and not scored; point-in-time event visibility |
| `test_jobs.py` | NYSE holidays and observance rules; last completed session; close-time cutoffs across DST; cron in New York time across DST; holiday skipping; missed-run catch-up; next-run calculation; job runner success, exponential backoff retries, duplicate prevention, pause, queue claiming, head-of-line blocking, restart recovery; Alembic downgrade/upgrade |
| `test_adapters.py` | Tiingo, Stooq, Finnhub, Alpha Vantage, RSS and SEC EDGAR (filings, XBRL company facts) parsers against payloads shaped like each provider's documented format, including error and quota responses |
| `test_api.py` | Auth, cookie flags, CSRF, login and API rate limits, roles; pair CRUD, validation, duplicate detection, archive-not-delete with history, lookups by uid and reversed slug; seeded pairs with LOW/HD merged; settings validation, secret redaction and weight versioning; jobs queue/pause/resume/schedule validation; backtest parameter validation; exports and PDF; watchlists; structured errors; database immutability triggers on predictions; production mode with no demo data and flagging of leftover demo rows |

Lint: `ruff check app worker tests`.

## Frontend (`cd frontend`)

* `npm run lint`: TypeScript type check.
* `npm test`: Vitest + Testing Library component tests (`tests/unit`): formatting never shows
  missing values as zero, risk levels, New York time display across DST, score cells (n/a and
  provisional), simulated-data labels, pair table sorting with unavailable scores last.
* `npm run test:e2e`: Playwright against a running stack in demo mode (`tests/e2e`): login and
  redirect, wrong password, dashboard, pair detail tabs, news filtering and evidence, performance
  separation of live/backtest/reported, pause/resume, add/archive/remove a pair, no secrets in
  Settings. Configure with `E2E_BASE_URL`, `E2E_EMAIL`, `E2E_PASSWORD`; set
  `E2E_IGNORE_HTTPS_ERRORS=1` when testing through Caddy with `SITE_ADDRESS=localhost`.

## What has not been tested against live services

The provider adapters (Tiingo, Stooq, SEC EDGAR, Finnhub, Alpha Vantage, RSS) and the live Claude
call are implemented against the providers' documented formats and their parsers are tested with
sample payloads in those formats, but they were not run against the real services in the build
environment (its network blocks those hosts). Use
**Settings → System & account → Health check** after adding keys to verify each one, then run the
ingestion jobs from Bot activity and check their results.
