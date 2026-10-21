# Running the dashboard on GitHub Pages

This is the free, server-less way to run the monitor. A scheduled GitHub Actions workflow
(`.github/workflows/pages.yml`) runs the same pipeline as the full app after each US market close
and publishes a read-only dashboard to GitHub Pages:

1. Starts a temporary PostgreSQL and restores the database from the previous run (the
   `db-latest.dump` asset of the `state` release in this repository).
2. Runs `sync_reference`, `ingest_prices`, `ingest_fundamentals`, `ingest_news`, `compute_scores`,
   `snapshot_predictions` and `evaluate_outcomes`. On the first run it also runs the walk-forward
   backtest.
3. Builds the site with `python -m app.static_site` (pairs, per-pair detail, predictions,
   performance, about, plus CSV downloads). The builder refuses to publish if any simulated data is
   present.
4. Saves the database back to the `state` release (plus a dated copy, kept for 30 days) and deploys
   the site.

Predictions are never edited: the database triggers that make them immutable are part of the
restored schema.

## One-time setup

1. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
2. Add whichever data sources you want under **Settings → Secrets and variables → Actions**. Each one
   is optional; anything missing shows as "unavailable" on the site, never as zero.

   | Name | Kind | Enables |
   |---|---|---|
   | `SEC_USER_AGENT` | variable | SEC filings and fundamentals (F, part of S, filing news). Value like `SplittabilityMonitor research you@example.com`; SEC requires a contact address. |
   | `TIINGO_API_KEY` | secret | **Needed for prices.** Dividend-adjusted daily prices; the free key at tiingo.com is enough. Stooq is still tried as a fallback, but on 2026-10-15 its download endpoint returned HTTP 404 for every ticker from GitHub's runners. |
   | `ANTHROPIC_API_KEY` | secret | Claude reads the news (N and P). Costs money per call; capped by the monthly budget (default $50). |
   | `FINNHUB_API_KEY`, `ALPHAVANTAGE_API_KEY` | secret | Company news feeds (free keys). |

3. **Actions → Update dashboard → Run workflow** for the first run. After that it runs every weekday
   evening on its own.

SI-v2 needs at least 60% of its weight available, including the market component. Market,
fundamentals and structure together are 55%, so SI-v2 shows as unavailable until news analysis
(`ANTHROPIC_API_KEY` plus at least one news source) is on. SI-v1 needs only prices, so with just `TIINGO_API_KEY` set the dashboard shows SI-v1 scores and records predictions.

## How the site is organised

The site root lists every experiment, in two groups: the market experiments (Exp1, Exp2 and Exp7, traded with
paper money) and the resonance research pages (Exp4 to Exp6). Every page starts with the same navigation bar, so
each experiment links to all the others. `experiments/hub/build_hub.py` adds the bar when the site is assembled,
from one list (`CATALOG`): to add an experiment, add its row there.

The 3D pages (Exp4, Exp6 and Exp6 live) share `experiments/hub/gl_runtime.js`. It renders only while the tab is
visible and the 3D view is on screen, caps the pixel ratio, and releases the graphics context whenever a page is
left. Chrome turns WebGL off for a whole site after its pages lose the context a few times, which is why a crash
used to need a browser restart. New 3D pages should use `LabGL.start(canvas)` and `gl.loop(frame)` instead of
creating a `THREE.WebGLRenderer` and calling `requestAnimationFrame` themselves.

## Privacy

The whole site is behind a password (the `PASSWORD` secret): `experiments/hub/lock_site.py` encrypts every page
and data file, and the browser decrypts them once the password is typed. The site sends `noindex` and a
`robots.txt` that asks search engines to stay out.

The repository is public, and so are its releases. Every file the workflows keep in a release (database dumps,
site builds, experiment state, the paper portfolio) is encrypted with `experiments/hub/state_crypt.sh` under the
`STATE_KEY` secret, or under `PASSWORD` when `STATE_KEY` is not set. Set `STATE_KEY` to its own random value before
you ever change the site password: files sealed under one key can't be opened with another. Files saved before
this change are read as they are and replaced with encrypted copies on the next run.

What stays public whatever the password: the code, the research notes and the market-data caches in this
repository and its releases, and the prediction log on the `exp2-ledger` branch. Making the repository private
(Settings → General → Danger zone) hides all of that; GitHub Pages can still publish from a private repository on
a paid plan.

API keys stay in GitHub's encrypted secrets and never appear on the site or in the database dump.
