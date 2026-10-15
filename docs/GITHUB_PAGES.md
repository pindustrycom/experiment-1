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
   | `TIINGO_API_KEY` | secret | Dividend-adjusted prices (free key at tiingo.com). Without it, Stooq is used (no key, price return only). |
   | `ANTHROPIC_API_KEY` | secret | Claude reads the news (N and P). Costs money per call; capped by the monthly budget (default $50). |
   | `FINNHUB_API_KEY`, `ALPHAVANTAGE_API_KEY` | secret | Company news feeds (free keys). |

3. **Actions → Update dashboard → Run workflow** for the first run. After that it runs every weekday
   evening on its own.

SI-v2 needs at least 60% of its weight available, including the market component. Market,
fundamentals and structure together are 55%, so SI-v2 shows as unavailable until news analysis
(`ANTHROPIC_API_KEY` plus at least one news source) is on. SI-v1 needs only prices.

## Privacy

GitHub Pages sites are public, even from a private repository, and the dashboard has no login. It
contains only scores, predictions, public prices and links to public news. API keys stay in
GitHub's encrypted secrets and never appear on the site or in the database dump.
