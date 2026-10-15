# Deploying on a Hostinger VPS

The whole stack runs from one command, `docker compose up -d --build`, and consists of:

| Container | Role | Public? |
|---|---|---|
| `caddy` | Reverse proxy, automatic HTTPS (Let's Encrypt) | ports 80/443 |
| `frontend` | Next.js production server (`node server.js`) | internal |
| `backend` | FastAPI on uvicorn (2 worker processes) | internal |
| `worker` | Scheduler + job runner (news, prices, scores, snapshots, outcomes, backups) | internal |
| `db` | PostgreSQL 17 with a persistent volume | internal, never published |
| `migrate` | One-shot: `alembic upgrade head` + idempotent bootstrap, runs before backend/worker | — |

Every long-running container has `restart: unless-stopped`, so the stack comes back after a reboot
as long as Docker itself starts at boot (it does by default on Ubuntu/Debian).

## Why a VPS (not shared hosting)

The monitor must keep working after you close the browser: a Python worker process runs all the
time, wakes up on a schedule (hourly news, daily prices after the close, nightly backups), keeps
state in PostgreSQL and makes outbound API calls. Shared web hosting only runs code while a web
request is being served, kills long-running processes, does not offer Docker, PostgreSQL or a
process supervisor, and usually cannot run Python background jobs at all. A VPS (or any machine
where you control long-lived processes) is required. Hostinger's KVM VPS plans work; 2 vCPU / 4 GB
RAM / 50 GB disk is comfortable (the KVM 1 plan with 1 vCPU / 4 GB also works for a few dozen
pairs). The images are built for x86-64 (amd64), which is what Hostinger KVM VPS use.

## 1. Connect to the VPS with SSH

In hPanel open **VPS → Overview** and note the server IP and root password (or add your SSH key
under **VPS → Settings → SSH keys**). Pick Ubuntu 24.04 as the OS template if you are starting
fresh.

```bash
ssh root@YOUR_SERVER_IP
# recommended: a non-root user for day-to-day work
adduser deploy && usermod -aG sudo deploy
```

Open the firewall for SSH and the web only (hPanel → VPS → Firewall, or ufw):

```bash
ufw allow OpenSSH && ufw allow 80/tcp && ufw allow 443/tcp && ufw allow 443/udp && ufw enable
```

Do not open 5432 (PostgreSQL), 3000 or 8000; nothing in the stack needs them reachable.

## 2. Install or verify Docker and Compose

Hostinger offers an "Ubuntu 24.04 with Docker" template; otherwise install Docker Engine from the
official repository:

```bash
docker --version && docker compose version   # both must work; Compose v2 is required
# if missing:
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker
usermod -aG docker deploy   # then log out and in again as deploy
```

## 3. Clone the repository

```bash
cd /opt
git clone https://github.com/YOUR_ACCOUNT/splittability-monitor.git
cd splittability-monitor
```

For a private repository, use a deploy key (`ssh-keygen -t ed25519`, add the public key under the
repository's **Settings → Deploy keys**) and clone with the `git@github.com:` URL.

## 4. Configure environment variables

```bash
cp .env.example .env
chmod 600 .env
nano .env
```

Required:

* `SITE_ADDRESS`: your domain or subdomain, e.g. `monitor.example.com`.
* `ACME_EMAIL`: e-mail for certificate expiry notices.
* `POSTGRES_PASSWORD`: `openssl rand -hex 24`.
* `SECRET_KEY`: `openssl rand -hex 32` (signs session cookies; at least 32 characters).
* `BOOTSTRAP_ADMIN_EMAIL` / `BOOTSTRAP_ADMIN_PASSWORD`: the first admin (12+ characters).

Data and AI (each optional; whatever is missing makes the affected score components
**unavailable**, never zero):

* `ANTHROPIC_API_KEY`: Claude news interpretation. Model and effort are `CLAUDE_MODEL` /
  `CLAUDE_EFFORT` and can be changed later in Settings. The monthly AI budget (default $50) is set in
  Settings → AI & budget.
* `TIINGO_API_KEY`: dividend-adjusted daily prices (recommended). `ENABLE_STOOQ=true` adds the free
  Stooq fallback (split-adjusted; recorded as price-return because dividends are not guaranteed).
* `SEC_USER_AGENT`: SEC requires a descriptive User-Agent with a contact address, e.g.
  `SplittabilityMonitor research you@example.com`. Enables filings and point-in-time fundamentals.
* `FINNHUB_API_KEY`, `ALPHAVANTAGE_API_KEY`: company news feeds.
* `APP_MODE`: start with `demo` to see the full application with SIMULATED data, or `production` to
  use only real providers. See "Switching from demo to production" below.

Keys are read only by the backend and worker containers; they are never sent to the browser and
never written to the database. Never commit `.env` (it is in `.gitignore`).

## 5. Build and start all services

```bash
docker compose up -d --build
docker compose ps
```

The first build takes a few minutes. `migrate` runs once and exits with code 0; the other
containers should show `healthy` after about a minute. In demo mode the first start also generates
the simulated data set (2–4 minutes; follow it with `docker compose logs -f migrate`).

## 6. Initialise the database

This happens automatically: the `migrate` service runs `alembic upgrade head` (schema, constraints,
immutability triggers) and `python -m app.cli init` (companies, the initial pairs with LOW/HD
deduplicated, model versions, schedules, the reported research benchmark and the bootstrap admin)
before the backend and worker start. It is idempotent and runs again on every `up`. To run it by
hand:

```bash
docker compose run --rm migrate
```

## 7. Create an administrator account

If you set `BOOTSTRAP_ADMIN_EMAIL`/`PASSWORD`, that account exists now; sign in and change the
password in Settings → System & account, then remove `BOOTSTRAP_ADMIN_PASSWORD` from `.env`. To
create another admin (or reset a password) at any time:

```bash
scripts/create-admin.sh you@example.com     # prompts for the password
```

Read-only "viewer" users can be added from Settings → System & account.

## 8. Configure the domain or subdomain and DNS

In the DNS zone of your domain (hPanel → Domains → DNS / Nameservers, or your registrar), create:

| Type | Name | Value | TTL |
|---|---|---|---|
| A | `monitor` (or `@` for the bare domain) | YOUR_SERVER_IP | 300 |
| AAAA (optional) | `monitor` | the VPS IPv6 address | 300 |

Set `SITE_ADDRESS=monitor.example.com` in `.env` to the same name. Check propagation with
`dig +short monitor.example.com` before the next step.

## 9. Enable HTTPS

Caddy obtains and renews a Let's Encrypt certificate automatically the first time the domain
resolves to the server and ports 80/443 are open. After changing `SITE_ADDRESS`:

```bash
docker compose up -d caddy
docker compose logs caddy | grep -i -E "certificate|error"
```

HTTP redirects to HTTPS, HSTS is enabled, and session cookies are `Secure`, `HttpOnly` and
`SameSite=Strict`. For a quick test without a domain, `SITE_ADDRESS=localhost` uses Caddy's
internal certificate authority (your browser will warn).

## 10. Check backend and worker health

```bash
docker compose ps                                   # all "healthy"
curl -s https://monitor.example.com/api/v1/health   # {"status":"ok",...,"mode":"production"}
```

In the dashboard, **Bot activity** shows the scheduler state, each job's next run (New York time),
the run history with durations, retries and errors, provider health, Claude model, tokens and
costs. **Settings → System & account → Health check** makes a live test call to every configured
provider. The worker container is marked unhealthy if it stops writing its heartbeat for 5 minutes.

API documentation (OpenAPI) is at `https://monitor.example.com/api/docs`.

## 11. Inspect logs

```bash
docker compose logs -f worker            # scheduler and jobs (JSON lines)
docker compose logs -f backend           # API requests with request ids
docker compose logs --since 1h caddy     # access log
docker compose logs migrate              # last migration/bootstrap
```

Logs are rotated by Docker's default json-file driver; to cap disk use add
`"log-opts": {"max-size": "20m", "max-file": "5"}` to `/etc/docker/daemon.json`.

## 12. Update without deleting historical data

```bash
cd /opt/splittability-monitor
scripts/deploy.sh
```

It pulls the latest code, takes a backup, rebuilds the images and restarts. Migrations run
automatically. Historical data lives in the `pgdata` Docker volume and is untouched by rebuilds.
**Never** run `docker compose down -v` (the `-v` deletes volumes, including the database).

## 13. Back up and restore

Automatic: the worker runs `pg_dump` every night at 02:00 New York time into the `backups` volume and
keeps 14 days (`BACKUP_RETENTION_DAYS`). It is visible on Bot activity and can be run there with
**Run now**.

On demand, and to copy the automatic backups to the host:

```bash
scripts/backup.sh          # writes ./backups/splittability-<UTC timestamp>.dump
```

Copy backups off the server regularly (for example `rsync -a backups/ you@other-host:sm-backups/`,
or Hostinger's VPS snapshot feature as an additional layer).

Restore (replaces all current data; asks for confirmation):

```bash
scripts/restore.sh backups/splittability-20261015T060000Z.dump
```

It stops the app containers, runs `pg_restore --clean --single-transaction`, re-applies any newer
migrations and starts everything again.

## Switching from demo to production

Demo mode fills the database with clearly labelled SIMULATED prices, news and predictions. Real
and simulated data are never mixed: in production mode the worker refuses to start while
simulated rows exist, and the dashboard says so.

```bash
nano .env                          # APP_MODE=production, add your API keys
scripts/switch-to-production.sh    # wipes simulated research data, restarts
```

The worker then fetches price history (from `history_start`, default 2009-01-01, enough for SI-v1's five-year window before the 2015 backtest), filings,
fundamentals and news on its schedule; use **Bot activity → Run now** on `ingest_prices`,
`ingest_fundamentals` and `ingest_news` to start immediately, then `compute_scores`.

## Troubleshooting

* **`migrate` exited with an error**: `docker compose logs migrate`. A wrong `POSTGRES_PASSWORD` after
  the volume was created is the usual cause; the password is fixed when the volume is first
  initialised.
* **Certificate not issued**: DNS must point at the VPS and ports 80/443 must be open (hPanel
  firewall and ufw).
* **Scores "unavailable"**: open the pair, then the score inspector; each component states why (for
  example "only 2 comparable fundamental metrics" or "AI not configured").
* **AI analysis stopped**: the monthly budget was reached; raise it in Settings and run
  `analyze_news`.
* **Login rate limited**: wait a minute; limits are per client IP and per backend process.
