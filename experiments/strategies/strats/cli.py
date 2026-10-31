"""python -m strats live --data data --state state/state.json --out site

Fetches prices, the S&P 500 list, earnings and insider filings, new 13Fs and Pelosi PTRs, moves every pot to the latest
close, replays the pots that have a history, and writes the pages. Prints a JSON report."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

from . import copy, copy_data, crypto, data, live, news, rules, site, stops, straddle


def closed(px: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop today's row while the US session is still open: Yahoo's latest bar is a live quote until the close settles."""
    now = now or pd.Timestamp.now(tz="America/New_York")
    if now.hour * 60 + now.minute < 16 * 60 + 30:
        return px[px.index < now.tz_localize(None).normalize()]
    return px


def run(data_dir: Path, state_f: Path, out: Path, fetch: bool = True) -> dict:
    rep: dict = {}
    members = data.sp500_members(data_dir) if fetch else (sorted(pd.read_csv(data_dir / "sp500.csv")["ticker"]) if (data_dir / "sp500.csv").exists() else [])
    key = os.environ.get("FINNHUB_API_KEY", "")
    if fetch and (data_dir / "copy").exists():
        rep["copy_data"] = copy_data.update(data_dir, pd.Timestamp.today().normalize())
    cp = copy.load(data_dir) if (data_dir / "copy").exists() else None
    tickers = sorted(set(rules.TREND_ETFS) | {m.replace(".", "-") for m in members} | (copy.tickers(cp) if copy.has_data(cp) else set()))
    if fetch:
        rep["prices"] = data.update_prices(data_dir, tickers)
        rep["prices"]["failed"] = rep["prices"]["failed"][:30]
    bars = data.load_bars(data_dir, tickers)
    px = closed(bars.pop("close"))
    spy = px["SPY"].dropna()
    d = spy.index[-1]
    px = px.loc[px.index.isin(spy.index)]
    rep["last_close"] = str(d.date())
    members = [m.replace(".", "-") for m in members]
    if fetch and key:
        for name, fn in (("earnings", lambda: data.update_earnings(data_dir, key, d)),
                         ("insiders", lambda: data.update_insiders(data_dir, key, members, d))):
            try:
                rep[name] = fn()
            except Exception as ex:  # noqa: BLE001 - a pot without fresh events keeps its holdings
                rep[name] = f"failed: {type(ex).__name__}: {str(ex)[:200]}"
    elif fetch:
        rep["finnhub"] = "FINNHUB_API_KEY not set: earnings and insider pots get no new events"
    read = lambda f: pd.read_csv(f) if f.exists() else pd.DataFrame()  # noqa: E731
    earnings, insiders = read(data_dir / "earnings.csv"), read(data_dir / "insiders.csv")
    if not earnings.empty:
        earnings["symbol"] = earnings["symbol"].astype(str).str.replace(".", "-")
    if not insiders.empty:
        insiders["symbol"] = insiders["symbol"].astype(str).str.replace(".", "-")
    if fetch:
        rep["btc"] = crypto.update(data_dir)
    state = live.load_state(state_f)
    btc = crypto.load(data_dir)
    straddle_in = None
    try:
        ranked = straddle.flags(bars, data.load_volume(data_dir, tickers), px, d, members)
        names = []
        for k in straddle.KEYS:
            last = (state.get("pots", {}).get(k) or [None])[-1]
            if not (last and last["date"] >= str(d.date())):
                names += [t for t in straddle.tickers_needed(last, ranked) if t not in names]
        chains = straddle.fetch_chains(names) if fetch else {}
        leans = news.leans(straddle.book_names(ranked, chains, d), d) if fetch and names else {}
        straddle_in = {"ranked": ranked, "chains": chains, "leans": leans}
        rep["straddle_news"] = leans or "no news leans (no ANTHROPIC_API_KEY, or nothing to buy)"
        rep["straddle_flags"] = {t: round(float(p), 4) for t, p in ranked.head(straddle.SCAN).items()}
        rep["straddle_chains"] = {t: (f"{c[2]}: {len(c[0])} contracts" if c else "none") for t, c in chains.items()}
    except Exception as ex:  # noqa: BLE001 - the other pots still trade
        rep["straddle_flags"] = f"failed: {type(ex).__name__}: {str(ex)[:300]}"
    rep["step"] = live.step(state, px, d, members, earnings, insiders, cp, btc, bars=bars, straddle_in=straddle_in)
    rep["what_moved"] = {str(x.date()): stops.explain(state, px, x) for x in px.index[-2:]}
    if os.environ.get("STOP_BACKTEST"):       # pull requests: how the stop levels would have done since 2019
        try:
            lv = [0.0, 0.03, 0.05, 0.08, 0.10]
            rep["stop_backtest"] = stops.backtest(px, bars, members, live.REPLAY_FROM, {"reversal": lv, "trend": [0.0, 0.02, 0.03, 0.04, 0.06]}, live.COST)
            rep["stop_backtest"] |= stops.backtest_btc(btc, [0.0, 0.04, 0.06, 0.08, 0.10])
        except Exception as ex:  # noqa: BLE001 - a report, not the run
            rep["stop_backtest"] = f"failed: {type(ex).__name__}: {str(ex)[:300]}"
    live.save_state(state_f, state)
    rp = None
    try:
        rp = live.replay_summary(live.replay(px, members, cp=cp))
        rep["replay"] = {k: {m: round(v, 4) for m, v in rp[k].items() if m != "years"} for k in rp if k in rules.STRATEGIES or k == "SPY"}
    except Exception as ex:  # noqa: BLE001 - the pages still build
        rep["replay"] = f"failed: {type(ex).__name__}: {str(ex)[:200]}"
    jf = data_dir / "crypto" / "journal.json"
    journal = json.loads(jf.read_text()) if jf.exists() else None
    rep["site"] = site.write(out, state, rp, {"members": len(members), "priced": int(px.loc[d].notna().sum())},
                             btc_trades=crypto.journal_rows(journal, btc), btc_journal=journal)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("live")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--state", type=Path, default=Path("state/state.json"))
    p.add_argument("--out", type=Path, default=Path("site"))
    p.add_argument("--no-network", action="store_true")
    q = sub.add_parser("quant", help="the Quant lab page: option chains to vol surfaces, market regimes, event-driven backtest (quant.py)")
    q.add_argument("--data", type=Path, default=Path("data"))
    q.add_argument("--options", type=Path, help="options_daily.csv from the options-data release (optional)")
    q.add_argument("--out", type=Path, default=Path("quant_site"))
    q.add_argument("--no-network", action="store_true")
    t = sub.add_parser("btc-tick", help="the 4-hour BTC run: update the bars and write down a trade when the rule flips")
    t.add_argument("--data", type=Path, default=Path("data"))
    t.add_argument("--journal", type=Path, default=Path("data/crypto/journal.json"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if a.cmd == "quant":
        from . import quant
        print(json.dumps(quant.run(a.data, a.options, a.out, fetch=not a.no_network), indent=1, default=str))
        return
    if a.cmd == "btc-tick":
        print(json.dumps(crypto.tick(a.data, a.journal, stop=stops.STOPS.get(crypto.KEY)), indent=1, default=str))
        return
    print(json.dumps(run(a.data, a.state, a.out, fetch=not a.no_network), indent=1, default=str))
