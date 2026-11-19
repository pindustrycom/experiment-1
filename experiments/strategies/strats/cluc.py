"""Crypto dip-buyer pot: ClucHAnix_5m (ssssi/freqtrade_strs) on a 12-coin basket, the one open-source bot in
research/bot_edge that beat random timing and holding out of sample (2023-01 to 2026-10, 6 slots, 10 bp a side:
+21.7%/yr, Sharpe 1.64, worst drop -11.7%, against the equal-weight basket +24.1%/yr, Sharpe 0.66, -68.6%).

The strategy is run by freqtrade itself (its backtester, offline, cluc_ftrun.py), on Binance spot 5-minute bars of the
12 USDT pairs, from the ORIGIN close (live.ORIGIN, 16:00 New York on 2026-10-16) to the last finished bar. Up to SLOTS
coins are held at once, each with an equal share of the pot's cash; freqtrade fills at the next bar's open, FEE a side.
Because a backtest only uses finished bars, running it over the bars so far is what trading it on every 5-minute bar
would have done: past trades never change when new bars arrive.

cluc-5m.yml runs every hour: it brings the bars up to date (Binance's public market-data API, or its daily files when
the API is unreachable), reruns the backtest and saves freqtrade's equity (cash plus each coin at the bar's price, every
5 minutes) and trade list to the encrypted "cluc-live" release. The daily strategies run reads that equity: the pot's
return between two US closes is the equity at the second over the equity at the first.

Paper only: no exchange keys, no orders.
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)
KEY = "cluc_basket"
KEYS = (KEY,)
STRATEGY = "ClucHAnix_5m"
SSSSI_SHA = "3ed699f6773bd3db4526eb31bd6d459b95fd2bc2"   # ssssi/freqtrade_strs, the commit research/bot_edge tested
BASES = ("BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LTC", "LINK", "AVAX", "DOT", "TRX", "ATOM")
SLOTS, FEE, WALLET = 6, 0.001, 1000.0
BAR = pd.Timedelta(minutes=5)
START = pd.Timestamp("2026-10-16 20:00")       # live.ORIGIN's US close in UTC: the pot's first moment
SEED = START - pd.Timedelta(days=9)            # the 1-hour filter needs 168 hours of bars before START
API = "https://data-api.binance.vision/api/v3/klines"
DAILY = "https://data.binance.vision/data/spot/daily/klines/{s}USDT/5m/{s}USDT-5m-{d}.zip"
UA = {"User-Agent": "experiment-1 research (https://github.com/pindustrycom/experiment-1)"}
COLS = ["open", "high", "low", "close", "volume"]


def _dir(data: Path) -> Path:
    return data / "cluc"


def bars_file(data: Path, base: str) -> Path:
    return _dir(data) / "bars" / f"{base}.csv"


def load_bars(data: Path, base: str) -> pd.DataFrame:
    f = bars_file(data, base)
    if not f.exists():
        return pd.DataFrame(columns=COLS, dtype=float, index=pd.DatetimeIndex([], name="t"))
    return pd.read_csv(f, index_col="t", parse_dates=["t"]).sort_index()


def _frame(rows: list) -> pd.DataFrame:
    d = pd.DataFrame([r[:6] for r in rows], columns=["t", *COLS])
    t = d["t"].astype("int64")
    d["t"] = pd.to_datetime(t.where(t > 10 ** 14, t * 1000), unit="us")     # ms, or us in the 2025+ daily files
    return d.set_index("t").astype(float)


def _api(base: str, since: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
    frames, t = [], since
    while t < now:
        r = requests.get(API, params={"symbol": f"{base}USDT", "interval": "5m", "limit": 1000,
                                      "startTime": int(t.timestamp() * 1000)}, headers=UA, timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        f = _frame(rows)
        frames.append(f)
        if len(rows) < 1000:
            break
        t = f.index[-1] + BAR
        time.sleep(0.2)
    return pd.concat(frames) if frames else pd.DataFrame(columns=COLS, dtype=float)


def _daily(base: str, since: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
    """Binance's daily 5-minute files (published the next day) for each whole day from since's day to yesterday."""
    frames = []
    for day in pd.date_range(since.normalize(), now.normalize() - pd.Timedelta(days=1), freq="D"):
        r = requests.get(DAILY.format(s=base, d=day.strftime("%Y-%m-%d")), headers=UA, timeout=60)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        raw = pd.read_csv(z.open(z.namelist()[0]), header=None)
        frames.append(_frame(raw.to_numpy().tolist()))
    return pd.concat(frames) if frames else pd.DataFrame(columns=COLS, dtype=float)


def update(data: Path, now: pd.Timestamp | None = None) -> dict:
    """Add the bars each coin is missing, from its last cached bar (or SEED) up to the last finished one. A coin whose
    sources both fail keeps its cache; the cache only grows, so the backtest's past stays the same."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    rep = {}
    for b in BASES:
        old = load_bars(data, b)
        since = old.index[-1] + BAR if len(old) else SEED
        new, src = None, "none"
        for name, fn in (("binance api", _api), ("binance daily files", _daily)):
            try:
                new, src = fn(b, since, now), name
                break
            except Exception as ex:  # noqa: BLE001 - try the next source
                log.warning("%s bars from %s failed: %s", b, name, ex)
        if new is not None and len(new):
            new = new[(new.index >= since) & (new.index + BAR <= now)]
            d = pd.concat([old, new[~new.index.isin(old.index)]]).sort_index() if len(old) else new.sort_index()
            bars_file(data, b).parent.mkdir(parents=True, exist_ok=True)
            d.to_csv(bars_file(data, b), index_label="t")
        else:
            d = old
        rep[b] = {"source": src, "added": 0 if new is None else len(new), "last": str(d.index[-1]) if len(d) else None}
    return rep


def common_end(data: Path) -> pd.Timestamp | None:
    """The last bar every coin has: the backtest stops there, so no coin is traded on a bar the others lack."""
    ends = [load_bars(data, b).index for b in BASES]
    return min(i[-1] for i in ends) if all(len(i) for i in ends) else None


def ft_datadir(data: Path, out: Path, end: pd.Timestamp) -> None:
    """The cached bars in freqtrade's layout (BASE_USDT-5m.feather and the 1-hour bars its filter reads)."""
    out.mkdir(parents=True, exist_ok=True)
    for b in BASES:
        d = load_bars(data, b)
        d = d[d.index <= end]
        for tf, rule in (("5m", None), ("1h", "1h")):
            x = d if rule is None else d.resample(rule, label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
            if tf == "1h":
                x = x[x.index + pd.Timedelta(hours=1) <= end + BAR]      # whole hours only
            f = x[COLS].astype(float).copy()
            f.index = f.index.tz_localize("UTC")
            f.rename_axis("date").reset_index().to_feather(out / f"{b}_USDT-{tf}.feather")


def config(datadir: Path) -> dict:
    return {"max_open_trades": SLOTS, "stake_currency": "USDT", "stake_amount": "unlimited", "tradable_balance_ratio": 1.0,
            "dry_run": True, "dry_run_wallet": WALLET, "trading_mode": "spot", "margin_mode": "", "fee": FEE,
            "timeframe": "5m", "datadir": str(datadir), "dataformat_ohlcv": "feather",
            "exchange": {"name": "binance", "key": "", "secret": "", "pair_whitelist": [f"{b}/USDT" for b in BASES],
                         "pair_blacklist": []},
            "pairlists": [{"method": "StaticPairList"}],
            "entry_pricing": {"price_side": "other", "use_order_book": False, "price_last_balance": 0.0},
            "exit_pricing": {"price_side": "other", "use_order_book": False},
            "internals": {"process_throttle_secs": 5}}


def backtest(data: Path, strategy_dir: Path, work: Path) -> dict:
    """Run freqtrade over the cached bars from START and save its equity and trades in data/cluc (needs freqtrade, which
    only cluc-5m.yml installs)."""
    end = common_end(data)
    if end is None or end <= START + pd.Timedelta(hours=1):
        return {"backtest": "not enough bars yet"}
    shutil.rmtree(work, ignore_errors=True)
    ft_datadir(data, work / "data", end)
    (work / "cfg.json").write_text(json.dumps(config(work / "data"), indent=1))
    tr = f"{int(START.tz_localize('UTC').timestamp())}-{int((end + BAR).tz_localize('UTC').timestamp())}"
    cmd = [sys.executable, "-m", "strats.cluc_ftrun", "backtesting", "-c", str(work / "cfg.json"), "-s", STRATEGY,
           "--strategy-path", str(strategy_dir), "--timeframe", "5m", "--timerange", tr, "--export", "trades",
           "--cache", "none", "--backtest-directory", str(work / "res"), "--userdir", str(work),
           "--datadir", str(work / "data")]
    env = dict(os.environ, FT_BASES=",".join(BASES))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=Path(__file__).resolve().parent.parent)
    zips = sorted(work.glob("res-*.zip"))           # --backtest-directory is used as a file name prefix
    if r.returncode != 0 or not zips:
        raise RuntimeError(f"freqtrade failed ({r.returncode}): {r.stderr[-2000:]}")
    eq, trades = parse(zips[-1])
    _dir(data).mkdir(parents=True, exist_ok=True)
    eq.to_csv(_dir(data) / "equity.csv", index_label="t")
    (_dir(data) / "trades.json").write_text(json.dumps({"end": str(end), "trades": trades}, indent=1))
    closed = [t for t in trades if t["open"] is False]
    return {"backtest": "ok", "last_bar": str(end), "equity": round(float(eq["equity"].iloc[-1]), 2),
            "trades_closed": len(closed), "open": [t["coin"] for t in trades if t["open"]]}


def parse(zpath: Path) -> tuple[pd.DataFrame, list[dict]]:
    """freqtrade's export: the wallet (cash plus each coin at the bar's open, captured as each 5-minute bar starts) as
    equity and per-coin value, and the trades (the ones still open at the end are force-closed by the backtest, so they
    are marked open here)."""
    z = zipfile.ZipFile(zpath)
    res = json.loads(z.read(next(n for n in z.namelist() if n.endswith(".json") and not n.endswith("_config.json"))))
    s = next(iter(res["strategy"].values()))
    w = pd.read_feather(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith("_wallet.feather")))))
    w["date"] = pd.to_datetime(w["date"], utc=True).dt.tz_localize(None)
    val = w.pivot_table(index="date", columns="currency", values="total_quote", aggfunc="sum").fillna(0.0)
    eq = pd.DataFrame({"equity": val.sum(axis=1)})
    for b in BASES:
        eq[b] = (val[b] / eq["equity"]).round(6) if b in val.columns else 0.0
    trades = []
    for t in s["trades"]:
        still = t.get("exit_reason") == "force_exit"
        trades.append({"coin": t["pair"].split("/")[0], "open_date": str(t["open_date"])[:16], "open_rate": t["open_rate"],
                       "close_date": None if still else str(t["close_date"])[:16],
                       "close_rate": None if still else t["close_rate"],
                       "profit": None if still else round(float(t["profit_ratio"]), 5),
                       "exit": None if still else t.get("exit_reason"), "stake": round(float(t["stake_amount"]), 2),
                       "open": still})
    return eq, trades


def load_equity(data: Path) -> pd.DataFrame | None:
    f = _dir(data) / "equity.csv"
    return pd.read_csv(f, index_col="t", parse_dates=["t"]).sort_index() if f.exists() else None


def load_trades(data: Path) -> dict | None:
    f = _dir(data) / "trades.json"
    return json.loads(f.read_text()) if f.exists() else None


def mark_time(us_close: pd.Timestamp) -> pd.Timestamp:
    """16:00 New York on the session day us_close, in UTC."""
    return (us_close.normalize() + pd.Timedelta(hours=16)).tz_localize("America/New_York").tz_convert("UTC").tz_localize(None)


def mark(eq: pd.DataFrame, us_close: pd.Timestamp) -> pd.Timestamp | None:
    """The last equity capture at or before 16:00 New York on us_close (a capture at time t values the pot at t). At
    START itself (the ORIGIN close) nothing has been captured yet, and the pot is its starting cash."""
    t = mark_time(us_close)
    done = eq.index[eq.index <= t]
    return done[-1] if len(done) else (START if t >= START else None)


def _equity_at(eq: pd.DataFrame, t: pd.Timestamp) -> float | None:
    if t <= START:
        return WALLET
    return float(eq.at[t, "equity"]) if t in eq.index else None


def step(rows: list[dict], last: dict | None, eq: pd.DataFrame | None, d: pd.Timestamp, ds: str) -> str:
    """Move the pot to the close of d: freqtrade's equity at the capture marked for d over the one marked last time."""
    b = mark(eq, d) if eq is not None and len(eq) else None
    if b is None:
        if not last:
            return "no ClucHAnix equity yet: starts once cluc-5m.yml has run"
        ret, w, at = 0.0, last["w"], last.get("bar")
    else:
        prev = pd.Timestamp(last["bar"]) if last and last.get("bar") else None
        e0, e1 = (_equity_at(eq, prev) if prev is not None else None), _equity_at(eq, b)
        ret = e1 / e0 - 1 if e0 and e1 else 0.0
        w = ({c: round(float(eq.at[b, c]), 4) for c in BASES if c in eq.columns and eq.at[b, c] > 0}
             if b in eq.index else {})
        at = str(b)
    idx = (last["index"] if last else 1.0) * (1 + ret)
    rows.append({"date": ds, "ret": ret, "index": idx, "w": w, "reb": False, "bar": at})
    return f"{ds}: {ret:+.4%}, {', '.join(w) if w else 'cash'} at {at} UTC"


def position_text(row: dict | None) -> str:
    if not row:
        return "No position yet: the pot starts once the 5-minute runs have its first equity."
    w = row.get("w") or {}
    if not w:
        return f"All cash at {row.get('bar', '?')} UTC: no coin has dipped under its band."
    legs = ", ".join(f"{c} {x:.0%}" for c, x in sorted(w.items(), key=lambda kv: -kv[1]))
    return f"Holding {legs} of the pot at {row.get('bar', '?')} UTC; the rest in cash."


def trade_rows(trades: dict | None, n: int = 12) -> list[dict]:
    """The latest trades, newest first (open ones first)."""
    if not trades:
        return []
    t = trades.get("trades", [])
    return ([x for x in t if x["open"]][::-1] + [x for x in t if not x["open"]][::-1])[:n]


def main() -> None:
    """python -m strats.cluc --data DIR --strategy-path DIR: the hourly run (needs only pandas, requests and freqtrade)."""
    import argparse
    ap = argparse.ArgumentParser(description="ClucHAnix pot: update the 12 coins' 5-minute bars and rerun freqtrade")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--strategy-path", type=Path, required=True, help="folder holding ClucHAnix_5m.py")
    ap.add_argument("--work", type=Path, default=Path("/tmp/cluc_work"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    rep = {"bars": update(a.data)}
    rep |= backtest(a.data, a.strategy_path, a.work)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
