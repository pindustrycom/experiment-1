"""Collect, score, build the state measurements, fit on 2018-2021, test on 2022-2025, write results.

    python -m exp3 run --data data --out results [--no-collect] [--no-score]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import collect, config, features, model, score

log = logging.getLogger("exp3")


def _step(report: dict, name: str, fn):
    t0 = time.time()
    try:
        res = fn()
        report["steps"][name] = {"status": "ok", "seconds": round(time.time() - t0, 1), **(res if isinstance(res, dict) else {"result": res})}
        return res
    except Exception as ex:  # noqa: BLE001 - each step reports and the run carries on where it can
        report["steps"][name] = {"status": "failed", "error": f"{type(ex).__name__}: {str(ex)[:300]}"}
        log.error("%s failed:\n%s", name, traceback.format_exc())
        return None


def load_prices(data: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for f in (data / "prices").glob("*.csv"):
        out[f.stem] = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
    return out


def coverage(items: pd.DataFrame, news: pd.DataFrame, reddit: pd.DataFrame, scores: dict) -> dict:
    out = {"collected": {}, "used": {}}
    for name, df in (("news", news), ("reddit", reddit)):
        if len(df):
            yr = pd.to_datetime(df["ts"], utc=True).dt.year
            out["collected"][name] = {int(k): int(v) for k, v in yr.value_counts().sort_index().items()}
            out["collected"][name + "_scored"] = int(df["id"].isin(scores).sum())
    if len(items):
        g = items.assign(year=pd.to_datetime(items["session"]).dt.year).groupby(["cat", "year"]).size()
        for (c, yr), v in g.items():
            out["used"].setdefault(c, {})[int(yr)] = int(v)
    return out


def signal_check(df: pd.DataFrame, cfg: config.Config) -> dict:
    """Rank correlation of each measurement with the next 5-session return, train vs test years, on non-overlapping days."""
    out = {}
    for period, (a, b) in {"train": (cfg.train_start, cfg.train_end), "test": (pd.Timestamp(cfg.train_end) + pd.Timedelta(days=1), cfg.test_end)}.items():
        d = df.loc[a:b].iloc[:: cfg.horizon]
        for c in features.PRICE_COLS + features.MARKET_COLS + features.SENT_COLS:
            x = d[[c, features.TARGET]].dropna()
            if len(x) >= 20 and x[c].std() > 0:
                out.setdefault(c, {})[period] = {"ic": round(float(x[c].rank().corr(x[features.TARGET].rank())), 3), "n": len(x)}
    return out


def run(data: Path, out_dir: Path, do_collect: bool = True, do_score: bool = True, cfg: config.Config = config.DEFAULT, scorer=None) -> dict:
    data.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"config_digest": cfg.digest(), "steps": {}, "warnings": []}
    if do_collect:
        tiingo = os.environ.get("TIINGO_API_KEY")
        if not (data / "prices" / f"{cfg.ticker}.csv").exists() or os.environ.get("EXP3_REFRESH_PRICES") == "1":
            _step(report, "prices", lambda: collect.prices(data, tiingo))
        if not (data / "shares.csv").exists():
            _step(report, "shares", lambda: collect.shares(data, os.environ.get("SEC_USER_AGENT", "")))
        if not (data / "analysts.csv").exists():
            _step(report, "analysts", lambda: collect.analysts(data))
        # Google News and Reddit are different hosts: collect them side by side
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(2) as ex:
            fn = ex.submit(collect.news, data)
            fr = ex.submit(collect.reddit, data)
            _step(report, "news", fn.result)
            _step(report, "reddit", fr.result)
        for k in ("news", "reddit"):
            st = report["steps"].get(k, {})
            if st.get("failed"):
                report["warnings"].append(f"{k}: {len(st['failed'])} windows could not be fetched; they count as unobserved.")
                st["failed"] = st["failed"][:10]

    news = collect.load_jsonl(data / "news.jsonl")
    reddit = collect.load_jsonl(data / "reddit.jsonl")
    to_score = score.items_to_score(news, reddit)
    if do_score:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if scorer is None and key:
            scorer = score.HaikuScorer(key)
        if scorer is None:
            report["warnings"].append("ANTHROPIC_API_KEY is not set, so items are not scored.")
        _step(report, "score", lambda: score.score_all(data, to_score, scorer))
    scores = score.load_scores(data)
    report["claude_usd_total"] = round(score.spent(data), 3)

    prices = load_prices(data)
    if cfg.ticker not in prices:
        report["warnings"].append("No NVDA prices, nothing to fit.")
        return report
    shares = pd.read_csv(data / "shares.csv")
    an = pd.read_csv(data / "analysts.csv") if (data / "analysts.csv").exists() else pd.DataFrame()
    sessions = prices[cfg.ticker].index
    items = features.category_items(news[news["id"].isin(to_score["id"])] if len(news) else news,
                                    reddit[reddit["id"].isin(to_score["id"])] if len(reddit) else reddit,
                                    an, scores, sessions, cfg.relevance_min)
    df = features.build(prices, shares, items, cfg)
    df.to_csv(out_dir / "features.csv", float_format="%.5g")
    report["coverage"] = coverage(items, news, reddit, scores)
    report["signal_check"] = signal_check(df, cfg)

    res = _step(report, "model", lambda: model.run(df, cfg))
    if res is not None:
        preds, info = res
        preds.to_csv(out_dir / "predictions.csv", index=False, float_format="%.5g")
        report["model"] = info
        report["results"] = model.summarize(preds, cfg.horizon)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=1, default=str))
    write_page(out_dir, report)
    return report


def _fmt(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{100 * x:+.1f}%" if pct else f"{x:.2f}"


def write_page(out_dir: Path, s: dict) -> None:
    res = s.get("results") or {}
    test = res.get("test", {})
    vz = res.get("test_vs_zero", {})
    labels = {"ekf": "EKF + SSM (price, market cap, sentiment)", "ekf_price": "EKF + SSM without sentiment", "ridge": "Ridge (all)",
              "ridge_price": "Ridge without sentiment", "momentum": "Momentum", "drift": "Drift", "zero": "No change"}
    rows = "".join(f"<tr><td>{labels.get(m, m)}</td><td>{_fmt(test[m].get('skill'), True)}</td><td>{_fmt((vz.get(m) or {}).get('t'))}</td>"
                   f"<td>{_fmt(test[m].get('hit'), True) if test[m].get('hit') is not None else '–'}</td><td>{_fmt(test[m].get('ic'))}</td>"
                   f"<td>{_fmt(test[m].get('cover80'), True)}</td></tr>" for m in labels if m in test)
    sa = res.get("sentiment_adds", {})
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Experiment 3: NVDA</title><style>body{{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#1d2433;background:#fff}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}td,th{{border-bottom:1px solid #e3e6ec;padding:6px 8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}
.note{{color:#5b6475;font-size:13px}}</style></head><body>
<h1>Experiment 3: NVDA, EKF + SSM with sentiment by category</h1>
<p>Trained on {s.get('model', {}).get('split', {}).get('tr0', '?')} to {s.get('model', {}).get('split', {}).get('tr1', '?')}, then frozen and tested on
{s.get('model', {}).get('split', {}).get('te0', '?')} to {s.get('model', {}).get('split', {}).get('te1', '?')}. Target: NVDA's log return over the next 5 sessions.
Scored on non-overlapping weeks. Paper research only.</p>
<table><tr><th>Model (test years)</th><th>Skill vs no change</th><th>t vs no change</th><th>Direction hit</th><th>Rank corr.</th><th>80% band coverage</th></tr>{rows}</table>
<p class="note">Skill is the error reduction against predicting no change; t below -2 means significantly better than no change.</p>
<p>Does sentiment add anything? EKF with vs without sentiment: t = {_fmt((sa.get('ekf_vs_ekf_price') or {}).get('t'))};
ridge with vs without: t = {_fmt((sa.get('ridge_vs_ridge_price') or {}).get('t'))} (negative = sentiment helps).</p>
<p class="note">Claude spend so far: ${s.get('claude_usd_total', 0):.2f}. Data: <a href="summary.json">summary.json</a>, <a href="predictions.csv">predictions.csv</a>,
<a href="features.csv">features.csv</a>.</p></body></html>"""
    (out_dir / "index.html").write_text(html)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="exp3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", type=Path, default=Path("data"))
    r.add_argument("--out", type=Path, default=Path("results"))
    r.add_argument("--no-collect", action="store_true")
    r.add_argument("--no-score", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    s = run(a.data, a.out, not a.no_collect, not a.no_score)
    print(json.dumps({k: s.get(k) for k in ("steps", "warnings", "claude_usd_total", "coverage", "results", "signal_check")}, indent=1, default=str))
    sys.exit(0 if s.get("results") else 1)
