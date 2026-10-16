"""Daily run: update data, score news, score due predictions, retrain if due, predict, build pages.

    python -m exp2 run --state state --site site [--backtest] [--retrain] [--no-network]

Each step reports its outcome in state/last_run.json. A failing data source leaves the previous
data in place and shows as a warning on the page; it never fabricates numbers.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch

from . import backtest as bt_mod
from . import config, data, features, ledger, news, site
from .train import Fitted, fit

log = logging.getLogger("exp2")


def _step(report: dict, name: str, fn):
    t0 = time.time()
    try:
        res = fn()
        report["steps"][name] = {"status": "ok", "seconds": round(time.time() - t0, 1), **(res if isinstance(res, dict) else {})}
        return res
    except Exception as ex:  # noqa: BLE001 - every step reports and the run carries on where it can
        report["steps"][name] = {"status": "failed", "error": f"{type(ex).__name__}: {str(ex)[:300]}"}
        log.error("%s failed:\n%s", name, traceback.format_exc())
        return None


def news_tables(state: Path, sessions: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(all screened stories with their session, the capped subset that goes to Claude)."""
    raw = data.load_news(state)
    scr = news.screen(raw)
    scr["session"] = news.assign_sessions(scr["ts"], sessions) if len(scr) else pd.Series(dtype="datetime64[ns]")
    capped = news.cap_per_session(scr.dropna(subset=["session"]), config.MAX_HEADLINES_PER_DAY)
    return scr, capped


def run(state: Path, site_dir: Path, do_backtest: bool = False, force_retrain: bool = False, network: bool = True,
        scorer=None, cfg: config.Config = config.DEFAULT, backtest_kw: dict | None = None) -> dict:
    state.mkdir(parents=True, exist_ok=True)
    report = {"started": datetime.now(UTC).isoformat(timespec="seconds"), "steps": {}, "warnings": []}
    tickers = [cfg.target, *cfg.constituents, *cfg.related]

    if network:
        pr = _step(report, "prices", lambda: data.update_prices(state, tickers, cfg.data_start, data.secret("TIINGO_API_KEY")))
        if pr and pr["failed"]:
            report["warnings"].append("Prices could not be updated for " + ", ".join(sorted(pr["failed"]))
                                      + "; the last stored prices are used.")
        nw = _step(report, "news_fetch", lambda: data.update_news(state, list(cfg.constituents), data.secret("FINNHUB_API_KEY"),
                                                                    backfill_days=config.NEWS_BACKFILL_DAYS))
        if nw and nw["failed"]:
            report["warnings"].append("News could not be fetched for " + ", ".join(sorted(nw["failed"])) + ".")
        key = data.secret("ANTHROPIC_API_KEY")
        if scorer is None and key:
            scorer = news.HaikuScorer(key)
        elif scorer is None:
            report["warnings"].append("ANTHROPIC_API_KEY is not set, so new headlines are not scored.")

    prices = data.load_prices(state, tickers)
    if cfg.target not in prices:
        report["warnings"].append("No price history yet, so there is nothing to predict.")
        return _finish(state, site_dir, report, None, ledger.load(state / "ledger" / "predictions.csv"), None, prices, None, None)
    sessions = prices[cfg.target].index[prices[cfg.target].index >= pd.Timestamp(cfg.data_start)]

    scr, capped = news_tables(state, sessions)
    if network or scorer is not None:
        sc = _step(report, "news_score", lambda: news.score_pending(state, capped, scorer))
        if sc and sc.get("budget_stop"):
            report["warnings"].append(f"This month's ${config.CLAUDE_MONTHLY_BUDGET_USD:.0f} news-scoring budget is used up; "
                                      "remaining headlines are scored next month.")
    meta_f = state / "news" / "fetched.json"
    coverage = json.loads(meta_f.read_text()).get("coverage_start") if meta_f.exists() else None
    scores = news.load_scores(state)
    nf = news.session_features(scr, scores, sessions, coverage)

    df = features.build(prices, nf, cfg)
    led_path = state / "ledger" / "predictions.csv"
    led = ledger.load(led_path)
    led, n_scored = ledger.score(led, df, cfg.horizon)
    report["steps"]["score_outcomes"] = {"status": "ok", "scored": n_scored}

    # model: retrain weekly, or when the setup changed
    model_f = state / "models" / "current.pt"
    fitted = None
    as_of = len(df) - 1
    if model_f.exists() and not force_retrain:
        try:
            st = torch.load(model_f, weights_only=False)
            if st["info"]["config_digest"] == cfg.digest() and st["cols"] == features.feature_columns(df):
                trained_pos = df.index.searchsorted(pd.Timestamp(st["info"]["as_of"]))
                if as_of - trained_pos < cfg.model.retrain_every:
                    fitted = Fitted.from_state(st, cfg)
        except Exception as ex:  # noqa: BLE001
            report["warnings"].append(f"The saved model could not be loaded ({type(ex).__name__}); retraining.")
    if fitted is None:
        def _fit():
            f = fit(df, as_of, cfg)
            model_f.parent.mkdir(parents=True, exist_ok=True)
            torch.save(f.to_state(), model_f)
            return {"seconds": f.info["seconds"], "train_end": f.info["train_end"]}
        _step(report, "train", _fit)
        if model_f.exists():
            fitted = Fitted.from_state(torch.load(model_f, weights_only=False), cfg)
    else:
        report["steps"]["train"] = {"status": "skipped", "reason": f"model trained {fitted.info['as_of']} is current"}

    latest = None
    if fitted is not None:
        as_of_s = df.index[as_of].strftime("%Y-%m-%d")
        rows = fitted.predict_row(df, as_of)
        model_id = f"{cfg.digest()}-{fitted.info['as_of']}"
        led, added = ledger.append(led, as_of_s, rows, model_id, cfg.horizon, float(df["close"].iloc[as_of]))
        report["steps"]["predict"] = {"status": "ok" if added else "skipped", "as_of": as_of_s,
                                      **({} if added else {"reason": "this session already has predictions"})}
        cur = led[led["as_of"] == as_of_s]
        latest = {"as_of": as_of_s, "close": float(df["close"].iloc[as_of]), "rows": cur[["model", "target", "pred", "sd", "p_up", "lo80", "hi80"]].to_dict("records")}
    ledger.save(led, led_path)

    bt_f, bt_csv = state / "backtest.json", state / "backtest.csv"
    bt_summary = json.loads(bt_f.read_text()) if bt_f.exists() else None
    bt_df = pd.read_csv(bt_csv) if bt_csv.exists() else None
    if do_backtest or bt_summary is None or bt_summary.get("config_digest") != cfg.digest() or not bt_summary.get("complete"):
        def _bt():
            nonlocal bt_df, bt_summary
            bt_df, bt_summary = bt_mod.run(df, cfg, **(backtest_kw or {}))
            bt_summary["config_digest"] = cfg.digest()
            bt_df.to_csv(bt_csv, index=False, float_format="%.6g")
            bt_f.write_text(json.dumps(bt_summary, indent=1))
            return {"blocks": bt_summary["blocks"], "complete": bt_summary["complete"]}
        _step(report, "backtest", _bt)

    return _finish(state, site_dir, report, df, led, fitted, prices, (scr, scores), latest, bt_df, bt_summary)


def _finish(state, site_dir, report, df, led, fitted, prices, news_bits, latest, bt_df=None, bt_summary=None) -> dict:
    summary = {**report, "finished": datetime.now(UTC).isoformat(timespec="seconds"), "latest": latest,
               "forward_scorecard": ledger.scorecard(led) if len(led) else {},
               "ledger_rows": led.where(pd.notna(led), None).to_dict("records"),
               "model": fitted.info if fitted else None, "features": fitted.cols if fitted else [],
               "backtest": bt_summary, "claude_month_usd": news.month_spend(state, datetime.now(UTC).strftime("%Y-%m"))}
    if df is not None:
        tail = df["close"].iloc[-260:]
        summary["price_tail"] = [(d.strftime("%Y-%m-%d"), float(v)) for d, v in tail.items()]
        nd = df[news.NEWS_FEATURES].iloc[-130:] if "news_avail" in df else None
        if nd is not None:
            summary["news_daily"] = [{"date": d.strftime("%Y-%m-%d"), "avail": bool(r.news_avail), "count": float(r.news_count),
                                      "tone": float(r.news_tone), "tone_ewm": float(r.news_tone_ewm)} for d, r in nd.iterrows()]
    if news_bits and df is not None:
        scr, scores = news_bits
        last = df.index[-1]
        cur = scr[scr["session"] == last].copy()
        cur["relevance"] = cur["id"].map(lambda i: scores.get(i, {}).get("relevance"))
        cur["sentiment"] = cur["id"].map(lambda i: scores.get(i, {}).get("sentiment"))
        cur = cur.dropna(subset=["relevance"]).sort_values("relevance", ascending=False).head(12)
        summary["top_headlines"] = cur[["ticker", "headline", "relevance", "sentiment"]].to_dict("records")
        summary["news_today"] = {"session": last.strftime("%Y-%m-%d"), "stories": int((scr["session"] == last).sum())}
    (state / "last_run.json").write_text(json.dumps(report, indent=1, default=str))
    site.build(site_dir, summary)
    site.write_data(site_dir, led, bt_df, summary)
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="exp2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--state", type=Path, default=Path("state"))
    r.add_argument("--site", type=Path, default=Path("site"))
    r.add_argument("--backtest", action="store_true", help="re-run the walk-forward backtest")
    r.add_argument("--retrain", action="store_true", help="retrain even if the model is current")
    r.add_argument("--no-network", action="store_true", help="use stored data only")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    s = run(a.state, a.site, a.backtest, a.retrain, not a.no_network)
    bt = s.get("backtest") or {}
    print(json.dumps({"steps": s["steps"], "warnings": s["warnings"], "latest": s.get("latest"), "model": s.get("model"),
                      "backtest": {k: bt.get(k) for k in ("first", "last", "blocks", "complete", "scorecard_nonoverlap", "vs_zero", "prune_compare", "val_nll_full")}},
                     indent=1, default=str))
    # A trial run (pull request) that produced no model is a failure, not a quiet success.
    sys.exit(1 if os.environ.get("TRIAL") == "true" and not s.get("model") else 0)
