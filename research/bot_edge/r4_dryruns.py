"""Forward evidence: TheoBrigitte/freqtrade keeps the trade databases of his own dry runs (paper trading live on
Binance/KuCoin prices, 2024-08 to 2026-08). Summarise each: closed trades, win rate, profit on the 1,000 USDT dry-run
wallet (closed trades only; trades still open at the end are left out, so losses are if anything understated).

  python r4_dryruns.py THEO_REPO_DIR OUT.csv
"""
import glob
import json
import os
import sqlite3
import sys

import pandas as pd

repo, out = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(f"{repo}/strategies/**/dry-run/**/*.sqlite", recursive=True)):
    t = pd.read_sql("select pair, is_open, open_date, close_date, close_profit, close_profit_abs, strategy, leverage, is_short "
                    "from trades", sqlite3.connect(f))
    if t.empty:
        continue
    try:
        wallet = json.load(open(os.path.join(os.path.dirname(f), "config_base.json"))).get("dry_run_wallet")
    except (OSError, ValueError):
        wallet = None
    t["open_date"], t["close_date"] = pd.to_datetime(t.open_date), pd.to_datetime(t.close_date)
    cl = t[t.is_open == 0]
    days = max((t.close_date.max() - t.open_date.min()).days, 1)
    ret = cl.close_profit_abs.sum() / wallet if isinstance(wallet, (int, float)) else None
    rows.append({"run": os.path.relpath(os.path.dirname(f), f"{repo}/strategies"), "strategy": t.strategy.iloc[0],
                 "start": str(t.open_date.min().date()), "days": days, "trades": len(cl), "still_open": int(t.is_open.sum()),
                 "win_rate": float((cl.close_profit > 0).mean()), "profit_usdt": round(float(cl.close_profit_abs.sum()), 2),
                 "wallet": wallet, "ret": ret, "per_month": (1 + ret) ** (30.44 / days) - 1 if ret is not None else None,
                 "max_leverage": float(t.leverage.max()), "short_trades": int(t.is_short.sum())})
pd.DataFrame(rows).to_csv(out, index=False)
print(pd.DataFrame(rows).to_string())
