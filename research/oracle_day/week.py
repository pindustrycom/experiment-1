"""Perfect-hindsight week: $20 reinvested day after day (cash overnight), each day switching to whichever ticker rises.

python week.py DATA_DIR   (every regular session found in DATA_DIR/<TICKER>/bars.csv). Paper research only.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from oracle import COSTS_BP, FREQS, START, load, oracle, path, switch_oracle

files = sorted(Path(sys.argv[1]).glob("*/bars.csv"))
days = sorted(set(pd.to_datetime(pd.read_csv(files[0])["date"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")))
out, wealth, best_single = {"start_usd": START, "days": []}, {f"{f}_{c}bp": START for f in FREQS for c in COSTS_BP}, {}
for day in days:
    series = {f.parent.name: load(f, day) for f in files}
    series = {t: s for t, s in series.items() if len(s)}
    if len(series) < 5:
        continue
    idx = sorted(set.intersection(*[set(s.index) for s in series.values()]))
    al = {t: s.reindex(idx).ffill().bfill() for t, s in series.items()}
    rec = {"day": day, "spy_day_pct": round(100 * (series["SPY"].iloc[-1] / series["SPY"].iloc[0] - 1), 2)}
    for fq in FREQS:
        ps = {t: path(s, fq) for t, s in al.items()}
        for c in COSTS_BP:
            m = switch_oracle(ps, c)
            wealth[f"{fq}_{c}bp"] *= m
            rec[f"{fq}_{c}bp_day_x"] = round(m, 3)
            rec[f"{fq}_{c}bp_end_usd"] = round(wealth[f"{fq}_{c}bp"], 2)
    singles = {t: oracle(path(s, "1min"), 1)[0] for t, s in series.items()}
    top = max(singles, key=singles.get)
    rec["best_single_1min_1bp"] = [top, round(singles[top], 3)]
    for t, m in singles.items():
        best_single[t] = best_single.get(t, START) * m
    out["days"].append(rec)
    print(day, rec["spy_day_pct"], "switch 1m 0bp x", rec["1min_0bp_day_x"], "->", rec["1min_0bp_end_usd"],
          "| 1bp", rec["1min_1bp_end_usd"], "| 5m", rec["5min_0bp_end_usd"], "| best single", rec["best_single_1min_1bp"])
out["final_usd"] = {k: round(v, 2) for k, v in wealth.items()}
out["single_ticker_week_1min_1bp_usd"] = dict(sorted(((t, round(v, 2)) for t, v in best_single.items()), key=lambda x: -x[1]))
print(json.dumps(out["final_usd"], indent=1))
print(json.dumps(out["single_ticker_week_1min_1bp_usd"]))
Path("week_result.json").write_text(json.dumps(out, indent=1))
