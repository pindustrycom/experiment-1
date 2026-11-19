"""Combine every scored row, apply Holm across all of them, and write results.csv + table.md.

  python report.py OUT_DIR rows1.csv rows2.csv ...
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import evalkit as ek  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
df = pd.concat([pd.read_csv(f) for f in sys.argv[2:]], ignore_index=True)
df["tested"] = df.trades >= 10
df["beats_hold_sharpe"] = df.sharpe > df.hold_sharpe
df["net_positive"] = df.cagr > 0
t = df[df.tested].copy()
t["holm_p"] = ek.holm(t.p_random.to_numpy())
df = df.merge(t[["holm_p"]], left_index=True, right_index=True, how="left")
df["verdict"] = [ek.verdict(r, r["holm_p"] if not np.isnan(r["holm_p"]) else 1.0) if r["tested"] else "too few trades"
                 for _, r in df.iterrows()]
# A strategy written after the data it is tested on can only pass in-sample: report it, do not count it.
ins = df.source.str.contains("in-sample") & (df.verdict == "EDGE")
df.loc[ins, "verdict"] = "passes in-sample only (not counted)"
df.to_csv(out / "results.csv", index=False)

n = int(df.tested.sum())


def pct(x):
    return "n/a" if pd.isna(x) else f"{x:+.1%}"


lines = [f"Tested strategy x market rows (10+ trades): {n}", "",
         f"- raw p_random < 0.05 (beats random timing before any correction): {int((t.p_random < 0.05).sum())} "
         f"({(t.p_random < 0.05).mean():.0%}; chance alone gives ~5%)",
         f"- survive Holm across all {n}: {int((t.holm_p < 0.05).sum())}",
         f"- random-timing z >= 3.8 (project-wide bar): {int((t.z_random >= 3.8).sum())}",
         f"- net Sharpe above buy-and-hold: {int((t.sharpe > t.hold_sharpe).sum())}",
         f"- net CAGR above zero: {int((t.cagr > 0).sum())}; above buy-and-hold CAGR: {int((t.cagr > t.hold_cagr).sum())}",
         f"- EDGE (all checks): {int((df.verdict == 'EDGE').sum())}"
         + (f" (plus {int(ins.sum())} that pass only in-sample, not counted)" if ins.any() else ""), ""]
src = t.groupby("source").agg(rows=("strategy", "size"), median_net_cagr=("cagr", "median"),
                              median_gross_cagr=("gross_cagr", "median"), beat_random_raw=("p_random", lambda p: int((p < .05).sum())),
                              beat_hold_sharpe=("beats_hold_sharpe", "sum"), median_win_rate=("win_rate", "median"))
lines += ["| source | rows | median net CAGR | median gross CAGR | raw p<0.05 | Sharpe > hold | median win rate |", "|---|---|---|---|---|---|---|"]
for s, r in src.iterrows():
    lines.append(f"| {s} | {int(r.rows)} | {r.median_net_cagr:+.1%} | {pct(r.median_gross_cagr)} | {int(r.beat_random_raw)} | "
                 f"{int(r.beat_hold_sharpe)} | {r.median_win_rate:.0%} |")
lines += ["", "Top 25 by random-timing z (all rows):", "",
          "| source | strategy | asset | tf | trades | net CAGR | gross | Sharpe | hold Sharpe | z vs random | p | Holm p | z 2019-22 | z 2023-26 | alpha t | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in t.sort_values("z_random", ascending=False).head(25).iterrows():
    lines.append(f"| {r.source} | {r.strategy} | {r.asset} | {r.tf} | {r.trades} | {r.cagr:+.1%} | {pct(r.gross_cagr)} | {r.sharpe:.2f} | "
                 f"{r.hold_sharpe:.2f} | {r.z_random:+.1f} | {r.p_random:.3f} | {r.holm_p:.2f} | {r.z_random_h1:+.1f} | "
                 f"{r.z_random_h2:+.1f} | {r.alpha_t:+.1f} | {df.loc[_, 'verdict']} |")
(out / "table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
