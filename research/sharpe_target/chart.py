"""Video-style chart: $5,000 in each book vs SPY and the cloud of 200 random 25-stock books. Args: out_dir png"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT, PNG = Path(sys.argv[1]), sys.argv[2]
b = pd.read_csv(OUT / "books_daily.csv", index_col=0, parse_dates=True)
rnd = pd.read_parquet(OUT / "random_daily.parquet")
SER = [("LGBM predicts return", "#2a78d6"), ("LGBM predicts Sharpe", "#eb6834"),
       ("LSTM predicts return", "#1baf7a"), ("LSTM predicts Sharpe", "#eda100"), ("SPY", "#0b0b0b")]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
fig.patch.set_facecolor("#fcfcfb")
for ax, (a, z, title, log) in zip(axes, [("2004-12-31", "2025-06-30", "Walk-forward 2005 to mid-2025 (log scale)", True),
                                         ("2025-06-30", "2026-09-30", "Sealed holdout, Jul 2025 to Sep 2026 (scored once)", False)]):
    sl = (b.index > a) & (b.index <= z)
    ax.set_facecolor("#fcfcfb")
    rw = 5000 * (1 + rnd[sl]).cumprod()
    ax.plot(rw.index, rw.to_numpy(), color="#b9b8b2", lw=0.5, alpha=0.35)
    ax.plot(rw.index, rw.median(axis=1), color="#52514e", lw=1.2, ls="--")
    ax.annotate("random median", (rw.index[-1], rw.median(axis=1).iloc[-1]), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color="#52514e", va="center")
    for name, c in SER:
        eq = 5000 * (1 + b.loc[sl, name]).cumprod()
        ax.plot(eq.index, eq, color=c, lw=2 if name != "SPY" else 1.6, label=f"{name}: ${eq.iloc[-1]:,.0f}")
    if log:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=11, color="#0b0b0b", loc="left")
    ax.grid(color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b9b8b2")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator(maxticks=6))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y" if log else "%b %Y"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
fig.suptitle("$5,000 in 25 stocks picked every 3 months, point-in-time S&P 500, 5 bp costs. Grey: 200 random 25-stock books",
             fontsize=10, color="#52514e", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 0.97, 0.95))
fig.savefig(PNG, dpi=150, facecolor=fig.get_facecolor())
