"""Survivorship-free check: the published factor series, built by their authors from every US stock (CRSP).

- AQR's Quality Minus Junk factor (USA), the paper's own series, monthly since 1957.
- Ken French's RMW (robust minus weak profitability) and HML (value), the closest public cousins of the quality and
  Piotroski ideas.
Each is split at its paper's publication and at 2010 (the start of our own backtest), so we see whether it kept working.
"""
from __future__ import annotations

import io
import logging

import pandas as pd
import requests

import stats as st

log = logging.getLogger("quantamental.factors")
AQR = "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Monthly.xlsx"


def aqr_qmj() -> pd.Series | None:
    try:
        r = requests.get(AQR, headers={"User-Agent": "Mozilla/5.0"}, timeout=90)
        r.raise_for_status()
        raw = pd.read_excel(io.BytesIO(r.content), sheet_name="QMJ Factors", header=None)
        head = raw.index[raw.iloc[:, 0].astype(str).str.upper().str.strip() == "DATE"][0]
        df = raw.iloc[head + 1:].copy()
        df.columns = [str(c).strip() for c in raw.iloc[head]]
        df = df.rename(columns={df.columns[0]: "date"}).dropna(subset=["date"])
        s = pd.Series(pd.to_numeric(df["USA"], errors="coerce").values, index=pd.to_datetime(df["date"], format="mixed")).dropna()
        s.index = s.index + pd.offsets.MonthEnd(0)
        return s
    except Exception as e:  # noqa: BLE001 - optional check
        log.warning("AQR QMJ: %s", e)
        return None


def periods(s: pd.Series, published: str) -> dict:
    out = {}
    for lab, a, b in (("before publication", None, published), ("after publication", published, None),
                      ("2010-2026 (our test window)", "2010-01-01", None), ("2018-2026", "2018-01-01", None)):
        part = s[a:b].dropna()
        if len(part) > 24:
            out[lab] = {k: v for k, v in st.summary(part, ppy=12).items() if k in ("start", "end", "ann", "vol", "sharpe", "t", "max_dd")}
    return out


def run(F: pd.DataFrame | None) -> dict:
    res = {}
    q = aqr_qmj()
    if q is not None:
        res["AQR QMJ factor (USA), published 2013 working paper"] = periods(q, "2013-06-30")
    if F is not None:
        res["French RMW profitability (Novy-Marx 2013, FF 2015)"] = periods(F["RMW"], "2013-01-31")
        res["French HML value (Fama-French 1992)"] = periods(F["HML"], "1992-06-30")
    return res


def table(res: dict) -> str:
    lines = ["| Factor | Period | Per year | Vol | Sharpe | t | Worst drop |", "|---|---|---|---|---|---|---|"]
    for f, ps in res.items():
        for p, v in ps.items():
            lines.append(f"| {f} | {p} ({v['start'][:4]}-{v['end'][:4]}) | {v['ann']:+.1%} | {v['vol']:.1%} | {v['sharpe']:.2f} | {v['t']:.1f} | {v['max_dd']:.0%} |")
    return "\n".join(lines) + "\n"



