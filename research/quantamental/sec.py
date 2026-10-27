"""Point-in-time annual fundamentals from SEC XBRL company facts (data.sec.gov), for slides 1 and 3.

Every reported number carries the date it was filed. A fiscal year's record becomes usable on the day its 10-K was
first filed, and every number in it (this year's and the earlier years it is compared with) is the latest value
filed on or before that day: restatements filed later are never seen early.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("quantamental.sec")
UA = os.environ.get("SEC_USER_AGENT") or "experiment-1 research (github.com/pindustrycom/experiment-1)"
_lock = threading.Lock()
_last = [0.0]


def get(url: str, rate: float = 8.0, tries: int = 4) -> requests.Response:
    """GET from sec.gov under its fair-access limit (10 requests a second), retrying throttles."""
    for k in range(tries):
        with _lock:
            wait = _last[0] + 1 / rate - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}, timeout=60)
            if r.status_code in (429, 503):
                time.sleep(2 ** (k + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(2 ** (k + 1))
    raise RuntimeError(f"failed: {url}")


def ticker_ciks() -> dict[str, int]:
    js = get("https://www.sec.gov/files/company_tickers.json").json()
    return {v["ticker"].upper().replace("-", "."): int(v["cik_str"]) for v in js.values()}


def company_facts(cik: int, cache: Path) -> dict | None:
    f = cache / f"{cik}.json.gz"
    if f.exists():
        return json.loads(gzip.decompress(f.read_bytes()))
    r = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
    if r.status_code != 200:
        return None
    cache.mkdir(parents=True, exist_ok=True)
    f.write_bytes(gzip.compress(r.content))
    return r.json()


# Each item: concepts in order of preference. A year takes the first concept that has a value for it.
DUR = {
    "ni": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "rev": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"],
    "gp": ["GrossProfit"],
    "shares": ["WeightedAverageNumberOfSharesOutstandingBasic"],
    "div": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "buyback": ["PaymentsForRepurchaseOfCommonStock"],
    "issue": ["ProceedsFromIssuanceOfCommonStock", "ProceedsFromStockOptionsExercised"],
}
INST = {
    "assets": ["Assets"],
    "ca": ["AssetsCurrent"],
    "cl": ["LiabilitiesCurrent"],
    "ltd": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "eq": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "liab": ["Liabilities"],
}
YEAR = pd.Timedelta(days=365)


def _rows(facts: dict, concept: str, taxonomy: str = "us-gaap") -> pd.DataFrame:
    c = facts.get("facts", {}).get(taxonomy, {}).get(concept)
    if not c:
        return pd.DataFrame()
    units = c["units"].get("USD") or c["units"].get("shares") or next(iter(c["units"].values()))
    df = pd.DataFrame(units)
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df["start"] = pd.to_datetime(df["start"]) if "start" in df else pd.NaT
    return df[["start", "end", "val", "filed", "form"]].dropna(subset=["val"]).sort_values("filed")


class Facts:
    def __init__(self, facts: dict):
        self.dur = {k: [d for c in v if not (d := self._annual(_rows(facts, c))).empty] for k, v in DUR.items()}
        self.inst = {k: [d for c in v if not (d := _rows(facts, c)).empty] for k, v in INST.items()}
        self.float = _rows(facts, "EntityPublicFloat", "dei")

    @staticmethod
    def _annual(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        days = (df["end"] - df["start"]).dt.days
        return df[(days >= 340) & (days <= 390)]

    def value(self, item: str, end: pd.Timestamp, asof: pd.Timestamp) -> float:
        for df in (self.dur.get(item) or self.inst.get(item) or []):
            m = df[((df["end"] - end).abs() <= pd.Timedelta(days=12)) & (df["filed"] <= asof)]
            if len(m):
                return float(m["val"].iloc[-1])
        return np.nan

    def years(self) -> pd.DataFrame:
        """Fiscal year ends and the day each year's 10-K was first filed."""
        rows = []
        for df in self.dur["ni"] + self.inst["assets"]:
            k = df[df["form"].str.startswith("10-K")]
            if "start" in k and k["start"].notna().any():
                k = k[k["start"].notna()]
            rows.append(k[["end", "filed"]])
        if not rows:
            return pd.DataFrame(columns=["end", "avail"])
        a = pd.concat(rows)
        a = a.groupby("end")["filed"].min().reset_index().rename(columns={"filed": "avail"})
        # a 10-K reports this year and last year: a year counts only from the 10-K filed for it (within 15 months)
        a = a[(a["avail"] - a["end"]).dt.days.between(20, 460)]
        return a.sort_values("end").reset_index(drop=True)


ITEMS = list(DUR) + list(INST)


def annual_table(facts: dict, lags: tuple[int, ...] = (0, 1, 2, 3, 4)) -> pd.DataFrame:
    """One row per fiscal year: items for this year (_0) and earlier years (_1.._4), as known on the 10-K's filing day."""
    F = Facts(facts)
    out = []
    for y in F.years().itertuples(index=False):
        row = {"end": y.end, "avail": y.avail}
        for lag in lags:
            e = y.end - lag * YEAR
            for it in ITEMS:
                row[f"{it}_{lag}"] = F.value(it, e, y.avail)
        fl = F.float[F.float["filed"] <= y.avail] if len(F.float) else F.float
        row["float_usd"] = float(fl["val"].iloc[-1]) if len(fl) else np.nan
        row["float_date"] = fl["end"].iloc[-1] if len(fl) else pd.NaT
        out.append(row)
    return pd.DataFrame(out)


def fundamentals(ciks: dict[str, int], cache: Path, out_csv: Path) -> pd.DataFrame:
    if out_csv.exists():
        return pd.read_csv(out_csv, parse_dates=["end", "avail", "float_date"])
    frames, miss = [], []
    for i, (t, cik) in enumerate(sorted(ciks.items())):
        js = company_facts(cik, cache)
        if not js:
            miss.append(t)
            continue
        try:
            a = annual_table(js)
        except Exception as e:  # noqa: BLE001 - one odd filer must not stop the run
            log.warning("%s: %s", t, e)
            miss.append(t)
            continue
        if len(a):
            a.insert(0, "ticker", t)
            a.insert(1, "cik", cik)
            frames.append(a)
        if i % 100 == 0:
            log.info("fundamentals %d/%d", i, len(ciks))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_csv, index=False)
    log.info("fundamentals: %d firm-years, %d tickers without facts", len(df), len(miss))
    return df
