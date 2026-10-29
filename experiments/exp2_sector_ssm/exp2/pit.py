"""Point-in-time S&P 500 membership, and prices of the companies that left the index, for the papers-data cache.

Source: the public dataset github.com/Johnbrick123/sp500-data. Membership 1970 to today as [start, end) spans per label;
every member priced on every day it was a member (companies that were taken over or went bankrupt recovered from Tiingo,
CRSP and archives); a dead company's label carries a -YYYYMM suffix, so a reused ticker never points at the wrong firm.
Rebuilt equal weight from this data, the index tracks RSP with a 0.994 daily correlation (2004-2026); today's members alone
would have earned about 2.5%/yr more. That gap is the survivorship bias this module removes.

refresh() writes, in the layout the rest of exp2 already reads:
  sp500_current.csv   today's members (ticker, sector)
  sp500_changes.csv   date, added, removed: one row per join or exit, so papers.membership() rebuilds any month
  sp500_sectors.csv   ticker, sector for every member since START (today's members: GICS from Wikipedia; former
                      members: the sector SPDR their weekly returns follow most closely while in the index)
  prices/<label>.csv  date, open, close, adj_close, volume for every former member (their histories have ended)
  pit/former.txt      the former members' labels, which papers.update() leaves alone (Yahoo does not have them)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger("exp2.pit")
URL = "https://github.com/Johnbrick123/sp500-data/releases/download/data/"
MAX_AGE_DAYS = 7
SPDR = {"XLB": "Materials", "XLE": "Energy", "XLF": "Financials", "XLI": "Industrials", "XLK": "Information Technology",
        "XLP": "Consumer Staples", "XLU": "Utilities", "XLV": "Health Care", "XLY": "Consumer Discretionary"}


def download(name: str, f: Path, max_age_days: float = MAX_AGE_DAYS) -> bool:
    if f.exists() and time.time() - f.stat().st_mtime < max_age_days * 86400:
        return False
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".part")
    with requests.get(URL + name, stream=True, timeout=120, headers={"User-Agent": "experiment-1 research"}) as r:
        r.raise_for_status()
        with tmp.open("wb") as out:
            for chunk in r.iter_content(1 << 20):
                out.write(chunk)
    with tmp.open("rb") as h:
        if h.read(4) != b"PAR1":
            tmp.unlink()
            raise ValueError(f"{name} is not a Parquet file")
    tmp.replace(f)
    return True


def changes_from_intervals(iv: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Today's members and the dated list of joins and exits that papers.membership() walks backwards."""
    cur = sorted(iv.loc[iv["end"].isna(), "ticker"])
    add = pd.DataFrame({"date": iv["start"], "added": iv["ticker"], "removed": None})
    out = iv[iv["end"].notna()]
    rem = pd.DataFrame({"date": out["end"], "added": None, "removed": out["ticker"]})
    ch = pd.concat([add, rem], ignore_index=True).sort_values(["date", "added"], ascending=[False, True]).reset_index(drop=True)
    return cur, ch


def sector_by_returns(px: pd.DataFrame, spans: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> dict[str, str]:
    """For each label, the sector SPDR whose weekly returns correlate most with its own while it was in the index (the
    first two years of SPDR history for a span that ended before the SPDRs began)."""
    wk = px.resample("W-FRI").last().pct_change(fill_method=None)
    etf = wk[[c for c in SPDR if c in wk.columns]]
    out = {}
    for t, (a, b) in spans.items():
        if t not in wk.columns:
            continue
        s = wk[t].loc[a:b].dropna()
        if len(s) < 52:
            s = wk[t].dropna().iloc[:104]
        e = etf.reindex(s.index).dropna(how="all")
        if len(e) < 26:
            continue
        c = e.corrwith(s.reindex(e.index))
        if c.notna().any():
            out[t] = SPDR[c.idxmax()]
    return out


def due(data: Path, max_age_days: float = MAX_AGE_DAYS) -> bool:
    f = data / "pit" / "refreshed.txt"
    return not f.exists() or not (data / "sp500_changes.csv").exists() or time.time() - f.stat().st_mtime > max_age_days * 86400


def refresh(data: Path, start: str, gics: pd.DataFrame | None = None, fetch: bool = True, force: bool = False) -> dict:
    """Bring the membership tables and the former members' prices up to date (see the module docstring), at most once a
    week. The 235 MB price file is deleted afterwards so the caches that carry the data folder stay small."""
    pit = data / "pit"
    if not (force or due(data)):
        return {"status": "fresh"}
    rep: dict = {}
    pq = pit / "prices.parquet"
    if fetch:
        for name in ("membership_intervals.parquet", "prices.parquet"):
            try:
                rep[name] = "downloaded" if download(name, pit / name, 0) else "cached"
            except Exception as ex:  # noqa: BLE001 - last week's tables stay in place
                rep[name] = f"failed: {type(ex).__name__}: {str(ex)[:200]}"
    if not (pit / "membership_intervals.parquet").exists() or not pq.exists():
        rep["error"] = "no point-in-time data"
        return rep
    iv = pd.read_parquet(pit / "membership_intervals.parquet")
    iv["start"], iv["end"] = pd.to_datetime(iv["start"]), pd.to_datetime(iv["end"])
    iv = iv[iv["end"].isna() | (iv["end"] > pd.Timestamp(start))]
    cur, ch = changes_from_intervals(iv)
    former = sorted(set(iv["ticker"]) - set(cur))
    p = pd.read_parquet(pq, columns=["date", "ticker", "open", "close", "adj_close", "volume"],
                        filters=[("ticker", "in", [*former, *SPDR, *cur]), ("date", ">=", pd.Timestamp(start) - pd.Timedelta(days=400))])
    p["date"] = pd.to_datetime(p["date"])
    d = data / "prices"
    d.mkdir(parents=True, exist_ok=True)
    for t, g in p[p["ticker"].isin(former)].groupby("ticker"):
        g = g.sort_values("date").drop_duplicates("date", keep="last")
        g.assign(date=g["date"].dt.strftime("%Y-%m-%d"))[["date", "open", "close", "adj_close", "volume"]].to_csv(d / f"{t}.csv", index=False)
    px = p.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
    spans = {r.ticker: (r.start, r.end if pd.notna(r.end) else px.index[-1]) for r in iv.itertuples()}
    by_ret = sector_by_returns(px, spans)
    g = dict(zip(gics["ticker"].str.replace(".", "-"), gics["sector"])) if gics is not None and len(gics) else {}
    sectors = {t: g.get(t) or by_ret.get(t, "Unknown") for t in sorted(set(iv["ticker"]))}
    pd.DataFrame({"ticker": cur, "sector": [sectors[t] for t in cur]}).to_csv(data / "sp500_current.csv", index=False)
    ch.to_csv(data / "sp500_changes.csv", index=False)
    pd.DataFrame({"ticker": list(sectors), "sector": list(sectors.values())}).to_csv(data / "sp500_sectors.csv", index=False)
    (pit / "former.txt").write_text("\n".join(former) + "\n")
    rep.update({"members_today": len(cur), "former_members": len(former), "former_priced": int(p.loc[p["ticker"].isin(former), "ticker"].nunique()),
                "changes": len(ch), "sector_by_returns": sum(t not in g for t in sectors),
                "unknown_sector": sum(v == "Unknown" for v in sectors.values()), "data_to": str(p["date"].max().date())})
    if fetch:
        pq.unlink()
    (pit / "refreshed.txt").write_text(rep["data_to"] + "\n")
    return rep


def former(data: Path) -> set[str]:
    f = data / "pit" / "former.txt"
    return set(f.read_text().split()) if f.exists() else set()


def sectors(data: Path) -> dict[str, str]:
    """ticker -> sector for every member, past or present; falls back to today's list."""
    for name in ("sp500_sectors.csv", "sp500_current.csv"):
        f = data / name
        if f.exists():
            s = pd.read_csv(f)
            return dict(zip(s["ticker"], s["sector"]))
    return {}


def check(data: Path, start: str = "2004-01-01") -> dict:
    """Equal weight of the point-in-time members against RSP and against today's members only: the size of the bias."""
    pit = data / "pit"
    download("membership_intervals.parquet", pit / "membership_intervals.parquet")
    download("prices.parquet", pit / "prices.parquet")
    iv = pd.read_parquet(pit / "membership_intervals.parquet")
    p = pd.read_parquet(pit / "prices.parquet", columns=["date", "ticker", "adj_close"], filters=[("date", ">=", pd.Timestamp(start))])
    r = p.pivot_table(index="date", columns="ticker", values="adj_close").sort_index().pct_change(fill_method=None)
    days = r.index
    mem = pd.DataFrame(False, index=days, columns=r.columns)
    for t in iv.itertuples():
        if t.ticker in mem.columns:
            mem.loc[(days >= t.start) & ((days < t.end) if pd.notna(t.end) else True), t.ticker] = True
    held = mem.shift(1, fill_value=False)
    ew = r.where(held).mean(axis=1)
    cur = [t for t in iv.loc[iv["end"].isna(), "ticker"] if t in r.columns]
    surv = r[cur].where(mem[cur].cummax().shift(1, fill_value=False)).mean(axis=1)
    ann = lambda s: float((1 + s.dropna()).prod() ** (252 / s.dropna().size) - 1)  # noqa: E731
    out = {"ew_point_in_time": ann(ew), "ew_todays_members": ann(surv), "members_per_day_min": int(mem.sum(axis=1).min())}
    if "RSP" in r.columns:
        j = pd.concat([ew, r["RSP"]], axis=1).dropna()
        out |= {"rsp": ann(j.iloc[:, 1]), "corr_rsp": float(j.corr().iloc[0, 1]), "ew_same_days": ann(j.iloc[:, 0])}
    out["member_days_priced"] = float((r.notna() & held).sum().sum() / held.sum().sum())
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()} | {"from": start, "to": str(days[-1].date())}

