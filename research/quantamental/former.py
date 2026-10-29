"""SEC CIKs for companies that left the S&P 500, so their 10-Ks count while they were members.

SEC's ticker file lists today's tickers only. A former member that still trades under its symbol is found there (and
kept only if that company filed a 10-K while it was in the index, so a reused ticker cannot point at another firm). A
company that is gone has a label like ABC-201905 (exp2/pit.py); its name comes from Wikipedia's list of index changes
(a 2025 revision: the live page dropped the table in 2026), and the name is matched to SEC's list of every filer name,
again kept only with a 10-K filed while it was a member.
"""
from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path

import pandas as pd
import requests

import sec

log = logging.getLogger("quantamental.former")
WIKI_API = "https://en.wikipedia.org/w/api.php"
TITLE = "List of S&P 500 companies"
REVISION_BEFORE = "2025-12-31T00:00:00Z"
DEAD = re.compile(r"^(.+)-(\d{6})$")
DROP = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC", "THE", "LLC", "LP", "NV", "SA", "AG",
        "HOLDING", "HOLDINGS", "GROUP", "DE", "NEW", "CLASS", "A", "B"}
UA = {"User-Agent": "experiment-1 research (https://github.com/pindustrycom/experiment-1)"}


def norm(name: str) -> str:
    w = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper().replace("&", " AND ")).split()
    return " ".join(x for x in w if x not in DROP)


def wiki_removed(html: str | None = None) -> pd.DataFrame:
    """date, ticker, name of every removal in Wikipedia's change table (an older revision that still has it)."""
    if html is None:
        js = requests.get(WIKI_API, params={"action": "query", "prop": "revisions", "titles": TITLE, "rvlimit": 1, "rvstart": REVISION_BEFORE,
                                            "rvdir": "older", "rvprop": "ids", "format": "json"}, headers=UA, timeout=30).json()
        rid = next(iter(js["query"]["pages"].values()))["revisions"][0]["revid"]
        r = requests.get("https://en.wikipedia.org/w/index.php", params={"oldid": rid}, headers=UA, timeout=60)
        r.raise_for_status()
        html = r.text
    for t in pd.read_html(io.StringIO(html)):
        cols = ["_".join(dict.fromkeys(str(x) for x in c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        t.columns = cols
        tick = [c for c in cols if "Removed" in c and ("Ticker" in c or "Symbol" in c)]
        name = [c for c in cols if "Removed" in c and "Security" in c]
        if tick and name:
            out = pd.DataFrame({"date": pd.to_datetime(t[cols[0]], errors="coerce", format="mixed"), "ticker": t[tick[0]], "name": t[name[0]]})
            return out.dropna()
    return pd.DataFrame(columns=["date", "ticker", "name"])


def filer_names(cache: Path) -> pd.DataFrame:
    """SEC's list of every filer name ever used, with its CIK (cik-lookup-data.txt), cached."""
    f = cache / "cik-lookup-data.txt"
    if not f.exists():
        cache.mkdir(parents=True, exist_ok=True)
        f.write_bytes(sec.get("https://www.sec.gov/Archives/edgar/cik-lookup-data.txt").content)
    rows = []
    for line in f.read_text(encoding="latin-1").splitlines():
        name, _, rest = line.rpartition(":")[0].rpartition(":")
        if rest.strip().isdigit():
            rows.append((name, int(rest)))
    df = pd.DataFrame(rows, columns=["name", "cik"])
    df["key"] = df["name"].map(norm)
    return df


def tenk_years(cik: int, cache: Path) -> list[str]:
    """Filing dates of the company's 10-Ks (submissions API), cached."""
    f = cache / "tenk" / f"{cik}.json"
    if f.exists():
        return json.loads(f.read_text())
    js = sec.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    parts = [js.get("filings", {}).get("recent", {})]
    parts += [sec.get(f"https://data.sec.gov/submissions/{x['name']}").json() for x in js.get("filings", {}).get("files", [])]
    dates = sorted({d for p in parts for fm, d in zip(p.get("form", []), p.get("filingDate", [])) if fm in ("10-K", "10-K405")})
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dates))
    return dates


def filed_while_member(cik: int, start: pd.Timestamp, end: pd.Timestamp, cache: Path) -> bool:
    lo, hi = (start - pd.Timedelta(days=200)).strftime("%Y-%m-%d"), (end + pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    try:
        return any(lo <= d <= hi for d in tenk_years(cik, cache))
    except Exception:  # noqa: BLE001 - an unreachable record is not a match
        return False


def resolve(iv: pd.DataFrame, current: dict[str, int], cache: Path, wiki: pd.DataFrame | None = None, names: pd.DataFrame | None = None) -> tuple[dict[str, int], dict]:
    """label -> CIK for every former member in iv (ticker, start, end), with a report of how each was found."""
    gone = iv[iv["end"].notna()].sort_values("end").drop_duplicates("ticker", keep="last")
    found, how = {}, {"alive_ticker": 0, "wikipedia_name": 0, "unmatched": []}
    dead = []
    for r in gone.itertuples():
        m = DEAD.match(r.ticker)
        if not m:
            c = current.get(r.ticker.replace("-", "."))
            if c and filed_while_member(c, r.start, r.end, cache):
                found[r.ticker] = c
                how["alive_ticker"] += 1
            else:
                dead.append((r, r.ticker))
        else:
            dead.append((r, m.group(1)))
    if dead:
        if wiki is None:
            try:
                wiki = wiki_removed()
            except Exception as ex:  # noqa: BLE001 - gone companies then stay without fundamentals
                log.warning("Wikipedia change list: %s", ex)
                wiki = pd.DataFrame(columns=["date", "ticker", "name"])
        names = filer_names(cache) if names is None else names
        by_key = names.groupby("key")["cik"].apply(lambda s: sorted(set(s)))
        for r, base in dead:
            w = wiki[(wiki["ticker"].astype(str).str.replace(".", "-") == base) & ((wiki["date"] - r.end).abs() <= pd.Timedelta(days=45))]
            cands = []
            for nm in w["name"]:
                cands += by_key.get(norm(nm), [])
            ok = [c for c in list(dict.fromkeys(cands))[:6] if filed_while_member(c, r.start, r.end, cache)] if cands else []
            if ok:
                found[r.ticker] = ok[0]
                how["wikipedia_name"] += 1
            else:
                how["unmatched"].append(r.ticker)
    how["former_members"] = len(gone)
    how["unmatched_count"] = len(how["unmatched"])
    how["unmatched"] = how["unmatched"][:60]
    return found, how
