"""Keeps data/copy current for the copy pots: new 13Fs of the 23 funds (SEC EDGAR), CUSIP -> ticker for new holdings
(OpenFIGI, then SEC's ticker list), and Nancy Pelosi's new PTRs (House Clerk PDFs, read with pdftotext).

It uses the readers of the trader-skill pack (github.com/Jakeschincariol/trader-skill, MIT) at the commit the workflow
pins, as research/trader_skill did to build the history; TRADER_SKILL is the folder it was checked out to.
Needs SEC_USER_AGENT. Anything that fails leaves the cached data as it was.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
KEEP = 25            # holdings kept per 13F
MAP_TOP = 15         # holdings mapped to tickers: enough for every 7.5%+ position


def _skill(folder: str, name: str):
    root = Path(os.environ["TRADER_SKILL"])
    spec = importlib.util.spec_from_file_location("ts_" + name, root / folder / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def update(data: Path, today: pd.Timestamp) -> dict:
    if not os.environ.get("TRADER_SKILL"):
        return {"skipped": "TRADER_SKILL not set"}
    rep = {}
    for name, fn in (("13f", update_13f), ("pelosi", update_pelosi)):
        try:
            rep[name] = fn(data / "copy", today)
        except Exception as ex:  # noqa: BLE001 - the pots keep their books
            rep[name] = f"failed: {type(ex).__name__}: {str(ex)[:200]}"
    return rep


def update_13f(d: Path, today: pd.Timestamp) -> dict:
    e = _skill("trader-copycat", "edgar13f")
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        return {"skipped": "SEC_USER_AGENT not set"}
    fetcher = e.Fetcher(ua=ua, pause=0.15)
    added = 0
    for path in sorted((d / "13f").glob("*.json")):
        f = json.loads(path.read_text())
        done = {q["accession"] for q in f["quarters"]}
        periods = {q["period"] for q in f["quarters"]}
        try:
            filings = e.list_filings(fetcher.get_json(e.SUBMISSIONS % e.cik10(f["cik"])))
        except Exception as ex:  # noqa: BLE001
            log.warning("%s: %s", f["name"], ex)
            continue
        for x in sorted(filings, key=lambda x: (x["filed"], x["accession"])):
            # originals only, and the first one filed for a period (a refiling is not news to a copier)
            if x["form"] != "13F-HR" or x["accession"] in done or x["period"] in periods:
                continue
            try:
                base = e.archive_url(f["cik"], x["accession"])
                table = e.info_table_name(fetcher.get_json(base + "index.json"))
                rows = e.parse_info_table(fetcher.get(base + table), multiplier=e.value_multiplier(x["filed"]))
                positions, total = e.aggregate(rows)
                stocks = [p for p in positions if e.is_stock(p)]
                f["quarters"].append({"period": x["period"], "filed": x["filed"], "accession": x["accession"], "total": total,
                                      "stock_total": sum(p["value"] for p in stocks), "n_positions": len(stocks),
                                      "top": [{"cusip": p["cusip"], "issuer": p["issuer"], "title": p["title"], "value": p["value"]}
                                              for p in stocks[:KEEP]]})
                periods.add(x["period"])
                added += 1
            except Exception as ex:  # noqa: BLE001 - one bad filing must not stop the rest
                log.warning("%s %s: %s", f["name"], x["accession"], ex)
        f["quarters"].sort(key=lambda q: q["filed"])
        path.write_text(json.dumps(f))
    mapped = map_cusips(e, fetcher, d)
    return {"new_13f": added, "new_cusips": mapped}


def map_cusips(e, fetcher, d: Path) -> int:
    path = d / "cusips.json"
    known = json.loads(path.read_text()) if path.exists() else {}
    issuers = {}
    for p in sorted((d / "13f").glob("*.json")):
        for q in json.loads(p.read_text())["quarters"]:
            for h in q["top"][:MAP_TOP]:
                issuers.setdefault(h["cusip"], h)
    todo = [c for c in issuers if c not in known]
    if not todo:
        return 0
    figi = {}
    for i in range(0, len(todo), 10):
        try:
            figi.update(e.figi_lookup(fetcher, todo[i:i + 10]))
        except e.EdgarError as ex:              # 25 requests a minute without a key: the next run retries
            log.warning("openfigi: %s", ex)
            break
        time.sleep(2.5)
    index = e.ticker_index(fetcher)
    for c in todo:
        p = dict(issuers[c], put_call="", share_type="SH", figi="")
        e.resolve_tickers([p], index, {}, figi)
        known[c] = p.get("ticker")
    path.write_text(json.dumps(known, indent=0, sort_keys=True))
    return len(todo)


def update_pelosi(d: Path, today: pd.Timestamp) -> dict:
    ptr = _skill("trader-pelosi", "ptr")
    years = [today.year - 1, today.year] if today.month <= 2 else [today.year]   # late January filings of last year
    (d / "ptr").mkdir(parents=True, exist_ok=True)
    new = 0
    for year in years:
        path = d / "ptr" / f"{year}.json"
        docs = json.loads(path.read_text())["docs"] if path.exists() else {}
        index = ptr.select(ptr.load_index(year), members=["Pelosi, Nancy"], kind="P")
        with tempfile.TemporaryDirectory() as tmp:
            for r in index:
                if r["doc_id"] in docs:
                    continue
                rec = {"name": r["name"], "last": r["last"], "first": r["first"], "filed": r["filed"], "rows": []}
                if not r["doc_id"].startswith("2"):
                    rec["skip"] = "scanned"      # a scanned paper form: no text to read
                else:
                    _, report = ptr._fetch_and_read(r["doc_id"], r["year"], tmp, filed=r["filed"])
                    rec["rows"] = [{k: x.get(k) for k in ("owner", "asset", "ticker", "asset_type", "type", "date", "amount_low", "amount_high")}
                                   for x in (report or {}).get("rows", [])]
                docs[r["doc_id"]] = rec
                new += 1
        path.write_text(json.dumps({"year": year, "docs": docs}))
    return {"new_ptrs": new}
