"""Slide 6: tone of 10-K filings with the Loughran-McDonald finance word lists vs a generic list (Harvard IV-4).

For every 10-K of the universe since 2009, count words on the finance negative/positive/uncertainty lists and on the
generic Harvard negative list. Only the counts are kept (text_counts.csv), so a rerun skips filings already counted.
Word lists ship with the pysentiment2 package (LM.csv, HIV-4.csv).
"""
from __future__ import annotations

import html
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from importlib import resources
from pathlib import Path

import pandas as pd

import sec

log = logging.getLogger("quantamental.text")
FIRST_YEAR = 2009
COLS = ["words", "lm_neg", "lm_pos", "lm_unc", "h4_neg", "h4_pos"]


def word_lists() -> dict[str, frozenset[str]]:
    base = resources.files("pysentiment2") / "static"
    lm = pd.read_csv(base / "LM.csv")
    hv = pd.read_csv(base / "HIV-4.csv", low_memory=False)
    hv["w"] = hv["Entry"].astype(str).str.split("#").str[0].str.upper()
    lm["Word"] = lm["Word"].astype(str).str.upper()
    return {"lm_neg": frozenset(lm.loc[lm["Negative"] > 0, "Word"]), "lm_pos": frozenset(lm.loc[lm["Positive"] > 0, "Word"]),
            "lm_unc": frozenset(lm.loc[lm["Uncertainty"] > 0, "Word"]),
            "h4_neg": frozenset(hv.loc[hv["Negativ"].notna(), "w"]), "h4_pos": frozenset(hv.loc[hv["Positiv"].notna(), "w"])}


_TAG = re.compile(r"<[^>]+>")
_HIDDEN = re.compile(r"<ix:header>.*?</ix:header>|<script.*?</script>|<style.*?</style>", re.DOTALL | re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z]{2,}")


def clean(raw: str) -> str:
    return html.unescape(_TAG.sub(" ", _HIDDEN.sub(" ", raw)))


def count(text: str, lists: dict[str, frozenset[str]]) -> dict[str, int]:
    words = [w.upper() for w in _WORD.findall(text)]
    out = {"words": len(words)}
    for k, s in lists.items():
        out[k] = sum(w in s for w in words)
    return out


def filings(cik: int) -> pd.DataFrame:
    """The company's 10-K filings since FIRST_YEAR: filing date, accession number, main document."""
    js = sec.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    parts = [js.get("filings", {}).get("recent", {})]
    for f in js.get("filings", {}).get("files", []):
        if f.get("filingTo", "9999") >= f"{FIRST_YEAR}-01-01":
            parts.append(sec.get(f"https://data.sec.gov/submissions/{f['name']}").json())
    df = pd.concat([pd.DataFrame({k: p.get(k, []) for k in ("form", "filingDate", "accessionNumber", "primaryDocument")}) for p in parts])
    df = df[df["form"].isin(["10-K", "10-K405"]) & (df["filingDate"] >= f"{FIRST_YEAR}-01-01")]
    return df.assign(cik=cik).drop_duplicates("accessionNumber")


_LISTS: dict | None = None


def _one(job: tuple[int, str, str, str]) -> dict | None:
    global _LISTS
    if _LISTS is None:
        _LISTS = word_lists()
    cik, accn, filed, doc = job
    try:
        r = sec.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/{doc}", rate=2.0)
        if r.status_code != 200:
            return None
        c = count(clean(r.text), _LISTS)
    except Exception:  # noqa: BLE001 - a missing document just drops that filing
        return None
    return {"cik": cik, "accn": accn, "filed": filed, **c}


def collect(ciks: list[int], out_csv: Path, workers: int = 4, max_docs: int = 20000) -> pd.DataFrame:
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["cik", "accn", "filed", *COLS])
    have = set(done["accn"])
    jobs = []
    for i, cik in enumerate(ciks):
        try:
            f = filings(cik)
        except Exception as e:  # noqa: BLE001
            log.warning("filings %s: %s", cik, e)
            continue
        jobs += [(cik, a, d, p) for a, d, p in zip(f["accessionNumber"], f["filingDate"], f["primaryDocument"]) if a not in have and p]
        if i % 100 == 0:
            log.info("listed %d/%d companies, %d filings to read", i, len(ciks), len(jobs))
    jobs = jobs[:max_docs]
    rows = []
    with ProcessPoolExecutor(workers) as ex:     # 4 processes x 2 requests a second stays under sec.gov's limit
        for k, r in enumerate(ex.map(_one, jobs, chunksize=4)):
            if r:
                rows.append(r)
            if k % 500 == 0:
                log.info("read %d/%d filings", k, len(jobs))
                if rows:
                    pd.concat([done, pd.DataFrame(rows)]).to_csv(out_csv, index=False)
    df = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(out_csv, index=False)
    return df
