"""SEC EDGAR adapters (free, documented JSON APIs; requires a descriptive User-Agent).

* company_tickers.json  ticker -> CIK map
* submissions/CIK##########.json  recent filings (8-K items, accession, acceptance time), SIC code
* Archives filing documents (8-K primary document + EX-99 press-release exhibits) as news-like documents
* api/xbrl/companyfacts  point-in-time fundamentals (each fact carries its `filed` date)
"""
from __future__ import annotations

import re
from datetime import UTC, date, datetime
from html import unescape

from app.core.config import get_settings
from app.data_sources.base import CompanyReference, FactRecord, ProviderNotConfigured, RawDocument
from app.data_sources.http import request

ITEM_TITLES = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate a Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure/Election of Directors or Officers",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}
# Items that are routine and rarely asymmetric; still ingested, but not sent to the AI by default.
ROUTINE_ITEMS = {"5.07", "9.01", "3.02", "5.03", "5.08"}

METRIC_TAGS = {
    "revenue": [
        ("us-gaap", "Revenues", "USD"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
        ("us-gaap", "SalesRevenueNet", "USD"),
        ("us-gaap", "RevenuesNetOfInterestExpense", "USD"),
    ],
    "operating_income": [("us-gaap", "OperatingIncomeLoss", "USD")],
    "net_income": [("us-gaap", "NetIncomeLoss", "USD")],
    "eps_diluted": [("us-gaap", "EarningsPerShareDiluted", "USD/shares")],
}


def _ua() -> dict:
    ua = get_settings().sec_user_agent
    if not ua:
        raise ProviderNotConfigured("sec", "SEC_USER_AGENT (name + contact e-mail)")
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def pad_cik(cik: str | int) -> str:
    return str(int(cik)).zfill(10)


class SecEdgar:
    name = "sec"

    def configured(self) -> bool:
        return bool(get_settings().sec_user_agent)

    def ticker_map(self) -> dict[str, CompanyReference]:
        r = request(self.name, "GET", "https://www.sec.gov/files/company_tickers.json", headers=_ua(), operation="ticker_map")
        out = {}
        for row in r.json().values():
            t = str(row["ticker"]).upper()
            out[t] = CompanyReference(ticker=t, cik=pad_cik(row["cik_str"]), name=row.get("title"))
        return out

    def submissions(self, cik: str) -> dict:
        return request(self.name, "GET", f"https://data.sec.gov/submissions/CIK{pad_cik(cik)}.json", headers=_ua(), operation="submissions").json()

    def company_facts(self, cik: str) -> dict:
        return request(self.name, "GET", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{pad_cik(cik)}.json", headers=_ua(), operation="companyfacts").json()

    def filing_text(self, cik: str, accession: str, primary_document: str, max_chars: int = 30000) -> str:
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"
        parts = [html_to_text(request(self.name, "GET", f"{base}/{primary_document}", headers=_ua(), operation="filing_doc").text)]
        try:
            idx = request(self.name, "GET", f"{base}/index.json", headers=_ua(), operation="filing_index").json()
            for item in idx.get("directory", {}).get("item", []):
                nm = item.get("name", "")
                if re.search(r"ex-?99", nm, re.I) and nm.lower().endswith((".htm", ".html", ".txt")):
                    parts.append(html_to_text(request(self.name, "GET", f"{base}/{nm}", headers=_ua(), operation="filing_exhibit").text))
                    break
        except Exception:
            pass  # exhibits are optional
        return "\n\n".join(parts)[:max_chars]

    def recent_filings(self, ticker: str, cik: str, since: date, forms=("8-K",), fetch_text: bool = True) -> tuple[list[RawDocument], dict]:
        sub = self.submissions(cik)
        docs = parse_recent_filings(ticker, cik, sub, since, forms)
        if fetch_text:
            for d in docs:
                if set(d.filing_items) - ROUTINE_ITEMS:
                    d.content_text = self.filing_text(cik, d.accession or "", d.raw["primary_document"])
        meta = {"sic": sub.get("sic"), "sic_description": sub.get("sicDescription"), "name": sub.get("name")}
        return docs, meta


def parse_recent_filings(ticker: str, cik: str, sub: dict, since: date, forms=("8-K",)) -> list[RawDocument]:
    recent = sub.get("filings", {}).get("recent", {})
    n = len(recent.get("accessionNumber", []))
    out = []
    for i in range(n):
        form = recent["form"][i]
        if form not in forms:
            continue
        fdate = date.fromisoformat(recent["filingDate"][i])
        if fdate < since:
            continue
        acc = recent["accessionNumber"][i]
        accepted = recent.get("acceptanceDateTime", [None] * n)[i]
        published = datetime.fromisoformat(accepted.replace("Z", "+00:00")) if accepted else datetime(fdate.year, fdate.month, fdate.day, tzinfo=UTC)
        items = [x.strip() for x in (recent.get("items", [""] * n)[i] or "").split(",") if x.strip()]
        primary = recent["primaryDocument"][i]
        titles = "; ".join(f"Item {it} {ITEM_TITLES.get(it, '')}".strip() for it in items) or form
        out.append(RawDocument(
            provider="sec", doc_type="filing",
            url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{primary}",
            title=f"{ticker} {form}: {titles}", published_at=published, publisher="SEC EDGAR",
            tickers=[ticker], filing_form=form, filing_items=items, accession=acc,
            raw={"primary_document": primary, "filing_date": str(fdate)},
        ))
    return out


def parse_company_facts(payload: dict) -> list[FactRecord]:
    out: list[FactRecord] = []
    facts = payload.get("facts", {})
    for metric, tags in METRIC_TAGS.items():
        seen: set[tuple] = set()
        for taxonomy, tag, unit in tags:
            for row in facts.get(taxonomy, {}).get(tag, {}).get("units", {}).get(unit, []):
                try:
                    end = date.fromisoformat(row["end"])
                    start = date.fromisoformat(row["start"]) if row.get("start") else None
                    key = (start, end, row["accn"])
                    if key in seen:  # earlier tag in the list wins for the same period/filing
                        continue
                    seen.add(key)
                    out.append(FactRecord(
                        metric=metric, period_start=start, period_end=end, value=float(row["val"]), unit=unit,
                        filed_at=date.fromisoformat(row["filed"]), accession=row["accn"], form=row.get("form"),
                        fiscal_year=row.get("fy"), fiscal_period=row.get("fp"), tag=tag,
                    ))
                except (KeyError, ValueError):
                    continue
    return out


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", html)
    text = unescape(_TAG.sub(" ", html))
    text = _WS.sub(" ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()
