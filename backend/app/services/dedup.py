"""Deduplication of syndicated / repeated coverage."""
from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING = re.compile(r"^(utm_|fbclid$|gclid$|mc_|ref$|src$|cmpid$|ncid$|yptr$|guccounter)", re.I)
STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "at", "by", "with", "as", "is", "its", "from", "after", "amid", "says", "said"}


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False) if not TRACKING.match(k)]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", parts.netloc.lower().removeprefix("www."), path, urlencode(sorted(q)), ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()


def content_hash(text: str | None) -> str | None:
    if not text or len(text.strip()) < 80:
        return None
    norm = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha256(norm.encode()).hexdigest()


def normalize_title(title: str) -> str:
    t = title.strip()
    m = re.match(r"^(.*\S)\s+[-|–—]\s+([^-|–—]{2,40})$", t)  # drop trailing " - Publisher"
    if m:
        t = m.group(1)
    t = re.sub(r"[^\w\s]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def title_tokens(title: str) -> set[str]:
    return {w for w in normalize_title(title).split() if w not in STOP and len(w) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def same_story(title_a: str, title_b: str, threshold: float = 0.6) -> bool:
    return jaccard(title_tokens(title_a), title_tokens(title_b)) >= threshold


def cluster_key_for(title: str, published_iso_day: str, tickers: list[str]) -> str:
    return hashlib.sha256(f"{normalize_title(title)}|{published_iso_day}|{','.join(sorted(tickers))}".encode()).hexdigest()
