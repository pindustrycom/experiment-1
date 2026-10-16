"""Claude Haiku reads each headline and Reddit post once and scores it for NVDA.

For every item: relevance to NVIDIA's stock (0..1), sentiment for the stock (-1..1), and a separate
market-saturation reading (-1 = the market is saturating: demand peaking, customers cutting orders,
competition or custom chips taking share, overcapacity; +1 = the market is expanding: demand
outstripping supply, new markets opening; 0 = the item says nothing about it).

Items are sent without their dates, so the model has less to anchor hindsight on (it may still
recognise famous events; that is a known limit of scoring past text with a model trained later).
Scores are cached by item id, so an item is only paid for once, and spending stops at a budget.
Items are capped per day before scoring (a deterministic sample), which bounds the cost of busy days.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from . import config

log = logging.getLogger("exp3.score")
PRICING = {"claude-haiku-5-5": (0.10, 0.50)}  # USD per million input/output tokens

SYSTEM = """You score short texts for a research model of NVIDIA's stock (NVDA). Texts are news headlines or Reddit posts.

For each numbered item return:
- relevance: 0 to 1, how much the item bears on NVDA's share price (1 = directly about NVIDIA's business, results, products,
  customers or stock; 0.5 = about its industry, competitors or customers in a way that matters for NVIDIA; 0 = unrelated, a
  product review or game benchmark with no business angle, or a page that only lists quotes).
- sentiment: -1 to 1, the item's implication for NVDA's share price (negative = bearish, positive = bullish, 0 = neutral or unclear).
- saturation: -1 to 1, what the item says about saturation of NVIDIA's markets. Negative = signs the market is saturating:
  demand peaking or slowing, customers cutting or delaying orders, inventory gluts, competitors or in-house chips taking share,
  overcapacity, spending cuts. Positive = signs of expansion: demand outstripping supply, backlogs, new markets or customers,
  rising capital spending on its products. 0 = says nothing about this.

Judge only what the text says; do not use knowledge of what happened later. Return one entry per item, using its number."""


class ItemScore(BaseModel):
    i: int
    relevance: float = Field(ge=0, le=1)
    sentiment: float = Field(ge=-1, le=1)
    saturation: float = Field(ge=-1, le=1)


class BatchScores(BaseModel):
    scores: list[ItemScore]


class HaikuScorer:
    def __init__(self, api_key: str, model: str = config.CLAUDE_MODEL, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, timeout=120, max_retries=3)
        self.client, self.model = client, model

    def score(self, items: list[dict]) -> tuple[list[ItemScore], float]:
        lines = []
        for k, it in enumerate(items, 1):
            src = "Reddit r/" + it["sub"] if it["kind"] == "reddit" else "News"
            body = (it.get("text") or "")[:240].replace("\n", " ")
            lines.append(f"{k}. [{src}] {it['title']}" + (f" | {body}" if body else ""))
        msg = "<items>\n" + "\n".join(lines) + "\n</items>\nScore every item."
        resp = self.client.beta.messages.parse(
            model=self.model, max_tokens=4000,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": msg}], output_format=BatchScores)
        parsed = getattr(resp, "parsed_output", None)
        if parsed is None:
            raise RuntimeError(f"no structured output (stop_reason={getattr(resp, 'stop_reason', None)})")
        pin, pout = PRICING.get(self.model, (1.0, 5.0))
        u = resp.usage
        cost = ((u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0) * 0.1
                + (getattr(u, "cache_creation_input_tokens", 0) or 0) * 1.25) * pin / 1e6 + (u.output_tokens or 0) * pout / 1e6
        return BatchScores.model_validate(parsed.model_dump()).scores, cost


def items_to_score(news: pd.DataFrame, reddit: pd.DataFrame) -> pd.DataFrame:
    """All collected items, deduplicated, junk dropped, capped per day and kind (deterministic sample)."""
    parts = []
    if len(news):
        n = news.assign(kind="news", text="", sub="")
        n["key"] = n["title"].str.lower().str.replace(r"[^a-z0-9 ]+", "", regex=True).str.strip()
        n = n.drop_duplicates("key")
        parts.append((n, config.MAX_NEWS_PER_DAY))
    if len(reddit):
        r = reddit.assign(kind="reddit")
        r = r[~r["title"].str.lower().isin(["[deleted by user]", "[removed]", "[deleted]", ""])]
        parts.append((r, config.MAX_REDDIT_PER_DAY))
    out = []
    for df, cap in parts:
        df = df.copy()
        df["day"] = pd.to_datetime(df["ts"], utc=True).dt.strftime("%Y-%m-%d")
        df["h"] = df["id"].map(lambda s: hashlib.sha1(s.encode()).hexdigest())
        df = df.sort_values(["day", "h"]).groupby("day", group_keys=False).head(cap)
        out.append(df[["id", "ts", "kind", "title", "text", "sub"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["id", "ts", "kind", "title", "text", "sub"])


def load_scores(data: Path) -> dict[str, dict]:
    f = data / "scores.jsonl"
    if not f.exists():
        return {}
    return {r["id"]: r for r in (json.loads(x) for x in f.read_text().splitlines() if x.strip())}


def spent(data: Path) -> float:
    f = data / "usage.json"
    return float(json.loads(f.read_text()).get("cost_usd", 0.0)) if f.exists() else 0.0


def score_all(data: Path, items: pd.DataFrame, scorer, budget_usd: float = config.CLAUDE_BUDGET_USD, batch: int = config.ITEMS_PER_CALL,
              workers: int = 6) -> dict:
    """Score every item without a score, `workers` calls at a time, until the budget is used."""
    from concurrent.futures import ThreadPoolExecutor

    done = load_scores(data)
    todo = items[~items["id"].isin(done)].sort_values("ts")
    out = {"items": len(items), "pending": len(todo), "scored": 0, "calls": 0, "cost_usd": 0.0, "errors": 0, "budget_stop": False}
    if scorer is None or todo.empty:
        return out
    recs = todo.to_dict("records")
    chunks = [recs[k:k + batch] for k in range(0, len(recs), batch)]
    uf = data / "usage.json"
    usage = json.loads(uf.read_text()) if uf.exists() else {"cost_usd": 0.0, "calls": 0, "items": 0}

    def call(chunk):
        try:
            return chunk, *scorer.score(chunk)
        except Exception as e:  # noqa: BLE001
            log.warning("scoring batch failed: %s", e)
            return chunk, None, 0.0

    with open(data / "scores.jsonl", "a") as fh, ThreadPoolExecutor(workers) as ex:
        for g in range(0, len(chunks), workers):
            if usage["cost_usd"] >= budget_usd:
                out["budget_stop"] = True
                break
            for chunk, scores, cost in ex.map(call, chunks[g:g + workers]):
                if scores is None:
                    out["errors"] += 1
                    continue
                by_i = {s.i: s for s in scores}
                stamp = datetime.now(UTC).isoformat(timespec="seconds")
                for j, it in enumerate(chunk, 1):
                    if j in by_i:
                        s = by_i[j]
                        fh.write(json.dumps({"id": it["id"], "relevance": s.relevance, "sentiment": s.sentiment, "saturation": s.saturation,
                                             "model": scorer.model, "scored_at": stamp}) + "\n")
                        out["scored"] += 1
                usage["cost_usd"] += cost
                usage["calls"] += 1
                usage["items"] += len(chunk)
                out["calls"] += 1
                out["cost_usd"] += cost
            fh.flush()
            uf.write_text(json.dumps(usage))
            if out["errors"] >= 10 and out["scored"] == 0:
                break
            if out["calls"] and out["calls"] % 120 < workers:
                log.info("scored %d items, $%.2f", out["scored"], usage["cost_usd"])
    return out
