"""Experiment 3: one stock (NVDA), fixed train / test split, sentiment by category.

Everything that defines the experiment is here; its hash is stored with the results.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

TICKER = "NVDA"
COMPANY = "NVIDIA"
CIK = "0001045810"
MARKET = ["QQQ", "SPY"]        # cap-weighted indices: Nasdaq-100 and S&P 500

PRICE_START = "2016-01-04"     # warm-up for 200/250-session features
SENTIMENT_START = "2017-06-01"  # warm-up for the long sentiment memory
TRAIN_START = "2018-01-02"
TRAIN_END = "2021-12-31"
TEST_END = "2025-12-31"
HORIZON = 5                    # trading sessions ahead

# Public speculation: Reddit (Twitter/X has no free history). Searched by title.
SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket"]
REDDIT_TERMS = ["NVDA", "Nvidia"]
NEWS_QUERY = "NVIDIA"           # Google News search

CATEGORIES = ["news", "saturation", "speculation", "brokers"]

CLAUDE_MODEL = os.environ.get("EXP3_CLAUDE_MODEL", "claude-haiku-5-5")
CLAUDE_BUDGET_USD = float(os.environ.get("EXP3_CLAUDE_BUDGET_USD", "4"))   # one-off backfill
ITEMS_PER_CALL = 25
MAX_NEWS_PER_DAY = 40
MAX_REDDIT_PER_DAY = 40


@dataclass(frozen=True)
class Config:
    ticker: str = TICKER
    horizon: int = HORIZON
    train_start: str = TRAIN_START
    train_end: str = TRAIN_END
    test_end: str = TEST_END
    val_sessions: int = 126       # end of the training period, used only for early stopping
    week: int = 5                 # "last week" window, sessions
    long_halflife: int = 63       # long memory of what was said, sessions (about a quarter)
    relevance_min: float = 0.3
    d_state: int = 8
    d_hidden: int = 16
    steps: int = 300
    window: int = 128
    batch: int = 8
    lr: float = 1e-2
    patience: int = 80
    seeds: tuple = (0, 1, 2)
    carry: int = 128              # sessions filtered without gradient before each training window (slow modes start filled)
    spectrum_seg: int = 64        # frequency grid for the resonance index
    ekf_variants: tuple = ("ekf", "ekf_price", "ekf_slow", "ekf_h2", "ekf_res", "ekf_res_price")

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


DEFAULT = Config()
