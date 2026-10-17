"""Experiment 2: universe, data window, model and training settings.

Everything that defines the experiment lives here, and a hash of it is stored with each trained
model, so a prediction can always be traced back to the exact setup that produced it.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

SECTOR = "Information Technology"
TARGET = "XLK"          # the sector being tracked
BENCHMARK = "SPY"       # the market, for the relative target

# Large US tech companies, all trading since before DATA_START. Their news is the news input.
CONSTITUENTS = ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO", "QCOM", "INTC", "TXN"]

# Related series the sector depends on: the market, semis, neighbouring sectors, rates, dollar, gold.
# (XLC, communication services, only exists since 2018 and is left out to keep a long clean history.)
RELATED = {
    "SPY": "S&P 500", "SMH": "Semiconductors", "XLY": "Consumer discretionary", "XLF": "Financials",
    "XLE": "Energy", "XLI": "Industrials", "XLU": "Utilities", "TLT": "20y+ Treasuries", "UUP": "US dollar",
    "GLD": "Gold",
}

TICKERS = [TARGET, *CONSTITUENTS, *RELATED]
DATA_START = "2010-01-04"
HORIZON = 5              # trading sessions ahead
NEWS_BACKFILL_DAYS = int(os.environ.get("EXP2_NEWS_BACKFILL_DAYS", "365"))  # Finnhub's free plan returns about a year of news

CLAUDE_MODEL = os.environ.get("EXP2_CLAUDE_MODEL", "claude-haiku-5-5")
CLAUDE_MONTHLY_BUDGET_USD = float(os.environ.get("EXP2_CLAUDE_BUDGET_USD", "5"))
MAX_HEADLINES_PER_DAY = 150
HEADLINES_PER_CALL = 25


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 32
    n_modes: int = 32          # complex modes per channel (conjugate pairs are implicit)
    n_layers: int = 2
    dropout: float = 0.1
    dt_min: float = 0.002      # memory from ~5 to ~500 sessions
    dt_max: float = 0.2
    lr: float = 3e-3
    ssm_lr: float = 1e-3       # lower rate and no weight decay on the state matrices, as in S4D
    weight_decay: float = 0.02
    steps: int = 600
    batch: int = 8
    window: int = 1024
    burn_in: int = 256         # positions at the start of each window not scored (state still warming up)
    val_sessions: int = 250
    patience: int = 120        # steps without validation improvement before stopping
    seeds: tuple = (0, 1, 2)
    prune_keep: tuple = (0.125, 0.25, 0.5)  # fractions of modes to try keeping, smallest first
    prune_tolerance: float = 0.005   # a less pruned network must validate better by this much to be preferred
    prune_criterion: str = "resonance"  # Philco's resonance index (see model.S4D.scores)
    compare_criteria: tuple = ("resonance", "flat", "h2", "hinf", "l1", "random")
    spectrum_seg: int = 256    # periodogram segment length (frequency grid) for the resonance index
    finetune_steps: int = 150
    retrain_every: int = 5     # sessions between retrains (weekly)
    # EKF + SSM (exp2/ekf.py): latent sector state filtered from the daily vector
    ekf: bool = True
    ekf_d_state: int = 8
    ekf_d_hidden: int = 16
    ekf_steps: int = 150
    ekf_window: int = 128
    ekf_batch: int = 8
    ekf_lr: float = 1e-2
    ekf_context: int = 512     # sessions the filter runs over before the predicted day


@dataclass(frozen=True)
class Config:
    target: str = TARGET
    benchmark: str = BENCHMARK
    constituents: tuple = tuple(CONSTITUENTS)
    related: tuple = tuple(RELATED)
    horizon: int = HORIZON
    data_start: str = DATA_START
    model: ModelConfig = field(default_factory=ModelConfig)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


DEFAULT = Config()
