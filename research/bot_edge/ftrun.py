"""Run freqtrade's own CLI fully offline: exchange markets come from a static table instead of the network.

freqtrade asks the exchange for its market list at start-up, even for a backtest. The sandbox has no route to
exchanges and Binance answers US GitHub runners with HTTP 451, so this launcher swaps that call for a fixed table of
spot (BASE/USDT) or linear-perpetual (BASE/USDT:USDT) markets. Everything else is freqtrade's own code.

  FT_BASES=BTC,ETH python ftrun.py backtesting -c cfg.json -s Strategy ...
"""
import os
import sys

import warnings

import pandas as pd

# NFI X5/X6 (2024-25) write booleans into columns pandas 3 now types as str; restore the pandas 2 behaviour they assume
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pd.options.future.infer_string = False

import numpy as np  # noqa: E402

for _k, _v in (("NAN", np.nan), ("NaN", np.nan), ("Inf", np.inf), ("PINF", np.inf), ("NINF", -np.inf)):
    if not hasattr(np, _k):          # numpy 2 removed these aliases; 2021-22 strategies still use them
        setattr(np, _k, _v)

import freqtrade.exchange.exchange as E  # noqa: E402
from freqtrade.enums import TradingMode

BASES = os.environ.get("FT_BASES", "BTC,ETH,SPY,QQQ").split(",")
MAXLEV = float(os.environ.get("FT_MAXLEV", "1"))   # futures leverage allowed (round 4 leverage runs)


def _market(base, futures):
    sym = f"{base}/USDT" + (":USDT" if futures else "")
    return {"id": base + "USDT", "symbol": sym, "base": base, "quote": "USDT", "settle": "USDT" if futures else None,
            "baseId": base, "quoteId": "USDT", "settleId": "USDT" if futures else None,
            "type": "swap" if futures else "spot", "spot": not futures, "margin": False, "swap": futures,
            "future": False, "option": False, "active": True, "contract": futures, "linear": True if futures else None,
            "inverse": False if futures else None, "contractSize": 1.0 if futures else None, "taker": 0.001, "maker": 0.001,
            "precision": {"amount": 1e-8, "price": 1e-8}, "limits": {"amount": {"min": 1e-8, "max": None},
            "price": {"min": None, "max": None}, "cost": {"min": 1.0, "max": None}, "leverage": {"min": 1, "max": MAXLEV}},
            "info": {}}


def reload_markets(self, force=False, *, load_leverage_tiers=True):
    fut = self.trading_mode == TradingMode.FUTURES
    mk = [_market(b, fut) for b in BASES]
    self._api.set_markets(mk)
    self._api_async.set_markets(mk)
    self._markets = self._api.markets
    self._last_markets_refresh = E.dt_ts()
    if fut:
        self._leverage_tiers = {m["symbol"]: [{"minNotional": 0, "maxNotional": 1e12, "maintenanceMarginRate": 0.004,
                                              "maxLeverage": MAXLEV, "maintAmt": 0.0}] for m in mk}


E.Exchange.reload_markets = reload_markets
E.Exchange.validate_pricing = lambda self, pricing: None     # ticker pricing is never used in a backtest
E.Exchange.fetch_trading_fees = lambda self: {}
E.Exchange.additional_exchange_init = lambda self: None

if __name__ == "__main__":
    from freqtrade.main import main
    sys.argv = ["freqtrade"] + sys.argv[1:]
    main()
