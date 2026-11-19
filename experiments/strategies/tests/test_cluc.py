"""The ClucHAnix pot's freqtrade side (strats/cluc.py). Imports only cluc, so cluc-5m.yml can run it with freqtrade
installed; needs pyarrow for freqtrade's feather files (skipped where it is missing)."""
import io
import json
import zipfile

import pandas as pd
import pytest

from strats import cluc

pytest.importorskip("pyarrow")


def test_cluc_parses_a_freqtrade_export(tmp_path):
    w = pd.DataFrame({"date": pd.to_datetime(["2026-10-16 20:05", "2026-10-16 20:05", "2026-10-16 20:10"], utc=True),
                      "currency": ["USDT", "SOL", "USDT"], "rate": [1, 150, 1], "total_quote": [833.3, 166.7, 1001.0]})
    buf = io.BytesIO()
    w.to_feather(buf)
    trades = [{"pair": "SOL/USDT", "open_date": "2026-10-16 20:05:00+00:00", "close_date": "2026-10-16 20:10:00+00:00",
               "open_rate": 150.0, "close_rate": 151.0, "profit_ratio": 0.0047, "exit_reason": "roi", "stake_amount": 166.7},
              {"pair": "BTC/USDT", "open_date": "2026-10-16 20:10:00+00:00", "close_date": "2026-10-16 20:15:00+00:00",
               "open_rate": 1.0, "close_rate": 1.0, "profit_ratio": 0.0, "exit_reason": "force_exit", "stake_amount": 10.0}]
    z = tmp_path / "res-x.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("res-x.json", json.dumps({"strategy": {"ClucHAnix_5m": {"trades": trades}}}))
        f.writestr("res-x_config.json", "{}")
        f.writestr("res-x_ClucHAnix_5m_wallet.feather", buf.getvalue())
    eq, tr = cluc.parse(z)
    assert list(eq["equity"]) == pytest.approx([1000.0, 1001.0]) and eq["SOL"].iloc[0] == pytest.approx(0.1667)
    assert tr[0]["profit"] == 0.0047 and not tr[0]["open"] and tr[1]["open"] and tr[1]["close_rate"] is None
    assert [t["coin"] for t in cluc.trade_rows({"trades": tr})] == ["BTC", "SOL"]      # open trades first
