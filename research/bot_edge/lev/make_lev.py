"""Leveraged copies of the one strategy that passed (ClucHAnix_5m): same rules, run as a perpetual-futures long at
2x, 3x or 5x (isolated margin, zero funding assumed, like the other futures runs). freqtrade then measures profit and
the strategy's own stops on the leveraged position, as it would live.

  python lev/make_lev.py ClucHAnix_5m.py OUTDIR
"""
import re
import sys

src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
s = s[:s.index("class Cluc5mDCA")] if "class Cluc5mDCA" in s else s      # the DCA subclass is not tested
# futures mode refuses freqtrade's old buy/sell names, which spot backtests still accept: the same settings, new names
for old, new in (("use_sell_signal", "use_exit_signal"), ("sell_profit_only", "exit_profit_only"),
                 ("ignore_roi_if_buy_signal", "ignore_roi_if_entry_signal"), ("'buy': 'limit'", "'entry': 'limit'"),
                 ("'sell': 'limit'", "'exit': 'limit'"), ("'emergencysell': 'limit'", "'emergency_exit': 'limit'"),
                 ("'forcebuy': \"limit\"", "'force_entry': \"limit\""), ("'forcesell': 'limit'", "'force_exit': 'limit'"),
                 ("def populate_buy_trend", "def populate_entry_trend"), ("def populate_sell_trend", "def populate_exit_trend"),
                 ("'buy_tag'", "'enter_tag'"), ("\n            'buy'\n", "\n            'enter_long'\n"),
                 ("\n            'sell'\n", "\n            'exit_long'\n")):
    assert old in s, old
    s = s.replace(old, new)
for x in (1, 2, 3, 5):  # x1 = futures control
    t = re.sub(r"^class ClucHAnix_5m\(IStrategy\):", f"class ClucHAnix_5m_x{x}(IStrategy):\n    can_short = True   # futures mode; the rules only go long", s, flags=re.M)
    t += f'''

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs):
        return {x}.0
'''
    open(f"{out}/ClucHAnix_5m_x{x}.py", "w").write(t)
