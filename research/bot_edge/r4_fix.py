"""Honest copies of the NOTankAi strategies (TheoBrigitte/freqtrade), the round-4 rows that looked like 18-26% a month.

They mark local minima and maxima with scipy's argrelextrema(close, order=5): a bar is a minimum only if it is lower
than the 5 bars before AND the 5 bars after it, so the mark uses 5 future bars, and the entries trade on it one bar
later. Live, that mark only exists 5 bars after the fact. The copies shift both marks by those 5 bars (to the bar when
they could first be known) and change nothing else; class names get a _known suffix.

  python r4_fix.py R4_DIR OUT_DIR
"""
import glob
import os
import re
import sys

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
ANCHOR = '        dataframe["minima"] = minima\n'
for name in ("NOTankAi_15", "NOTankAi_15_Cleaned_v2"):
    f = glob.glob(f"{src}/*/{name}.py")[0]
    s = open(f).read()
    assert s.count(ANCHOR) == 1 and "order=5" in s, name
    s = s.replace(ANCHOR, ANCHOR + '        # known 5 bars later (argrelextrema order=5 looks 5 bars ahead)\n'
                  '        dataframe["maxima"] = dataframe["maxima"].shift(5).fillna(0)\n'
                  '        dataframe["minima"] = dataframe["minima"].shift(5).fillna(0)\n')
    s = re.sub(rf"^class {name}\(", f"class {name}_known(", s, flags=re.M)
    open(f"{out}/{name}_known.py", "w").write(s)
    print("wrote", f"{out}/{name}_known.py")
