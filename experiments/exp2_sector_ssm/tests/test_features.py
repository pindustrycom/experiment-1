import numpy as np
import pandas as pd

from exp2 import features
from exp2.config import DEFAULT

from synth import prices


def test_no_lookahead():
    p = prices(1200)
    full = features.build(p, None, DEFAULT)
    cut = full.index[900]
    part = features.build({t: d[d.index <= cut] for t, d in p.items()}, None, DEFAULT)
    cols = features.feature_columns(full)
    pd.testing.assert_frame_equal(full.loc[:cut, cols], part.loc[:cut, cols])
    # targets: known for rows with HORIZON sessions after them, unknown for the last ones
    assert part["y_sector"].iloc[-DEFAULT.horizon:].isna().all()
    assert part["y_sector"].iloc[: -DEFAULT.horizon].notna().all()


def test_target_is_forward_log_return():
    p = prices(400)
    f = features.build(p, None, DEFAULT)
    px = p[DEFAULT.target]["adj_close"]
    d = f.index[100]
    i = px.index.get_loc(d)
    assert np.isclose(f.loc[d, "y_sector"], 100 * np.log(px.iloc[i + DEFAULT.horizon] / px.iloc[i]))


def test_scaler_fits_on_given_rows_and_clips():
    x = np.array([[0.0], [1.0], [2.0], [3.0], [1000.0]])
    s = features.Scaler().fit(x[:4])
    z = s.transform(x)
    assert z[-1, 0] == 5 and abs(z[1:3, 0].mean()) < 0.5
