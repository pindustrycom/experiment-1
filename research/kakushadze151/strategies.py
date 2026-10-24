"""Strategies from Kakushadze & Serur, "151 Trading Strategies" (2018, SSRN 3247865), as target-weight builders.

Every function returns a DataFrame of target weights on the price index: a row of numbers rebalances at that close,
a NaN row leaves the book alone. Section names follow the paper; parameters are the paper's examples where it gives
them, otherwise the usual textbook choice (noted inline)."""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- helpers


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(idx.to_period("M")).max().to_numpy())


def week_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(idx.to_period("W-FRI")).max().to_numpy())


def empty(P: pd.DataFrame, cols=None) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=P.index, columns=cols if cols is not None else P.columns)


def long_short(score: pd.Series, n: int, long_only=False) -> pd.Series:
    """Equal weights: +1/n on the n highest scores, -1/n on the n lowest (dollar neutral, gross 2)."""
    s = score.dropna()
    w = pd.Series(0.0, index=score.index)
    if len(s) < 2 * n:
        return w
    s = s.sort_values()
    w[s.index[-n:]] = 1.0 / n
    if not long_only:
        w[s.index[:n]] = -1.0 / n
    return w


def monthly(P: pd.DataFrame, score_fn, n: int, long_only=False, min_hist=252) -> pd.DataFrame:
    W = empty(P)
    for d in month_ends(P.index):
        i = P.index.get_loc(d)
        if i < min_hist:
            continue
        W.loc[d] = long_short(score_fn(i), n, long_only)
    return W


def signal_weights(sig: pd.Series, col: str) -> pd.DataFrame:
    """Daily +1/0/-1 position in one asset, rebalanced every close."""
    return pd.DataFrame({col: sig.astype(float)})


def available(P: pd.DataFrame, i: int, need: int) -> pd.Index:
    """Assets with a full price history over the last `need` days at row i."""
    win = P.iloc[i - need:i + 1]
    return win.columns[win.notna().all()]


# ---------------------------------------------------------------- 3. Stocks


def price_momentum(P, n=10, long_only=False):
    """3.1 Price-momentum: 12-month return skipping the last month, top vs bottom decile, monthly."""
    def score(i):
        c = available(P, i, 252)
        return (P.iloc[i - 21][c] / P.iloc[i - 252][c] - 1).reindex(P.columns)
    return monthly(P, score, n, long_only)


def low_vol(P, n=10, long_only=False):
    """3.4 Low-volatility anomaly: long the calmest stocks (1-year daily vol), short the wildest."""
    r = P.pct_change(fill_method=None)

    def score(i):
        c = available(P, i, 252)
        return (-r.iloc[i - 251:i + 1][c].std()).reindex(P.columns)
    return monthly(P, score, n, long_only)


def residual_momentum(P, mkt: pd.Series, n=10):
    """3.7 Residual momentum: momentum of the part of each stock's return the market does not explain
    (1-year daily regression on SPY, months t-12..t-1, scaled by residual vol)."""
    r = P.pct_change(fill_method=None)
    m = mkt.pct_change().reindex(P.index)

    def score(i):
        c = available(P, i, 252)
        y = r.iloc[i - 251:i + 1][c]
        x = m.iloc[i - 251:i + 1]
        xm = x - x.mean()
        beta = (y - y.mean()).mul(xm, axis=0).sum() / (xm ** 2).sum()
        alpha = y.mean() - beta * x.mean()
        e = y - alpha - np.outer(x, beta)
        e = e.iloc[:-21]
        return (e.sum() / e.std()).reindex(P.columns)
    return monthly(P, score, n)


def pairs(P, n_pairs=20, form=252, z_open=2.0):
    """3.8 Pairs trading (distance method): each month pick the 20 pairs whose normalised prices tracked each other
    best over the last year; during the next month open when the gap is 2 formation sigmas wide (short the rich leg,
    long the cheap one), close when it crosses back. Everything is closed at the month end and re-formed."""
    W = pd.DataFrame(0.0, index=P.index, columns=P.columns)
    ends = month_ends(P.index)
    for k in range(len(ends) - 1):
        i = P.index.get_loc(ends[k])
        j = P.index.get_loc(ends[k + 1])
        if i < form:
            continue
        c = available(P, i, form)
        base = P.iloc[i - form][c]
        norm = (P.iloc[i - form:i + 1][c] / base).to_numpy()
        # sum of squared distance for all pairs at once
        sq = (norm ** 2).sum(0)
        ssd = sq[:, None] + sq[None, :] - 2 * norm.T @ norm
        iu = np.triu_indices(len(c), 1)
        best = np.argsort(ssd[iu])[:n_pairs]
        trade = P.iloc[i + 1:j + 1][c] / base
        for b in best:
            a1, a2 = c[iu[0][b]], c[iu[1][b]]
            spread_f = norm[:, iu[0][b]] - norm[:, iu[1][b]]
            mu, sd = spread_f.mean(), spread_f.std()
            s = ((trade[a1] - trade[a2]) - mu) / sd
            pos = 0
            for t, z in zip(trade.index[:-1], s.to_numpy()[:-1]):    # flat at the month end
                if pos == 0 and abs(z) > z_open:
                    pos = -np.sign(z)
                elif pos != 0 and np.sign(z) == pos:                   # crossed the mean
                    pos = 0
                W.loc[t, a1] += pos / n_pairs
                W.loc[t, a2] -= pos / n_pairs
    return W


def mean_reversion(P, clusters: pd.DataFrame | None = None):
    """3.9 / 3.10 Mean-reversion: each day short yesterday's winners and buy losers in proportion to their return
    minus the average (of everything, or of the stock's cluster). Dollar neutral, gross 2, rebalanced daily."""
    r = np.log(P).diff()
    if clusters is None:
        dm = r.sub(r.mean(axis=1), axis=0)
    else:
        cl = clusters.reindex(P.index).ffill()
        dm = r.copy() * np.nan
        for g in pd.unique(cl.to_numpy().ravel()):
            if not isinstance(g, str):
                continue
            mask = cl.eq(g)
            rg = r.where(mask)
            dm = dm.fillna(rg.sub(rg.mean(axis=1), axis=0))
    w = -dm
    return w.div(w.abs().sum(axis=1), axis=0).mul(2).fillna(0.0)


def sector_clusters(P, sectors: pd.DataFrame) -> pd.DataFrame:
    """Cluster = the sector ETF each stock moved with most over the last year, re-assigned monthly."""
    r = P.pct_change(fill_method=None)
    rs = sectors.pct_change(fill_method=None).reindex(P.index)
    out = pd.DataFrame(np.nan, index=P.index, columns=P.columns, dtype=object)
    for d in month_ends(P.index):
        i = P.index.get_loc(d)
        if i < 252:
            continue
        x = r.iloc[i - 251:i + 1]
        y = rs.iloc[i - 251:i + 1].dropna(axis=1)
        if y.shape[1] == 0:
            continue
        xs, ys = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
        corr = xs.fillna(0).T @ ys.fillna(0)
        lab = corr.idxmax(1).where(x.notna().sum() > 200)
        out.loc[d] = lab
    return out.ffill()


def alpha_combo(P, mkt, n=10):
    """3.21 Alpha combos: equal-weight average of cross-sectional z-scores of momentum, residual momentum and
    low volatility; top vs bottom decile, monthly."""
    r = P.pct_change(fill_method=None)
    m = mkt.pct_change().reindex(P.index)

    def z(s):
        return (s - s.mean()) / s.std()

    def score(i):
        c = available(P, i, 252)
        mom = P.iloc[i - 21][c] / P.iloc[i - 252][c] - 1
        y = r.iloc[i - 251:i + 1][c]
        x = m.iloc[i - 251:i + 1]
        xm = x - x.mean()
        beta = (y - y.mean()).mul(xm, axis=0).sum() / (xm ** 2).sum()
        e = (y - y.mean() - np.outer(xm, beta)).iloc[:-21]
        res = e.sum() / e.std()
        vol = -y.std()
        return ((z(mom) + z(res) + z(vol)) / 3).reindex(P.columns)
    return monthly(P, score, n)


def knn(close: pd.Series, k=50, train=1000, col="SPY"):
    """3.18 Machine learning, single-stock k-nearest neighbours: features are the last 5 daily returns and the
    20/60-day vol ratio; label is the next day's direction. Refit monthly on the last 1000 days; long when the
    neighbours say up, else in cash."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    r = close.pct_change()
    X = pd.concat([r.shift(j) for j in range(5)] + [r.rolling(20).std() / r.rolling(60).std()], axis=1)
    X.columns = range(X.shape[1])
    y = (r.shift(-1) > 0).astype(int)
    sig = pd.Series(np.nan, index=close.index)
    ends = set(month_ends(close.index))
    model = sc = None
    for i, d in enumerate(close.index):
        if i < train + 60:
            continue
        if model is None or d in ends:
            Xt, yt = X.iloc[i - train:i - 1], y.iloc[i - train:i - 1]   # labels known by day i
            ok = Xt.notna().all(axis=1)
            sc = StandardScaler().fit(Xt[ok])
            model = KNeighborsClassifier(k).fit(sc.transform(Xt[ok]), yt[ok])
        sig.iloc[i] = float(model.predict(sc.transform(X.iloc[[i]]))[0])
    return signal_weights(sig, col)


def ann_crypto(close: pd.Series, train=730, col="BTC"):
    """18.2 Artificial neural network on bitcoin: a small MLP on the last 10 daily returns predicts tomorrow's
    direction; refit monthly on the last two years; long when it says up, else in cash."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    r = close.pct_change()
    X = pd.concat([r.shift(j) for j in range(10)], axis=1)
    X.columns = range(X.shape[1])
    y = (r.shift(-1) > 0).astype(int)
    sig = pd.Series(np.nan, index=close.index)
    ends = set(month_ends(close.index))
    model = sc = None
    first = close.first_valid_index()
    start = close.index.get_loc(first) + train + 20
    for i in range(start, len(close)):
        d = close.index[i]
        if model is None or d in ends:
            Xt, yt = X.iloc[i - train:i - 1], y.iloc[i - train:i - 1]
            ok = Xt.notna().all(axis=1)
            sc = StandardScaler().fit(Xt[ok])
            model = MLPClassifier((16, 8), max_iter=500, random_state=0, alpha=1e-3).fit(sc.transform(Xt[ok]), yt[ok])
        sig.iloc[i] = float(model.predict(sc.transform(X.iloc[[i]].fillna(0)))[0])
    return signal_weights(sig, col)


# ---------------------------------------------------------------- 3.12-3.16 technical rules (any one asset)


def sma_rule(p: pd.Series, n=200, short=True):
    """3.12 Single moving average: long above the 200-day average, short (or cash) below."""
    m = p.rolling(n).mean()
    return ((p > m).astype(float) - ((p < m) & short).astype(float)).where(m.notna())


def two_ma(p: pd.Series, fast=50, slow=200, short=True):
    """3.13 Two moving averages: long when the 50-day is above the 200-day, short (or cash) when below."""
    f, s = p.rolling(fast).mean(), p.rolling(slow).mean()
    return ((f > s).astype(float) - ((f < s) & short).astype(float)).where(s.notna())


def three_ma(p: pd.Series, a=10, b=50, c=200, short=True):
    """3.14 Three moving averages: long when 10 > 50 > 200, short when 10 < 50 < 200, else flat."""
    x, y, z = p.rolling(a).mean(), p.rolling(b).mean(), p.rolling(c).mean()
    return (((x > y) & (y > z)).astype(float) - (((x < y) & (y < z)) & short).astype(float)).where(z.notna())


def pivot(d: pd.DataFrame, short=True):
    """3.15 Support and resistance: pivot C = (H+L+close)/3 of the previous day, R = 2C-L, S = 2C-H.
    Go long when the close is above C and take profit at R; go short below C and cover at S."""
    C = ((d.high + d.low + d.close) / 3).shift()
    R = 2 * C - d.low.shift()
    S = 2 * C - d.high.shift()
    pos, out = 0.0, []
    for p, c, r_, s_ in zip(d.close, C, R, S):
        if np.isnan(c):
            out.append(np.nan)
            continue
        if pos > 0 and p >= r_ or pos < 0 and p <= s_:
            pos = 0.0
        elif pos == 0:
            pos = 1.0 if p > c else (-1.0 if (p < c and short) else 0.0)
        out.append(pos)
    return pd.Series(out, index=d.index)


def channel(d: pd.DataFrame, n=20, short=True):
    """3.16 Channel: buy when the price touches the 20-day floor, sell (short) when it touches the 20-day ceiling."""
    up = d.high.rolling(n).max()
    lo = d.low.rolling(n).min()
    pos, out = 0.0, []
    for p, u, lw in zip(d.close, up, lo):
        if np.isnan(u):
            out.append(np.nan)
            continue
        if p <= lw:
            pos = 1.0
        elif p >= u:
            pos = -1.0 if short else 0.0
        out.append(pos)
    return pd.Series(out, index=d.index)


def ts_momentum(p: pd.Series, n=252, short=True):
    """Time-series momentum (10.4 trend following, one asset): sign of the 12-month return, checked monthly."""
    s = np.sign(p / p.shift(n) - 1)
    if not short:
        s = s.clip(lower=0)
    out = pd.Series(np.nan, index=p.index)
    me = month_ends(p.index)
    out[me] = s[me]
    return out.ffill()


# ---------------------------------------------------------------- 4. ETFs


def sector_rotation(P, n=3, mkt: pd.Series | None = None, rf_price: pd.Series | None = None, mode="plain"):
    """4.1 Sector momentum rotation: hold the 3 sectors with the best 12-month return, monthly.
    mode="ma": only sectors above their own 200-day average (rest in cash). mode="dual": only sectors whose 12-month
    return beats T-bills (dual momentum)."""
    W = empty(P)
    for d in month_ends(P.index):
        i = P.index.get_loc(d)
        if i < 252:
            continue
        c = available(P, i, 252)
        mom = P.iloc[i][c] / P.iloc[i - 252][c] - 1
        top = mom.sort_values().index[-n:]
        w = pd.Series(0.0, index=P.columns)
        for t in top:
            if mode == "ma" and P.iloc[i][t] < P.iloc[i - 199:i + 1][t].mean():
                continue
            if mode == "dual" and mom[t] <= rf_price.iloc[i] / rf_price.iloc[i - 252] - 1:
                continue
            w[t] = 1.0 / n
        W.loc[d] = w
    return W


def alpha_rotation(P, mkt: pd.Series, n=3, r2_filter=False):
    """4.2 Alpha rotation: hold the 3 ETFs with the highest 1-year Jensen alpha against SPY, monthly.
    4.3 R-squared: same, but only among ETFs whose regression R-squared is at least the median (alpha that the
    market model explains well is more likely to persist, per the paper's reference)."""
    r = P.pct_change(fill_method=None)
    m = mkt.pct_change().reindex(P.index)
    W = empty(P)
    for d in month_ends(P.index):
        i = P.index.get_loc(d)
        if i < 252:
            continue
        c = available(P, i, 252)
        y, x = r.iloc[i - 251:i + 1][c], m.iloc[i - 251:i + 1]
        xm = x - x.mean()
        beta = (y - y.mean()).mul(xm, axis=0).sum() / (xm ** 2).sum()
        alpha = y.mean() - beta * x.mean()
        r2 = beta ** 2 * x.var() / y.var()
        if r2_filter:
            alpha = alpha[r2 >= r2.median()]
        top = alpha.sort_values().index[-n:]
        w = pd.Series(0.0, index=P.columns)
        w[top] = 1.0 / n
        W.loc[d] = w
    return W


def ibs(d: pd.DataFrame, lo=0.2, hi=0.8, short=True):
    """4.4 ETF mean-reversion with internal bar strength IBS = (close-low)/(high-low): long below 0.2, short above
    0.8, flat between, decided at the close and held one day."""
    x = (d.close - d.low) / (d.high - d.low)
    return (x < lo).astype(float) - ((x > hi) & short).astype(float)


def leveraged_pair(P, a="TQQQ", b="SQQQ", size=0.5):
    """4.5 Leveraged ETFs: short both the 3x bull and the 3x bear fund (half the capital each), let them decay,
    reset monthly."""
    W = empty(P, [a, b])
    for d in month_ends(P.index):
        W.loc[d] = [-size, -size]
    return W


def trend_multi(P, n=252, vol_n=60):
    """4.6 Multi-asset trend following / 10.4 futures trend: each asset long if its 12-month return beats zero,
    inverse-vol weights over the assets that are in, monthly."""
    r = P.pct_change(fill_method=None)
    W = empty(P)
    for d in month_ends(P.index):
        i = P.index.get_loc(d)
        if i < n:
            continue
        c = available(P, i, n)
        mom = P.iloc[i][c] / P.iloc[i - n][c] - 1
        iv = 1 / r.iloc[i - vol_n + 1:i + 1][c].std()
        iv = iv.where(mom > 0, 0.0)
        w = pd.Series(0.0, index=P.columns)
        if iv.sum() > 0:
            w[c] = iv / iv.sum()
        W.loc[d] = w
    return W


# ---------------------------------------------------------------- 6. Indexes, 7. Volatility


def vol_target(p: pd.Series, target=0.15, n=60, cap=1.0, col="SPY"):
    """6.4 Index volatility targeting with a risk-free asset: SPY weight = 15% / realised 60-day vol, capped at 1,
    rest in T-bills, rebalanced weekly."""
    v = p.pct_change().rolling(n).std() * np.sqrt(252)
    w = (target / v).clip(upper=cap)
    out = pd.Series(np.nan, index=p.index)
    we = week_ends(p.index)
    out[we] = w[we]
    return pd.DataFrame({col: out})


def vix_basis(vix: pd.Series, vix3m: pd.Series, both=True, col="VIXY"):
    """7.1 VIX futures basis: short VIX futures (VIXY) while the curve is in contango (VIX below VIX3M), long in
    backwardation (or flat when both=False), half the capital, checked daily."""
    ratio = vix / vix3m
    s = -(ratio < 1).astype(float) + ((ratio > 1) & both).astype(float)
    return pd.DataFrame({col: 0.5 * s.where(ratio.notna())})


def vol_carry_etns(P, short="VIXY", hedge="VIXM", n=60, size=0.5):
    """7.2 Volatility carry with two ETNs: short the front-month VIX fund, long the mid-term one with a rolling
    beta hedge ratio (the mid curve decays less), rebalanced weekly."""
    r = P[[short, hedge]].pct_change(fill_method=None)
    beta = r[short].rolling(n).cov(r[hedge]) / r[hedge].rolling(n).var()
    W = empty(P, [short, hedge])
    for d in week_ends(P.index):
        if np.isnan(beta[d]):
            continue
        W.loc[d] = [-size, size * beta[d]]
    return W


def short_vol(size=0.2, col="VIXY", index=None):
    """7.3 Volatility risk premium: always short front-month VIX futures (VIXY) with 20% of capital, weekly reset."""
    out = pd.Series(np.nan, index=index)
    out[week_ends(index)] = -size
    return pd.DataFrame({col: out})


# ---------------------------------------------------------------- 8. FX


def hp_trend(p: pd.Series, lam=1600 * 30, win=500, fast=10, slow=50):
    """8.1 Moving averages with HP filter: run a one-sided Hodrick-Prescott filter (trailing 500 days, so no look
    ahead), then a 10/50 moving-average crossover on the filtered series. Long when fast > slow, else short."""
    from scipy.sparse import diags, eye
    from scipy.sparse.linalg import spsolve
    x = np.log(p.to_numpy())
    D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(win - 2, win))
    A = (eye(win) + lam * D.T @ D).tocsc()
    filt = np.full(len(x), np.nan)
    for t in range(win, len(x)):
        seg = x[t - win + 1:t + 1]
        if np.isnan(seg).any():
            continue
        filt[t] = spsolve(A, seg)[-1]
    f = pd.Series(filt, index=p.index)
    a, b = f.rolling(fast).mean(), f.rolling(slow).mean()
    return (np.sign(a - b)).where(b.notna())


def fx_carry(rates: pd.DataFrame, P: pd.DataFrame, n=2, mode="carry", mom_n=63):
    """8.2 Carry: long the 2 highest-rate currencies, short the 2 lowest (USD counts as a currency with zero
    excess return), monthly. 8.3 Dollar carry: long all foreign currencies if their average rate beats the US
    rate, else short them all. 8.4 Momentum & carry combo: rank on carry rank + 3-month momentum rank."""
    W = empty(P)
    for d in month_ends(P.index):
        i = P.index.get_loc(d)
        if i < mom_n:
            continue
        rt = rates.loc[:d].iloc[-1] if len(rates.loc[:d]) else None
        if rt is None or rt.isna().any():
            continue
        diff = rt[P.columns] - rt["USD"]
        w = pd.Series(0.0, index=P.columns)
        if mode == "dollar":
            w[:] = (1.0 if diff.mean() > 0 else -1.0) / len(diff)
        else:
            score = diff.copy()
            score["USD"] = 0.0
            if mode == "combo":
                mom = (P.iloc[i] / P.iloc[i - mom_n] - 1)
                mom["USD"] = 0.0
                score = score.rank() + mom.rank()
            s = score.sort_values()
            for t in s.index[-n:]:
                if t != "USD":
                    w[t] += 1.0 / n
            for t in s.index[:n]:
                if t != "USD":
                    w[t] -= 1.0 / n
        W.loc[d] = w
    return W


# ---------------------------------------------------------------- 9. Commodities, 10. Futures


def commodity_value(P, n=3):
    """9.4 Value: 'cheap' = price now vs the average price 4.5-5.5 years ago; long the 3 cheapest, short the 3
    dearest, monthly."""
    def score(i):
        c = available(P, i, 1386)
        past = P.iloc[i - 1386:i - 1134][c].mean()
        return (past / P.iloc[i][c]).reindex(P.columns)
    return monthly(P, score, n, min_hist=1386)


def skewness(P, n=3):
    """9.5 Skewness premium: long the 3 commodities with the most negative 1-year daily-return skew, short the 3
    with the most positive, monthly."""
    r = P.pct_change(fill_method=None)

    def score(i):
        c = available(P, i, 252)
        return (-r.iloc[i - 251:i + 1][c].skew()).reindex(P.columns)
    return monthly(P, score, n)


def contrarian(P):
    """10.3 Contrarian (mean-reversion) on futures: each week long last week's losers and short its winners in
    proportion to the return minus the average, dollar neutral, gross 2."""
    W = empty(P)
    for d in week_ends(P.index):
        i = P.index.get_loc(d)
        if i < 6:
            continue
        c = available(P, i, 5)
        x = P.iloc[i][c] / P.iloc[i - 5][c] - 1
        w = -(x - x.mean())
        W.loc[d] = (w / w.abs().sum() * 2).reindex(P.columns).fillna(0.0)
    return W


def equal_weight(P):
    """Benchmark: equal weight in every stock that trades, rebalanced monthly."""
    W = empty(P)
    for d in month_ends(P.index):
        ok = P.loc[d].notna()
        W.loc[d] = ok / ok.sum()
    return W


def fixed_mix(P, weights: dict):
    """9.3 Portfolio diversification with commodities: fixed weights, rebalanced monthly."""
    W = empty(P, list(weights))
    for d in month_ends(P.index):
        W.loc[d] = list(weights.values())
    return W


# ---------------------------------------------------------------- 19. Global macro

FOMC = """2013-01-30 2013-03-20 2013-05-01 2013-06-19 2013-07-31 2013-09-18 2013-10-30 2013-12-18
2014-01-29 2014-03-19 2014-04-30 2014-06-18 2014-07-30 2014-09-17 2014-10-29 2014-12-17
2015-01-28 2015-03-18 2015-04-29 2015-06-17 2015-07-29 2015-09-17 2015-10-28 2015-12-16
2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14
2017-02-01 2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01 2017-12-13
2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19
2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11
2020-01-29 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16"""


def pre_fomc(index: pd.DatetimeIndex, col="SPY"):
    """19.5 Trading on economic announcements: buy SPY at the close the day before a scheduled FOMC decision, sell
    at the decision day's close (the pre-FOMC drift). Cash otherwise."""
    days = pd.DatetimeIndex(FOMC.split())
    pos = pd.Series(0.0, index=index)
    for d in days:
        if d in index:
            i = index.get_loc(d)
            if i:
                pos.iloc[i - 1] = 1.0
    return pd.DataFrame({col: pos})
