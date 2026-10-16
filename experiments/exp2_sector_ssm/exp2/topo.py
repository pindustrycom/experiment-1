"""Topological stat arb on a correlation graph of large US stocks, tested against plain reversal.

The idea (from a clip Joaquin shared): build a weighted graph of stocks from their correlations, smooth
each day's cross-section of returns over the graph, h = (I - alpha L)^J x, and trade the residual
e = x - h back towards its neighbours (market neutral). Persistent homology of the correlation
distances, d_ij = sqrt(2 (1 - rho_ij)), summarises the shape of the market (Betti curves, persistence
diagram) and sets how much to trade: w = f(phi_topo, constraints).

Mixed with what Experiments 2 and 3 already use:
- resonance normalisation: the graph filter's frequency response is (1 - alpha lambda)^J on each Laplacian
  eigenmode; the residual's components are divided by the square root of the mode's measured power
  (the graph analogue of the resonance-index input gain), so the most energetic modes don't dominate;
- volatility as a state: the book is scaled to a constant risk with an exponentially weighted variance
  of its own past returns (only information available before each day).

Everything is chosen on 2012-2018 and frozen for 2019 onwards. The comparisons that decide whether the
graph and the topology add anything: plain cross-sectional reversal, reversal within GICS sectors,
reversal of the residual after 3 principal components (Avellaneda-Lee), the graph residual on a
shuffled graph (placebo), and SPY. Paper backtest only; costs are charged on turnover.

    python -m exp2 topo --data topo_data --out topo_results
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("exp2.topo")

SECTORS = {
    "tech": ["AAPL", "MSFT", "NVDA", "ORCL", "INTC", "CSCO", "AMD", "QCOM", "TXN", "ADBE", "CRM", "IBM"],
    "comm": ["GOOGL", "META", "DIS", "T", "VZ"],
    "discretionary": ["AMZN", "HD", "MCD", "NKE", "SBUX"],
    "financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP"],
    "health": ["JNJ", "PFE", "MRK", "UNH", "ABT", "LLY"],
    "staples": ["PG", "KO", "PEP", "WMT", "COST"],
    "energy": ["XOM", "CVX", "COP"],
    "industrials": ["CAT", "BA", "HON", "GE", "UNP"],
    "utilities": ["NEE", "DUK"],
}
UNIVERSE = [t for v in SECTORS.values() for t in v]
MARKET = "SPY"
START = "2011-01-01"
TRAIN_END = "2018-12-31"
EPS_MAX = 1.4                       # largest correlation distance the Rips complex is built to (rho = 0.02)
EPS_GRID = np.linspace(0.2, EPS_MAX, 13)
GRID = {"window": (60, 120), "lookback": (1, 5), "alpha": (0.5, 0.9), "J": (1, 3, 10)}
TOPO_EVERY = 5                      # sessions between persistent-homology snapshots
VOL_TARGET = 0.10                   # annual volatility of the vol-targeted book
VOL_HALFLIFE = 20


# ---------------------------------------------------------------- data

def fetch_yahoo(ticker: str, start: str) -> pd.DataFrame:
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    r = requests.get(f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
                     params={"period1": t0, "period2": int(time.time()), "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]
    dates = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York").strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "close": q["quote"][0]["close"], "adj_close": q["adjclose"][0]["adjclose"]}).dropna()


def fetch_tiingo(ticker: str, start: str) -> pd.DataFrame:
    import os

    from .data import fetch_tiingo as ft
    key = os.environ.get("TIINGO_API_KEY")
    if not key:
        raise RuntimeError("no TIINGO_API_KEY")
    return ft(ticker.replace(".", "-"), start, key)[["date", "close", "adj_close"]]


def update_prices(data: Path, tickers: list[str], fetchers=(fetch_yahoo, fetch_tiingo), pause: float = 1.0, max_age_days: int = 3) -> dict:
    """One CSV per ticker; a ticker is refetched (whole history, so adjustments stay consistent) when it is stale."""
    d = data / "prices"
    d.mkdir(parents=True, exist_ok=True)
    out = {"fetched": [], "failed": {}}
    for t in tickers:
        f = d / f"{t}.csv"
        if f.exists() and (time.time() - f.stat().st_mtime) < max_age_days * 86400:
            continue
        for fetch in fetchers:
            try:
                df = fetch(t, START)
                if len(df) < 500:
                    raise RuntimeError(f"only {len(df)} rows")
                df.to_csv(f, index=False)
                out["fetched"].append(t)
                out["failed"].pop(t, None)
                break
            except Exception as ex:  # noqa: BLE001 - try the next source, report what failed
                out["failed"][t] = f"{fetch.__name__}: {type(ex).__name__}: {str(ex)[:120]}"
                time.sleep(pause)
        time.sleep(pause)
    return out


def load_returns(data: Path, tickers: list[str]) -> pd.DataFrame:
    """Daily log returns (in %), one column per ticker, on the market's sessions."""
    px = {}
    for t in tickers:
        f = data / "prices" / f"{t}.csv"
        if f.exists():
            s = pd.read_csv(f, parse_dates=["date"]).drop_duplicates("date").set_index("date")["adj_close"].astype(float)
            px[t] = s
    p = pd.DataFrame(px).sort_index()
    return 100 * np.log(p).diff().iloc[1:]


# ---------------------------------------------------------------- persistent homology

def rips_persistence(D: np.ndarray, eps_max: float = EPS_MAX) -> tuple[np.ndarray, np.ndarray]:
    """Vietoris-Rips persistence in dimensions 0 and 1 of a distance matrix (exact, by column reduction
    over Z/2 of the 2-skeleton up to eps_max). Returns H0 deaths (all births are 0; the one infinite bar
    is left out) and H1 (birth, death) pairs; cycles still open at eps_max die at eps_max."""
    n = len(D)
    iu, ju = np.triu_indices(n, 1)
    el = D[iu, ju]
    order = np.argsort(el, kind="stable")
    rank = np.empty(len(el), int)
    rank[order] = np.arange(len(el))
    erank = np.full((n, n), -1)
    erank[iu, ju] = rank
    erank[ju, iu] = rank
    length = el[order]                                 # edge length by rank
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    h0, positive = [], np.zeros(len(el), bool)
    for k in range(len(el)):
        a, b = find(iu[order[k]]), find(ju[order[k]])
        if a == b:
            positive[k] = True                         # closes a cycle: an H1 class is born
        else:
            parent[a] = b
            h0.append(length[k])
    m = int(np.searchsorted(length, eps_max, side="right"))
    # triangles with every edge in the complex, ordered by their longest edge
    tri = []
    ok = erank < m
    for i in range(n):
        for j in range(i + 1, n):
            if not ok[i, j]:
                continue
            ks = np.nonzero(ok[i, j + 1:] & ok[j, j + 1:])[0] + j + 1
            for k in ks:
                e = sorted((erank[i, j], erank[i, k], erank[j, k]))
                tri.append((e[2], e[1], e[0]))
    tri.sort()
    pivots, h1 = {}, []
    for t in tri:
        col = set(t)
        while col:
            low = max(col)
            if low in pivots:
                col ^= pivots[low]
            else:
                pivots[low] = col
                if length[t[0]] > length[low]:
                    h1.append((length[low], length[t[0]]))
                break
    for k in np.nonzero(positive[:m])[0]:
        if k not in pivots:
            h1.append((length[k], eps_max))
    return np.array(h0), np.array(h1).reshape(-1, 2)


def topo_features(h0: np.ndarray, h1: np.ndarray, n: int, grid: np.ndarray = EPS_GRID) -> dict:
    """phi_topo: Betti curves over epsilon and summaries of the persistence diagram."""
    b0 = np.array([n - np.sum(h0 <= e) for e in grid])
    b1 = np.array([np.sum((h1[:, 0] <= e) & (h1[:, 1] > e)) for e in grid]) if len(h1) else np.zeros(len(grid))
    p1 = h1[:, 1] - h1[:, 0] if len(h1) else np.zeros(0)
    q = h0 / h0.sum() if h0.sum() > 0 else np.ones(1)
    return {"h0_total": float(h0.sum()), "h0_mean": float(h0.mean()), "h0_entropy": float(-(q * np.log(q)).sum()),
            "h1_count": float(len(p1)), "h1_total": float(p1.sum()), "h1_max": float(p1.max()) if len(p1) else 0.0,
            "betti0_area": float(b0.mean()), "betti1_area": float(b1.mean()), "betti1_peak_eps": float(grid[int(np.argmax(b1))]) if b1.any() else 0.0}


TOPO_COLS = ["h0_total", "h0_mean", "h0_entropy", "h1_count", "h1_total", "h1_max", "betti0_area", "betti1_area", "betti1_peak_eps"]


def topo_series(R: pd.DataFrame, window: int, every: int = TOPO_EVERY) -> pd.DataFrame:
    """phi_topo every `every` sessions from the trailing correlation window, carried forward between snapshots."""
    X = R.to_numpy()
    rows = {}
    for t in range(window, len(R), every):
        C = np.corrcoef(X[t - window + 1:t + 1].T)
        D = np.sqrt(np.clip(2 * (1 - C), 0, None))
        h0, h1 = rips_persistence(D)
        rows[R.index[t]] = topo_features(h0, h1, len(D))
    return pd.DataFrame.from_dict(rows, orient="index").reindex(R.index).ffill()


# ---------------------------------------------------------------- signals

def graph_laplacian(C: np.ndarray, k: int = 6) -> np.ndarray:
    """Normalised Laplacian of a symmetric k-nearest-neighbour graph, weights exp(-d^2) on d = sqrt(2(1 - rho))."""
    D2 = np.clip(2 * (1 - C), 0, None)
    n = len(C)
    W = np.exp(-D2)
    np.fill_diagonal(W, 0)
    keep = np.zeros_like(W, bool)
    nn = np.argsort(-W, axis=1)[:, :k]
    keep[np.repeat(np.arange(n), k), nn.ravel()] = True
    W = np.where(keep | keep.T, W, 0.0)
    deg = W.sum(1)
    s = 1 / np.sqrt(np.where(deg > 0, deg, 1))
    return np.eye(n) - s[:, None] * W * s[None, :]


def _neutral(w: np.ndarray) -> np.ndarray:
    w = w - w.mean()
    g = np.abs(w).sum()
    return w / g if g > 0 else w


def signals(R: pd.DataFrame, window: int, lookback: int, params: list[tuple[float, int]], perm: np.ndarray | None = None) -> dict:
    """Target weights at each close (rows) for the reversal signals; each row sums to 0 with gross 1.

    x is each stock's last-`lookback`-session return over its own volatility. Every signal trades the
    part of x that its reference doesn't explain back towards it: the cross-sectional mean (xs), the
    GICS sector mean (sector), 3 principal components (pca), or the graph-smoothed x for each (alpha, J)
    (graph), also with each Laplacian mode's residual divided by its measured power (graph_res)."""
    X = R.to_numpy()
    T, n = X.shape
    names = list(R.columns)
    sec = np.array([next(s for s, v in SECTORS.items() if t in v) for t in names])
    same = (sec[:, None] == sec[None, :]).astype(float)
    same /= same.sum(1, keepdims=True)
    out = {k: np.full((T, n), np.nan) for k in ["xs", "sector", "pca"]}
    for a, J in params:
        out[f"graph_a{a}_J{J}"] = np.full((T, n), np.nan)
        out[f"graphres_a{a}_J{J}"] = np.full((T, n), np.nan)
    cum = np.cumsum(np.nan_to_num(X), 0)
    power = None
    for t in range(window, T):
        Y = X[t - window + 1:t + 1]
        vol = Y.std(0) + 1e-9
        x = (cum[t] - cum[t - lookback]) / (vol * math.sqrt(lookback))
        out["xs"][t] = _neutral(-(x - x.mean()))
        out["sector"][t] = _neutral(-(x - same @ x))
        C = np.corrcoef(Y.T)
        ev, U = np.linalg.eigh(C)
        P = U[:, -3:]
        out["pca"][t] = _neutral(-(x - P @ (P.T @ x)))
        Cg = C if perm is None else C[np.ix_(perm, perm)]
        lam, V = np.linalg.eigh(graph_laplacian(Cg))
        # each graph mode's power in the cross-section of standardised returns over the window
        Zw = (Y / vol) @ V
        pw = (Zw ** 2).mean(0)
        power = pw / pw.mean()
        xv = V.T @ x
        for a, J in params:
            resp = 1 - (1 - a * lam) ** J                    # the residual's response on each graph mode
            out[f"graph_a{a}_J{J}"][t] = _neutral(-(V @ (resp * xv)))
            out[f"graphres_a{a}_J{J}"][t] = _neutral(-(V @ (resp * xv / np.sqrt(power))))
    return out


def backtest(W: np.ndarray, R: pd.DataFrame, cost_bp: float = 5.0, delay: int = 0, scale: np.ndarray | None = None) -> pd.Series:
    """Daily P&L (in %) of holding the weights set at close t over session t+1+delay, net of cost_bp per unit traded."""
    X = np.nan_to_num(R.to_numpy())
    Wd = np.nan_to_num(W)
    if scale is not None:
        Wd = Wd * np.nan_to_num(scale)[:, None]
    held = np.roll(Wd, 1 + delay, axis=0)
    held[: 1 + delay] = 0
    pnl = (held * X).sum(1)
    turn = np.abs(np.diff(held, axis=0, prepend=0)).sum(1)
    return pd.Series(pnl - turn * cost_bp / 100, index=R.index)


def vol_scale(pnl: pd.Series, target: float = VOL_TARGET, halflife: int = VOL_HALFLIFE) -> np.ndarray:
    """Exposure that targets a constant annual volatility, from the book's own past variance (lagged a day)."""
    v = (pnl ** 2).ewm(halflife=halflife, min_periods=halflife).mean().shift(1)
    s = (target * 100 / math.sqrt(252)) / np.sqrt(v)
    return s.clip(upper=5 * s.median()).to_numpy()


def stats(p: pd.Series) -> dict:
    p = p.dropna()
    if len(p) < 20 or p.std() == 0:
        return {"n": len(p)}
    eq = np.exp(np.log1p(p / 100).cumsum())
    dd = float((eq / eq.cummax() - 1).min())
    yrs = len(p) / 252
    return {"n": len(p), "sharpe": float(p.mean() / p.std() * math.sqrt(252)), "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
            "vol": float(p.std() * math.sqrt(252) / 100), "max_dd": dd, "t": float(p.mean() / p.std() * math.sqrt(len(p)))}


def deflated_sharpe(sr: float, n: int, trials: list[float], skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probability that the true Sharpe is above zero after picking the best of `trials` (Bailey & Lopez de Prado), daily units."""
    from statistics import NormalDist
    N = NormalDist()
    k = max(len(trials), 2)
    var = float(np.var(trials, ddof=1)) if len(trials) > 1 else 0.0
    g = 0.5772156649
    sr0 = math.sqrt(var) * ((1 - g) * N.inv_cdf(1 - 1 / k) + g * N.inv_cdf(1 - 1 / (k * math.e)))
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(N.cdf((sr - sr0) * math.sqrt(n - 1) / den))


def topo_gate(phi: pd.DataFrame, pnl: pd.Series, train: np.ndarray, horizon: int = 5) -> tuple[np.ndarray, dict]:
    """w = f(phi_topo): exposure in [0, 2] from a ridge regression, fitted on the training years, of the
    strategy's next-`horizon`-session P&L on the standardised topology features."""
    F = phi[TOPO_COLS].to_numpy(float)
    fwd = pnl.rolling(horizon).sum().shift(-horizon).to_numpy()
    ok = train & ~np.isnan(F).any(1) & ~np.isnan(fwd)
    mu, sd = F[ok].mean(0), F[ok].std(0) + 1e-9
    Z = (F - mu) / sd
    A = Z[ok]
    b = np.linalg.solve(A.T @ A + 50.0 * np.eye(A.shape[1]), A.T @ (fwd[ok] - fwd[ok].mean()))
    pred = np.nan_to_num(Z @ b)
    s = pred[ok].std() + 1e-12
    expo = np.clip(1 + pred / s, 0, 2)
    ic_tr = float(pd.Series(pred[ok]).rank().corr(pd.Series(fwd[ok]).rank()))
    return expo, {"coef": dict(zip(TOPO_COLS, np.round(b, 4).tolist())), "train_ic": ic_tr}


def topo_vol_ic(phi: pd.DataFrame, R: pd.DataFrame, test: np.ndarray, horizon: int = 5) -> dict:
    """Does the shape of the market lead its volatility? Rank correlation of each phi_topo feature with SPY's realised vol over the next week."""
    m = R[MARKET] if MARKET in R else R.mean(axis=1)
    fv = np.log((m ** 2).rolling(horizon).mean().shift(-horizon))
    out = {}
    for c in TOPO_COLS:
        for name, sel in (("train", ~test), ("test", test)):
            d = pd.DataFrame({"x": phi[c], "y": fv})[sel].iloc[::horizon].dropna()
            if len(d) > 30:
                out.setdefault(c, {})[name] = round(float(d["x"].rank().corr(d["y"].rank())), 3)
    return out


# ---------------------------------------------------------------- run

def run(data: Path, out: Path, fetch: bool = True, grid: dict = GRID, train_end: str = TRAIN_END, universe: list[str] = UNIVERSE) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    rep = {"built": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"), "universe": universe, "train_end": train_end, "warnings": []}
    if fetch:
        rep["fetch"] = update_prices(data, [*universe, MARKET])
    Rall = load_returns(data, [*universe, MARKET])
    have = [t for t in universe if t in Rall]
    missing = sorted(set(universe) - set(have))
    if missing:
        rep["warnings"].append(f"no prices for {missing}; left out")
    R = Rall[have].dropna()
    spy = Rall[MARKET].reindex(R.index) if MARKET in Rall else None
    rep["sessions"] = [R.index[0].strftime("%Y-%m-%d"), R.index[-1].strftime("%Y-%m-%d"), len(R)]
    first_trade = R.index[max(grid["window"]) + 5]
    train = np.asarray((R.index >= first_trade) & (R.index <= pd.Timestamp(train_end)))
    test = np.asarray(R.index > pd.Timestamp(train_end))
    params = list(product(grid["alpha"], grid["J"]))

    pnls, trials = {}, []
    for window, lookback in product(grid["window"], grid["lookback"]):
        W = signals(R, window, lookback, params)
        for name, w in W.items():
            key = f"{name}_w{window}_l{lookback}"
            p = backtest(w, R)
            pnls[key] = (p, w)
            trials.append(key)
        log.info("signals window %d lookback %d done (%.0fs)", window, lookback, time.time() - t0)

    def tr_sharpe(k):
        return stats(pnls[k][0][train]).get("sharpe", -9)

    # pick each family's settings on the training years only
    fams = {"xs": "xs_", "sector": "sector_", "pca": "pca_", "graph": "graph_a", "graph_res": "graphres_a"}
    chosen = {f: max((k for k in trials if k.startswith(pre)), key=tr_sharpe) for f, pre in fams.items()}
    rep["chosen"] = chosen
    best_graph = chosen["graph"]
    gw = int(best_graph.split("_w")[1].split("_")[0])
    gl = int(best_graph.split("_l")[1])
    ga, gJ = float(best_graph.split("_a")[1].split("_")[0]), int(best_graph.split("_J")[1].split("_")[0])

    # placebo: the same graph filter on a graph whose stocks are shuffled
    rng = np.random.default_rng(0)
    placebo = []
    for _ in range(5):
        Wp = signals(R, gw, gl, [(ga, gJ)], perm=rng.permutation(len(have)))[f"graph_a{ga}_J{gJ}"]
        placebo.append(backtest(Wp, R))
    rep["placebo_test_sharpe"] = [round(stats(p[test]).get("sharpe", float("nan")), 3) for p in placebo]

    # persistent homology of the correlation distances, and the gate it sets
    phi = topo_series(R, gw)
    phi.to_csv(out / "phi_topo.csv", float_format="%.5g")
    log.info("persistent homology done (%.0fs)", time.time() - t0)
    rep["topo_vol_ic"] = topo_vol_ic(phi, pd.concat([R, spy.rename(MARKET)], axis=1) if spy is not None else R, test)

    books = {f: pnls[k][0] for f, k in chosen.items()}
    wts = {f: pnls[k][1] for f, k in chosen.items()}
    gp, gwts = pnls[best_graph]
    expo, rep["topo_gate"] = topo_gate(phi, gp, train)
    books["graph_topo"] = backtest(gwts, R, scale=expo)
    rp, rwts = pnls[chosen["graph_res"]]
    expo_r, _ = topo_gate(phi, rp, train)
    books["graph_res_topo"] = backtest(rwts, R, scale=expo_r)
    books["graph_res_topo_vol"] = backtest(rwts, R, scale=expo_r * vol_scale(books["graph_res_topo"]))
    books["graph_vol"] = backtest(gwts, R, scale=vol_scale(gp))
    books["graph_delay1"] = backtest(gwts, R, delay=1)
    books["graph_cost0"] = backtest(gwts, R, cost_bp=0.0)
    books["graph_cost10"] = backtest(gwts, R, cost_bp=10.0)
    if spy is not None:
        books["spy"] = spy.fillna(0.0)

    all_train_sr = [stats(pnls[k][0][train]).get("sharpe", 0.0) / math.sqrt(252) for k in trials]
    res = {}
    for name, p in books.items():
        r = {"train": stats(p[train]), "test": stats(p[test])}
        pt = p[test].dropna()
        if len(pt) > 30 and name != "spy":
            sr = pt.mean() / pt.std()
            r["test"]["deflated_sharpe_prob"] = deflated_sharpe(sr, len(pt), all_train_sr, float(pt.skew()), float(pt.kurt() + 3))
            w = np.nan_to_num(wts.get(name, gwts))
            r["turnover_per_day"] = float(np.abs(np.diff(w, axis=0)).sum(1).mean())
        r["by_year"] = {str(y): round(float(np.expm1(np.log1p(g / 100).sum())), 4) for y, g in p[p.index >= first_trade].groupby(p.index[p.index >= first_trade].year)}
        res[name] = r
    rep["results"] = res
    rep["n_trials"] = len(trials)
    # does the graph beat plain reversal day by day in the test years?
    rep["graph_minus"] = {}
    for b in ("xs", "sector", "pca"):
        d = (books["graph"] - books[b])[test]
        rep["graph_minus"][b] = {"ann_diff": float(d.mean() * 252 / 100), "t": float(d.mean() / d.std() * math.sqrt(len(d)))}
    eq = pd.DataFrame({k: np.exp(np.log1p(v[v.index >= first_trade] / 100).cumsum()) for k, v in books.items()})
    eq.to_csv(out / "equity.csv", float_format="%.5g")
    rep["seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(rep, indent=1, default=str))
    write_page(out, rep)
    return rep


LABELS = {"spy": "SPY (buy and hold)", "xs": "Plain reversal (vs market average)", "sector": "Reversal within sectors", "pca": "Reversal of 3-factor PCA residual",
          "graph": "Graph residual (h = (I - aL)^J x)", "graph_res": "Graph residual, resonance-normalised", "graph_topo": "Graph + topology gate",
          "graph_res_topo": "Graph res + topology gate", "graph_res_topo_vol": "Graph res + topology + vol target", "graph_vol": "Graph + vol target",
          "graph_delay1": "Graph, traded a day late", "graph_cost0": "Graph, no costs", "graph_cost10": "Graph, 10 bp costs"}


def write_page(out: Path, rep: dict) -> None:
    def f(x, pct=False):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "–"
        return f"{100 * x:+.1f}%" if pct else f"{x:.2f}"

    rows = ""
    for k, lab in LABELS.items():
        r = rep["results"].get(k)
        if not r:
            continue
        te, tr = r["test"], r["train"]
        rows += (f"<tr><td>{lab}</td><td>{f(tr.get('sharpe'))}</td><td>{f(te.get('sharpe'))}</td><td>{f(te.get('cagr'), True)}</td>"
                 f"<td>{f(te.get('max_dd'), True)}</td><td>{f(te.get('deflated_sharpe_prob'))}</td></tr>")
    years = sorted({y for r in rep["results"].values() for y in r["by_year"]})
    yr_rows = "".join(f"<tr><td>{LABELS[k]}</td>" + "".join(f"<td>{f(rep['results'][k]['by_year'].get(y), True)}</td>" for y in years) + "</tr>"
                      for k in ("spy", "xs", "pca", "graph", "graph_res_topo_vol") if k in rep["results"])
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Topological stat arb</title><style>body{{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1d2433;background:#fff}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}}td,th{{border-bottom:1px solid #e3e6ec;padding:5px 6px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}.note{{color:#5b6475;font-size:13px}}.wrap{{overflow-x:auto}}</style></head><body>
<h1>Topological stat arb on {len(rep['universe'])} large US stocks</h1>
<p>Settings chosen on {rep['sessions'][0]} to {rep['train_end']}, then frozen; test from {rep['train_end']} to {rep['sessions'][1]}.
Market neutral, gross exposure 1 (except vol-targeted books, 10% a year), 5 bp per unit traded unless stated. Paper backtest only.</p>
<div class="wrap"><table><tr><th>Book</th><th>Train Sharpe</th><th>Test Sharpe</th><th>Test CAGR</th><th>Test max DD</th><th>P(real) after {rep['n_trials']} trials</th></tr>{rows}</table></div>
<p class="note">Placebo (graph filter on shuffled graphs), test Sharpe: {', '.join(f(x) for x in rep['placebo_test_sharpe'])}.</p>
<h2>By year</h2><div class="wrap"><table><tr><th>Book</th>{''.join(f'<th>{y}</th>' for y in years)}</tr>{yr_rows}</table></div>
<p class="note">Data: <a href="summary.json">summary.json</a>, <a href="equity.csv">equity.csv</a>, <a href="phi_topo.csv">phi_topo.csv</a>.</p></body></html>"""
    (out / "index.html").write_text(html)
