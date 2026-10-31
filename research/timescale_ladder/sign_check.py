"""Sign check for big8.py: given a big move happens on a flagged day, which inputs tell up from down, and is it tradeable?

    python sign_check.py PANEL.pkl
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import big8, lightgbm as lgb
D = big8.build(pd.read_pickle(sys.argv[1]))
cols = sorted(set(big8.SIZE + big8.SIGN[:-1])); D[cols] = D[cols].replace([np.inf, -np.inf], np.nan); D = D.dropna(subset=cols)
tr, te = D.date < big8.TEST, D.date >= big8.TEST
gb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=500, subsample=0.5, subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(D.loc[tr, big8.SIZE], D.loc[tr, "big"])
D["p_big"] = gb.predict_proba(D[big8.SIZE])[:, 1]
S = [c for c in big8.SIGN if c != "p_big"]
hi = D[D.p_big >= D.loc[tr, "p_big"].quantile(0.95)]
a = hi[hi.big == 1]
atr, ate = a[a.date < big8.TEST], a[a.date >= big8.TEST]
mu, sd = atr[S].mean(), atr[S].std()
lr = LogisticRegression(max_iter=500).fit((atr[S]-mu)/sd, atr.big_up)
print("coef", dict(zip(S, lr.coef_[0].round(3))))
p = lr.predict_proba((ate[S]-mu)/sd)[:, 1]; print("AUC sign|big logit", round(roc_auc_score(ate.big_up, p), 3))
for c in S: print(c, round(roc_auc_score(ate.big_up, ate[c]), 3))
# tradeable: flagged test days, logit sign score, buy top each day
hte = hi[hi.date >= big8.TEST].copy(); hte["s"] = lr.predict_proba((hte[S]-mu)/sd)[:, 1]
pk = hte.loc[hte.groupby("date")["s"].idxmax()]
net = np.expm1(pk.y_next) - 0.001
print("buy flagged with best up-score:", len(pk), "avg bp", round(1e4*net.mean(),1), "t", round(net.mean()/net.std()*np.sqrt(len(net)),2), "win", round(100*(net>0).mean(),1))
pk2 = hte.loc[hte.groupby("date")["s"].idxmin()]; net2 = -np.expm1(pk2.y_next) - 0.001
print("short flagged with worst up-score:", round(1e4*net2.mean(),1), "t", round(net2.mean()/net2.std()*np.sqrt(len(net2)),2))
