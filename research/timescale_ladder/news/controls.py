"""Does the news signal survive controls for the stock's own recent returns? python controls.py prices.parquet scores.jsonl"""
import json
import sys

import numpy as np
import pandas as pd

P=pd.read_parquet(sys.argv[1],columns=['date','ticker','open','adj_close'])
P['date']=pd.to_datetime(P['date']).dt.strftime('%Y-%m-%d')
P=P.sort_values(['ticker','date'])
g=P.groupby('ticker')['adj_close']
P['r0']=g.pct_change(); P['r3']=g.pct_change(3); P['r20']=g.pct_change(20); P['nx']=g.shift(-1)/P['adj_close']-1
fl=pd.read_csv('flagged.csv')
with open(sys.argv[2]) as fh:
    sc=pd.DataFrame([json.loads(x) for x in fh]).drop(columns=['titles'])
D=fl.merge(sc,on=['date','ticker']).merge(P[['date','ticker','r0','r3','r20','nx']],on=['date','ticker'],how='left')
print('next_ret check corr',D[['next_ret','nx']].corr().iloc[0,1])
def auc(y,s):
    m=~np.isnan(s); y,s=y[m],s[m]
    r=pd.Series(s).rank().to_numpy(); p=y==1
    return (r[p].sum()-p.sum()*(p.sum()+1)/2)/(p.sum()*(~p).sum()), m.sum()
B=D[D.big==1].copy(); y=(B.next_ret>0).to_numpy().astype(int)
print('big days',len(B),'up share',y.mean())
for c in ['sentiment','lean','words','r0','r3','r20']:
    print(c, auc(y,B[c].to_numpy(float)), 'corr w r0', round(B[[c,'r0']].astype(float).corr().iloc[0,1],2))
# residual sentiment after r0,r3 via logistic on big days
import statsmodels.api as sm

for c in ['sentiment','lean']:
    X=B[[c,'r0','r3','r20']].astype(float).dropna(); yy=(B.loc[X.index,'next_ret']>0).astype(int)
    m=sm.Logit(yy,sm.add_constant((X-X.mean())/X.std())).fit(disp=0); print(c, m.params.round(2).to_dict(), 'z', m.tvalues.round(2).to_dict())
# all flagged days regression of next_ret on sentiment controlling r0
X=D[['sentiment','r0','r3']].astype(float).dropna(); yy=D.loc[X.index,'next_ret']
m=sm.OLS(yy,sm.add_constant(X)).fit(cov_type='HC1'); print('OLS all', m.tvalues.round(2).to_dict())
# tilted straddle proxy: on all flagged days with sentiment !=0, share of |move| captured... simple: sign(sent)*next_ret mean
for c in ['sentiment','lean']:
    s=np.sign(D[c].astype(float)); m=s!=0
    v=(s*D.next_ret)[m]; print(c,'sign*ret mean bp',round(v.mean()*1e4,1),'t',round(v.mean()/v.std()*np.sqrt(len(v)),2),len(v))
