#!/usr/bin/env python3
"""Point-in-time 100-pair predictive validation of Splittability Index market subscore.
Historical dividend/split-adjusted prices required: date, ticker, adj_close in Parquet or CSV.
Not a test of the original full 100-point index: archived point-in-time structural/news
scores are unavailable. 65-point market-only index is rescaled to 100.
Usage: python run_experiment.py (automatically downloads prices.parquet if missing)
Or supply a local dataset: python run_experiment.py --prices prices.parquet
Or run logic checks: python run_experiment.py --self-test
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd

DATA_URL = 'https://github.com/Johnbrick123/sp500-data/releases/download/data/prices.parquet'
DATA_PAGE = 'https://github.com/Johnbrick123/sp500-data/releases/tag/data'

def download_prices(destination):
    """Download the real historical dataset, with streaming progress and atomic write."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + '.part')
    print(f'Historical price dataset not found: {destination}', flush=True)
    print('Downloading public price history (around 108 MB) from GitHub...', flush=True)
    print(DATA_URL, flush=True)
    request = urllib.request.Request(DATA_URL, headers={'User-Agent': 'Mozilla/5.0 SplittabilityExperiment/1.1'})
    try:
        with urllib.request.urlopen(request, timeout=45) as response, temp.open('wb') as target:
            total = int(response.headers.get('Content-Length', '0'))
            received = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk: break
                target.write(chunk)
                received += len(chunk)
                if received % (10 * 1024 * 1024) < len(chunk):
                    print(f'  Downloaded {received / (1024 ** 2):.0f} MB' + (f' of {total/(1024**2):.0f} MB' if total else ''), flush=True)
        with temp.open('rb') as file:
            header = file.read(4)
            file.seek(-4, 2)
            footer = file.read(4)
        if header != b'PAR1' or footer != b'PAR1':
            raise ValueError('Downloaded file is not a Parquet file (possible HTML/error page).')
        temp.replace(destination)
        print(f'Download complete: {destination} ({received / (1024 ** 2):.1f} MB)', flush=True)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        temp.unlink(missing_ok=True)
        raise RuntimeError(
            'Could not download the required historical dataset automatically.\n'
            f'Cause: {exc}\n'
            f'Open this page to manually download prices.parquet: {DATA_PAGE}\n'
            f'Place prices.parquet next to run_experiment.py ({destination.parent}) and rerun.\n'
            'Alternatively provide your own file: python3 run_experiment.py --prices /path/to/prices.parquet'
        ) from exc

def clip(x): return min(1., max(0., float(x)))
def index_components(rho52, rho5, gap_pct_points):
    c=25*clip((.90-rho52)/.90)
    d=15*clip((rho5-rho52)/.30)
    r=25*clip(gap_pct_points/50.)
    return c, d, r, (c+d+r)*100/65

def load_prices(path):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f'Historical dataset unavailable: {p}. Download it from the source in README.md.')
    if p.suffix.lower()=='.parquet': df=pd.read_parquet(p,columns=['date','ticker','adj_close'])
    elif p.suffix.lower()=='.csv': df=pd.read_csv(p,usecols=['date','ticker','adj_close'])
    else: raise ValueError('Prices must be a .csv or .parquet file')
    df['date']=pd.to_datetime(df['date']).dt.normalize()
    df['ticker']=df['ticker'].astype(str).str.upper().str.strip()
    df['adj_close']=pd.to_numeric(df['adj_close'],errors='coerce')
    df=df.loc[(df['adj_close']>0)&np.isfinite(df['adj_close'])]
    df=df.drop_duplicates(['date','ticker'],keep='last')
    prices=df.pivot(index='date',columns='ticker',values='adj_close').sort_index()
    prices=prices.loc[(prices.index>='2010-01-01')&(prices.index<='2026-10-14')]
    if len(prices)<2000: raise ValueError('Too few observations to reconstruct five-year correlations and future 24-month outcomes')
    return prices

def read_pairs(path):
    p=pd.read_csv(path)
    needed={'pair_id','sector','ticker_a','ticker_b'}
    if not needed.issubset(p.columns): raise ValueError(f'Pairs need {sorted(needed)}')
    if len(p)!=100 or p.pair_id.nunique()!=100: raise ValueError('Must contain exactly 100 unique predeclared pairs')
    if (p.ticker_a==p.ticker_b).any(): raise ValueError('Identical tickers')
    return p

def calculate_rows(prices,pairs,years=range(2015,2025),horizon_years=2):
    """Score January end of each year. Future gap measured at 24-month horizon.
    Require five years >=1200 valid common return days. Daily correlation from log returns.
    One-year return gap from dividend-adjusted total-return prices. No forward prices in features.
    """
    idx=prices.index
    rec=[]
    rejected=[]
    for year in years:
        candidates=idx[idx<=pd.Timestamp(year,1,31)]
        if len(candidates)==0: continue
        formation=candidates[-1]
        target=formation+pd.DateOffset(years=horizon_years)
        future_cand=idx[idx>=target]
        if len(future_cand)==0 or (future_cand[0]-target).days>7: continue
        horizon=future_cand[0]
        i=idx.get_loc(formation)
        if i<1260: continue
        for pair in pairs.itertuples(index=False):
            ta,tb=str(pair.ticker_a),str(pair.ticker_b)
            if ta not in prices.columns or tb not in prices.columns:
                rejected.append((year,pair.pair_id,'missing ticker'));continue
            past=prices[[ta,tb]].iloc[i-1260:i+1]
            # Reject sporadic missing data instead of using stale fills across listing gaps
            if past.isna().values.sum()>10 or (past<=0).values.any():
                rejected.append((year,pair.pair_id,'missing historical prices'));continue
            rets=np.log(past).diff().dropna()
            if len(rets)<1250: rejected.append((year,pair.pair_id,'missing returns'));continue
            rho5=rets.iloc[-1260:].corr().iloc[0,1]
            rho52=rets.iloc[-252:].corr().iloc[0,1]
            if not np.isfinite(rho5) or not np.isfinite(rho52):
                rejected.append((year,pair.pair_id,'undefined correlation'));continue
            gap_pp=abs(100*(past[ta].iloc[-1]/past[ta].iloc[-253]-1)-100*(past[tb].iloc[-1]/past[tb].iloc[-253]-1))
            c,d,r,score=index_components(rho52,rho5,gap_pp)
            ahead=prices.loc[horizon,[ta,tb]]
            if ahead.isna().any() or (ahead<=0).any():
                rejected.append((year,pair.pair_id,'missing horizon close'));continue
            future_gap=abs(np.log(ahead[ta]/past[ta].iloc[-1])-np.log(ahead[tb]/past[tb].iloc[-1]))
            rec.append({'pair_id':pair.pair_id,'sector':pair.sector,'ticker_a':ta,'ticker_b':tb,
                        'year':year,'score_date':str(formation.date()),'outcome_date':str(horizon.date()),
                        'corr52':rho52,'corr5y':rho5,'past_return_gap_pp':gap_pp,
                        'SI_C_25':c,'SI_D_15':d,'SI_R_25':r,'SI_market_100':score,
                        'future_24mo_abs_log_return_gap':future_gap,
                        'future_24mo_diverge20':int(future_gap>=.20)})
    return pd.DataFrame(rec),pd.DataFrame(rejected,columns=['year','pair_id','reason'])

def eval_model(y, scores):
    from sklearn.metrics import roc_auc_score,brier_score_loss,average_precision_score
    if len(y)==0: return {'n':0}
    return {'n':int(len(y)),'event_rate':round(float(np.mean(y)),4),
            'roc_auc':round(float(roc_auc_score(y,scores)),4) if len(np.unique(y))>1 else None,
            'avg_precision':round(float(average_precision_score(y,scores)),4),
            'brier':round(float(brier_score_loss(y,scores)),4)}

def calibrate(train,test,xcol):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000))
    model.fit(train[[xcol]],train['future_24mo_diverge20'])
    return model.predict_proba(test[[xcol]])[:,1]

def bootstrap_sector_diff(test,seed=20261015,iterations=1000):
    from sklearn.metrics import roc_auc_score
    rng=np.random.default_rng(seed)
    sector_names=np.asarray(sorted(test.sector.unique()))
    grouped={s:test[test.sector==s] for s in sector_names}
    aucdif=[]; highlow=[]
    for _ in range(iterations):
        block=pd.concat([grouped[s] for s in rng.choice(sector_names,size=len(sector_names),replace=True)],ignore_index=True)
        y=block.future_24mo_diverge20.to_numpy()
        if len(np.unique(y))>1:
            aucdif.append(roc_auc_score(y,block.p_si)-roc_auc_score(y,block.p_gap))
        high=block.loc[block.SI_market_100>=65,'future_24mo_diverge20']
        low=block.loc[block.SI_market_100<40,'future_24mo_diverge20']
        if len(high)>=5 and len(low)>=5: highlow.append(float(high.mean()-low.mean()))
    return {'auc_delta_si_minus_past_gap_95pct_CI':list(map(float,np.quantile(aucdif,[.025,.975]))) if len(aucdif)>=100 else None,
            'high_minus_low_event_rate_95pct_CI':list(map(float,np.quantile(highlow,[.025,.975]))) if len(highlow)>=100 else None,
            'bootstrap_iterations':iterations,'resampling_unit':'industry sector'}

def evaluate(rows,source,out):
    train=rows[rows.year.between(2015,2019)].copy()
    test=rows[rows.year.between(2022,2024)].copy()
    if len(train)<150 or len(test)<150 or train.future_24mo_diverge20.nunique()<2:
        raise RuntimeError('Insufficient eligible pairs/outcomes for honest train/test analysis')
    test['p_si']=calibrate(train,test,'SI_market_100')
    test['p_gap']=calibrate(train,test,'past_return_gap_pp')
    test['p_corr']=calibrate(train,test,'corr52')
    base=train.future_24mo_diverge20.mean()
    test['p_base']=base
    metrics={name:eval_model(test.future_24mo_diverge20.values,test[f'p_{name}'].values) for name in ('si','gap','corr','base')}
    risk={}
    for name,mask in [('low_<40',test.SI_market_100<40),('medium_40-65',test.SI_market_100.between(40,65,inclusive='left')),('high_>=65',test.SI_market_100>=65)]:
        ss=test[mask]
        risk[name]={'n':int(len(ss)),'divergence_rate':float(ss.future_24mo_diverge20.mean()) if len(ss) else None}
    qbins=pd.qcut(test.SI_market_100,4,labels=False,duplicates='drop')
    quartile=[{'quartile':int(j+1),'n':int((qbins==j).sum()),'event_rate':float(test.loc[qbins==j,'future_24mo_diverge20'].mean()),'mean_si':float(test.loc[qbins==j,'SI_market_100'].mean())} for j in sorted(qbins.dropna().unique())]
    summary={'source':source,'sample':{'pairs_predeclared':100,'train_2015_2019':len(train),'embargo_2020_2021':int(rows.year.between(2020,2021).sum()),'out_of_sample_2022_2024':len(test),'out_of_sample_unique_pairs':int(test.pair_id.nunique()),'score_years':list(range(2015,2025))},
             'outcome':'absolute difference in 24-month cumulative log total returns >=0.20',
             'index_variant':'(C+D+R)*100/65; original news/fundamentals S+A excluded',
             'results':metrics,'risk_bands':risk,'quartiles':quartile,
             'sector_bootstrap_95pct':bootstrap_sector_diff(test)}
    out.mkdir(parents=True,exist_ok=True)
    rows.to_csv(out/'all_annual_pair_observations.csv',index=False)
    test.to_csv(out/'out_of_sample_predictions.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return summary

def self_test():
    assert index_components(.9,.9,0)[-1]==0
    assert abs(index_components(-1.,.9,80)[-1]-100)<1e-10
    assert index_components(-.1,.9,60)[-1]==100
    # No-lookahead: future prices never influence the feature score from fixed history
    dates=pd.bdate_range('2009-01-01','2026-10-01')
    rng=np.random.default_rng(91)
    x=100*np.exp(np.cumsum(rng.normal(0,.012,len(dates))))
    y=85*np.exp(np.cumsum(rng.normal(0,.012,len(dates))))
    prices=pd.DataFrame({'A':x,'B':y},index=dates)
    pairs=pd.DataFrame([{'pair_id':'A/B','sector':'test','ticker_a':'A','ticker_b':'B'}])
    before,_=calculate_rows(prices,pairs,years=[2020])
    altered=prices.copy()
    altered.loc[altered.index>'2020-01-31','A']*=4
    after,_=calculate_rows(altered,pairs,years=[2020])
    for col in ['corr52','corr5y','past_return_gap_pp','SI_market_100']:
        assert abs(before.iloc[0][col]-after.iloc[0][col])<1e-10,(col,before.iloc[0][col],after.iloc[0][col])
    assert before.iloc[0]['future_24mo_abs_log_return_gap']!=after.iloc[0]['future_24mo_abs_log_return_gap']
    print('PASS: component limits, identical past scores after altering future, future outcome independent')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prices',type=Path,help='CSV/Parquet with date,ticker,adj_close; defaults to prices.parquet next to this script')
    ap.add_argument('--pairs',type=Path,default=Path(__file__).with_name('pairs.csv'))
    ap.add_argument('--out',type=Path,default=Path(__file__).with_name('results'))
    ap.add_argument('--self-test',action='store_true',help='Run calculation/leakage tests without market data')
    ap.add_argument('--no-download',action='store_true',help='Fail with instructions if dataset is missing; never attempt internet access')
    args=ap.parse_args()
    if args.self_test: self_test();return
    if args.prices is None:
        args.prices=Path(__file__).with_name('prices.parquet')
        if not args.prices.exists():
            if args.no_download:
                ap.error(f'{args.prices} is missing. Download from {DATA_PAGE} or rerun without --no-download.')
            download_prices(args.prices)
    try:
        print(f'Loading adjusted price history from {args.prices}...',flush=True)
        prices=load_prices(args.prices)
        pairs=read_pairs(args.pairs)
        print('Scoring 100 pairs on historical formation dates...',flush=True)
        rows,rejected=calculate_rows(prices,pairs)
        args.out.mkdir(parents=True,exist_ok=True)
        rejected.to_csv(args.out/'rejected_observations.csv',index=False)
        if rows.empty: raise RuntimeError('No historical observations could be scored. Check ticker coverage and price dates.')
        print(f'Scored {len(rows)} pair-years; excluded {len(rejected)} due to missing data',flush=True)
        print('Evaluating 2022–2024 out-of-sample results...',flush=True)
        evaluate(rows,str(args.prices),args.out)
        print(f'\nSUCCESS: report is saved in {args.out / "summary.json"}',flush=True)
    except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f'\nERROR: {exc}',file=sys.stderr,flush=True)
        print('Try: python3 -m pip install -r requirements.txt',file=sys.stderr,flush=True)
        raise SystemExit(1) from exc

if __name__=='__main__': main()
