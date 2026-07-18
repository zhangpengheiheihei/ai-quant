# -*- coding: utf-8 -*-
"""
TASK 6 — 智能决策者：机器学习定制专属策略
读取 task05 里的 4 只股票日频行情，构造技术因子作为自变量，
以未来 63 个交易日累计收益率作为因变量，训练回归模型，
按季度调仓 Top-N，回测并与市场平均收益率对比。
"""
import os, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

# 中文字体
for f in ['Microsoft YaHei','SimHei','SimSun']:
    try:
        fm.findfont(f, fallback_to_default=False); plt.rcParams['font.sans-serif']=[f]; break
    except: pass
plt.rcParams['axes.unicode_minus']=False

DATA_DIR = Path(r'D:/task/task05/data')
OUT_DIR  = Path(r'D:/task/task06')
CHART    = OUT_DIR/'charts'; CHART.mkdir(parents=True, exist_ok=True)
DATA_OUT = OUT_DIR/'data';   DATA_OUT.mkdir(parents=True, exist_ok=True)

STOCK_FILES = {
    '600519.SH': DATA_DIR/'600519_SH_贵州茅台.csv',
    '002594.SZ': DATA_DIR/'002594_SZ_比亚迪.csv',
    '300750.SZ': DATA_DIR/'300750_SZ_宁德时代.csv',
    '601318.SH': DATA_DIR/'601318_SH_中国平安.csv',
}

# ---------- 1. 读取 & 因子工程 ----------
def make_factors(df):
    df = df.sort_values('trade_date').reset_index(drop=True).copy()
    close = df['close']; high = df['high']; low = df['low']; vol = df['volume']
    df['ret_1']  = close.pct_change()
    df['ret_5']  = close.pct_change(5)
    df['ret_20'] = close.pct_change(20)
    df['ret_60'] = close.pct_change(60)
    df['ma5']    = close.rolling(5).mean()
    df['ma20']   = close.rolling(20).mean()
    df['ma60']   = close.rolling(60).mean()
    df['bias20'] = close/df['ma20']-1
    df['bias60'] = close/df['ma60']-1
    df['vol20']  = df['ret_1'].rolling(20).std()
    df['vol60']  = df['ret_1'].rolling(60).std()
    df['hl_20']  = (high.rolling(20).max()-low.rolling(20).min())/close
    df['vchg20'] = vol.pct_change(20)
    d = close.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    df['rsi14'] = 100 - 100/(1+up.rolling(14).mean()/(dn.rolling(14).mean()+1e-9))
    ema12 = close.ewm(span=12).mean(); ema26 = close.ewm(span=26).mean()
    df['macd']  = ema12-ema26
    df['macd_sig'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']= df['macd']-df['macd_sig']
    # 因变量：未来 63 日累计收益率
    df['y_fwd63'] = close.shift(-63)/close - 1
    return df

frames = []
for code, path in STOCK_FILES.items():
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lstrip('\ufeff') for c in raw.columns]
    raw['code'] = code
    frames.append(make_factors(raw))
panel = pd.concat(frames, ignore_index=True)
panel['trade_date'] = pd.to_datetime(panel['trade_date'])
panel['quarter'] = panel['trade_date'].dt.to_period('Q')

FEATURES = ['ret_5','ret_20','ret_60','bias20','bias60','vol20','vol60',
            'hl_20','vchg20','rsi14','macd','macd_sig','macd_hist']
TARGET   = 'y_fwd63'

panel_clean = panel.dropna(subset=FEATURES+[TARGET]).copy()
panel_clean.to_csv(DATA_OUT/'feature_panel.csv', index=False, encoding='utf-8-sig')
print(f'样本量: {len(panel_clean)}, 特征数: {len(FEATURES)}')
print(f'时间范围: {panel_clean.trade_date.min().date()} ~ {panel_clean.trade_date.max().date()}')

# ---------- 2. 训练集/测试集 (时序切分) ----------
split_date = pd.Timestamp('2024-08-01')
train = panel_clean[panel_clean.trade_date <  split_date]
test  = panel_clean[panel_clean.trade_date >= split_date]
X_tr, y_tr = train[FEATURES].values, train[TARGET].values
X_te, y_te = test[FEATURES].values,  test[TARGET].values
print(f'训练集: {len(train)}  测试集: {len(test)}  切分日: {split_date.date()}')

MODELS = {
    'LinearReg' : LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=6,
                                          min_samples_leaf=20, random_state=42, n_jobs=-1),
    'GBDT'      : GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                            learning_rate=0.05, random_state=42),
}

metrics_rows = []
pred_dict = {}
for name, mdl in MODELS.items():
    mdl.fit(X_tr, y_tr)
    p_tr, p_te = mdl.predict(X_tr), mdl.predict(X_te)
    rmse_tr = np.sqrt(mean_squared_error(y_tr, p_tr))
    rmse_te = np.sqrt(mean_squared_error(y_te, p_te))
    r2_tr, r2_te = r2_score(y_tr, p_tr), r2_score(y_te, p_te)
    # IC = 皮尔逊相关
    ic_te = np.corrcoef(p_te, y_te)[0,1]
    rank_ic = pd.Series(p_te).rank().corr(pd.Series(y_te).rank())
    metrics_rows.append(dict(model=name, rmse_train=rmse_tr, rmse_test=rmse_te,
                             r2_train=r2_tr, r2_test=r2_te,
                             ic_test=ic_te, rank_ic_test=rank_ic))
    pred_dict[name] = p_te

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(DATA_OUT/'model_metrics.csv', index=False, encoding='utf-8-sig')
print('\n=== 模型指标 ===')
print(metrics_df.round(4).to_string(index=False))

# ---------- 3. 交易策略回测 ----------
TOP_N = 2  # 4 只股票池，Top2 演示，等价于 Top-30/全市场
def backtest(pred_series, base_df, name):
    df = base_df.copy(); df['pred'] = pred_series
    # 每季度第一个交易日 rebalance
    rebal = df.groupby(['code','quarter']).head(1).copy()
    rebal = rebal.sort_values(['quarter','pred'], ascending=[True, False])
    picks = rebal.groupby('quarter').head(TOP_N).copy()
    # 未来 63 日实际收益 = y_fwd63
    stra_q = picks.groupby('quarter')['y_fwd63'].mean().rename('strategy_ret')
    mkt_q  = rebal.groupby('quarter')['y_fwd63'].mean().rename('market_ret')
    res = pd.concat([stra_q, mkt_q], axis=1).dropna()
    res['excess'] = res['strategy_ret'] - res['market_ret']
    res['strategy_nav'] = (1+res['strategy_ret']).cumprod()
    res['market_nav']   = (1+res['market_ret']).cumprod()
    return res, picks[['quarter','code','name','pred','y_fwd63']]

results, picks_map = {}, {}
for name, p in pred_dict.items():
    r, picks = backtest(p, test, name)
    results[name] = r
    picks_map[name] = picks
    r.to_csv(DATA_OUT/f'backtest_{name}.csv', encoding='utf-8-sig')

# 核心回测指标
def perf_metrics(r):
    q_ret = r['strategy_ret']; m_ret = r['market_ret']
    ann   = (r['strategy_nav'].iloc[-1])**(4/len(r))-1
    ann_m = (r['market_nav'].iloc[-1])**(4/len(r))-1
    sharpe = q_ret.mean()/(q_ret.std()+1e-9)*np.sqrt(4)
    winrate = (q_ret>m_ret).mean()
    nav = r['strategy_nav']; peak = nav.cummax(); mdd = ((nav-peak)/peak).min()
    return dict(annual_ret=ann, market_annual=ann_m,
                excess_annual=ann-ann_m, sharpe=sharpe,
                win_rate=winrate, max_dd=mdd, quarters=len(r))

perf_rows = []
for name, r in results.items():
    row = perf_metrics(r); row['model']=name; perf_rows.append(row)
perf_df = pd.DataFrame(perf_rows)[['model','annual_ret','market_annual','excess_annual',
                                   'sharpe','win_rate','max_dd','quarters']]
perf_df.to_csv(DATA_OUT/'performance_summary.csv', index=False, encoding='utf-8-sig')
print('\n=== 回测绩效 ===')
print(perf_df.round(4).to_string(index=False))
