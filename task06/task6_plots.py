# -*- coding: utf-8 -*-
"""TASK6 绘图脚本"""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

for f in ['Microsoft YaHei','SimHei']:
    try:
        fm.findfont(f, fallback_to_default=False)
        plt.rcParams['font.sans-serif']=[f]; break
    except: pass
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['axes.grid']=True; plt.rcParams['grid.alpha']=0.35

D = Path(r'D:/task/task06/data'); C = Path(r'D:/task/task06/charts')
metrics = pd.read_csv(D/'model_metrics.csv')
perf    = pd.read_csv(D/'performance_summary.csv')
models = ['LinearReg','DecisionTree','RandomForest','GBDT']
labels = ['线性回归','决策树','随机森林','GBDT']
color_map = dict(zip(models, ['#1f77b4','#ff7f0e','#2ca02c','#d62728']))

# ============ 图1：净值曲线 ============
fig, ax = plt.subplots(figsize=(9,4.6))
first = pd.read_csv(D/f'backtest_{models[0]}.csv')
qs = first['quarter'].tolist()
ax.plot(range(len(qs)), first['market_nav'], marker='s', color='#555', lw=2.2, label='市场基准（等权）')
for m,l in zip(models, labels):
    r = pd.read_csv(D/f'backtest_{m}.csv')
    ax.plot(range(len(r)), r['strategy_nav'], marker='o', lw=2, color=color_map[m], label=f'{l} 策略')
ax.set_xticks(range(len(qs))); ax.set_xticklabels(qs)
ax.set_ylabel('累计净值（初始=1.0）'); ax.set_xlabel('季度')
ax.set_title('图1  各模型 Top-N 策略净值 vs 市场基准（测试集 2024Q3–2026Q2）')
ax.legend(loc='upper left', fontsize=9, ncol=2)
plt.tight_layout(); plt.savefig(C/'fig1_nav_curve.png', dpi=160); plt.close()

# ============ 图2：季度超额收益 ============
fig, ax = plt.subplots(figsize=(9,4.2))
x = np.arange(len(qs)); w = 0.2
for i,(m,l) in enumerate(zip(models,labels)):
    r = pd.read_csv(D/f'backtest_{m}.csv')
    ax.bar(x+(i-1.5)*w, r['excess']*100, width=w, color=color_map[m], label=l)
ax.axhline(0, color='k', lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(qs)
ax.set_ylabel('季度超额收益 (%)'); ax.set_xlabel('季度')
ax.set_title('图2  各模型每季度相对市场基准的超额收益')
ax.legend(loc='upper right', fontsize=9, ncol=4)
plt.tight_layout(); plt.savefig(C/'fig2_excess_bar.png', dpi=160); plt.close()

# ============ 图3：年化收益 & 夏普 ============
fig, (a1,a2) = plt.subplots(1,2, figsize=(10,4))
p = perf.set_index('model').reindex(models)
bars1 = a1.bar(labels, p['annual_ret']*100, color=[color_map[m] for m in models])
a1.axhline(p['market_annual'].iloc[0]*100, color='k', ls='--', lw=1.4,
           label=f"市场基准 {p['market_annual'].iloc[0]*100:.2f}%")
for b,v in zip(bars1, p['annual_ret']*100):
    a1.text(b.get_x()+b.get_width()/2, v+0.4, f'{v:.2f}%', ha='center', fontsize=9)
a1.set_ylabel('年化收益率 (%)'); a1.set_title('图3-a  年化收益率对比')
a1.legend(fontsize=9)
bars2 = a2.bar(labels, p['sharpe'], color=[color_map[m] for m in models])
for b,v in zip(bars2, p['sharpe']):
    a2.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.2f}', ha='center', fontsize=9)
a2.set_ylabel('夏普比率'); a2.set_title('图3-b  夏普比率对比')
plt.tight_layout(); plt.savefig(C/'fig3_perf_bar.png', dpi=160); plt.close()

# ============ 图4：IC / Rank IC ============
fig, ax = plt.subplots(figsize=(8,4))
m = metrics.set_index('model').reindex(models)
x = np.arange(len(models)); w = 0.35
ax.bar(x-w/2, m['ic_test'],       width=w, color='#1f77b4', label='Pearson IC')
ax.bar(x+w/2, m['rank_ic_test'],  width=w, color='#ff7f0e', label='Rank IC (Spearman)')
for i,(a,b) in enumerate(zip(m['ic_test'], m['rank_ic_test'])):
    ax.text(i-w/2, a+0.005, f'{a:.3f}', ha='center', fontsize=8)
    ax.text(i+w/2, b+0.005, f'{b:.3f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('IC / Rank IC')
ax.set_title('图4  各模型在测试集的信息系数 IC 与排序系数 Rank IC')
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(C/'fig4_ic.png', dpi=160); plt.close()

# ============ 图5：随机森林特征重要性 ============
panel = pd.read_csv(D/'feature_panel.csv')
panel['trade_date'] = pd.to_datetime(panel['trade_date'])
FEATURES = ['ret_5','ret_20','ret_60','bias20','bias60','vol20','vol60',
            'hl_20','vchg20','rsi14','macd','macd_sig','macd_hist']
train = panel[panel.trade_date < '2024-08-01']
test  = panel[panel.trade_date >= '2024-08-01']
X_tr, y_tr = train[FEATURES].values, train['y_fwd63'].values
X_te, y_te = test[FEATURES].values,  test['y_fwd63'].values

rf = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=20,
                           random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8,4.5))
ax.barh(imp.index, imp.values, color='#2ca02c')
for i,v in enumerate(imp.values):
    ax.text(v+0.002, i, f'{v:.3f}', va='center', fontsize=8)
ax.set_xlabel('重要性 (Gini)')
ax.set_title('图5  随机森林特征重要性排序（13 个技术因子）')
plt.tight_layout(); plt.savefig(C/'fig5_feat_imp.png', dpi=160); plt.close()

# ============ 图6：预测 vs 真实收益散点 2x2 ============
fig, axes = plt.subplots(2,2, figsize=(9,7.4))
_mdls = {'LinearReg':LinearRegression(),
         'DecisionTree':DecisionTreeRegressor(max_depth=5,random_state=42),
         'RandomForest':RandomForestRegressor(n_estimators=200,max_depth=6,
                                              min_samples_leaf=20,random_state=42,n_jobs=-1),
         'GBDT':GradientBoostingRegressor(n_estimators=200,max_depth=3,
                                          learning_rate=0.05,random_state=42)}
for ax,(mn,ml),lb in zip(axes.flat, _mdls.items(), labels):
    ml.fit(X_tr, y_tr); pr = ml.predict(X_te)
    ax.scatter(pr, y_te, s=6, alpha=0.35, color=color_map[mn])
    lo, hi = min(pr.min(), y_te.min()), max(pr.max(), y_te.max())
    ax.plot([lo,hi],[lo,hi],'k--', lw=1)
    ic = np.corrcoef(pr, y_te)[0,1]
    ax.set_title(f'{lb}  |  IC={ic:.3f}', fontsize=10)
    ax.set_xlabel('预测收益'); ax.set_ylabel('实际63日收益')
plt.suptitle('图6  各模型预测收益 vs 真实63日收益（测试集）', y=1.00, fontsize=13)
plt.tight_layout(); plt.savefig(C/'fig6_pred_scatter.png', dpi=160); plt.close()

# ============ 图7：GBDT 季度收益 vs 市场 ============
gb = pd.read_csv(D/f'backtest_GBDT.csv')
fig, ax = plt.subplots(figsize=(8.5,3.8))
x = np.arange(len(gb))
ax.bar(x-0.2, gb['strategy_ret']*100, width=0.4, color='#d62728', label='GBDT策略')
ax.bar(x+0.2, gb['market_ret']*100,   width=0.4, color='#888',   label='市场基准')
for i,(s,mk) in enumerate(zip(gb['strategy_ret']*100, gb['market_ret']*100)):
    ax.text(i-0.2, s+(0.4 if s>=0 else -1.2), f'{s:.1f}%', ha='center', fontsize=8)
    ax.text(i+0.2, mk+(0.4 if mk>=0 else -1.2), f'{mk:.1f}%', ha='center', fontsize=8, color='#333')
ax.axhline(0, color='k', lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(gb['quarter'])
ax.set_ylabel('季度收益率 (%)'); ax.set_xlabel('季度')
ax.set_title('图7  GBDT 策略与市场基准的每季度收益率对比')
ax.legend(fontsize=9, loc='upper left')
plt.tight_layout(); plt.savefig(C/'fig7_gbdt_quarter.png', dpi=160); plt.close()

print('全部 7 张图已生成')
