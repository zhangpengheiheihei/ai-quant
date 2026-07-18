"""
Task05 - Part B: 股票次日涨跌预测（二分类）
使用比亚迪/宁德时代/贵州茅台/中国平安 4 只股票的日K线，
构造技术指标特征，预测次日涨跌，4 个算法对比 + 混淆矩阵 + ROC + AUC。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = r'D:\task\task05\data'
OUT = r'D:\task\task05\charts'
os.makedirs(OUT, exist_ok=True)

STOCKS = [
    ('002594_SZ_比亚迪.csv',   '比亚迪 (002594)'),
    ('300750_SZ_宁德时代.csv', '宁德时代 (300750)'),
    ('600519_SH_贵州茅台.csv', '贵州茅台 (600519)'),
    ('601318_SH_中国平安.csv', '中国平安 (601318)'),
]


def build_features(df):
    """构造技术指标特征 + 标签（次日涨=1，跌=0）"""
    df = df.sort_values('trade_date').reset_index(drop=True).copy()
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)

    df['ret_1']  = df['close'].pct_change()
    df['ret_2']  = df['close'].pct_change(2)
    df['ret_5']  = df['close'].pct_change(5)
    df['ret_10'] = df['close'].pct_change(10)

    df['ma5']  = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma_diff_5_20']  = df['ma5']  / df['ma20'] - 1
    df['ma_diff_10_20'] = df['ma10'] / df['ma20'] - 1

    df['vol_5']  = df['ret_1'].rolling(5).std()
    df['vol_20'] = df['ret_1'].rolling(20).std()

    df['vol_ma5']   = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']
    df['vol_chg_1'] = df['volume'].pct_change()

    delta = df['close'].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / (dn + 1e-9)
    df['rsi_14'] = 100 - 100 / (1 + rs)

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']   = ema12 - ema26
    df['macd_s'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_h'] = df['macd'] - df['macd_s']

    df['amp_pct'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['label'] = (df['close'].shift(-1) > df['close']).astype(int)

    feature_cols = [
        'ret_1', 'ret_2', 'ret_5', 'ret_10',
        'ma_diff_5_20', 'ma_diff_10_20',
        'vol_5', 'vol_20',
        'vol_ratio', 'vol_chg_1',
        'rsi_14', 'macd', 'macd_s', 'macd_h', 'amp_pct',
    ]
    df = df.dropna(subset=feature_cols + ['label']).reset_index(drop=True)
    return df, feature_cols


all_feats = []
feature_cols = None
for fname, disp in STOCKS:
    raw = pd.read_csv(os.path.join(DATA_DIR, fname))
    fe, feature_cols = build_features(raw)
    fe['stock'] = disp
    all_feats.append(fe)
    print(f"{disp}: 样本={len(fe)}, 涨={int((fe['label']==1).sum())}/跌={int((fe['label']==0).sum())}")

data = pd.concat(all_feats, ignore_index=True)
print(f"\n合并总样本: {len(data)}, 涨={int((data.label==1).sum())} / 跌={int((data.label==0).sum())}")

X = data[feature_cols].values
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    '逻辑回归 LR':    LogisticRegression(max_iter=3000, C=1.0, random_state=42),
    '随机森林 RF':    RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1),
    '支持向量机 SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    '梯度提升 GBDT':  GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
}

results = []
roc_data = {}
fitted = {}

for name, clf in models.items():
    clf.fit(X_train_sc, y_train)
    fitted[name] = clf
    y_pred = clf.predict(X_test_sc)
    y_proba = clf.predict_proba(X_test_sc)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_v = auc(fpr, tpr)
    roc_data[name] = (fpr, tpr, auc_v)

    results.append({
        'model': name,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'auc':       auc_v,
        'TN': int(cm[0,0]), 'FP': int(cm[0,1]),
        'FN': int(cm[1,0]), 'TP': int(cm[1,1]),
    })

res_df = pd.DataFrame(results)
print("\n=== 股票次日涨跌预测：模型评估结果 ===")
print(res_df.round(4).to_string(index=False))
res_df.to_csv(os.path.join(DATA_DIR, 'stock_results.csv'), index=False, encoding='utf-8-sig')

# ---- 混淆矩阵 ----
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.flatten()
for i, (name, clf) in enumerate(fitted.items()):
    y_pred = clf.predict(X_test_sc)
    cm = confusion_matrix(y_test, y_pred)
    ax = axes[i]
    ax.imshow(cm, cmap='Oranges')
    ax.set_title(f'{name}\nAUC={roc_data[name][2]:.4f}', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['预测:跌(0)', '预测:涨(1)'])
    ax.set_yticklabels(['实际:跌(0)', '实际:涨(1)'])
    for r in range(2):
        for c in range(2):
            val = cm[r, c]
            color = 'white' if val > cm.max()/2 else '#222'
            ax.text(c, r, str(val), ha='center', va='center',
                    fontsize=17, fontweight='bold', color=color)
    ax.set_xlabel('预测标签'); ax.set_ylabel('真实标签')
plt.suptitle('股票次日涨跌 — 混淆矩阵对比', fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'stock_confusion_matrix.png'), dpi=150, facecolor='white', bbox_inches='tight')
plt.close()
print('[OK] stock_confusion_matrix.png')

# ---- ROC ----
plt.figure(figsize=(9, 7))
colors = ['#e74c3c', '#27ae60', '#2980b9', '#9b59b6']
for (name, (fpr, tpr, auc_v)), color in zip(roc_data.items(), colors):
    plt.plot(fpr, tpr, lw=2.2, color=color, label=f'{name} (AUC = {auc_v:.4f})')
plt.plot([0, 1], [0, 1], color='#7f8c8d', lw=1.2, linestyle='--', label='随机猜测 AUC=0.5')
plt.xlim([-0.01, 1.01]); plt.ylim([-0.01, 1.02])
plt.xlabel('假阳性率 FPR', fontsize=11)
plt.ylabel('真阳性率 TPR', fontsize=11)
plt.title('股票次日涨跌 — ROC 曲线对比', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'stock_roc_curve.png'), dpi=150, facecolor='white')
plt.close()
print('[OK] stock_roc_curve.png')

# ---- 指标对比柱状图 ----
fig, ax = plt.subplots(figsize=(11, 5.5))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
metric_names = ['准确率', '精确率', '召回率', 'F1', 'AUC']
x_pos = np.arange(len(metrics))
width = 0.2
for i, row in res_df.iterrows():
    ax.bar(x_pos + i*width, [row[m] for m in metrics], width, label=row['model'], color=colors[i])
    for j, m in enumerate(metrics):
        ax.text(x_pos[j] + i*width, row[m] + 0.003, f'{row[m]:.3f}',
                ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x_pos + 1.5*width)
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylim(0.35, max(res_df[metrics].values.max()+0.05, 0.7))
ax.set_ylabel('得分')
ax.set_title('股票次日涨跌 — 各模型评估指标对比', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9, ncol=2)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'stock_metrics_bar.png'), dpi=150, facecolor='white')
plt.close()
print('[OK] stock_metrics_bar.png')

# ---- 特征重要性（用 RandomForest） ----
rf = fitted['随机森林 RF']
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(importances.index, importances.values, color='#3498db')
for i, v in enumerate(importances.values):
    ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
ax.set_title('股票预测 — 随机森林特征重要性排序', fontsize=13, fontweight='bold')
ax.set_xlabel('重要性')
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'stock_feature_importance.png'), dpi=150, facecolor='white')
plt.close()
print('[OK] stock_feature_importance.png')

print('\n=== Part B 完成 ===')
