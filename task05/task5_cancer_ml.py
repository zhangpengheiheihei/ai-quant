"""
Task05 - Part A: 癌症二分类数据集
用 sklearn breast_cancer 训练 4 个主流分类模型，做混淆矩阵、ROC、AUC 评估。
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
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

OUT = r'D:\task\task05\charts'
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 加载数据 ----------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')  # 1=良性, 0=恶性
print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
print(f"良性(1)={int((y==1).sum())}, 恶性(0)={int((y==0).sum())}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ---------- 2. 定义模型 ----------
models = {
    '逻辑回归 LR':      LogisticRegression(max_iter=2000, random_state=42),
    '随机森林 RF':      RandomForestClassifier(n_estimators=200, random_state=42),
    '支持向量机 SVM':   SVC(kernel='rbf', probability=True, random_state=42),
    '梯度提升 GBDT':    GradientBoostingClassifier(random_state=42),
}

results = []
roc_data = {}

for name, clf in models.items():
    # LR/SVM 用标准化数据；树模型也无害，统一用标准化
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)
    y_proba = clf.predict_proba(X_test_sc)[:, 1]

    cm = confusion_matrix(y_test, y_pred)  # [[TN, FP], [FN, TP]]
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

results_df = pd.DataFrame(results)
print("\n=== 癌症数据集：模型评估结果 ===")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(r'D:\task\task05\data\cancer_results.csv', index=False, encoding='utf-8-sig')

# ---------- 3. 混淆矩阵可视化 (2x2 subplot) ----------
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
axes = axes.flatten()
for i, (name, clf) in enumerate(models.items()):
    y_pred = clf.predict(X_test_sc)
    cm = confusion_matrix(y_test, y_pred)
    ax = axes[i]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(f'{name}\nAUC={roc_data[name][2]:.4f}', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['预测:恶性(0)', '预测:良性(1)'])
    ax.set_yticklabels(['实际:恶性(0)', '实际:良性(1)'])
    for r in range(2):
        for c in range(2):
            val = cm[r, c]
            txt_color = 'white' if val > cm.max() / 2 else '#222'
            ax.text(c, r, str(val), ha='center', va='center',
                    fontsize=18, fontweight='bold', color=txt_color)
    ax.set_xlabel('预测标签'); ax.set_ylabel('真实标签')
plt.suptitle('癌症二分类 — 混淆矩阵对比', fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'cancer_confusion_matrix.png'), dpi=150, facecolor='white', bbox_inches='tight')
plt.close()
print('[OK] 混淆矩阵图: cancer_confusion_matrix.png')

# ---------- 4. ROC 曲线对比图 ----------
plt.figure(figsize=(9, 7))
colors = ['#e74c3c', '#27ae60', '#2980b9', '#9b59b6']
for (name, (fpr, tpr, auc_v)), color in zip(roc_data.items(), colors):
    plt.plot(fpr, tpr, lw=2.2, color=color, label=f'{name} (AUC = {auc_v:.4f})')
plt.plot([0, 1], [0, 1], color='#7f8c8d', lw=1.2, linestyle='--', label='随机猜测 AUC=0.5')
plt.xlim([-0.01, 1.01]); plt.ylim([-0.01, 1.02])
plt.xlabel('假阳性率 FPR (False Positive Rate)', fontsize=11)
plt.ylabel('真阳性率 TPR (True Positive Rate)', fontsize=11)
plt.title('癌症二分类 — ROC 曲线对比', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'cancer_roc_curve.png'), dpi=150, facecolor='white')
plt.close()
print('[OK] ROC 曲线图: cancer_roc_curve.png')

# ---------- 5. 指标柱状图 ----------
fig, ax = plt.subplots(figsize=(11, 5.5))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
metric_names = ['准确率', '精确率', '召回率', 'F1', 'AUC']
x = np.arange(len(metrics))
width = 0.2
for i, row in results_df.iterrows():
    ax.bar(x + i * width, [row[m] for m in metrics], width, label=row['model'], color=colors[i])
    for j, m in enumerate(metrics):
        ax.text(x[j] + i * width, row[m] + 0.005, f'{row[m]:.3f}',
                ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylim(0.85, 1.02)
ax.set_ylabel('得分')
ax.set_title('癌症二分类 — 各模型评估指标对比', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9, ncol=2)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'cancer_metrics_bar.png'), dpi=150, facecolor='white')
plt.close()
print('[OK] 指标对比图: cancer_metrics_bar.png')

print('\n=== Part A 完成 ===')
