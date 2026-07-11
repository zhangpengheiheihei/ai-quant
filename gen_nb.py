import json

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# 双均线量化策略研究报告 - TASK3"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 任务要求\n1. 理解双均线策略，解释金叉/死叉\n2. 理解量化指标（累计回报、最大回撤、夏普比率）\n3. Python编程实现全部功能\n4. 多股票多周期对比实验与总结"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 一、双均线策略原理\n\n### 金叉（Golden Cross）\n短周期均线由下向上穿越长周期均线 → 买入信号，表示市场由弱转强\n\n### 死叉（Death Cross）\n短周期均线由上向下穿越长周期均线 → 卖出信号，表示市场由强转弱\n\n### 完整交易逻辑\n空仓 → 金叉买入 → 持有 → 死叉卖出 → 空仓"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 二、量化评价指标\n\n- **累计回报（Cumulative Return）**：策略在整个回测区间内的总收益率，公式：期末净值 / 期初净值 - 1\n- **最大回撤（Maximum Drawdown, MDD）**：历史最高点到最低点的最大亏损幅度，衡量策略风险\n- **夏普比率（Sharpe Ratio）**：每承担一单位风险所获得的超额收益（年化），衡量风险调整后收益"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 三、Python 编程实现\n\n### 3.1 导入依赖包"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": ["import os\nimport numpy as np\nimport pandas as pd\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n\nplt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']\nplt.rcParams['axes.unicode_minus'] = False\n\nOUTPUT_DIR = os.path.abspath('.')\nprint('依赖导入完成，工作目录:', OUTPUT_DIR)"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### 3.2 加载已存储的股价数据"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": ["def load_csv(path):\n    df = pd.read_csv(path)\n    df = df[['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'pct_chg', 'turnover']].copy()\n    return df\n\n# 加载主样本：高能环境\ndf = load_csv('gaoneng_huanjing_daily.csv')\nprint(f'数据加载完成: {len(df)} 条记录')\nprint(f'时间范围: {df[\"date\"].iloc[0]} 至 {df[\"date\"].iloc[-1]}')\ndf.head()"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### 3.3 设定短均线和长均线周期，计算均线数据"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": ["# 设定均线周期\nSHORT_PERIOD = 5\nLONG_PERIOD = 15\n\nclose = df['close'].astype(float).values\ndates = df['date'].values\n\n# 计算移动平均线\nma_s = df['close'].rolling(SHORT_PERIOD).mean().values  # 短均线\nma_l = df['close'].rolling(LONG_PERIOD).mean().values   # 长均线\n\nprint(f'短均线 MA{SHORT_PERIOD} 和长均线 MA{LONG_PERIOD} 计算完成')\nprint(f'价格序列长度: {len(close)}')"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### 3.4 计算买入卖出的交易信号"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": ["# 计算均线差值\ndiff = ma_s - ma_l\n\n# 金叉：短均线上穿长均线（由负转正）\ngolden = (diff > 0) & (np.roll(diff, 1) <= 0)\n# 死叉：短均线下穿长均线（由正转负）\ndeath = (diff < 0) & (np.roll(diff, 1) >= 0)\n\n# 第一个点不判断（roll后第一个点是最后一个点的值）\ngolden[0] = False\ndeath[0] = False\n\nbuy_idx = np.where(golden)[0]   # 买入信号位置\nsell_idx = np.where(death)[0]   # 卖出信号位置\n\n# 持仓状态（1=持有，0=空仓）\npos = np.where(ma_s > ma_l, 1.0, 0.0)\npos[:LONG_PERIOD - 1] = 0.0  # 均线未成形时空仓\n\nprint(f'买入信号（金叉）数量: {len(buy_idx)} 个')\nprint(f'卖出信号（死叉）数量: {len(sell_idx)} 个')\nprint(f'\\n买入日期: {dates[buy_idx]}')\nprint(f'卖出日期: {dates[sell_idx]}')"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### 3.5 绘制可视化图形（股价+长短均线+买卖信号）"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": ["fig, ax = plt.subplots(figsize=(14, 6.2))\nx = range(len(close))\n\n# 绘制价格和均线\nax.plot(x, close, color='#34495e', lw=1.3, label='收盘价', zorder=1)\nax.plot(x, ma_s, color='#e67e22', lw=1.3, label=f'短均线 MA{SHORT_PERIOD}')\nax.plot(x, ma_l, color='#2980b9', lw=1.3, label=f'长均线 MA{LONG_PERIOD}')\n\n# 标记买卖信号\nif len(buy_idx) > 0:\n    ax.scatter(buy_idx, close[buy_idx], marker='^', color='#27ae60',\n               s=120, zorder=5, label='买入(金叉)', edgecolors='white', linewidths=0.6)\nif len(sell_idx) > 0:\n    ax.scatter(sell_idx, close[sell_idx], marker='v', color='#c0392b',\n               s=120, zorder=5, label='卖出(死叉)', edgecolors='white', linewidths=0.6)\n\nax.set_title('双均线策略信号图（MA5 / MA15）', fontsize=14, fontweight='bold')\nax.set_ylabel('价格（元）')\nax.legend(loc='best', fontsize=10, ncol=3)\nax.grid(alpha=0.3, linestyle='--')\n\n# 设置x轴刻度\nstep = max(1, len(close) // 12)\nax.set_xticks(list(range(0, len(close), step)))\nax.set_xticklabels([dates[i][5:] for i in range(0, len(close), step)], \n                   rotation=45, fontsize=8, ha='right')\n\nplt.tight_layout()\nplt.savefig('notebook_signal.png', dpi=150, facecolor='white')\nprint('信号图已保存: notebook_signal.png')\nplt.close()"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### 3.6 模拟交易与回测，计算量化指标"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": ["n = len(close)\n\n# 日收益率\nret = np.zeros(n)\nret[1:] = close[1:] / close[:-1] - 1.0\n\n# 策略日收益（前一日持仓 × 当日收益率）\npos_prev = np.zeros(n)\npos_prev[1:] = pos[:-1]\nstrat_ret = pos_prev * ret\n\n# 净值曲线\nequity = np.ones(n)          # 策略净值\nbnh_equity = np.ones(n)      # 买入持有净值\nfor t in range(1, n):\n    equity[t] = equity[t - 1] * (1.0 + strat_ret[t])\n    bnh_equity[t] = bnh_equity[t - 1] * (1.0 + ret[t])\n\n# 计算量化指标\nrf_annual = 0.02\nrf_daily = rf_annual / 252.0\n\ncum_ret = equity[-1] - 1.0                          # 累计回报\nrunmax = np.maximum.accumulate(equity)\nmdd = (equity / runmax - 1.0).min()                 # 最大回撤\n\nif strat_ret[1:].std(ddof=1) > 1e-12:\n    sharpe = (strat_ret[1:].mean() - rf_daily) / strat_ret[1:].std(ddof=1) * np.sqrt(252.0)\nelse:\n    sharpe = 0.0\n\nprint('=' * 50)\nprint('双均线策略回测结果 (MA5/MA15)')\nprint('=' * 50)\nprint(f'累计回报: {cum_ret*100:.2f}%')\nprint(f'最大回撤: {mdd*100:.2f}%')\nprint(f'夏普比率: {sharpe:.2f}')\nprint(f'买入持有基准: {(bnh_equity[-1]-1)*100:.2f}%')\nprint('=' * 50)"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["### 绘制净值曲线对比图"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": ["fig, ax = plt.subplots(figsize=(14, 5.5))\n\nax.plot(range(n), equity, color='#27ae60', lw=1.6, label='双均线策略净值')\nax.plot(range(n), bnh_equity, color='#7f8c8d', lw=1.4, linestyle='--', label='买入持有净值')\n\nax.set_title('策略净值曲线 vs 买入持有', fontsize=14, fontweight='bold')\nax.set_ylabel('净值（起点=1.0）')\nax.legend(loc='best', fontsize=10)\nax.grid(alpha=0.3, linestyle='--')\n\nstep = max(1, n // 12)\nax.set_xticks(list(range(0, n, step)))\nax.set_xticklabels([dates[i][5:] for i in range(0, n, step)], \n                   rotation=45, fontsize=8, ha='right')\n\nplt.tight_layout()\nplt.savefig('notebook_equity.png', dpi=150, facecolor='white')\nprint('净值曲线图已保存: notebook_equity.png')\nplt.close()"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 四、不同股票与均线周期对比实验"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": ["# 多股票列表\nstocks = [\n    ('高能环境', 'gaoneng_huanjing_daily.csv'),\n    ('贵州茅台', 'stock_600519.csv'),\n    ('比亚迪', 'stock_002594.csv'),\n    ('中国平安', 'stock_601318.csv'),\n    ('宁德时代', 'stock_300750.csv'),\n]\n\n# 多周期组合\nperiods = [(5, 15), (5, 20), (10, 30), (20, 60)]\n\nprint('加载多只股票数据:')\nfor name, filename in stocks:\n    if os.path.exists(filename):\n        df_stock = load_csv(filename)\n        print(f'  {name}: {len(df_stock)} 条记录')\n    else:\n        print(f'  {name}: 文件不存在')\n\nprint(f'\\n均线周期组合: {periods}')"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 五、适用场景与应用心得总结\n\n### 适用场景\n1. **平滑趋势市场**：在趋势方向明确、回调不频繁、走势连贯的标的上，双均线策略能\"顺势而为\"\n2. **中长线持仓**：选用10/30、20/60等较长周期更适合中长线投资者\n3. **组合风控**：将死叉作为离场/止损纪律，能机械地规避\"扛单\"的人性弱点\n\n### 不适用场景\n1. **震荡市**：价格在区间内来回震荡，均线频繁交叉，产生大量假信号\n2. **高交易成本环境**：短周期参数交易频繁，手续费滑点会显著侵蚀收益\n3. **突发拐点**：均线天然滞后，无法在尖顶/尖底处及时反应\n\n### 应用建议\n- 参数没有\"最优\"，只有\"最适配\"，需结合标的特性选择\n- 务必与买入持有基准对比，只有风险调整后收益真正改善时才有实盘价值\n- 单一策略信号较粗糙，实战中常配合成交量、MACD、布林带等过滤器减少假突破"]
  }
 ],
 'metadata': {
  'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
  'language_info': {'name': 'python', 'version': '3.0'}
 },
 'nbformat': 4,
 'nbformat_minor': 4
}

with open('双均线策略_TASK3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print('Notebook文件已成功生成: 双均线策略_TASK3.ipynb')
