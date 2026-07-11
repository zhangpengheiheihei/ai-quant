import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 双均线（Dual Moving Average）量化策略研究报告\n",
    "## TASK3 - 策略原理、回测指标与多股票实证分析"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 任务要求\n",
    "\n",
    "1. **理论学习**：了解双均线策略，解释金叉/死叉概念\n",
    "2. **指标理解**：解释最大回撤（MDD）、夏普比率（Sharpe Ratio）、累计回报等量化指标\n",
    "3. **Python编程实现**：\n",
    "   - 加载已存储的股价数据\n",
    "   - 设定短均线和长均线周期（如5和15），计算均线数据\n",
    "   - 计算买入卖出的交易信号\n",
    "   - 绘制可视化图形（股价、长短均线、交易信号）\n",
    "   - 模拟交易与回测，计算量化指标\n",
    "4. **对比实验**：尝试不同股票、不同均线周期，观察收益变化，总结适用场景与应用心得"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 一、双均线策略原理与金叉/死叉概念\n",
    "\n",
    "### 1.1 移动平均线（MA）\n",
    "移动平均线是将某段时间内的收盘价求算术平均并连成的曲线。周期越短，均线越贴近价格、越敏感；周期越长，均线越平滑、滞后越明显。\n",
    "\n",
    "### 1.2 金叉（Golden Cross）—— 买入信号\n",
    "当短周期均线由下向上穿越长周期均线时，形成\"黄金交叉\"，简称**金叉**。意味着近期价格的平均水平已经高于远期平均水平，短期买入力量增强，市场可能由弱转强。\n",
    "\n",
    "### 1.3 死叉（Death Cross）—— 卖出信号\n",
    "当短周期均线由上向下穿越长周期均线时，形成\"死亡交叉\"，简称**死叉**。意味着短期平均价格已跌破长期平均价格，短期抛压加重，市场可能由强转弱。\n",
    "\n",
    "### 1.4 策略逻辑\n",
    "完整交易循环：**空仓 → 金叉买入 → 持有 → 死叉卖出 → 空仓**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 二、量化策略效果的基础评价指标\n",
    "\n",
    "### 2.1 累计回报（Cumulative Return）\n",
    "策略在整个回测区间内的总收益率：\n",
    "- 累计回报 = 期末净值 / 期初净值 - 1\n",
    "- 净值曲线由每日收益连乘得到\n",
    "\n",
    "### 2.2 最大回撤（Maximum Drawdown, MDD）\n",
    "衡量策略在历史上出现过的最糟糕的\"从峰值到谷底\"的亏损幅度：\n",
    "- 回撤 = 当前净值 / 历史最高净值 - 1\n",
    "- 最大回撤即所有回撤中的最小值（最大亏损）\n",
    "\n",
    "### 2.3 夏普比率（Sharpe Ratio）\n",
    "衡量\"每承担一单位总风险所获得的超额收益\"：\n",
    "- 夏普比率 = (策略日均收益率 - 日均无风险利率) / 策略日均收益率标准差 × √252\n",
    "- √252是将日频结果年化（A股约252个交易日/年）"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 三、Python 编程实现\n",
    "\n",
    "### 3.0 导入依赖包"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\nimport numpy as np\nimport pandas as pd\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n\nplt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']\nplt.rcParams['axes.unicode_minus'] = False\n\nOUTPUT_DIR = os.path.abspath('.')\nprint(f\"工作目录: {OUTPUT_DIR}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.1 加载已存储的股价数据"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_csv(path):\n    df = pd.read_csv(path)\n    df = df[['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'pct_chg', 'turnover']].copy()\n    return df\n\nCSV_STORED = os.path.join(OUTPUT_DIR, 'gaoneng_huanjing_daily.csv')\ndf_gn = load_csv(CSV_STORED)\nprint(f\"高能环境数据: {len(df_gn)} 条记录\")\nprint(f\"时间范围: {df_gn['date'].iloc[0]} 至 {df_gn['date'].iloc[-1]}\")\ndf_gn.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.2 设定短均线和长均线周期，计算均线数据"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "SHORT_PERIOD = 5\nLONG_PERIOD = 15\n\nclose = df_gn['close'].astype(float).values\ndates = df_gn['date'].values\nma_s = df_gn['close'].rolling(SHORT_PERIOD).mean().values\nma_l = df_gn['close'].rolling(LONG_PERIOD).mean().values\n\nprint(f\"短均线 MA{SHORT_PERIOD} 和长均线 MA{LONG_PERIOD} 已计算\")\nprint(f\"价格序列长度: {len(close)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.3 计算买入卖出的交易信号"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "diff = ma_s - ma_l\ngolden = (diff > 0) & (np.roll(diff, 1) <= 0)\ndeath = (diff < 0) & (np.roll(diff, 1) >= 0)\ngolden[0] = False\ndeath[0] = False\n\nbuy_idx = np.where(golden)[0]\nsell_idx = np.where(death)[0]\npos = np.where(ma_s > ma_l, 1.0, 0.0)\npos[:LONG_PERIOD - 1] = 0.0\n\nprint(f\"买入信号 (金叉) 数量: {len(buy_idx)}\")\nprint(f\"卖出信号 (死叉) 数量: {len(sell_idx)}\")\nprint(f\"买入日期: {dates[buy_idx]}\")\nprint(f\"卖出日期: {dates[sell_idx]}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.4 绘制可视化图形"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(14, 6.2))\nx = range(len(close))\n\nax.plot(x, close, color='#34495e', lw=1.3, label='收盘价', zorder=1)\nax.plot(x, ma_s, color='#e67e22', lw=1.3, label=f'MA{SHORT_PERIOD}')\nax.plot(x, ma_l, color='#2980b9', lw=1.3, label=f'MA{LONG_PERIOD}')\n\nif len(buy_idx) > 0:\n    ax.scatter(buy_idx, close[buy_idx], marker='^', color='#27ae60',\n               s=110, zorder=5, label='买入(金叉)', edgecolors='white', linewidths=0.6)\nif len(sell_idx) > 0:\n    ax.scatter(sell_idx, close[sell_idx], marker='v', color='#c0392b',\n               s=110, zorder=5, label='卖出(死叉)', edgecolors='white', linewidths=0.6)\n\nax.set_title('双均线策略信号图', fontsize=14, fontweight='bold')\nax.set_ylabel('价格（元）')\nax.legend(loc='best', fontsize=9, ncol=3)\nax.grid(alpha=0.3, linestyle='--')\n\nstep = max(1, len(close) // 12)\nax.set_xticks(list(range(0, len(close), step)))\nax.set_xticklabels([dates[i][5:] for i in range(0, len(close), step)], \n                   rotation=45, fontsize=8, ha='right')\n\nplt.tight_layout()\nplt.savefig(os.path.join(OUTPUT_DIR, 'notebook_chart_signal.png'), dpi=150, facecolor='white')\nprint('信号图已保存')\nplt.close()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.5 模拟交易与回测，计算量化指标"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "n = len(close)\nret = np.zeros(n)\nret[1:] = close[1:] / close[:-1] - 1.0\n\npos_prev = np.zeros(n)\npos_prev[1:] = pos[:-1]\nstrat_ret = pos_prev * ret\n\nequity = np.ones(n)\nfor t in range(1, n):\n    equity[t] = equity[t - 1] * (1.0 + strat_ret[t])\n\nbnh_equity = np.ones(n)\nfor t in range(1, n):\n    bnh_equity[t] = bnh_equity[t - 1] * (1.0 + ret[t])\n\nrf_annual = 0.02\nrf_daily = rf_annual / 252.0\ncum_ret = equity[-1] - 1.0\nbnh_cum = bnh_equity[-1] - 1.0\n\nrunmax = np.maximum.accumulate(equity)\ndd = equity / runmax - 1.0\nmdd = dd.min()\n\nif strat_ret[1:].std(ddof=1) > 1e-12:\n    sharpe = (strat_ret[1:].mean() - rf_daily) / strat_ret[1:].std(ddof=1) * np.sqrt(252.0)\nelse:\n    sharpe = 0.0\n\nann_ret = (equity[-1]) ** (252.0 / n) - 1.0\nann_vol = strat_ret[1:].std(ddof=1) * np.sqrt(252.0)\n\nprint('=' * 50)\nprint('双均线策略回测结果 (MA5/MA15)')\nprint('=' * 50)\nprint(f'累计回报: {cum_ret*100:.2f}%')\nprint(f'年化收益: {ann_ret*100:.2f}%')\nprint(f'年化波动: {ann_vol*100:.2f}%')\nprint(f'夏普比率: {sharpe:.2f}')\nprint(f'最大回撤: {mdd*100:.2f}%')\nprint(f'买入持有基准: {bnh_cum*100:.2f}%')\nprint('=' * 50)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 绘制净值曲线对比图"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(14, 5.6))\nax.plot(range(n), equity, color='#27ae60', lw=1.6, label='双均线策略净值')\nax.plot(range(n), bnh_equity, color='#7f8c8d', lw=1.4, linestyle='--', label='买入持有净值')\nax.set_title('策略净值曲线 vs 买入持有', fontsize=14, fontweight='bold')\nax.set_ylabel('净值（起点=1.0）')\nax.legend(loc='best', fontsize=10)\nax.grid(alpha=0.3, linestyle='--')\n\nstep = max(1, n // 12)\nax.set_xticks(list(range(0, n, step)))\nax.set_xticklabels([dates[i][5:] for i in range(0, n, step)], \n                   rotation=45, fontsize=8, ha='right')\n\nplt.tight_layout()\nplt.savefig(os.path.join(OUTPUT_DIR, 'notebook_chart_equity.png'), dpi=150, facecolor='white')\nprint('净值曲线图已保存')\nplt.close()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 四、不同股票与均线周期对比实验\n",
    "\n",
    "### 加载多只股票数据"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "stocks = [\n    ('高能环境', 'gaoneng_huanjing_daily.csv'),\n    ('贵州茅台', 'stock_600519.csv'),\n    ('比亚迪', 'stock_002594.csv'),\n    ('中国平安', 'stock_601318.csv'),\n    ('宁德时代', 'stock_300750.csv'),\n]\n\nperiods = [(5, 15), (5, 20), (10, 30), (20, 60)]\n\nprint('已加载股票:')\nfor name, filename in stocks:\n    csv_path = os.path.join(OUTPUT_DIR, filename)\n    if os.path.exists(csv_path):\n        df = load_csv(csv_path)\n        print(f'  {name}: {len(df)} 条记录')\n    else:\n        print(f'  {name}: 文件不存在 ({filename})')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 五、适用场景与应用心得总结\n",
    "\n",
    "### 适用场景\n",
    "1. **平滑趋势市场**：在趋势方向明确、回调不频繁、走势连贯的标的上，双均线策略能\"顺势而为\"\n",
    "2. **中长线持仓**：选用10/30、20/60等较长周期更适合中长线投资者\n",
    "3. **组合风控**：将死叉作为离场/止损纪律，能机械地规避\"扛单\"的人性弱点\n",
    "\n",
    "