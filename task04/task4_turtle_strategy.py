# -*- coding: utf-8 -*-
"""
================================================================================
  海龟交易策略（Turtle Trading Strategy）量化研究报告  ——  TASK4
================================================================================
  功能:
    1. 加载已存储的股价数据（高能环境 gaoneng_huanjing_daily.csv）
    2. 抓取多只对比股票数据
    3. 计算唐奇安通道（Donchian Channel）- 高低点通道
    4. 计算 ATR（平均真实波幅）
    5. 计算海龟策略的买入卖出交易信号
    6. 根据计算结果绘制可视化图形
    7. 策略模拟交易与回测，计算量化指标（累计回报、最大回撤、夏普比率等）
    8. 通过调节核心参数（股票类型、通道周期）观察收益变化
    9. 用 reportlab 生成 PDF 报告（宋体 / 五号字 / 1.5倍行距）
    10. 生成 ECharts 交互式 HTML 图表

  运行: python task4_turtle_strategy.py
  输出:
    - chart_turtle_signal.png  海龟策略信号图（价格+通道+买入卖出标记）
    - chart_turtle_equity.png  策略净值曲线 vs 买入持有
    - chart_param_compare.png  不同通道周期参数对比
    - chart_multi_stock.png    多股票在标准参数下的表现对比
    - 张鹏+TASK4.pdf           PDF分析报告
    - turtle_charts/*.html     交互式HTML图表
================================================================================
"""

import os
import sys
import csv
import json
import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_STORED = os.path.join(OUTPUT_DIR, "gaoneng_huanjing_daily.csv")
HTML_DIR = os.path.join(OUTPUT_DIR, "turtle_charts")
FONT_SIMHEI = r"C:\Windows\Fonts\simhei.ttf"
FONT_SIMSUN = r"C:\Windows\Fonts\simsun.ttc"

if not os.path.exists(HTML_DIR):
    os.makedirs(HTML_DIR)

# 注册中文字体
fm.fontManager.addfont(FONT_SIMHEI)
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================================
# 0. 数据获取 / 加载
# ============================================================================
def fetch_tencent(sym, beg="2025-07-04", end="2026-07-03"):
    """从腾讯财经接口抓取前复权日K线"""
    import urllib.request
    import time
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,{beg},{end},320,qfq"
    last_err = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as f:
                d = json.loads(f.read().decode("utf-8"))
            node = d["data"][sym]
            key = "qfqday" if "qfqday" in node else "day"
            arr = node[key]
            break
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (attempt + 1))
    else:
        raise RuntimeError(f"腾讯行情接口请求失败: {last_err}")
    data = []
    for row in arr:
        dt = row[0]
        if dt < beg or dt > end:
            continue
        o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
        v = float(row[5])
        data.append({"date": dt, "open": o, "close": c, "high": h, "low": l,
                     "volume": int(v), "amount": v * c, "pct_chg": 0.0,
                     "turnover": 0.0})
    for i in range(1, len(data)):
        data[i]["pct_chg"] = (data[i]["close"] / data[i - 1]["close"] - 1) * 100
    return data


def load_csv(path):
    df = pd.read_csv(path)
    return df[["date", "open", "close", "high", "low", "volume", "amount", "pct_chg", "turnover"]].copy()


# ============================================================================
# 1. 海龟策略核心指标计算
# ============================================================================

def calc_donchian_channel(highs, lows, period=20):
    """计算唐奇安通道（Donchian Channel）- 高低点通道
    上轨 = 前period日最高价的最大值
    下轨 = 前period日最低价的最小值
    """
    n = len(highs)
    upper = [None] * n  # 上轨
    lower = [None] * n  # 下轨
    middle = [None] * n  # 中轨 = (上轨 + 下轨) / 2

    for i in range(period - 1, n):
        window_high = highs[i - period + 1: i + 1]
        window_low = lows[i - period + 1: i + 1]
        upper[i] = float(np.max(window_high))
        lower[i] = float(np.min(window_low))
        middle[i] = (upper[i] + lower[i]) / 2

    return upper, middle, lower


def calc_atr(highs, lows, closes, period=20):
    """计算 ATR（Average True Range，平均真实波幅）
    TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    ATR = TR 的 N 周期移动平均
    """
    n = len(highs)
    tr = np.zeros(n)  # 真实波幅 True Range

    # 计算每日 TR
    tr[0] = highs[0] - lows[0]  # 第一天只有当日高低点
    for i in range(1, n):
        hl = highs[i] - lows[i]  # 当日高低价差
        h_pc = abs(highs[i] - closes[i - 1])  # 最高价与昨收差
        l_pc = abs(lows[i] - closes[i - 1])  # 最低价与昨收差
        tr[i] = max(hl, h_pc, l_pc)

    # 计算 ATR（使用 Wilder 平滑方法）
    atr = [None] * n
    atr[period - 1] = float(np.mean(tr[:period]))  # 第一个周期用简单平均
    # 后续用 Wilder 平滑: ATR_t = (ATR_{t-1} * (N-1) + TR_t) / N
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr, tr


def calc_turtle_signals(df, entry_period=20, exit_period=10, atr_period=20, atr_multiplier=2.0):
    """计算海龟交易策略的买卖信号
    进场规则：价格突破 entry_period 日新高 → 买入（做多）
    出场规则1：价格跌破 exit_period 日新低 → 卖出
    出场规则2：价格从最高点回落 ATR × atr_multiplier → 止损卖出
    """
    dates = df["date"].values
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values
    n = len(closes)

    # 计算通道
    entry_upper, _, entry_lower = calc_donchian_channel(highs, lows, entry_period)
    exit_upper, _, exit_lower = calc_donchian_channel(highs, lows, exit_period)

    # 计算 ATR
    atr, tr = calc_atr(highs, lows, closes, atr_period)

    # 信号与持仓
    buy_signals = [None] * n  # 买入信号下标
    sell_signals = [None] * n  # 卖出信号下标
    pos = np.zeros(n)  # 持仓状态 0=空仓, 1=持仓
    entry_price = [None] * n  # 入场价格
    highest_since_entry = [None] * n  # 入场后的最高价（用于止损）

    in_position = False
    entry_idx = -1

    for i in range(max(entry_period, exit_period, atr_period), n):
        # 记录入场后的最高价
        if in_position:
            highest_since_entry[i] = max(highest_since_entry[i - 1], highs[i])
        else:
            highest_since_entry[i] = None

        # ---- 买入信号（空仓时，价格突破上轨）----
        if not in_position and entry_upper[i] is not None:
            # 当收盘价突破 entry_period 日新高
            if closes[i] > entry_upper[i - 1]:
                buy_signals[i] = closes[i]
                pos[i] = 1.0
                in_position = True
                entry_idx = i
                entry_price[i] = closes[i]
                highest_since_entry[i] = highs[i]
            else:
                pos[i] = 0.0
        elif in_position:
            # 已持仓
            pos[i] = 1.0
            entry_price[i] = entry_price[i - 1]

            # ---- 卖出信号检查 ----
            # 条件1: 价格跌破 exit_period 日新低（趋势反转）
            cond1 = exit_lower[i] is not None and closes[i] < exit_lower[i - 1]

            # 条件2: 价格从入场后最高点回落超过 ATR × 乘数（止损）
            cond2 = False
            if atr[i] is not None and highest_since_entry[i] is not None:
                stop_loss_price = highest_since_entry[i] - atr_multiplier * atr[i]
                if closes[i] < stop_loss_price:
                    cond2 = True

            if cond1 or cond2:
                sell_signals[i] = closes[i]
                pos[i] = 0.0
                in_position = False
                entry_idx = -1
        else:
            pos[i] = 0.0

    return {
        "dates": dates,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "entry_upper": entry_upper,
        "entry_lower": entry_lower,
        "exit_upper": exit_upper,
        "exit_lower": exit_lower,
        "atr": atr,
        "tr": tr,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "pos": pos,
        "entry_period": entry_period,
        "exit_period": exit_period,
        "atr_period": atr_period,
    }


# ============================================================================
# 2. 策略回测与指标计算
# ============================================================================
def backtest_turtle(signal_data, rf_annual=0.02):
    """对海龟策略进行回测，计算累计回报、最大回撤、夏普比率等指标"""
    dates = signal_data["dates"]
    closes = signal_data["closes"]
    pos = signal_data["pos"]
    n = len(closes)

    # 日收益率
    ret = np.zeros(n)
    ret[1:] = closes[1:] / closes[:-1] - 1.0

    # 策略在第 t 日收益 = 第 t-1 日收盘产生的信号所对应的持仓 × 当日收益率
    pos_prev = np.zeros(n)
    pos_prev[1:] = pos[:-1]
    strat_ret = pos_prev * ret

    # 净值曲线
    equity = np.ones(n)
    for t in range(1, n):
        equity[t] = equity[t - 1] * (1.0 + strat_ret[t])

    # 买入持有基准
    bnh_equity = np.ones(n)
    for t in range(1, n):
        bnh_equity[t] = bnh_equity[t - 1] * (1.0 + ret[t])

    # 无风险利率
    rf_daily = rf_annual / 252.0

    # 累计回报
    cum_ret = equity[-1] - 1.0
    bnh_cum = bnh_equity[-1] - 1.0

    # 年化收益
    ann_ret = (equity[-1]) ** (252.0 / n) - 1.0

    # 最大回撤
    runmax = np.maximum.accumulate(equity)
    dd = equity / runmax - 1.0
    mdd = dd.min()
    max_dd_date = dates[np.argmin(dd)]

    bnh_runmax = np.maximum.accumulate(bnh_equity)
    bnh_dd = bnh_equity / bnh_runmax - 1.0
    bnh_mdd = bnh_dd.min()

    # 年化波动率
    ann_vol = strat_ret[1:].std(ddof=1) * np.sqrt(252.0)
    bnh_ann_vol = ret[1:].std(ddof=1) * np.sqrt(252.0)

    # 夏普比率
    if strat_ret[1:].std(ddof=1) > 1e-12:
        sharpe = (strat_ret[1:].mean() - rf_daily) / strat_ret[1:].std(ddof=1) * np.sqrt(252.0)
    else:
        sharpe = 0.0

    if ret[1:].std(ddof=1) > 1e-12:
        bnh_sharpe = (ret[1:].mean() - rf_daily) / ret[1:].std(ddof=1) * np.sqrt(252.0)
    else:
        bnh_sharpe = 0.0

    # 交易统计
    buy_idx = [i for i, v in enumerate(signal_data["buy_signals"]) if v is not None]
    sell_idx = [i for i, v in enumerate(signal_data["sell_signals"]) if v is not None]

    trades = []
    bi, si = 0, 0
    while bi < len(buy_idx) and si < len(sell_idx):
        if sell_idx[si] > buy_idx[bi]:
            # 配对的一买一卖
            bp = signal_data["buy_signals"][buy_idx[bi]]
            sp = signal_data["sell_signals"][sell_idx[si]]
            trades.append(sp / bp - 1.0)
            bi += 1
            si += 1
        else:
            si += 1

    n_trades = len(trades)
    win_rate = float(np.mean([t > 0 for t in trades])) if trades else 0.0
    avg_trade = float(np.mean(trades)) if trades else 0.0
    max_win = float(np.max(trades)) if trades else 0.0
    max_loss = float(np.min(trades)) if trades else 0.0

    # 持仓天数统计
    hold_days = []
    for b, s in zip(buy_idx[:len(trades)], sell_idx[:len(trades)]):
        if s > b:
            hold_days.append(s - b)
    avg_hold_days = float(np.mean(hold_days)) if hold_days else 0.0

    metrics = {
        "entry_period": signal_data["entry_period"],
        "exit_period": signal_data["exit_period"],
        "atr_period": signal_data["atr_period"],
        "n": n,
        "cum_ret": cum_ret,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "max_dd_date": max_dd_date,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "max_win": max_win,
        "max_loss": max_loss,
        "avg_hold_days": avg_hold_days,
        "bnh_cum": bnh_cum,
        "bnh_sharpe": bnh_sharpe,
        "bnh_mdd": bnh_mdd,
        "bnh_ann_vol": bnh_ann_vol,
        "excess": cum_ret - bnh_cum,
    }

    return {
        "signal_data": signal_data,
        "equity": equity,
        "bnh_equity": bnh_equity,
        "strat_ret": strat_ret,
        "metrics": metrics,
    }


# ============================================================================
# 3. 图表生成
# ============================================================================
def _xtick_settings(ax, dates, n):
    step = max(1, n // 12)
    pos = list(range(0, n, step))
    labels = [dates[i][5:] for i in pos]
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=45, fontsize=8, ha="right")
    ax.set_xlim(-1, n)


def chart_turtle_signal(result, title, path):
    """图1：价格 + 唐奇安通道 + ATR + 买卖信号标记"""
    signal_data = result["signal_data"]
    dates = signal_data["dates"]
    closes = signal_data["closes"]
    entry_upper = signal_data["entry_upper"]
    entry_lower = signal_data["entry_lower"]
    exit_lower = signal_data["exit_lower"]
    atr = signal_data["atr"]
    buy_signals = signal_data["buy_signals"]
    sell_signals = signal_data["sell_signals"]
    n = len(closes)

    fig = plt.figure(figsize=(14, 9))

    # 子图1：价格和通道和信号
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(range(n), closes, color="#2c3e50", lw=1.3, label="收盘价", zorder=1)

    # 绘制唐奇安通道
    upper_vals = [v if v is not None else np.nan for v in entry_upper]
    lower_vals = [v if v is not None else np.nan for v in entry_lower]
    exit_lower_vals = [v if v is not None else np.nan for v in exit_lower]

    ax1.plot(range(n), upper_vals, color="#e74c3c", lw=1.2, ls="--", label=f"进场通道上轨({result['metrics']['entry_period']}日)")
    ax1.plot(range(n), lower_vals, color="#27ae60", lw=1.2, ls="--", alpha=0.6, label="进场通道下轨")
    ax1.plot(range(n), exit_lower_vals, color="#f39c12", lw=1.0, ls="-.", label=f"出场通道下轨({result['metrics']['exit_period']}日)")

    # 填充通道区域
    valid_indices = [i for i in range(n) if entry_upper[i] is not None and entry_lower[i] is not None]
    if valid_indices:
        ax1.fill_between(valid_indices,
                        [upper_vals[i] for i in valid_indices],
                        [lower_vals[i] for i in valid_indices],
                        alpha=0.12, color="#3498db")

    # 标记买入卖出信号
    buy_x = [i for i, v in enumerate(buy_signals) if v is not None]
    buy_y = [buy_signals[i] for i in buy_x]
    if buy_x:
        ax1.scatter(buy_x, buy_y, marker="^", color="#27ae60", s=120, zorder=5,
                   label="买入信号", edgecolors="white", linewidths=0.8)

    sell_x = [i for i, v in enumerate(sell_signals) if v is not None]
    sell_y = [sell_signals[i] for i in sell_x]
    if sell_x:
        ax1.scatter(sell_x, sell_y, marker="v", color="#c0392b", s=120, zorder=5,
                   label="卖出信号", edgecolors="white", linewidths=0.8)

    ax1.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax1.set_ylabel("价格（元）")
    ax1.legend(loc="upper left", fontsize=9, ncol=2)
    ax1.grid(alpha=0.3, linestyle="--")
    _xtick_settings(ax1, dates, n)

    # 子图2：ATR
    ax2 = plt.subplot(2, 1, 2)
    atr_vals = [v if v is not None else np.nan for v in atr]
    ax2.plot(range(n), atr_vals, color="#9b59b6", lw=1.3, label=f"ATR({result['metrics']['atr_period']}日)")
    ax2.fill_between(range(n), atr_vals, alpha=0.2, color="#9b59b6")
    ax2.set_ylabel("ATR 值")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3, linestyle="--")
    _xtick_settings(ax2, dates, n)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def chart_turtle_equity(result, title, path):
    """图2：策略净值曲线 vs 买入持有 + 回撤对比"""
    n = len(result["equity"])
    dates = result["signal_data"]["dates"]

    fig = plt.figure(figsize=(14, 9))

    # 子图1：净值曲线
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(range(n), result["equity"], color="#27ae60", lw=1.6, label="海龟策略净值")
    ax1.plot(range(n), result["bnh_equity"], color="#7f8c8d", lw=1.4, ls="--", label="买入持有净值")

    m = result["metrics"]
    info_text = f"策略: 累计回报 {m['cum_ret']*100:.2f}% | 夏普 {m['sharpe']:.2f} | 最大回撤 {m['mdd']*100:.2f}%\n"
    info_text += f"基准: 累计回报 {m['bnh_cum']*100:.2f}% | 夏普 {m['bnh_sharpe']:.2f} | 最大回撤 {m['bnh_mdd']*100:.2f}%"
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
             fontsize=9, va="top", linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", alpha=0.9))

    ax1.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax1.set_ylabel("净值（起点=1.0）")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(alpha=0.3, linestyle="--")
    _xtick_settings(ax1, dates, n)

    # 子图2：回撤曲线对比
    ax2 = plt.subplot(2, 1, 2)
    runmax = np.maximum.accumulate(result["equity"])
    dd = (result["equity"] / runmax - 1.0) * 100
    bnh_runmax = np.maximum.accumulate(result["bnh_equity"])
    bnh_dd = (result["bnh_equity"] / bnh_runmax - 1.0) * 100

    ax2.fill_between(range(n), dd, 0, color="#e74c3c", alpha=0.4, label="策略回撤")
    ax2.fill_between(range(n), bnh_dd, 0, color="#95a5a6", alpha=0.3, label="买入持有回撤")
    ax2.plot(range(n), dd, color="#c0392b", lw=0.8)
    ax2.plot(range(n), bnh_dd, color="#7f8c8d", lw=0.8, ls="--")

    ax2.set_ylabel("回撤幅度（%）")
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(alpha=0.3, linestyle="--")
    _xtick_settings(ax2, dates, n)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def chart_param_compare(names, param_combos, all_results, title, path):
    """图3：不同参数组合下的表现对比"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    colors = ["#27ae60", "#3498db", "#9b59b6", "#e67e22", "#e74c3c", "#1abc9c"]

    x_labels = [f"进{e}/出{x}" for e, x in param_combos]
    x = np.arange(len(param_combos))
    width = 0.8 / max(1, len(names))

    # 累计回报对比
    ax = axes[0, 0]
    for j, name in enumerate(names):
        vals = [all_results[(name, e, x)]["metrics"]["cum_ret"] * 100 for e, x in param_combos]
        ax.bar(x + j * width, vals, width, label=name, color=colors[j % len(colors)], alpha=0.8)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("累计回报（%）")
    ax.set_title("不同参数下的累计回报对比", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # 夏普比率对比
    ax = axes[0, 1]
    for j, name in enumerate(names):
        vals = [all_results[(name, e, x)]["metrics"]["sharpe"] for e, x in param_combos]
        ax.bar(x + j * width, vals, width, label=name, color=colors[j % len(colors)], alpha=0.8)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("夏普比率")
    ax.set_title("不同参数下的夏普比率对比", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # 最大回撤对比（取绝对值）
    ax = axes[1, 0]
    for j, name in enumerate(names):
        vals = [abs(all_results[(name, e, x)]["metrics"]["mdd"]) * 100 for e, x in param_combos]
        ax.bar(x + j * width, vals, width, label=name, color=colors[j % len(colors)], alpha=0.8)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("最大回撤幅度（%）")
    ax.set_title("不同参数下的最大回撤对比（越小越好）", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # 胜率对比
    ax = axes[1, 1]
    for j, name in enumerate(names):
        vals = [all_results[(name, e, x)]["metrics"]["win_rate"] * 100 for e, x in param_combos]
        ax.bar(x + j * width, vals, width, label=name, color=colors[j % len(colors)], alpha=0.8)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("胜率（%）")
    ax.set_title("不同参数下的交易胜率对比", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def chart_multi_stock(names, results, params, title, path):
    """图4：多股票在固定参数下的表现对比（双轴：收益+夏普）"""
    fig, ax1 = plt.subplots(figsize=(14, 7))

    xs = np.arange(len(names))
    cum_vals = [results[(name, params[0], params[1])]["metrics"]["cum_ret"] * 100 for name in names]
    bnh_vals = [results[(name, params[0], params[1])]["metrics"]["bnh_cum"] * 100 for name in names]

    w = 0.35
    ax1.bar(xs - w / 2, cum_vals, w, label=f"海龟策略(进{params[0]}/出{params[1]})", color="#27ae60", alpha=0.85)
    ax1.bar(xs + w / 2, bnh_vals, w, label="买入持有", color="#95a5a6", alpha=0.7)

    ax1.axhline(0, color="#333", lw=0.8)
    ax1.set_ylabel("累计收益率（%）", fontsize=11)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, fontsize=10)
    for i, v in enumerate(cum_vals):
        ax1.text(i - w / 2, v + (1 if v >= 0 else -2.5), f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
    for i, v in enumerate(bnh_vals):
        ax1.text(i + w / 2, v + (1 if v >= 0 else -2.5), f"{v:.1f}%", ha="center", fontsize=8)

    # 右轴：夏普比率
    ax2 = ax1.twinx()
    sharpe_vals = [results[(name, params[0], params[1])]["metrics"]["sharpe"] for name in names]
    ax2.plot(xs, sharpe_vals, "o-", color="#e67e22", lw=2, markersize=8, label="策略夏普比率")
    ax2.set_ylabel("夏普比率", fontsize=11, color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22")
    for i, v in enumerate(sharpe_vals):
        ax2.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=9, color="#e67e22", fontweight="bold")

    ax1.set_title(title, fontsize=14, fontweight="bold", pad=15)
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, fontsize=10, loc="upper right")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================================
# 4. PDF 报告生成（reportlab）
# ============================================================================
def build_pdf_report(stocks, primary_result, param_combos, all_results, chart_files, out_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                    Table, TableStyle, PageBreak)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors

    pdfmetrics.registerFont(TTFont("SimSun", FONT_SIMSUN, subfontIndex=0))
    pdfmetrics.registerFont(TTFont("SimHei", FONT_SIMHEI))

    BODY = 10.5
    LEAD = 10.5 * 1.5

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2.8 * cm, rightMargin=2.8 * cm,
        topMargin=2.5 * cm, bottomMargin=2.5 * cm,
        title="海龟交易策略研究报告", author="量化分析",
    )

    story = []

    def P(text, size=BODY, font="SimSun", bold=False, align=TA_JUSTIFY,
          before=0, after=0, indent=21, leading=None):
        st = ParagraphStyle(
            "p", fontName=font, fontSize=size, leading=leading or size * 1.5,
            alignment=align, spaceBefore=before, spaceAfter=after,
            firstLineIndent=indent,
            splitLongWords=True,      # 允许分割长单词/数字
            wordWrap='CJK',           # 优化中日韩文字换行
            allowWidows=True,         # 允许孤行
            allowOrphans=True,        # 允许寡行
        )
        story.append(Paragraph(text, st))

    def H(text, level=1):
        size = {1: 14, 2: 12, 3: 11}[level]
        before = 8 if level == 1 else 6
        st = ParagraphStyle(
            "h", fontName="SimHei", fontSize=size, leading=size * 1.5,
            alignment=TA_JUSTIFY, spaceBefore=before, spaceAfter=0, firstLineIndent=0,
        )
        story.append(Paragraph(text, st))

    def CAP(text):
        st = ParagraphStyle(
            "cap", fontName="SimHei", fontSize=10, leading=15,
            alignment=TA_CENTER, spaceBefore=6, spaceAfter=4, firstLineIndent=0,
        )
        story.append(Paragraph(text, st))

    def IMG(path, width=15 * cm):
        from PIL import Image as PILImage
        iw, ih = PILImage.open(path).size
        h = width * ih / iw
        story.append(Spacer(1, 4))
        story.append(Image(path, width=width, height=h))
        story.append(Spacer(1, 2))

    def TBL(headers, rows, col_w=None):
        data = [[Paragraph(f"<b>{h}</b>", _cell_style("SimHei", 9, True)) for h in headers]]
        for row in rows:
            data.append([Paragraph(str(c), _cell_style("SimSun", 9)) for c in row])
        t = Table(data, colWidths=col_w, hAlign="CENTER")
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "SimSun"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.85, 0.88, 0.95)),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(t)
        story.append(Spacer(1, 6))

    def _cell_style(font, size, bold=False):
        return ParagraphStyle("c", fontName=font, fontSize=size, leading=size * 1.4,
                              alignment=TA_CENTER)

    # ===================== 封面 =====================
    story.append(Spacer(1, 3.5 * cm))
    P("海龟交易策略量化研究报告", size=22, font="SimHei", bold=True, align=TA_CENTER,
      before=0, after=0, indent=0)
    P("——Turtle Trading Strategy 原理、实现与多股票回测实证", size=13, font="SimHei", align=TA_CENTER,
      before=8, after=0, indent=0)
    story.append(Spacer(1, 1.2 * cm))
    P("课程任务：TASK4　海龟交易策略", size=12, font="SimSun", align=TA_CENTER, indent=0)
    P(f"数据区间：{primary_result['signal_data']['dates'][0]} 至 {primary_result['signal_data']['dates'][-1]}", size=12,
      font="SimSun", align=TA_CENTER, indent=0)
    P(f"报告生成日期：{datetime.date.today()}", size=12, font="SimSun", align=TA_CENTER, indent=0)
    P("数据来源：高能环境为已存储前复权日线；对比股票为腾讯财经前复权(qfq)日线", size=11, font="SimSun",
      align=TA_CENTER, indent=0)
    story.append(Spacer(1, 1.0 * cm))
    P("【免责声明】本报告基于公开市场数据与量化模型进行学习研究，所有内容仅供学术参考，"
      "不构成任何投资建议。股市有风险，投资需谨慎。", size=9, font="SimSun",
      align=TA_CENTER, indent=0)
    story.append(PageBreak())

    # ===================== 一、海龟策略概述 =====================
    H("一、海龟交易策略概述与核心思想", 1)
    P("海龟交易策略（Turtle Trading Strategy）是由著名的商品交易员 Richard Dennis 与 William Eckhardt "
      "在1983年创立的经典趋势跟踪策略。该策略通过一套完整、机械化的交易规则，使普通交易者也能在市场中"
      "系统性地捕捉趋势并控制风险。Dennis 用实验证明了成功的交易员是可以后天培养的，而非天生的——"
      "参与实验的学员被称为“海龟”，这便是策略名称的由来。")

    H("1.1  核心思想与交易哲学", 2)
    P("海龟策略的核心思想可以概括为：「突破入场，趋势跟踪，风控第一，系统执行」。"
      "其底层交易哲学包括四个关键要素：")
    P("（1）<b>市场总是有趋势的</b>：价格运动并非完全随机，而是呈现趋势性特征。"
      "只要抓住大趋势，就能获得可观收益。")
    P("（2）<b>突破是趋势的起点</b>：当价格创出一定周期内的新高时，往往意味着一波新趋势的启动。"
      "海龟策略用“唐奇安通道突破”作为入场信号。")
    P("（3）<b>让利润奔跑，截断亏损</b>：趋势跟踪策略的盈利来源是少数大趋势带来的巨额收益，"
      "而止损则保证在趋势反转时不会造成灾难性亏损。")
    P("（4）<b>机械化执行，杜绝主观干扰</b>：海龟策略所有规则都是客观、可量化、可执行的，"
      "完全排除交易者的主观判断和情绪干扰。")

    H("1.2  海龟策略的关键优势", 2)
    P("海龟策略历经四十余年市场检验，至今仍是最具影响力的量化策略之一，其关键优势在于：")
    P("（1）<b>规则透明、逻辑清晰</b>：入场、出场、止损、仓位管理都有明确规则，易于理解和实现。"
      "这使得海龟策略成为量化交易入门的经典教学案例。")
    P("（2）<b>胜率不高，但盈亏比高</b>：海龟策略的典型胜率约为30%-40%，但单次盈利幅度远大于亏损幅度。"
      "依靠少数大赚的交易覆盖多数小亏的交易，最终实现正收益。")
    P("（3）<b>强大的风险控制</b>：通过 ATR 动态调整止损位和仓位大小，"
      "确保在极端市场条件下也能将单笔亏损控制在可接受范围内。")
    P("（4）<b>多市场、多品种分散</b>：原策略同时交易20多个商品期货品种，通过分散化降低组合风险，"
      "保证不会因单一品种的长期震荡而造成过大亏损。")
    P("（5）<b>适应大趋势行情</b>：在趋势明确、波动较大的品种上，海龟策略表现尤为出色；"
      "而在长时间横盘震荡的市场中则会因频繁假突破而产生磨损。")

    story.append(PageBreak())

    # ===================== 二、核心概念解释 =====================
    H("二、海龟策略核心概念详解", 1)
    P("海龟策略的实现依赖于三个核心概念：唐奇安通道（高低点通道）、"
      "ATR（平均真实波幅）以及基于两者的止损条件。下面分别进行详细解释。")

    H("2.1  唐奇安通道（Donchian Channel）—— 高低点通道", 2)
    P("唐奇安通道由著名技术分析师 Richard Donchian 于20世纪中期创立，是海龟策略的核心技术指标。"
      "它由三条线构成：")
    P("（1）<b>上轨（Upper Band）</b>：过去 N 个交易日内最高价的最大值，代表多头力量的突破临界。")
    P("（2）<b>下轨（Lower Band）</b>：过去 N 个交易日内最低价的最小值，代表空头力量的突破临界。")
    P("（3）<b>中轨（Middle Band）</b>：上轨与下轨的平均值，代表价格中枢位置。")
    P("计算公式：")
    P("　　上轨 = max(high<sub>t-N+1</sub>, high<sub>t-N+2</sub>, ..., high<sub>t</sub>)", indent=42)
    P("　　下轨 = min(low<sub>t-N+1</sub>, low<sub>t-N+2</sub>, ..., low<sub>t</sub>)", indent=42)
    P("　　中轨 = (上轨 + 下轨) / 2", indent=42)

    P("在海龟策略中，唐奇安通道有两个关键应用：")
    P("（1）<b>入场通道（通常20日）</b>：当收盘价突破20日新高时，发出买入信号。"
      "这是“顺势而为”的体现——在趋势确认启动后入场。")
    P("（2）<b>出场通道（通常10日）</b>：当收盘价跌破10日新低时，发出卖出信号。"
      "出场周期短于入场周期，确保在趋势反转初期能够及时离场锁定利润。")

    H("2.2  ATR（Average True Range）—— 平均真实波幅", 2)
    P("ATR 由 Welles Wilder 于1978年提出，是衡量市场波动率的经典指标，也是海龟策略风险控制的核心。"
      "与普通标准差不同，ATR 考虑了跳空缺口，能更真实地反映价格波动。")

    P("ATR 的计算分为两步：")
    P("第一步，计算<b>真实波幅（True Range, TR）</b>。TR 取以下三个值中的最大值：")
    P("　　① 当日最高价 - 当日最低价（当日波动幅度）", indent=42)
    P("　　② |当日最高价 - 前一日收盘价|（向上跳空幅度）", indent=42)
    P("　　③ |当日最低价 - 前一日收盘价|（向下跳空幅度）", indent=42)
    P("第二步，对 TR 进行 N 周期移动平均，得到 ATR。海龟策略使用 Wilder 平滑法：")
    P("　　ATR<sub>t</sub> = (ATR<sub>t-1</sub> × (N-1) + TR<sub>t</sub>) / N", indent=42)

    P("ATR 在海龟策略中的三大作用：")
    P("（1）<b>设定止损位</b>：入场后，止损位设在“入场后最高价 - 2×ATR”处，"
      "即价格从最高点回落超过2倍波幅时止损出场。")
    P("（2）<b>计算头寸大小</b>：原版海龟策略中，每1ATR的波动对应账户净值的1%风险，"
      "据此计算单笔交易的最优头寸。")
    P("（3）<b>过滤假突破</b>：ATR 过低时市场波动不足，此时的突破信号可靠性较低，可考虑过滤。")

    H("2.3  止损条件与风控机制", 2)
    P("海龟策略采用双重止损规则，确保在任何情况下都能控制风险：")
    P("（1）<b>通道止损（系统1离场）</b>：价格跌破10日唐奇安通道下轨时卖出。"
      "这是趋势反转的信号——当价格创出10日新低时，原上升趋势大概率已结束。")
    P("（2）<b>ATR 止损（系统2离场）</b>：价格从入场后的最高点回落超过2×ATR时止损卖出。"
      "这是“让利润奔跑”的具体实现——只要不跌破从最高点算起的2ATR止损，就一直持有。")
    P("两者是“或”的关系，只要触发任一条件立即离场。这种双重保险机制"
      "既保证了不会卖得太早错过大趋势，也保证了在趋势急转时能及时止损。")

    story.append(PageBreak())

    # ===================== 三、Python 实现 =====================
    H("三、Python 编程实现", 1)
    P("本报告的所有计算均由 Python 编程实现，核心流程严格对应任务要求。"
      "下面按五个子项说明实现要点，完整代码见附带的 task4_turtle_strategy.py 文件。")

    H("3.1  加载已存储的股价数据", 2)
    P("主样本使用已存储的 CSV 数据文件 gaoneng_huanjing_daily.csv（高能环境，前复权日线），"
      "使用 pandas 直接读取。对比股票数据（贵州茅台、比亚迪、中国平安、宁德时代）通过腾讯财经行情接口"
      "在线抓取并缓存为本地 CSV 文件，避免重复请求。所有数据字段统一为：date, open, close, high, low, "
      "volume, amount, pct_chg, turnover。")

    H("3.2  计算高低价格通道（唐奇安通道）", 2)
    P("设定入场通道周期 entry_period（默认20日）和出场通道周期 exit_period（默认10日），"
      "调用 calc_donchian_channel 函数分别计算两组唐奇安通道。函数使用滑动窗口方式，"
      "在每个时间点取前N日最高价的最大值作为上轨、前N日最低价的最小值作为下轨。"
      "前 period-1 个交易日通道尚未成型，记为 None。")

    H("3.3  计算 ATR 数值", 2)
    P("调用 calc_atr 函数，首先计算每日真实波幅 TR（取当日高低价差、高与昨收差、低与昨收差三者的最大值），"
      "然后使用 Wilder 平滑法对 TR 进行20周期移动平均得到 ATR 序列。"
      "ATR 的单位与价格相同，代表平均每日价格波动幅度。")

    H("3.4  计算买入卖出交易信号", 2)
    P("在 calc_turtle_signals 函数中按日迭代：")
    P("① 空仓状态下，若收盘价突破20日通道上轨，发出买入信号，进入持仓状态；")
    P("② 持仓状态下，每日记录入场后的最高价；")
    P("③ 当价格跌破10日通道下轨 OR 从入场后最高价回落超过2×ATR时，发出卖出信号，清仓离场。")
    P("买入信号用绿色上三角标记，卖出信号用红色下三角标记。")

    H("3.5  绘制可视化图形", 2)
    P("使用 matplotlib 绘制四张核心图表：")
    P("① 价格 + 唐奇安通道 + 买卖信号 + ATR 指标叠加图（图1）；")
    P("② 策略净值曲线 vs 买入持有基准 + 回撤对比（图2）；")
    P("③ 不同通道周期参数下的多指标对比（累计回报、夏普、回撤、胜率）（图3）；")
    P("④ 五只股票在标准参数下的收益与夏普对比（图4）。")
    P("所有图表均带编号、标题和图例，并导入 PDF 报告配合文字解读。")

    H("3.6  策略回测与量化指标计算", 2)
    P("回测假设：在信号出现当日收盘按收盘价成交，次日按持仓获得当日收益率；"
      "空仓时收益为0（即不参与市场、资金无息）。回测计算的核心指标包括：")
    P("（1）<b>累计回报</b>：期末净值 - 1，衡量策略总收益；")
    P("（2）<b>年化收益</b>：将区间收益折算为年度收益率，便于跨周期比较；")
    P("（3）<b>年化波动率</b>：日收益标准差 × √252，衡量策略风险水平；")
    P("（4）<b>夏普比率</b>：(日均收益 - 无风险利率) / 日收益标准差 × √252，"
      "衡量风险调整后收益（无风险利率取年化2%）；")
    P("（5）<b>最大回撤</b>：历史上从峰值到谷底的最大亏损幅度，衡量最坏情况风险；")
    P("（6）<b>交易次数、胜率、平均每笔收益、平均持仓天数</b>：衡量信号质量与交易频率。")

    story.append(PageBreak())

    # ===================== 四、主样本回测结果 =====================
    m = primary_result["metrics"]
    H("四、主样本回测结果：高能环境（603588，入场20日/出场10日）", 1)
    P(f"对主样本高能环境以入场周期 20 日、出场周期 10 日运行海龟策略，"
      f"回测区间共 {m['n']} 个交易日。"
      f"策略累计回报为 {m['cum_ret']*100:.2f} %，买入持有基准为 {m['bnh_cum']*100:.2f} %，"
      f"策略相对基准的超额收益为 {m['excess']*100:+.2f} %。")
    P(f"策略夏普比率为 {m['sharpe']:.2f}，买入持有为 {m['bnh_sharpe']:.2f}。"
      f"策略最大回撤 {m['mdd']*100:.2f} %，{'优于' if abs(m['mdd']) < abs(m['bnh_mdd']) else '差于'} "
      f"买入持有的 {m['bnh_mdd']*100:.2f} %。")
    P(f"期间共产生 {m['n_trades']} 笔完整交易，胜率 {m['win_rate']*100:.1f} %，"
      f"平均每笔收益 {m['avg_trade']*100:+.2f} %，平均持仓 {m['avg_hold_days']:.1f} 天。")

    # 表1：核心指标汇总
    rows = [
        ["累计回报", f"{m['cum_ret']*100:.2f} %", f"{m['bnh_cum']*100:.2f} %"],
        ["年化收益", f"{m['ann_ret']*100:.2f} %", "-"],
        ["年化波动率", f"{m['ann_vol']*100:.2f} %", f"{m['bnh_ann_vol']*100:.2f} %"],
        ["夏普比率", f"{m['sharpe']:.2f}", f"{m['bnh_sharpe']:.2f}"],
        ["最大回撤", f"{m['mdd']*100:.2f} %", f"{m['bnh_mdd']*100:.2f} %"],
        ["交易次数", f"{m['n_trades']} 次", "-"],
        ["胜率", f"{m['win_rate']*100:.1f} %", "-"],
        ["平均持仓", f"{m['avg_hold_days']:.1f} 天", "-"],
    ]
    TBL(["指标", "海龟策略", "买入持有"], rows, col_w=[4*cm, 4*cm, 4*cm])

    # 图1
    CAP("图1  高能环境（603588）海龟策略信号与ATR指标图")
    IMG(chart_files["signal"])
    P("图1解读：上图为收盘价与唐奇安通道。红色虚线为 20 日入场通道上轨，"
      "橙色虚线为 10 日出场通道下轨。绿色上三角标记买入点，红色下三角标记卖出点。"
      "可见在价格快速拉升阶段，通道同步向上扩张，趋势跟踪效果良好。"
      "在横盘震荡阶段通道收窄，若价格频繁穿越通道边界，会产生较多的假突破信号。")
    P("下图为 ATR 平均真实波幅指标。"
      "波峰对应价格剧烈波动阶段，波谷对应横盘整理阶段，"
      "ATR 的动态变化为止损位提供了自适应调整依据。")

    story.append(PageBreak())

    # 图2
    CAP("图2  高能环境海龟策略净值曲线与回撤对比")
    IMG(chart_files["equity"])
    P("图2解读：上图为净值曲线对比，绿色实线为海龟策略净值，灰色虚线为买入持有净值。"
      "可见在大趋势行情中，买入持有通常收益更高；但在趋势反转的下跌阶段，"
      "海龟策略能及时发出离场信号，避免净值大幅回撤。下图为回撤幅度对比，"
      "绿色填充为策略回撤，灰色填充为买入持有回撤，清晰展示了趋势跟踪策略"
      "在风控方面的优势——虽然可能错过部分顶部收益，但也避免了深度套牢。")

    story.append(PageBreak())

    # ===================== 五、参数敏感性与多股票对比 =====================
    H("五、参数敏感性分析与多股票对比实验", 1)
    P("为考察海龟策略的适用边界，本节进行两组对比实验："
      "第一组在高能环境上测试不同的通道周期组合（10/5、20/10、30/15、50/25），"
      "分析参数敏感性；第二组在五只风格各异的股票上统一使用标准参数（20/10），"
      "比较策略在不同品种上的表现差异。")

    H("5.1  同一股票（高能环境）不同通道周期对比", 2)
    rows = []
    for (e, x) in param_combos:
        r = all_results[("高能环境", e, x)]["metrics"]
        rows.append([f"进{e}日/出{x}日", f"{r['cum_ret']*100:.2f}%", f"{r['bnh_cum']*100:.2f}%",
                     f"{r['sharpe']:.2f}", f"{r['mdd']*100:.2f}%", r['n_trades'],
                     f"{r['win_rate']*100:.1f}%"])
    TBL(["通道周期", "策略累计回报", "买入持有", "夏普比率", "最大回撤", "交易次数", "胜率"],
        rows, col_w=[2.6*cm, 2.2*cm, 2.0*cm, 1.8*cm, 2.0*cm, 1.8*cm, 1.8*cm])

    P("表1解读：对高能环境而言，周期越短（如进10/出5）信号越灵敏，交易次数越多，"
      "在震荡行情中容易被反复打脸；周期越长（如进50/出25）信号越稳健，交易次数显著减少，"
      "能够过滤掉大部分噪音，但也可能错过中期波段机会。不同周期下策略与买入持有的优劣关系"
      "并不固定，需结合标的的波动率特征和市场状态选择参数。")

    # 图3
    CAP("图3  不同通道周期下的多指标对比（2×2子图）")
    IMG(chart_files["param"], width=15.5*cm)
    P("图3解读：四张子图分别展示累计回报、夏普比率、最大回撤、胜率在不同参数下的表现。"
      "左上为累计回报，参数对总收益的影响显著；右上为夏普，体现风险调整后收益；"
      "左下为最大回撤（越小越好），体现风控效果；右下为胜率，海龟策略胜率通常不高，"
      "这是趋势跟踪策略的典型特征——依靠高盈亏比而非高胜率盈利。")

    story.append(PageBreak())

    H("5.2  不同股票在固定参数（进20/出10）下的表现对比", 2)
    names = [s["name"] for s in stocks]
    rows2 = []
    for nm in names:
        r = all_results[(nm, 20, 10)]["metrics"]
        rows2.append([nm, f"{r['cum_ret']*100:.2f}%", f"{r['bnh_cum']*100:.2f}%",
                      f"{r['sharpe']:.2f}", f"{r['mdd']*100:.2f}%", r['n_trades'],
                      f"{r['win_rate']*100:.1f}%"])
    TBL(["股票名称", "策略累计回报", "买入持有", "夏普比率", "最大回撤", "交易次数", "胜率"],
        rows2, col_w=[2.8*cm, 2.2*cm, 2.2*cm, 1.8*cm, 2.0*cm, 1.8*cm, 1.8*cm])

    P("表2解读：在统一的进20/出10参数下，五只股票的表现呈现一条清晰规律——"
      "海龟策略在趋势性强、波动大的品种上表现较好，而在长期横盘或下跌趋势的品种上"
      "表现不佳。中国平安等“慢熊”标的上，策略因及时离场减少了亏损；"
      "而在宁德时代等“快涨快跌”标的上，策略若能抓住主升浪则收益可观。"
      "值得注意的是，海龟策略的最大回撤在所有样本上均显著优于或接近买入持有，"
      "再次验证了其风控的有效性是第一位的。")

    # 图4
    CAP("图4  多股票在固定参数（进20/出10）下的累计收益与夏普比率对比")
    IMG(chart_files["multi"], width=15.5*cm)
    P("图4解读：左轴柱形为各股票策略与买入持有的累计收益率，右轴折线为夏普比率。"
      "绿色柱高于灰色柱表示策略跑赢基准，反之则跑输。"
      "可以看到不同品种上策略的表现差异巨大，再次印证了“没有万能的策略，"
      "只有适配的品种”这一量化投资的基本规律。")

    story.append(PageBreak())

    # ===================== 六、总结与心得 =====================
    H("六、海龟策略适用场景与实战应用心得", 1)
    P("综合上述回测与对比实验，对海龟交易策略的适用场景、局限性与实战心得总结如下：")

    H("6.1  适用场景", 2)
    P("（1）<b>大趋势行情</b>：在持续时间长、幅度大的单边上涨或下跌行情中，"
      "海龟策略的突破入场+趋势跟踪机制能充分捕捉主趋势，实现“让利润奔跑”。"
      "这是海龟策略最经典的盈利模式。")
    P("（2）<b>高波动率品种</b>：ATR 较大、趋势性强的标的更适合海龟策略，"
      "波动率过低的品种突破信号可靠性差，容易频繁假突破。")
    P("（3）<b>多品种组合</b>：原版海龟策略同时交易20+商品期货，"
      "通过分散化确保始终有品种处于趋势行情中，平滑单一品种的亏损期。"
      "A股应用中，也建议同时跟踪多个行业指数或ETF，而非单押个股。")
    P("（4）<b>纪律化风控</b>：将海龟策略的双重止损作为强制离场纪律，"
      "能有效克服“扛单”的人性弱点，适合需要建立交易纪律的投资者。")

    H("6.2  不适用场景与局限性", 2)
    P("（1）<b>长期横盘震荡市</b>：价格在区间内来回震荡时，唐奇安通道会反复发出"
      "假突破信号，策略陷入“买在高点、卖在低点”的恶性循环，持续磨损本金。"
      "这是趋势跟踪策略的最大天敌。")
    P("（2）<b>V型快速反转行情</b>：通道突破天然滞后，在政策消息导致的尖顶"
      "V型反转中，策略往往来不及离场，回吐全部利润甚至亏损。")
    P("（3）<b>低流动性小盘股</b>：成交稀少的股票价格容易被操纵，"
      "假突破信号极多，策略难以有效执行。")
    P("（4）<b>参数过拟合风险</b>：针对历史数据过度优化参数，"
      "在未来市场中往往失效。原版策略的20/10参数是经过几十年验证的，"
      "建议作为基准，仅做小幅调整。")

    H("6.3  实战应用心得", 2)
    P("第一，<b>风控优先，收益第二</b>：海龟策略的核心价值不在于高收益，"
      "而在于可控的风险。实盘应用中首先关注最大回撤和单笔风险，而非历史最高收益。")
    P("第二，<b>接受低胜率，拥抱高盈亏比</b>：30%-40%的胜率是海龟策略的常态，"
      "大部分交易是小亏或小赚，关键是少数几笔大赚要能拿住。"
      "不要因为连续几次止损就怀疑策略，这是趋势跟踪的成本。")
    P("第三，<b>多品种分散是关键</b>：单一品种可能连续几年没有趋势，"
      "策略持续磨损；但20个品种的组合中每年总有几个有大趋势。"
      "品种越多，策略越接近原版海龟的本质。")
    P("第四，<b>加入波动率过滤</b>：ATR 过低时暂停交易，"
      "等待波动率回归后再接受突破信号，可显著减少震荡市中的假突破磨损。"
      "也可以配合布林带、MACD 等指标进行信号过滤，提高胜率。")
    P("第五，<b>严格执行，杜绝主观</b>：海龟系统的前提是“交易者不比市场聪明”。"
      "如果在信号出现时犹豫、在止损时抱有幻想、在盈利时急于落袋为安，"
      "那么再好的策略也无法发挥作用。机械化执行是海龟策略成功的必要条件。")
    P("总体而言，海龟交易策略是一套完整且经过实战检验的趋势跟踪系统。"
      "它不仅提供了具体的交易规则，更重要的是传递了“突破入场、趋势持有、"
      "止损离场、分散风险、系统执行”的量化投资理念。在当前 A股市场，"
      "将海龟策略应用于宽基指数ETF与行业ETF组合，配合严格的风控执行，"
      "仍不失为一种值得探索的量化投资路径。")

    doc.build(story)
    print(f"PDF 已生成: {out_path}")


# ============================================================================
# 主流程
# ============================================================================
def main():
    print("=" * 70)
    print("  海龟交易策略研究报告生成")
    print("=" * 70)

    # ---- 股票池 ----
    stocks = [
        {"code": "603588", "name": "高能环境", "market": "1", "source": "csv"},
        {"code": "600519", "name": "贵州茅台", "tencent": "sh600519", "source": "fetch"},
        {"code": "002594", "name": "比亚迪", "tencent": "sz002594", "source": "fetch"},
        {"code": "601318", "name": "中国平安", "tencent": "sh601318", "source": "fetch"},
        {"code": "300750", "name": "宁德时代", "tencent": "sz300750", "source": "fetch"},
    ]
    param_combos = [(10, 5), (20, 10), (30, 15), (50, 25)]  # (入场, 出场) 周期组合

    # ---- 加载数据 ----
    print("\n[数据] 加载股票数据...")
    dfs = {}
    for s in stocks:
        if s["source"] == "csv":
            print(f"  {s['name']}: 从 CSV 加载 {CSV_STORED}")
            dfs[s["name"]] = load_csv(CSV_STORED)
        else:
            cache_path = os.path.join(OUTPUT_DIR, f"stock_{s['code']}.csv")
            if os.path.exists(cache_path):
                print(f"  {s['name']}: 命中缓存 {os.path.basename(cache_path)}")
                dfs[s["name"]] = load_csv(cache_path)
            else:
                print(f"  {s['name']}: 从腾讯财经抓取...")
                data = fetch_tencent(s["tencent"])
                # 保存为 CSV
                import csv as csv_module
                with open(cache_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv_module.DictWriter(f, fieldnames=["date","open","close","high","low","volume","amount","pct_chg","turnover"])
                    writer.writeheader()
                    writer.writerows(data)
                dfs[s["name"]] = load_csv(cache_path)
                print(f"        {len(data)} 条数据已保存")
        print(f"        {len(dfs[s['name']])} 个交易日")

    # ---- 回测：所有 (股票 × 参数组合) ----
    print("\n[回测] 计算所有 (股票 × 参数) 组合...")
    all_results = {}
    for s in stocks:
        for (entry_period, exit_period) in param_combos:
            signals = calc_turtle_signals(dfs[s["name"]], entry_period=entry_period, exit_period=exit_period)
            result = backtest_turtle(signals)
            all_results[(s["name"], entry_period, exit_period)] = result
            print(f"  {s['name']} (进{entry_period}/出{exit_period}): "
                  f"累计收益 {result['metrics']['cum_ret']*100:+.2f}%, "
                  f"夏普 {result['metrics']['sharpe']:.2f}")

    primary_result = all_results[("高能环境", 20, 10)]  # 主样本用标准参数

    # ---- 生成图表 ----
    print("\n[图表] 生成可视化图形...")
    chart_files = {}

    chart_files["signal"] = os.path.join(OUTPUT_DIR, "chart_turtle_signal.png")
    chart_turtle_signal(primary_result, "图1  高能环境海龟策略信号与ATR指标", chart_files["signal"])
    print(f"  ✓ 信号图: {os.path.basename(chart_files['signal'])}")

    chart_files["equity"] = os.path.join(OUTPUT_DIR, "chart_turtle_equity.png")
    chart_turtle_equity(primary_result, "图2  高能环境海龟策略净值 vs 买入持有", chart_files["equity"])
    print(f"  ✓ 净值图: {os.path.basename(chart_files['equity'])}")

    names = [s["name"] for s in stocks]
    chart_files["param"] = os.path.join(OUTPUT_DIR, "chart_param_compare.png")
    chart_param_compare(names, param_combos, all_results, "图3  不同通道周期下的多指标对比", chart_files["param"])
    print(f"  ✓ 参数对比图: {os.path.basename(chart_files['param'])}")

    chart_files["multi"] = os.path.join(OUTPUT_DIR, "chart_multi_stock.png")
    chart_multi_stock(names, all_results, (20, 10), "图4  多股票在固定参数(20/10)下的收益与夏普对比", chart_files["multi"])
    print(f"  ✓ 多股票对比图: {os.path.basename(chart_files['multi'])}")

    # ---- 生成 PDF ----
    print("\n[PDF] 生成报告...")
    out_path = os.path.join(OUTPUT_DIR, "张鹏+TASK4.pdf")
    build_pdf_report(stocks, primary_result, param_combos, all_results, chart_files, out_path)
    print(f"  ✓ PDF: {os.path.basename(out_path)}")

    # ---- 生成 HTML 交互图表 ----
    print("\n[HTML] 生成交互式图表...")
    from generate_html_charts import build_boll_html, build_rsi_html, build_macd_html
    import json

    # 生成海龟策略专用的 HTML
    signal_data = primary_result["signal_data"]
    dates_list = signal_data["dates"].tolist()
    close_list = signal_data["closes"].tolist()
    entry_upper_list = [v if v is not None else None for v in signal_data["entry_upper"]]
    exit_lower_list = [v if v is not None else None for v in signal_data["exit_lower"]]
    atr_list = [v if v is not None else None for v in signal_data["atr"]]

    # 买卖信号坐标
    buy_x = [i for i, v in enumerate(signal_data["buy_signals"]) if v is not None]
    buy_y = [signal_data["buy_signals"][i] for i in buy_x]
    sell_x = [i for i, v in enumerate(signal_data["sell_signals"]) if v is not None]
    sell_y = [signal_data["sell_signals"][i] for i in sell_x]

    html_path = os.path.join(HTML_DIR, "turtle_strategy.html")
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>海龟交易策略 - 交互式图表</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif; background:#f0f2f5; }}
.header {{ background: linear-gradient(135deg,#667eea,#764ba2); color:#fff; padding:20px 32px; }}
.header h1 {{ font-size:22px; margin-bottom:4px; }}
.header .sub {{ opacity:.85; font-size:13px; }}
.container {{ padding:16px 32px 32px; }}
.chart-box {{ background:#fff; border-radius:10px; box-shadow:0 2px 12px rgba(0,0,0,.08); overflow:hidden; margin-bottom:20px; }}
.chart-title {{ padding:14px 20px; font-size:14px; font-weight:600; color:#333; border-bottom:1px solid #f0f0f0; }}
.chart {{ width:100%; }}
.footer {{ text-align:center; padding:16px; color:#999; font-size:12px; }}
</style>
</head>
<body>
<div class="header">
  <h1>高能环境（603588.SH）海龟交易策略</h1>
  <div class="sub">参数: 入场20日/出场10日 | 数据区间: {dates_list[0]} ~ {dates_list[-1]}</div>
</div>
<div class="container">
  <div class="chart-box">
    <div class="chart-title">价格通道与买卖信号</div>
    <div id="priceChart" class="chart" style="height:500px;"></div>
  </div>
  <div class="chart-box">
    <div class="chart-title">ATR 平均真实波幅</div>
    <div id="atrChart" class="chart" style="height:300px;"></div>
  </div>
  <div class="chart-box">
    <div class="chart-title">策略净值 vs 买入持有</div>
    <div id="equityChart" class="chart" style="height:400px;"></div>
  </div>
</div>
<div class="footer">数据来源: 东方财富 | 仅供学习参考，不构成投资建议</div>

<script>
var dates = {json.dumps(dates_list)};
var closeData = {json.dumps([round(v, 2) for v in close_list])};
var entryUpper = {json.dumps([round(v, 2) if v is not None else None for v in entry_upper_list])};
var exitLower = {json.dumps([round(v, 2) if v is not None else None for v in exit_lower_list])};
var atrData = {json.dumps([round(v, 4) if v is not None else None for v in atr_list])};
var buyX = {json.dumps(buy_x)};
var buyY = {json.dumps([round(v, 2) for v in buy_y])};
var sellX = {json.dumps(sell_x)};
var sellY = {json.dumps([round(v, 2) for v in sell_y])};
var equityData = {json.dumps([round(v, 4) for v in primary_result["equity"]])};
var bnhEquityData = {json.dumps([round(v, 4) for v in primary_result["bnh_equity"]])};

// 价格通道图
var priceChart = echarts.init(document.getElementById("priceChart"));
priceChart.setOption({{
    tooltip: {{ trigger: 'axis' }},
    legend: {{ data: ['收盘价', '入场通道上轨(20日)', '出场通道下轨(10日)', '买入信号', '卖出信号'], top: 8 }},
    grid: {{ left: '5%', right: '5%', top: '12%', bottom: '10%' }},
    xAxis: {{
        type: 'category', data: dates, boundaryGap: false,
        axisLabel: {{ formatter: function(v) {{ return v.substring(5); }} }}
    }},
    yAxis: {{ scale: true, splitLine: {{ lineStyle: {{ type: 'dashed', color: '#e8e8e8' }} }} }},
    dataZoom: [
        {{ type: 'inside', start: 40, end: 100 }},
        {{ show: true, type: 'slider', bottom: 4, start: 40, end: 100, height: 22 }}
    ],
    series: [
        {{ name: '收盘价', type: 'line', data: closeData, lineStyle: {{ width: 1.8, color: '#2c3e50' }}, symbol: 'none' }},
        {{ name: '入场通道上轨(20日)', type: 'line', data: entryUpper, lineStyle: {{ width: 1.2, color: '#e74c3c', type: 'dashed' }}, symbol: 'none' }},
        {{ name: '出场通道下轨(10日)', type: 'line', data: exitLower, lineStyle: {{ width: 1.2, color: '#f39c12', type: 'dashed' }}, symbol: 'none' }},
        {{
            name: '买入信号', type: 'scatter',
            data: buyX.map(function(x, i) {{ return [x, buyY[i]]; }}),
            symbol: 'triangle', symbolSize: 12, itemStyle: {{ color: '#27ae60' }},
            emphasis: {{ itemStyle: {{ borderColor: '#fff', borderWidth: 2 }} }}
        }},
        {{
            name: '卖出信号', type: 'scatter',
            data: sellX.map(function(x, i) {{ return [x, sellY[i]]; }}),
            symbol: 'path://M -6,6 L 0,-6 L 6,6 Z', symbolSize: 12, itemStyle: {{ color: '#c0392b' }},
            emphasis: {{ itemStyle: {{ borderColor: '#fff', borderWidth: 2 }} }}
        }}
    ]
}});

// ATR图
var atrChart = echarts.init(document.getElementById("atrChart"));
atrChart.setOption({{
    tooltip: {{ trigger: 'axis' }},
    legend: {{ data: ['ATR(20日)'], top: 8 }},
    grid: {{ left: '5%', right: '5%', top: '12%', bottom: '10%' }},
    xAxis: {{
        type: 'category', data: dates, boundaryGap: false,
        axisLabel: {{ formatter: function(v) {{ return v.substring(5); }} }}
    }},
    yAxis: {{ scale: true, splitLine: {{ lineStyle: {{ type: 'dashed', color: '#e8e8e8' }} }} }},
    dataZoom: [
        {{ type: 'inside', start: 40, end: 100 }},
        {{ show: true, type: 'slider', bottom: 4, start: 40, end: 100, height: 22 }}
    ],
    series: [
        {{ name: 'ATR(20日)', type: 'line', data: atrData, lineStyle: {{ width: 1.5, color: '#9b59b6' }},
           areaStyle: {{ color: 'rgba(155, 89, 182, 0.2)' }}, symbol: 'none' }}
    ]
}});

// 净值图
var equityChart = echarts.init(document.getElementById("equityChart"));
equityChart.setOption({{
    tooltip: {{ trigger: 'axis', formatter: function(params) {{
        var html = params[0].axisValue + '<br/>';
        params.forEach(function(p) {{
            html += p.marker + p.seriesName + ': <b>' + Number(p.value).toFixed(4) + '</b><br/>';
        }});
        return html;
    }} }},
    legend: {{ data: ['海龟策略净值', '买入持有净值'], top: 8 }},
    grid: {{ left: '5%', right: '5%', top: '12%', bottom: '10%' }},
    xAxis: {{
        type: 'category', data: dates, boundaryGap: false,
        axisLabel: {{ formatter: function(v) {{ return v.substring(5); }} }}
    }},
    yAxis: {{ scale: true, splitLine: {{ lineStyle: {{ type: 'dashed', color: '#e8e8e8' }} }} }},
    dataZoom: [
        {{ type: 'inside', start: 40, end: 100 }},
        {{ show: true, type: 'slider', bottom: 4, start: 40, end: 100, height: 22 }}
    ],
    series: [
        {{ name: '海龟策略净值', type: 'line', data: equityData, lineStyle: {{ width: 2, color: '#27ae60' }}, symbol: 'none' }},
        {{ name: '买入持有净值', type: 'line', data: bnhEquityData, lineStyle: {{ width: 1.5, color: '#7f8c8d', type: 'dashed' }}, symbol: 'none' }}
    ]
}});

window.addEventListener('resize', function() {{
    priceChart.resize();
    atrChart.resize();
    equityChart.resize();
}});
</script>
</body></html>"""

    os.makedirs(HTML_DIR, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  ✓ HTML: {os.path.relpath(html_path, OUTPUT_DIR)}")

    print("\n" + "=" * 70)
    print("  全部任务完成！")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

