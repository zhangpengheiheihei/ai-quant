# -*- coding: utf-8 -*-
"""
生成技术指标图的 HTML 交互版本（ECharts）
输出:
  - chart_boll.html   布林带
  - chart_rsi.html    RSI
  - chart_macd.html   MACD
  - chart_kdj.html    KDJ
"""

import csv
import json
import os
import sys
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUTPUT_DIR, "gaoneng_huanjing_daily.csv")

UP_COLOR  = "#e74c3c"
DOWN_COLOR = "#27ae60"
LINE_COLOR = "#667eea"


# ---- 指标计算 (与 task2_report.py 一致) ----

def calc_rsi(closes, period=14):
    n = len(closes)
    if n <= period:
        return [None] * n
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    rsi = [None] * n
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


def calc_macd(closes, fast=12, slow=26, signal=9):
    n = len(closes)
    def ema(series, period):
        result = [None] * len(series)
        result[period - 1] = float(np.mean(series[:period]))
        k = 2.0 / (period + 1.0)
        for i in range(period, len(series)):
            result[i] = series[i] * k + result[i - 1] * (1.0 - k)
        return result
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    dif = [None] * n
    for i in range(slow - 1, n):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            dif[i] = round(ema_fast[i] - ema_slow[i], 4)
    dea = [None] * n
    valid_dif = [v for v in dif if v is not None]
    if len(valid_dif) >= signal:
        start_idx = slow - 1
        sma_dea = float(np.mean(valid_dif[:signal]))
        k_dea = 2.0 / (signal + 1.0)
        dea_idx = start_idx + signal - 1
        dea[dea_idx] = round(sma_dea, 4)
        for i in range(dea_idx + 1, n):
            if dif[i] is not None:
                dea[i] = round(dif[i] * k_dea + dea[i - 1] * (1.0 - k_dea), 4)
    macd_hist = []
    for i in range(n):
        if dif[i] is not None and dea[i] is not None:
            macd_hist.append(round(2.0 * (dif[i] - dea[i]), 4))
        else:
            macd_hist.append(None)
    return dif, dea, macd_hist


def calc_bollinger(closes, period=20, std_dev=2.0):
    n = len(closes)
    middle, upper, lower = [None] * n, [None] * n, [None] * n
    for i in range(period - 1, n):
        window = closes[i - period + 1: i + 1]
        ma = float(np.mean(window))
        std = float(np.std(window, ddof=1))
        middle[i] = round(ma, 2)
        upper[i] = round(ma + std_dev * std, 2)
        lower[i] = round(ma - std_dev * std, 2)
    return middle, upper, lower


def calc_kdj(highs, lows, closes, period=9, kp=3, dp=3):
    n = len(closes)
    k_vals, d_vals, j_vals = [None] * n, [None] * n, [None] * n
    rsv = [None] * n
    for i in range(period - 1, n):
        hh = max(highs[i - period + 1: i + 1])
        ll = min(lows[i - period + 1: i + 1])
        if hh == ll:
            rsv[i] = 50.0
        else:
            rsv[i] = round((closes[i] - ll) / (hh - ll) * 100.0, 2)
    start = period - 1
    k_vals[start] = 50.0
    d_vals[start] = 50.0
    for i in range(start + 1, n):
        if rsv[i] is not None:
            k_vals[i] = round((k_vals[i - 1] * (kp - 1) + rsv[i]) / kp, 2)
            d_vals[i] = round((d_vals[i - 1] * (dp - 1) + k_vals[i]) / dp, 2)
            j_vals[i] = round(3.0 * k_vals[i] - 2.0 * d_vals[i], 2)
    return k_vals, d_vals, j_vals


# ---- HTML 模板 ----

def html_head(title):
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif; background:#f0f2f5; }}
.header {{ background: linear-gradient(135deg,#667eea,#764ba2); color:#fff; padding:20px 32px; }}
.header h1 {{ font-size:22px; margin-bottom:4px; }}
.header .sub {{ opacity:.85; font-size:13px; }}
.container {{ padding:16px 32px 32px; }}
.chart-box {{ background:#fff; border-radius:10px; box-shadow:0 2px 12px rgba(0,0,0,.08); overflow:hidden; }}
.chart-title {{ padding:14px 20px; font-size:14px; font-weight:600; color:#333; border-bottom:1px solid #f0f0f0; }}
.chart {{ width:100%; }}
.footer {{ text-align:center; padding:16px; color:#999; font-size:12px; }}
</style>
</head>"""


def html_footer():
    return '<div class="footer">高能环境（603588.SH）| 数据来源: 东方财富 | 仅供学习参考，不构成投资建议</div>\n</body>\n</html>'


# ---- 各图表 HTML 生成 ----

def build_boll_html(dates, closes, bb_mid, bb_up, bb_low):
    n = len(dates)
    # 过滤 None 点
    c_data = []
    mid_data = []
    up_data = []
    low_data = []
    # 填充区域
    area_data = []
    for i in range(n):
        c_data.append(round(float(closes[i]), 2))
        if bb_mid[i] is not None:
            mid_data.append(round(bb_mid[i], 2))
            up_data.append(round(bb_up[i], 2))
            low_data.append(round(bb_low[i], 2))
            area_data.append([round(bb_low[i], 2), round(bb_up[i], 2)])
        else:
            mid_data.append(None)
            up_data.append(None)
            low_data.append(None)
            area_data.append(None)

    html = html_head("高能环境 - 布林带指标")
    html += f"""<body>
<div class="header">
  <h1>高能环境（603588.SH）布林带指标（BOLL）</h1>
  <div class="sub">参数: N=20, k=2 | 数据: {dates[0]} ~ {dates[-1]} | 共 {n} 个交易日</div>
</div>
<div class="container">
  <div class="chart-box">
    <div class="chart-title">图1  布林带指标 (中轨=MA20，上/下轨=MA20±2σ)</div>
    <div id="main-chart" class="chart" style="height:560px;"></div>
  </div>
</div>
{html_footer()}
<script>
var dates = {json.dumps(dates)};
var closeData = {json.dumps(c_data)};
var midData = {json.dumps(mid_data)};
var upData = {json.dumps(up_data)};
var lowData = {json.dumps(low_data)};

var chart = echarts.init(document.getElementById('main-chart'));
chart.setOption({{
    tooltip: {{
        trigger: 'axis',
        formatter: function(params) {{
            var d = params[0].axisValue;
            var html = '<div style="font-weight:600;margin-bottom:4px;">'+d+'</div>';
            params.forEach(function(p) {{
                if (p.value != null)
                    html += '<div>'+p.marker+' '+p.seriesName+': <b>'+Number(p.value).toFixed(2)+'</b></div>';
            }});
            return html;
        }}
    }},
    legend: {{ data: ['收盘价','BOLL中轨','BOLL上轨','BOLL下轨'], top:8 }},
    grid: {{ left:'8%', right:'8%', top:'14%', bottom:'10%' }},
    xAxis: {{
        type: 'category', data: dates, boundaryGap: false,
        axisLabel: {{ formatter: function(v){{ return v.substring(5); }} }}
    }},
    yAxis: {{ scale:true, splitLine:{{ lineStyle:{{ type:'dashed',color:'#e8e8e8' }} }} }},
    dataZoom: [
        {{ type:'inside', start:40, end:100 }},
        {{ show:true, type:'slider', bottom:4, start:40, end:100, height:22 }}
    ],
    series: [
        {{
            name: '收盘价', type: 'line', data: closeData, symbol: 'none',
            lineStyle: {{ width: 1.8, color: '#667eea' }},
            z: 3
        }},
        {{
            name: 'BOLL上轨', type: 'line', data: upData, symbol: 'none',
            lineStyle: {{ width: 1, color: '#e74c3c', type: 'solid' }},
            z: 1
        }},
        {{
            name: 'BOLL中轨', type: 'line', data: midData, symbol: 'none',
            lineStyle: {{ width: 1.2, color: '#ffa502', type: 'dashed' }},
            z: 2
        }},
        {{
            name: 'BOLL下轨', type: 'line', data: lowData, symbol: 'none',
            lineStyle: {{ width: 1, color: '#27ae60', type: 'solid' }},
            z: 1,
            areaStyle: {{
                color: new echarts.graphic.LinearGradient(0,0,0,1,[
                    {{ offset:0, color:'rgba(102,126,234,0.12)' }},
                    {{ offset:1, color:'rgba(102,126,234,0.02)' }}
                ])
            }}
        }}
    ]
}});
window.addEventListener('resize', function(){{ chart.resize(); }});
</script>
</body></html>"""
    return html


def build_rsi_html(dates, rsi_vals):
    n = len(dates)
    rsi_data = [round(v, 2) if v is not None else None for v in rsi_vals]
    line70 = [70] * n
    line30 = [30] * n
    line50 = [50] * n

    html = html_head("高能环境 - RSI指标")
    html += f"""<body>
<div class="header">
  <h1>高能环境（603588.SH）相对强弱指标（RSI）</h1>
  <div class="sub">参数: N=14 | 数据: {dates[0]} ~ {dates[-1]} | 共 {n} 个交易日</div>
</div>
<div class="container">
  <div class="chart-box">
    <div class="chart-title">图2  RSI(14) 相对强弱指标</div>
    <div id="main-chart" class="chart" style="height:480px;"></div>
  </div>
</div>
{html_footer()}
<script>
var dates = {json.dumps(dates)};
var rsiData = {json.dumps(rsi_data)};

var chart = echarts.init(document.getElementById('main-chart'));
chart.setOption({{
    tooltip: {{
        trigger: 'axis',
        formatter: function(params) {{
            var v = params[0].value;
            var d = params[0].axisValue;
            var status = v > 70 ? '（超买）' : (v < 30 ? '（超卖）' : '');
            return '<b>'+d+'</b><br>RSI: <b>'+v.toFixed(2)+'</b>'+status;
        }}
    }},
    legend: {{ data: ['RSI(14)','超买线(70)','超卖线(30)','中轴(50)'], top:8 }},
    grid: {{ left:'8%', right:'8%', top:'14%', bottom:'10%' }},
    xAxis: {{
        type: 'category', data: dates, boundaryGap: false,
        axisLabel: {{ formatter: function(v){{ return v.substring(5); }} }}
    }},
    yAxis: {{ min:0, max:100, splitLine:{{ lineStyle:{{ type:'dashed',color:'#e8e8e8' }} }} }},
    dataZoom: [
        {{ type:'inside', start:40, end:100 }},
        {{ show:true, type:'slider', bottom:4, start:40, end:100, height:22 }}
    ],
    visualMap: {{
        show: false, dimension:1, pieces: [
            {{ gt:70, lte:100, color:'#e74c3c' }},
            {{ gt:50, lte:70, color:'#667eea' }},
            {{ gt:30, lte:50, color:'#999' }},
            {{ gte:0, lte:30, color:'#27ae60' }}
        ]
    }},
    series: [
        {{
            name: 'RSI(14)', type: 'line', data: rsiData, symbol: 'none',
            lineStyle: {{ width: 2, color: '#764ba2' }},
            markArea: {{
                silent: true,
                data: [
                    [{{ yAxis:70, itemStyle:{{ color:'rgba(231,76,60,0.06)' }} }}, {{ yAxis:100 }}],
                    [{{ yAxis:0, itemStyle:{{ color:'rgba(39,174,96,0.06)' }} }}, {{ yAxis:30 }}]
                ]
            }}
        }},
        {{ name: '超买线(70)', type:'line', data: new Array(dates.length).fill(70),
            lineStyle: {{ width:1, color:'#e74c3c', type:'dashed' }},
            symbol:'none', emphasis:{{ disabled:true }} }},
        {{ name: '超卖线(30)', type:'line', data: new Array(dates.length).fill(30),
            lineStyle: {{ width:1, color:'#27ae60', type:'dashed' }},
            symbol:'none', emphasis:{{ disabled:true }} }},
        {{ name: '中轴(50)', type:'line', data: new Array(dates.length).fill(50),
            lineStyle: {{ width:0.5, color:'#999', type:'dotted' }},
            symbol:'none', emphasis:{{ disabled:true }} }}
    ]
}});
window.addEventListener('resize', function(){{ chart.resize(); }});
</script>
</body></html>"""
    return html


def build_macd_html(dates, closes, dif, dea, macd_hist):
    n = len(dates)
    c_data = [round(float(v), 2) for v in closes]
    dif_data = [round(v, 4) if v is not None else None for v in dif]
    dea_data = [round(v, 4) if v is not None else None for v in dea]
    hist_data = [round(v, 4) if v is not None else None for v in macd_hist]

    html = html_head("高能环境 - MACD指标")
    html += f"""<body>
<div class="header">
  <h1>高能环境（603588.SH）MACD指标</h1>
  <div class="sub">参数: (12,26,9) | 数据: {dates[0]} ~ {dates[-1]} | 共 {n} 个交易日</div>
</div>
<div class="container">
  <div class="chart-box">
    <div class="chart-title">图3  MACD指标 (DIF / DEA / 柱状图)</div>
    <div id="main-chart" class="chart" style="height:650px;"></div>
  </div>
</div>
{html_footer()}
<script>
var dates = {json.dumps(dates)};
var closeData = {json.dumps(c_data)};
var difData = {json.dumps(dif_data)};
var deaData = {json.dumps(dea_data)};
var histData = {json.dumps(hist_data)};

var chart = echarts.init(document.getElementById('main-chart'));
chart.setOption({{
    tooltip: {{
        trigger: 'axis',
        axisPointer: {{ type: 'cross' }},
        formatter: function(params) {{
            var d = params[0].axisValue;
            var html = '<div style="font-weight:600;margin-bottom:4px;">'+d+'</div>';
            params.forEach(function(p) {{
                if (p.value != null)
                    html += '<div>'+p.marker+' '+p.seriesName+': <b>'+Number(p.value).toFixed(4)+'</b></div>';
            }});
            return html;
        }}
    }},
    grid: [
        {{ left:'8%', right:'8%', top:'10%', height:'45%' }},
        {{ left:'8%', right:'8%', top:'59%', height:'35%' }}
    ],
    xAxis: [
        {{ type:'category', data:dates, gridIndex:0, boundaryGap:false,
           axisLabel:{{ show:false }}, min:'dataMin', max:'dataMax' }},
        {{ type:'category', data:dates, gridIndex:1, boundaryGap:true,
           axisLabel:{{ formatter:function(v){{return v.substring(5);}} }},
           min:'dataMin', max:'dataMax' }}
    ],
    yAxis: [
        {{ gridIndex:0, scale:true, name:'价格(元)', splitLine:{{ lineStyle:{{ type:'dashed',color:'#e8e8e8' }} }} }},
        {{ gridIndex:1, name:'MACD', splitLine:{{ lineStyle:{{ type:'dashed',color:'#e8e8e8' }} }} }}
    ],
    dataZoom: [
        {{ type:'inside', xAxisIndex:[0,1], start:40, end:100 }},
        {{ show:true, type:'slider', xAxisIndex:[0,1], bottom:4, start:40, end:100, height:22 }}
    ],
    series: [
        {{ name:'收盘价', type:'line', xAxisIndex:0, yAxisIndex:0, data:closeData,
           symbol:'none', lineStyle:{{ width:1.5, color:'#667eea' }} }},
        {{ name:'DIF', type:'line', xAxisIndex:1, yAxisIndex:1, data:difData,
           symbol:'none', lineStyle:{{ width:1.2, color:'#667eea' }} }},
        {{ name:'DEA', type:'line', xAxisIndex:1, yAxisIndex:1, data:deaData,
           symbol:'none', lineStyle:{{ width:1.2, color:'#ffa502' }} }},
        {{ name:'MACD柱', type:'bar', xAxisIndex:1, yAxisIndex:1,
           data: histData.map(function(v) {{
               if (v == null) return null;
               return {{ value:v, itemStyle:{{ color: v>=0?'#e74c3c':'#27ae60' }} }};
           }}) }}
    ]
}});
window.addEventListener('resize', function(){{ chart.resize(); }});
</script>
</body></html>"""
    return html


def build_kdj_html(dates, k, d, j):
    n = len(dates)
    k_data = [round(v, 2) if v is not None else None for v in k]
    d_data = [round(v, 2) if v is not None else None for v in d]
    j_data = [round(v, 2) if v is not None else None for v in j]

    html = html_head("高能环境 - KDJ指标")
    html += f"""<body>
<div class="header">
  <h1>高能环境（603588.SH）KDJ随机指标</h1>
  <div class="sub">参数: N=9, K=3, D=3 | 数据: {dates[0]} ~ {dates[-1]} | 共 {n} 个交易日</div>
</div>
<div class="container">
  <div class="chart-box">
    <div class="chart-title">图4  KDJ随机指标 (K / D / J)</div>
    <div id="main-chart" class="chart" style="height:520px;"></div>
  </div>
</div>
{html_footer()}
<script>
var dates = {json.dumps(dates)};
var kData = {json.dumps(k_data)};
var dData = {json.dumps(d_data)};
var jData = {json.dumps(j_data)};

var chart = echarts.init(document.getElementById('main-chart'));
chart.setOption({{
    tooltip: {{
        trigger: 'axis',
        formatter: function(params) {{
            var d = params[0].axisValue;
            var html = '<div style="font-weight:600;margin-bottom:4px;">'+d+'</div>';
            params.forEach(function(p) {{
                if (p.value != null)
                    html += '<div>'+p.marker+' '+p.seriesName+': <b>'+Number(p.value).toFixed(2)+'</b></div>';
            }});
            return html;
        }}
    }},
    legend: {{ data: ['K值','D值','J值','超买(80)','超卖(20)'], top:8 }},
    grid: {{ left:'8%', right:'8%', top:'14%', bottom:'10%' }},
    xAxis: {{
        type: 'category', data: dates, boundaryGap: false,
        axisLabel: {{ formatter: function(v){{ return v.substring(5); }} }}
    }},
    yAxis: {{ min:-10, max:110, splitLine:{{ lineStyle:{{ type:'dashed',color:'#e8e8e8' }} }} }},
    dataZoom: [
        {{ type:'inside', start:40, end:100 }},
        {{ show:true, type:'slider', bottom:4, start:40, end:100, height:22 }}
    ],
    series: [
        {{
            name:'K值', type:'line', data:kData, symbol:'none',
            lineStyle:{{ width:1.5, color:'#667eea' }}
        }},
        {{
            name:'D值', type:'line', data:dData, symbol:'none',
            lineStyle:{{ width:1.5, color:'#ffa502' }}
        }},
        {{
            name:'J值', type:'line', data:jData, symbol:'none',
            lineStyle:{{ width:1, color:'#e74c3c' }},
            markArea: {{
                silent: true,
                data: [
                    [{{ yAxis:80, itemStyle:{{ color:'rgba(231,76,60,0.06)' }} }}, {{ yAxis:110 }}],
                    [{{ yAxis:-10, itemStyle:{{ color:'rgba(39,174,96,0.06)' }} }}, {{ yAxis:20 }}]
                ]
            }}
        }},
        {{ name:'超买(80)', type:'line', data: new Array(dates.length).fill(80),
            lineStyle:{{ width:1, color:'#e74c3c', type:'dashed' }},
            symbol:'none', emphasis:{{ disabled:true }} }},
        {{ name:'超卖(20)', type:'line', data: new Array(dates.length).fill(20),
            lineStyle:{{ width:1, color:'#27ae60', type:'dashed' }},
            symbol:'none', emphasis:{{ disabled:true }} }}
    ]
}});
window.addEventListener('resize', function(){{ chart.resize(); }});
</script>
</body></html>"""
    return html


# ---- 主函数 ----

def main():
    print("加载数据 & 计算指标...")

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        data = list(csv.DictReader(f))

    dates = [d["date"] for d in data]
    closes = np.array([float(d["close"]) for d in data])
    highs = np.array([float(d["high"]) for d in data])
    lows = np.array([float(d["low"]) for d in data])

    # 计算指标
    rsi = calc_rsi(closes, 14)
    dif, dea, macd_hist = calc_macd(closes, 12, 26, 9)
    bb_mid, bb_up, bb_low = calc_bollinger(closes, 20, 2.0)
    k, d, j = calc_kdj(highs, lows, closes, 9, 3, 3)

    files = {
        "chart_boll.html": build_boll_html(dates, closes, bb_mid, bb_up, bb_low),
        "chart_rsi.html": build_rsi_html(dates, rsi),
        "chart_macd.html": build_macd_html(dates, closes, dif, dea, macd_hist),
        "chart_kdj.html": build_kdj_html(dates, k, d, j),
    }

    for name, content in files.items():
        path = os.path.join(OUTPUT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✓ {name}  ({len(content)} 字节)")

    print("\n全部 HTML 图表生成完毕！")


if __name__ == "__main__":
    main()
