# app.py
import streamlit as st
import pandas as pd
import numpy as np
import talib
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from data import fetch_historical, resample_candles, symbols_tuple

# ---------- App Layout ----------
st.set_page_config(layout="wide", page_title="Advanced Multi-Indicator Chart")
st.title("Advanced Multi-Indicator Chart")

# ---------- Sidebar: Chart Settings ----------
st.sidebar.header("Chart Settings")
symbol = st.sidebar.selectbox("Select Symbol", symbols_tuple)

period_options = ["1d", "2d", "5d", "15d"]
period_str = st.sidebar.selectbox("Select Period", period_options)
days = int(period_str.rstrip("d"))

interval_map = {
    "1 Minute": "1T", "5 Minutes": "5T", "15 Minutes": "15T",
    "30 Minutes": "30T", "1 Hour": "1H", "1 Day": "1D"
}
interval_labels = list(interval_map.keys())
interval_label = st.sidebar.selectbox("Chart Interval (base)", interval_labels)
chart_interval = interval_map[interval_label]

# ---------- Fetch 1-min (or high-resolution) cached data ----------
@st.cache_data
def _fetch_1min(symbol):
    # fetch_historical should return at least 1-min resolution data for resampling
    return fetch_historical(symbol, days=15)

data_1m = _fetch_1min(symbol)

# filter last N trading days
unique_dates = np.unique(data_1m.index.date)
last_trading_days = unique_dates[-days:]
df = data_1m[np.isin(data_1m.index.date, last_trading_days)].copy()

# keep trading hours (if intraday) and drop zero-volume days
try:
    df = df.between_time("09:15", "15:30")
except Exception:
    # non intraday index - ignore
    pass
df = df[df.groupby(df.index.date)["volume"].transform("sum") > 0]

# resample to chart interval for display
data = resample_candles(df, chart_interval)
if data.empty:
    st.warning("No data after resampling. Check symbol / period / data availability.")
    st.stop()

# a friendly Price column
data["Price"] = data["close"]

# ---------- Indicator Configuration UI ----------
st.sidebar.header("Indicators Configuration")
available_indicators = [
    "SMA","EMA","WMA","DEMA","TEMA","KAMA","RSI","MACD","ATR","VWAP","BBANDS",
    "STOCH","ADX","CCI","OBV","MFI","MOM","ROC","TRIX","AROON","AROONOSC","SAR"
]

num_indicators = st.sidebar.number_input("Number of Indicator Instances", min_value=1, max_value=20, value=1)
indicator_configs = []

for i in range(int(num_indicators)):
    with st.sidebar.expander(f"Indicator #{i+1} settings", expanded=(i < 3)):
        ind = st.selectbox("Type", available_indicators, key=f"type_{i}")
        period = None
        bb_params = None
        macd_params = None

        if ind in ["SMA","EMA","WMA","RSI","ATR","STOCH","CCI","MOM","TRIX","ROC","DEMA","TEMA","KAMA","MFI","ADX"]:
            period = st.number_input("Period", min_value=1, max_value=500, value=20, key=f"period_{i}")
        if ind == "BBANDS":
            period = st.number_input("BBANDS Period", min_value=2, max_value=200, value=20, key=f"bb_period_{i}")
            nbdevup = st.number_input("Upper Band Std Dev", min_value=0.1, max_value=5.0, value=2.0, step=0.1, key=f"bb_up_{i}")
            nbdevdn = st.number_input("Lower Band Std Dev", min_value=0.1, max_value=5.0, value=2.0, step=0.1, key=f"bb_dn_{i}")
            ma_type = st.selectbox("MA Type", ["SMA","EMA","WMA","DEMA","TEMA"], key=f"bb_ma_{i}")
            bb_params = {"nbdevup": nbdevup, "nbdevdn": nbdevdn, "matype": ma_type}
        if ind == "MACD":
            fast = st.number_input("MACD fast", min_value=1, max_value=200, value=12, key=f"macd_fast_{i}")
            slow = st.number_input("MACD slow", min_value=1, max_value=400, value=26, key=f"macd_slow_{i}")
            signal = st.number_input("MACD signal", min_value=1, max_value=200, value=9, key=f"macd_signal_{i}")
            macd_params = {"fastperiod": fast, "slowperiod": slow, "signalperiod": signal}

        tf_label = st.selectbox("Timeframe (resampled)", ["Same as Chart"] + interval_labels, key=f"tf_{i}")
        tf = interval_map[tf_label] if tf_label != "Same as Chart" else chart_interval

        indicator_configs.append({
            "type": ind,
            "period": period,
            "timeframe": tf,
            "bb_params": bb_params,
            "macd_params": macd_params
        })

# ---------- Compute Indicators (only for the timeframes user requested) ----------
computed_cols = []  # exact column names we compute; used for condition dropdowns
# which indicators will get separate subplots
oscillator_set = {"RSI","ATR","MACD","ADX","STOCH","CCI","MFI","TRIX","ROC","OBV"}

for cfg in indicator_configs:
    ind = cfg["type"]
    period = cfg.get("period")
    tf = cfg.get("timeframe")
    # unique col base including timeframe so multiple instances are distinct
    col_base = f"{ind}_{period}_{tf}" if period is not None else f"{ind}_{tf}"
    computed_cols.append(col_base)

    # choose the source (resample raw 1-min to indicator timeframe if needed)
    src = data if tf == chart_interval else resample_candles(df, tf)

    try:
        if ind == "SMA":
            src[col_base] = talib.SMA(src["close"], timeperiod=int(period))
        elif ind == "EMA":
            src[col_base] = talib.EMA(src["close"], timeperiod=int(period))
        elif ind == "WMA":
            src[col_base] = talib.WMA(src["close"], timeperiod=int(period))
        elif ind == "DEMA":
            src[col_base] = talib.DEMA(src["close"], timeperiod=int(period))
        elif ind == "TEMA":
            src[col_base] = talib.TEMA(src["close"], timeperiod=int(period))
        elif ind == "KAMA":
            src[col_base] = talib.KAMA(src["close"], timeperiod=int(period))
        elif ind == "RSI":
            src[col_base] = talib.RSI(src["close"], timeperiod=int(period))
        elif ind == "ATR":
            src[col_base] = talib.ATR(src["high"], src["low"], src["close"], timeperiod=int(period))
        elif ind == "VWAP":
            src[col_base] = (src["close"] * src["volume"]).cumsum() / src["volume"].cumsum()
        elif ind == "BBANDS":
            params = cfg["bb_params"]
            matype_map = {"SMA": 0, "EMA": 1, "WMA": 2, "DEMA": 3, "TEMA": 4}
            up, mid, low = talib.BBANDS(src["close"], timeperiod=int(period),
                                       nbdevup=params["nbdevup"], nbdevdn=params["nbdevdn"],
                                       matype=matype_map.get(params["matype"], 0))
            src[f"{col_base}_upper"] = up
            src[f"{col_base}_middle"] = mid
            src[f"{col_base}_lower"] = low
        elif ind == "MACD":
            params = cfg["macd_params"]
            macd_line, macd_signal, macd_hist = talib.MACD(src["close"],
                                                           fastperiod=int(params["fastperiod"]),
                                                           slowperiod=int(params["slowperiod"]),
                                                           signalperiod=int(params["signalperiod"]))
            src[col_base] = macd_line
            src[f"{col_base}_signal"] = macd_signal
            src[f"{col_base}_hist"] = macd_hist
        elif ind == "STOCH":
            slowk, slowd = talib.STOCH(src["high"], src["low"], src["close"],
                                       fastk_period=int(period), slowk_period=3, slowk_matype=0,
                                       slowd_period=3, slowd_matype=0)
            src[f"{col_base}_k"] = slowk
            src[f"{col_base}_d"] = slowd
        elif ind == "ADX":
            src[col_base] = talib.ADX(src["high"], src["low"], src["close"], timeperiod=int(period))
        elif ind == "CCI":
            src[col_base] = talib.CCI(src["high"], src["low"], src["close"], timeperiod=int(period))
        elif ind == "OBV":
            src[col_base] = talib.OBV(src["close"], src["volume"])
        elif ind == "MFI":
            src[col_base] = talib.MFI(src["high"], src["low"], src["close"], src["volume"], timeperiod=int(period))
        elif ind == "MOM":
            src[col_base] = talib.MOM(src["close"], timeperiod=int(period))
        elif ind == "ROC":
            src[col_base] = talib.ROC(src["close"], timeperiod=int(period))
        elif ind == "TRIX":
            src[col_base] = talib.TRIX(src["close"], timeperiod=int(period))
        elif ind == "AROON":
            aroon_dn, aroon_up = talib.AROON(src["high"], src["low"], timeperiod=int(period))
            src[f"{col_base}_down"] = aroon_dn
            src[f"{col_base}_up"] = aroon_up
        elif ind == "AROONOSC":
            src[col_base] = talib.AROONOSC(src["high"], src["low"], timeperiod=int(period))
        elif ind == "SAR":
            src[col_base] = talib.SAR(src["high"], src["low"], acceleration=0.02, maximum=0.2)
        else:
            # not recognized - skip
            continue
    except Exception as e:
        st.sidebar.warning(f"Failed computing {ind} ({period}) on TF {tf}: {e}")
        continue

    # align computed columns into main chart index
    to_copy = [c for c in src.columns if str(c).startswith(col_base)]
    for c in to_copy:
        try:
            data[c] = src[c].reindex(data.index, method="ffill")
        except Exception:
            data[c] = src[c]

st.markdown("## 🧩 Condition Builder")
with st.expander("Create conditions (LHS op RHS). RHS can be a number or an indicator. Default RHS = Price", expanded=True):
    # build dropdown list that contains only the indicators the user added + Price
    available_cols_cond = ["Price"] + computed_cols

    # number of conditions
    num_conditions = st.number_input("Number of Conditions", min_value=0, max_value=6, value=0, key="num_conditions_main")
    cond_specs = []
    connectors = []

    cols_for_buttons = st.columns([1, 1, 1, 1, 2])
    for i in range(int(num_conditions)):
        a, b, c, d, e = st.columns([3,1,3,1,2])
        lhs = a.selectbox(f"LHS {i+1}", options=available_cols_cond, index=0, key=f"lhs_{i}")
        op = b.selectbox(f"Op {i+1}", options=[">","<",">=","<=","=="], key=f"op_{i}")
        rhs_kind = c.selectbox(f"RHS kind {i+1}", options=["Indicator/Price","Number"], key=f"rhs_kind_{i}")
        if rhs_kind == "Number":
            rhs_val = c.number_input(f"Value {i+1}", value=0.0, key=f"rhs_num_{i}")
            rhs = float(rhs_val)
            rhs_is_num = True
        else:
            rhs = c.selectbox(f"RHS {i+1}", options=available_cols_cond, index=available_cols_cond.index("Price") if "Price" in available_cols_cond else 0, key=f"rhs_{i}")
            rhs_is_num = False

        cond_specs.append((lhs, op, rhs, rhs_is_num))
        if i < int(num_conditions)-1:
            connector = e.selectbox(f"Connector {i+1}", options=["AND","OR"], key=f"conn_{i}")
            connectors.append(connector)

    evaluate_btn = st.button("Evaluate Conditions")

# ---------- Evaluate Conditions (NaN-safe, bitwise-safe) ----------
final_mask = None
if int(num_conditions) > 0 and evaluate_btn:
    # build safe arrays for all available columns (price + computed)
    vars_safe = {}
    for col in available_cols_cond:
        if col in data.columns:
            arr = data[col].values.astype(float)
            arr = np.nan_to_num(arr, nan=0.0)  # replace NaN with 0 for safe comparisons
            vars_safe[col] = arr
        else:
            # column missing (shouldn't happen) -> zeros
            vars_safe[col] = np.zeros(len(data), dtype=float)

    masks = []
    for lhs, op, rhs, rhs_is_num in cond_specs:
        lhs_arr = vars_safe.get(lhs, np.zeros(len(data), dtype=float))
        rhs_arr = np.full(len(lhs_arr), float(rhs)) if rhs_is_num else vars_safe.get(rhs, np.zeros(len(data), dtype=float))

        if op == ">":
            mask = lhs_arr > rhs_arr
        elif op == "<":
            mask = lhs_arr < rhs_arr
        elif op == ">=":
            mask = lhs_arr >= rhs_arr
        elif op == "<=":
            mask = lhs_arr <= rhs_arr
        elif op == "==":
            mask = lhs_arr == rhs_arr
        else:
            mask = lhs_arr > rhs_arr

        mask = np.nan_to_num(mask, nan=0).astype(bool)
        masks.append(mask)

    # combine masks using connectors
    final_mask = masks[0]
    for idx, conn in enumerate(connectors):
        if conn == "AND":
            final_mask = final_mask & masks[idx + 1]
        else:
            final_mask = final_mask | masks[idx + 1]

    st.success(f"Condition evaluated: {final_mask.sum()} matching rows found. Highlighting applied to price chart.")

# ---------- Build Plot (price + oscillator subplots) ----------
# choose oscillator cfgs for separate subplots
oscillator_cfgs = [cfg for cfg in indicator_configs if cfg["type"] in oscillator_set]

rows = 1 + len(oscillator_cfgs)
if rows == 1:
    row_heights = [1.0]
else:
    row_heights = [0.62] + [0.38 / len(oscillator_cfgs)] * len(oscillator_cfgs)

subplot_titles = [f"Price ({chart_interval})"] + [
    f"{cfg['type']}_{cfg.get('period')} ({cfg.get('timeframe')})" for cfg in oscillator_cfgs
]

fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=row_heights, subplot_titles=subplot_titles)

# Price (row 1)
fig.add_trace(go.Candlestick(x=data.index, open=data["open"], high=data["high"], low=data["low"], close=data["close"], name="Price"), row=1, col=1)

# Overlay non-oscillator indicators on price
for cfg in indicator_configs:
    ind = cfg["type"]
    period = cfg.get("period")
    tf = cfg.get("timeframe")
    col_base = f"{ind}_{period}_{tf}" if period is not None else f"{ind}_{tf}"

    if ind in oscillator_set:
        continue  # oscillator placed separately

    # main line
    if col_base in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[col_base], name=col_base, line=dict(width=1.5)), row=1, col=1)

    # BBANDS parts
    for part_suffix, dash in [("_upper","dot"), ("_middle","dash"), ("_lower","dot")]:
        bbcol = f"{col_base}{part_suffix}"
        if bbcol in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[bbcol], name=bbcol, line=dict(width=1, dash=dash)), row=1, col=1)

    # MACD histogram (if user added macd but we put MACD as oscillator; keep overlays minimal)

# Highlight final_mask on price only
if final_mask is not None and final_mask.any():
    fig.add_trace(go.Scatter(x=data.index[final_mask], y=data["close"][final_mask], mode="markers",
                             marker=dict(size=9, color="magenta", symbol="circle"), name="Condition Met"), row=1, col=1)

# Add oscillator subplots
for idx, cfg in enumerate(oscillator_cfgs):
    row_num = idx + 2
    ind = cfg["type"]
    period = cfg.get("period")
    tf = cfg.get("timeframe")
    col_base = f"{ind}_{period}_{tf}" if period is not None else f"{ind}_{tf}"

    if ind == "MACD":
        # line, signal, hist
        if col_base in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[col_base], name=col_base, line=dict(width=1.2)), row=row_num, col=1)
        sig = f"{col_base}_signal"
        hist = f"{col_base}_hist"
        if sig in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[sig], name=sig, line=dict(width=1, color="red")), row=row_num, col=1)
        if hist in data.columns:
            fig.add_trace(go.Bar(x=data.index, y=data[hist], name=hist, marker=dict(color="rgba(120,120,120,0.5)")), row=row_num, col=1)

    elif ind == "STOCH":
        kcol = f"{col_base}_k"
        dcol = f"{col_base}_d"
        if kcol in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[kcol], name=kcol, line=dict(width=1.2)), row=row_num, col=1)
        if dcol in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[dcol], name=dcol, line=dict(width=1, color="red")), row=row_num, col=1)

    else:
        # single-line oscillators
        if col_base in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[col_base], name=col_base, line=dict(width=1.2)), row=row_num, col=1)

# finalize layout
fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False,
                  title=f"{symbol}  •  {period_str}  •  Base interval: {interval_label}",
                  height=900, width=1400)

st.plotly_chart(fig, use_container_width=True)
