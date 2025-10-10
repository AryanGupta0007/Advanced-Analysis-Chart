import streamlit as st
import pandas as pd
import numpy as np
import talib
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import json 
from data import fetch_historical, resample_candles, symbols_tuple
import plotly.express as px
import os
import traceback 

default_colors = px.colors.qualitative.Plotly

# ---------- App Layout ----------
st.set_page_config(layout="wide", page_title="Advanced Multi-Indicator Chart")
st.title("Advanced Multi-Indicator Chart")

st.session_state.setdefault("num_indicators", 1)
st.session_state.setdefault("num_conditions", 0)
st.session_state.setdefault("config_loaded", False)
st.session_state.setdefault("file_name", "nofile")
st.session_state.setdefault("indicator_configs", [])
st.session_state.setdefault("connectors", [])
st.session_state.setdefault("cond_specs", [])
# if st.session_state.num_indicators 
    # ---------- Rest of your code continues unchanged ----------
for i in range(st.session_state.num_indicators):
    st.session_state.setdefault(f"type_{i}", "SMA")
    st.session_state.setdefault(f"period_{i}", 20)
    st.session_state.setdefault(f"tf_{i}", "Same as Chart")
    st.session_state.setdefault(f"bb_period_{i}", 20)
    st.session_state.setdefault(f"bb_up_{i}", 2.0)
    st.session_state.setdefault(f"bb_dn_{i}", 2.0)
    st.session_state.setdefault(f"bb_ma_{i}", "SMA")
    st.session_state.setdefault(f"macd_fast_{i}", 12)
    st.session_state.setdefault(f"macd_slow_{i}", 26)
    st.session_state.setdefault(f"macd_signal_{i}", 9)
    st.session_state.setdefault(f"color_{i}", "#636EFA")
    st.session_state.setdefault(f"color_bbands_nbdevup_{i}", "#636EFA")
    st.session_state.setdefault(f"color_bbands_nbdevdn_{i}", "#636EFA")
    st.session_state.setdefault(f"color_bbands_ma_{i}", "#636EFA")
    st.session_state.setdefault(f"color_ma{i}", "#636EFA")
    st.session_state.setdefault(f"color_macd_signal{i}", "#636EFA")
    st.session_state.setdefault(f"color_macd_hist{i}", "#636EFA")
    st.session_state.setdefault(f"color_macd_line{i}", "#636EFA")
    
for i in range(6): 
    st.session_state.setdefault(f'lhs_{i}',  "Price") 
    st.session_state.setdefault(f'op_{i}',  ">") 
    st.session_state.setdefault(f"rhs_num_{i}",  3)  
    st.session_state.setdefault(f'rhs_{i}',  "Price")         
    st.session_state.setdefault(f"conn_{i}",  "AND")
st.session_state.setdefault("cond_specs", [])
uploaded_file = st.sidebar.file_uploader("Import Conditions", type="json")
indicator_configs =  st.session_state.get("indicator_configs")

if uploaded_file is not None:
    
    file_data = json.load(uploaded_file)
    if (uploaded_file.name != st.session_state.file_name):
        st.session_state["config_loaded"] = False  
        st.session_state["file_name"] = uploaded_file.name       
    if st.session_state["config_loaded"] is False:
        # Only process if new file or different file
        # print(file_data)
            # ---------- Immediately load the config into session_state ----------
        try:
            st.session_state.chart_interval = file_data["chart"]["interval"]
            st.session_state.chart_period = file_data["chart"]["period"]
            config_data = file_data.get("indicator_conf", [])
            conditions_data = file_data.get("conditions_conf", {})
            # ----------------- Load Indicators -----------------
            st.session_state.num_indicators = len(config_data)
            st.session_state.num_indicators_input = len(config_data)
            # print(config_data, conditions_data)
            # print(f'session_state: {st.session_state}')
            for i, cfg in enumerate(config_data):
                # print(f"{i} {cfg}")
                st.session_state[f"type_{i}"] = cfg["type"] 
                st.session_state[f"period_{i}"] = cfg.get("period", 20)
                st.session_state[f"tf_{i}"] = cfg.get("timeframe")

                st.session_state[f"color_{i}"] = cfg.get("color", "#636EFA")

                # BBANDS
                bb = cfg.get("bb_params", {})
                if cfg["bb_params"] is not None:
                    st.session_state[f"bb_period_{i}"] = bb.get("nbdevup", 20)
                    st.session_state[f"bb_up_{i}"] = bb.get("nbdevup", 2.0)
                    st.session_state[f"bb_dn_{i}"] = bb.get("nbdevdn", 2.0)
                    st.session_state[f"bb_ma_{i}"] = bb.get("matype", "SMA")
                    st.session_state[f"color_bbands_nbdevup_{i}"] = bb.get("color_up", "#636EFA")
                    st.session_state[f"color_bbands_nbdevdn_{i}"] = bb.get("color_down", "#636EFA")
                    st.session_state[f"color_bbands_ma_{i}"] = bb.get("color_ma", "#636EFA")

                # MACD
                macd = cfg.get("macd_params", {})
                if cfg["macd_params"] is not None:
                    st.session_state[f"macd_fast_{i}"] = macd.get("fastperiod", 12)
                    st.session_state[f"macd_slow_{i}"] = macd.get("slowperiod", 26)
                    st.session_state[f"macd_signal_{i}"] = macd.get("signalperiod", 9)
                    st.session_state[f"color_macd_fast{i}"] = macd.get("color_fast", "#636EFA")
                    st.session_state[f"color_macd_slow{i}"] = macd.get("color_slow", "#636EFA")
                    st.session_state[f"color_macd_signal{i}"] = macd.get("color_signal", "#636EFA")
                    st.session_state[f"color_macd_line{i}"] = macd.get("color_line", "#636EFA")
                    st.session_state[f"color_macd_hist{i}"] = macd.get("color_hist", "#636EFA")
                st.session_state.indicator_configs.append(cfg)
            # ----------------- Load Conditions -----------------
            conditions = conditions_data.get("conditions", [])
            connectors = conditions_data.get("connectors", [])
            st.session_state.cond_specs = conditions
            st.session_state.connectors = connectors
            
            st.session_state.num_conditions = len(conditions)
            st.session_state.num_conditions_input = len(conditions)
            for i, x in enumerate(conditions):
                print(f'condition: {x}')
                st.session_state[f'lhs_{i}'] = x[0]
                st.session_state[f'op_{i}'] = x[1]
                st.session_state[f"rhs_kind_{i}"] = x[3]
                if st.session_state[f"rhs_kind_{i}"]: 
                    st.session_state[f"rhs_num_{i}"] = x[2] 
                else:
                    st.session_state[f'rhs_{i}'] = x[2]
                    
            for i, x in enumerate(connectors):
                st.session_state[f"conn_{i}"] = x
                            
            st.session_state["config_loaded"] = True
            # print(f'session state after upload: {st.session_state}')
            st.success(f"✅ Loaded {len(config_data)} indicators and {len(conditions)} conditions from uploaded config!")
            st.rerun()

        except Exception as e:
            st.error(f"Failed to load config: {e}")
            st.error(traceback.format_exc())


# ---------- Sidebar: Chart Settings ----------
st.sidebar.header("Chart Settings")
symbol = st.sidebar.selectbox("Select Symbol", symbols_tuple)
st.session_state.setdefault("chart_period", "1d")
st.session_state.setdefault("chart_interval", "1 Minute")
period_options = ["1d", "2d", "5d", "15d"]
period_str = st.sidebar.selectbox("Select Period", period_options, index=period_options.index(st.session_state.get("chart_period", "1d")), key="chart_period")
days = int(period_str.rstrip("d"))

interval_map = {
    "1 Minute": "1T", "5 Minutes": "5T", "15 Minutes": "15T",
    "30 Minutes": "30T", "1 Hour": "1H", "1 Day": "1D"
}
interval_labels = list(interval_map.keys())
interval_label = st.sidebar.selectbox("Chart Interval (base)", interval_labels, index=interval_labels.index(st.session_state.get("chart_interval", "1d")), key="chart_interval")
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
def update_num_indicators():
    new_n = st.session_state['num_indicators_input']
    st.session_state.num_indicators = new_n 

num_indicators = st.sidebar.number_input("Number of Indicator Instances", min_value=1, max_value=20, value=st.session_state.get("num_indicators_input", 1),key="num_indicators_input", on_change=update_num_indicators)

def update_num_conditions():
    new_n = st.session_state['num_conditions_input']
    # print(new_n)
    st.session_state.num_conditions = new_n 

def update_rhs_kind():
    new_val = st.session_state[f'rhs_kind_input_{i}']
    if new_val == "Number":
        st.session_state[f'rhs_kind_{i}'] = True
    else:        
        st.session_state[f'rhs_kind_{i}'] = False
# num_indicators = st.sidebar.number_input("Number of Indicator Instances", min_value=1, max_value=20, value=st.session_state.get("num_indicators_input", 1),key="num_indicators_input", on_change=update_num_indicators)

indicator_configs = st.session_state.get("indicator_configs")
new_configs = []
for i in range(int(st.session_state.num_indicators)):
    exp_key = f"find_expander_{i}"
    # print(f"state while adding indicator settings {st.session_state}")
    
    with st.sidebar.expander(f"Indicator #{i+1} settings", expanded=(i < 3)):
        type_index = available_indicators.index(st.session_state.get(f"type_{i}", "SMA"))
        # print(type_index)
        ind = st.selectbox(
            "Type",
            available_indicators,
            index=type_index,
            key=f"type_{i}"
        )
        
        if ind in available_indicators:
            if st.session_state.get(f"type_{i}_index")  != available_indicators.index(ind):
                st.session_state[f"type_{i}_index"] = available_indicators.index(ind)
        period = None
        bb_params = None
        macd_params = None
        line_indicators = ["SMA","EMA","WMA","RSI","ATR","STOCH","CCI","MOM","TRIX","ROC","DEMA","TEMA","KAMA","MFI","ADX", "VWAP"]
        if ind in line_indicators:
            color = st.color_picker("Line Color", value=st.session_state.get(f"color_{i}"), key=f"color_{i}")
            period = st.number_input("Period", min_value=1, max_value=500, value=st.session_state.get(f"period_{i}", 20), key=f"period_{i}")
        if ind == "BBANDS":
            period = float(st.number_input("BBANDS Period", min_value=2.0, max_value=200.0, value=st.session_state.get(f"bb_period_{i}", 20), key=f"bb_period_{i}"))
            nbdevup = float(st.number_input("Upper Band Std Dev", min_value=0.1, max_value=5.0, value=st.session_state.get(f"bb_up_{i}", 2.0), step=0.1, key=f"bb_up_{i}"))
            color_up = st.color_picker("Upper Band Color", value=st.session_state.get(f"color_bbands_nbdevup_{i}"), key=f"color_bbands_nbdevup_{i}")
            
            nbdevdn = float(st.number_input("Lower Band Std Dev", min_value=0.1, max_value=5.0, value=st.session_state.get(f"bb_dn_{i}", 2.0), step=0.1, key=f"bb_dn_{i}"))
            color_down = st.color_picker("Lower Band Color", value=st.session_state.get(f"color_bbands_nbdevdn_{i}"), key=f"color_bbands_nbdevdn_{i}")
            
            ma_type = st.selectbox("MA Type", ["SMA","EMA","WMA","DEMA","TEMA"], key=f"bb_ma_{i}")
            color_ma = st.color_picker("Line Color", value=st.session_state.get(f"color_bbands_ma_{i}"), key=f"color_bbands_ma_{i}")
            
            bb_params = {"nbdevup": nbdevup, "nbdevdn": nbdevdn, "matype": ma_type, "color_up": color_up, "color_down": color_down, "color_ma": color_ma}
        
        if ind == "MACD":
            fast = st.number_input("MACD fast", min_value=1, max_value=200, value=st.session_state.get(f"macd_fast_{i}", 12), key=f"macd_fast_{i}")
            slow = st.number_input("MACD slow", min_value=1, max_value=400, value=st.session_state.get(f"macd_slow_{i}", 26), key=f"macd_slow_{i}")
            signal = st.number_input("MACD signal", min_value=1, max_value=200, value=st.session_state.get(f"macd_signal_{i}", 9), key=f"macd_signal_{i}")
            color_signal = st.color_picker("MACD signal color", value=st.session_state.get(f"color_macd_signal{i}"), key=f"color_macd_signal{i}")
            color_hist = st.color_picker("MACD hist color", value=st.session_state.get(f"color_macd_hist{i}"), key=f"color_macd_hist{i}")
            color_line = st.color_picker("MACD line color", value=st.session_state.get(f"color_macd_line{i}"), key=f"color_macd_line{i}")
            
            macd_params = {"fastperiod": fast, "slowperiod": slow, "signalperiod": signal, "color_hist": color_hist, "color_signal": color_signal, "color_line": color_line}
        options = ["Same as Chart"] + interval_labels
        tf_value = st.session_state.get(f"tf_{i}")
        for (k, v) in interval_map.items():
            if (v == tf_value):
                    option = k
        else:
            option = options[0]
        tf_index = options.index(option) 
        tf_label1 = st.selectbox(
            "Timeframe (resampled)",
            options,
            index=tf_index,
            key=f"tf_{i}"
        )
        tf_label = options[tf_index] 
        print(f'271 {tf_label} {tf_label1}')
        
        # Optional: store index separately if you need it later
        if tf_label1 in options:
            st.session_state[f"tf_{i}_index"] = options.index(tf_label1)
            tf = interval_map[tf_label] if tf_label != "Same as Chart" else chart_interval
            params_dict = {
                "type": ind,
                "period": period,
                "timeframe": tf,
                "bb_params": bb_params,
                "macd_params": macd_params
            }
            if ind in line_indicators:
                params_dict["color"] = color
            new_configs.append(params_dict)

if (new_configs != st.session_state.indicator_configs) and new_configs != []: 
    st.session_state.indicator_configs = new_configs            

indicator_configs = st.session_state.indicator_configs                
# print(f'292: {indicator_configs}, {new_configs}')
# ---------- Compute Indicators (only for the timeframes user requested) ----------
computed_cols = []  # exact column names we compute; used for condition dropdowns
# which indicators will get separate subplots
oscillator_set = {"RSI","ATR","MACD","ADX","STOCH","CCI","MFI","TRIX","ROC","OBV"}

for cfg in st.session_state.indicator_configs:
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
            src[col_base] = (src["close"] * src["volume"]).rolling(int(period)).sum() / src["volume"].rolling(int(period)).sum()
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

if not "num_conditions_input" in st.session_state: 
    st.session_state.setdefault("num_conditions_input", 0)
    st.session_state.setdefault("num_conditions", 0)

with st.expander("Create conditions (LHS op RHS). RHS can be a number or an indicator. Default RHS = Price", expanded=True):
    # build dropdown list that contains only the indicators the user added + Price
    available_cols_cond = ["Price"] + computed_cols

    # number of conditions
    num_conditions_input = st.number_input("Number of Conditions", min_value=0, max_value=6, value=st.session_state.get('num_conditions_input'), key="num_conditions_input", on_change=update_num_conditions)
    
    # cols_for_buttons = st.columns([1, 1, 1, 1, 2])
    new_cond_specs = []
    new_connectors = []
    for i in range(int(st.session_state.num_conditions)):
        a, b, c, d, e = st.columns([3,1,3,1,2])
        op_options = [">","<",">=","<=","=="]
        connector_options = ["AND","OR"]
        rhs_kind_options = ["Indicator/Number", "Number"]
        rhs_kind = f'rhs_kind_{i}'
        if rhs_kind: 
            r_kind = 0
        else: 
            r_kind = 1
        print(f'lhs, rhs: {st.session_state[f"lhs_{i}"]}"], {st.session_state[f"rhs_{i}"]}"]')
        lhs = a.selectbox(f"LHS {i+1}", options=available_cols_cond, index=available_cols_cond.index(st.session_state.get(f'lhs_{i}',available_cols_cond[0])), key=f"lhs_{i}")
        op = b.selectbox(f"Op {i+1}", options=op_options ,index=op_options.index(st.session_state.get(f'op_{i}', op_options[0])),  key=f"op_{i}")
        rhs_kind = c.selectbox(f"RHS kind {i+1}", options=rhs_kind_options, index=r_kind, key=f"rhs_kind_input_{i}", on_change=update_rhs_kind)
        if rhs_kind == "Number":
            rhs_val = c.number_input(f"Value {i+1}", value=st.session_state.get(f"rhs_num_{i}"), key=f"rhs_num_{i}")
            rhs = float(rhs_val)
            rhs_is_num = True
        else:
            rhs = c.selectbox(f"RHS {i+1}", options=available_cols_cond, index=available_cols_cond.index(st.session_state.get(f'rhs_{i}', available_cols_cond[0])), key=f"rhs_{i}")
            rhs_is_num = False

        new_cond_specs.append((lhs, op, rhs, rhs_is_num))
        if i < int(st.session_state.num_conditions)-1:
            connector = e.selectbox(f"Connector {i+1}", options=connector_options, index=connector_options.index(st.session_state.get(f"conn_{i}", connector_options[0])), key=f"conn_{i}")
            new_connectors.append(connector)

    evaluate_btn = st.button("Evaluate Conditions")
if st.session_state.cond_specs != new_cond_specs:
    st.session_state.cond_specs = new_cond_specs

if st.session_state.connectors != new_connectors:
    st.session_state.connectors = new_connectors
cond_specs = st.session_state.cond_specs
connectors = st.session_state.connectors = new_connectors
c_json = {
    "indicator_conf": st.session_state.indicator_configs,
    "conditions_conf": {
            "conditions": cond_specs,
            "connectors": connectors
                },
    "chart": {
        "interval": st.session_state.chart_interval,
        "period": st.session_state.chart_period
    } 
}
config_json = json.dumps(c_json, indent=4)
config_file_name = st.sidebar.text_input("Enter the config file name: ", value="config")
st.sidebar.download_button(
    label='Export Config',
    data=config_json,
    file_name=f"{config_file_name}.json", 
    mime="application/json"
)

# ---------- Evaluate Conditions (NaN-safe, bitwise-safe) ----------
final_mask = None
if int(st.session_state.num_conditions) > 0 and evaluate_btn:
    # build safe arrays for all available columns (price + computed)
    vars_safe = {}
    print(f'available_cols_cond: {available_cols_cond}')
    for col in available_cols_cond:
        if col in data.columns:
            arr = data[col].values.astype(float)
            arr = np.nan_to_num(arr, nan=0.0)  # replace NaN with 0 for safe comparisons
            vars_safe[col] = arr
        else:
            # column missing (shouldn't happen) -> zeros
            vars_safe[col] = np.zeros(len(data), dtype=float)

    masks = []
    print(f'470: {st.session_state.cond_specs}')
    for lhs, op, rhs, rhs_is_num in st.session_state.cond_specs:
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
indicator_configs = st.session_state["indicator_configs"]
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
    color = cfg.get("color")
        
    # main line
    if col_base in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[col_base], name=col_base, line=dict(width=1.5, color=color)), row=1, col=1)

    # BBANDS parts
    for part_suffix, dash in [("_upper","dot"), ("_middle","dash"), ("_lower","dot")]:
        bbcol = f"{col_base}{part_suffix}"
        if bbcol in data.columns:
            if part_suffix == "_upper":
                fig.add_trace(go.Scatter(x=data.index, y=data[bbcol], name=bbcol, line=dict(width=1, dash=dash, color=cfg["bb_params"]["color_up"])), row=1, col=1)
            if part_suffix == "_middle":
                fig.add_trace(go.Scatter(x=data.index, y=data[bbcol], name=bbcol, line=dict(width=1, dash=dash, color=cfg["bb_params"]["color_ma"])), row=1, col=1)
            if part_suffix == "_lower":
                fig.add_trace(go.Scatter(x=data.index, y=data[bbcol], name=bbcol, line=dict(width=1, dash=dash, color=cfg["bb_params"]["color_down"])), row=1, col=1)
            
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
    color = cfg.get("color")
    if ind == "MACD":
        # line, signal, hist
        if col_base in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[col_base], name=col_base, line=dict(width=1.2, color=cfg["macd_params"]["color_line"])), row=row_num, col=1)
        sig = f"{col_base}_signal"
        hist = f"{col_base}_hist"
        if sig in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[sig], name=sig, line=dict(width=1, color=cfg["macd_params"]["color_signal"])), row=row_num, col=1)
        if hist in data.columns:
            fig.add_trace(go.Bar(x=data.index, y=data[hist], name=hist, marker=dict(color=cfg["macd_params"]["color_hist"])), row=row_num, col=1)

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
            fig.add_trace(go.Scatter(x=data.index, y=data[col_base], name=col_base, line=dict(width=1.2, color=color)), row=row_num, col=1)

# finalize layout
fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False,
                  title=f"{symbol}  •  {period_str}  •  Base interval: {interval_label}",
                  height=900, width=1400)

st.plotly_chart(fig, use_container_width=True)
