import numpy as np
import pandas as pd 
import streamlit as st
import talib
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import json
import traceback

ohlc_cols = ['open', 'high', 'low', 'close']
line_indicators = ["SMA","EMA","WMA","RSI","ATR","STOCH","CCI","MOM","TRIX","ROC","DEMA","TEMA","KAMA","MFI","ADX", "VWAP"]
moving_averages = ["SMA","EMA","WMA","DEMA","TEMA","KAMA"]

op_options = [">","<",">=","<=","=="]
connector_options = ["AND","OR"]
rhs_kind_options = ["Indicator", "Number"]
oscillator_set = {"RSI","ATR","MACD","ADX","STOCH","CCI","MFI","TRIX","ROC","OBV"}
                
strategy_rules = {
    'LONG': {
        'ENTRY': {},
        'EXIT': {}
    },
    'SHORT':{
        'ENTRY': {},
        'EXIT': {}
    } 
}

def set_defaults():
    st.session_state.setdefault("data_store", {})    
    st.session_state.setdefault("strats", {})
    st.session_state.setdefault("chart_period", "1d")
    st.session_state.setdefault("chart_interval", "1 Minute")
    st.session_state.setdefault("num_indicators", 1)
    st.session_state.setdefault("computed_cols", ohlc_cols)
    st.session_state.setdefault("num_conditions", 0)
    st.session_state.setdefault("config_loaded", False)
    st.session_state.setdefault("file_name", "nofile")
    st.session_state.setdefault("indicator_configs", [])
    st.session_state.setdefault("connectors", [])
    st.session_state.setdefault("cond_specs", [])
    


    for x in strategy_rules.keys():
        for y in strategy_rules[x].keys():   
            x = x.lower()
            y = y.lower()     
            st.session_state.setdefault(f'num_rules_input_{x}_{y}', 0)
            st.session_state.setdefault(f'num_rules_{x}_{y}', 0)
            for i in range(int(st.session_state[f'num_rules_input_{x}_{y}'])):
                st.session_state.setdefault(f'rhs_kind_input_{i}_{x}_{y}', rhs_kind_options[0])
                st.session_state.setdefault(f'rhs_kind_{i}_{x}_{y}', True)
                st.session_state.setdefault(f'lhs_{i}_{x}_{y}', "close")
                st.session_state.setdefault(f'rhs_{i}_{x}_{y}', "close")
                st.session_state.setdefault(f"rhs_num_{i}_{x}_{y}", 0)
                st.session_state.setdefault(f'op_{i}_{x}_{y}', op_options[0])
                st.session_state.setdefault(f"conn_{i}_{x}_{y}", connector_options[0])        
                        

    # if st.session_state.num_indicators 
        # ---------- Rest of your code continues unchanged ----------
    for i in range(st.session_state["num_indicators"]):
        st.session_state.setdefault(f"type_{i}", "SMA")
        st.session_state.setdefault(f"ref_{i}", "close")
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
        st.session_state.setdefault(f'lhs_{i}',  "close") 
        st.session_state.setdefault(f'op_{i}',  ">") 
        st.session_state.setdefault(f"rhs_num_{i}",  3)  
        st.session_state.setdefault(f'rhs_{i}',  "close")         
        st.session_state.setdefault(f"conn_{i}",  "AND")
    st.session_state.setdefault("cond_specs", [])
    
    
def resample_candles(df, interval="5T"):
    """
    Resample intraday 1-minute data to a larger timeframe.
    
    Parameters:
        df (pd.DataFrame): 1-minute OHLCV data
        interval (str): Pandas offset alias ('5T' = 5 min, '15T', '30T', '1H', '1D')
    
    Returns:
        pd.DataFrame: resampled OHLCV data
    """
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    df_resampled = df.resample(interval).agg(ohlc_dict).dropna()
    return df_resampled

def compute_indicators(data, indicator_config, moving_averages, chart_interval, df):
    for i, cfg in enumerate(indicator_config):
        print(f"405. {i} config: ", cfg)
        ind = cfg["type"]
        period = cfg.get("period")
        tf = cfg.get("timeframe")
        
        # unique col base including timeframe so multiple instances are distinct
        col_base = f"{ind}_{period}_{tf}" if period is not None else f"{ind}_{tf}"
        # print(f"323: {computed_cols}")
        # print(f"308: {chart_interval}, {tf}")
        # # choose the source (resample raw 1-min to indicator timeframe if needed)
        src = data if tf == chart_interval else resample_candles(df, tf)
        if ind in moving_averages:
            ref_val = st.session_state.get(f"ref_{i}", "close")
            col_base = f"{ind}_{ref_val}_{period}_{tf}"
        to_copy = []
        print('419 ', col_base)
        try:
            if ind == "SMA":
                src[col_base] = talib.SMA(src[ref_val], timeperiod=int(period))
            elif ind == "EMA":
                src[col_base] = talib.EMA(src[ref_val], timeperiod=int(period))
            elif ind == "WMA":
                src[col_base] = talib.WMA(src[ref_val], timeperiod=int(period))
            elif ind == "DEMA":
                src[col_base] = talib.DEMA(src[ref_val], timeperiod=int(period))
            elif ind == "TEMA":
                src[col_base] = talib.TEMA(src[ref_val], timeperiod=int(period))
            elif ind == "KAMA":
                src[col_base] = talib.KAMA(src[ref_val], timeperiod=int(period))
            elif ind == "RSI":
                src[col_base] = talib.RSI(src[ref_val], timeperiod=int(period))
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
                src[f"{col_base}"] = mid
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
                src[col_base] = talib.MOM(src[ref_val], timeperiod=int(period))
            elif ind == "ROC":
                src[col_base] = talib.ROC(src[ref_val], timeperiod=int(period))
            elif ind == "TRIX":
                src[col_base] = talib.TRIX(src[ref_val], timeperiod=int(period))
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
            continue
        else:
            to_copy.append(col_base)
            
            
        for c in to_copy:
            print("496 to_copy: ", to_copy)
            try:
                data[c] = src[c].reindex(data.index, method="ffill")
            except Exception:
                data[c] = src[c]
    all_cols = list(data.columns)
    return all_cols, data

def evaluate_conditions(data, computed_cols, cond_specs, connectors):
    final_mask = None
    vars_safe = {}
    # print(f'available_cols_cond: {available_cols_cond}')
    for col in computed_cols:
        print("561.",  list(data.columns))
        if col in data.columns:
            arr = data[col].values.astype(float)
            arr = np.nan_to_num(arr, nan=0.0)  # replace NaN with 0 for safe comparisons
            vars_safe[col] = arr
        else:
            print('zeros')
            # column missing (shouldn't happen) -> zeros
            vars_safe[col] = np.zeros(len(data), dtype=float)

    masks = []
    print(f'470: {cond_specs}')
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
        # else:
        #     mask = lhs_arr > rhs_arr

        mask = np.nan_to_num(mask, nan=0).astype(bool)
        masks.append(mask)

    # combine masks using connectors
    final_mask = masks[0]
    for idx, conn in enumerate(connectors):
        if conn == "AND":
            final_mask = final_mask & masks[idx + 1]
        else:
            final_mask = final_mask | masks[idx + 1]
    return final_mask

def plot_charts_and_indicators(data, chart_interval, indicator_configs, final_mask, **kwargs):
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
    for cfg in st.session_state.indicator_configs:
        ind = cfg["type"]
        period = cfg.get("period")
        tf = cfg.get("timeframe")
        ref_val = cfg.get("ref_val", "close")
        col_base = f"{ind}_{period}_{tf}" if period is not None else f"{ind}_{tf}"
        if ind in moving_averages:
            col_base = f"{ind}_{ref_val}_{period}_{tf}"
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
                    title=f"{kwargs.get('symbol')}  •  {kwargs.get('period_str')}  •  Base interval: {kwargs.get('interval_label')}",
                    height=900, width=1400)
    return fig

def load_file(uploaded_file):
    file_data = json.load(uploaded_file)
    if (uploaded_file.name != st.session_state.file_name):
        st.session_state["config_loaded"] = False  
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["indicator_configs"] = []
        st.session_state["computed_cols"] = []            
    if st.session_state["config_loaded"] is False:
        # Only process if new file or different file
        # print(file_data)
            # ---------- Immediately load the config into session_state ----------
        try:
            st.session_state.brokerage = file_data["broker"]["brokerage"]
            st.session_state.slippage = file_data["broker"]["slippage"]
            st.session_state.cash = file_data["broker"]["cash"]
            new_rules = file_data.get("strategy_rules", {})
            st.session_state.computed_cols = file_data.get("computed_cols")
            if new_rules != {}:
                st.session_state.strategy_rules = new_rules
                for x in strategy_rules.keys():
                    for y in strategy_rules[x].keys():
                        
                        x = x.upper()
                        y = y.upper()   
                        if (st.session_state.strategy_rules[x][y].get("conditions", None) is None or st.session_state.strategy_rules[x][y]["conditions"] == []):
                            continue
                        conditions = st.session_state.strategy_rules[x][y]["conditions"]
                        connectors = st.session_state.strategy_rules[x][y]["connectors"]
                        x = x.lower()
                        y = y.lower()     
                    
                        st.session_state[f'num_rules_input_{x}_{y}'] =  len(conditions)
                        st.session_state[f'num_rules_{x}_{y}'] = len(conditions)
                        
                        for i, condition in enumerate(conditions):
                            rhs = condition[2]
                            lhs = condition[0]
                            op = condition[1]
                            rhs_kind = condition[3]
                            st.session_state[f'rhs_kind_{i}_{x}_{y}'] = rhs_kind
                            st.session_state[f'lhs_{i}_{x}_{y}']  = lhs
                            if rhs_kind:
                                r_kind = 1
                            else:
                                r_kind = 0
                            rhs_kind = rhs_kind_options[r_kind]
                            if rhs_kind == "Indicator":
                                st.session_state[f'rhs_{i}_{x}_{y}'] = rhs
                            else:
                                st.session_state[f"rhs_num_{i}_{x}_{y}"] = float(rhs) 
        
                            st.session_state[f'rhs_kind_input_{i}_{x}_{y}']  = rhs_kind_options[r_kind]
                                
                            
                            st.session_state[f'op_{i}_{x}_{y}'] = op 
                            if (i > 0):
                                conn = connectors[i-1]
                                st.session_state[f"conn_{i}_{x}_{y}"] = conn

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
                st.session_state[f"ref_{i}"] = cfg.get("ref_val")
                st.session_state[f"color_{i}"] = cfg.get("color", "#636EFA")

                # BBANDS
                bb = cfg.get("bb_params", {})
                if cfg["bb_params"] is not None:
                    st.session_state[f"bb_period_{i}"] = st.session_state.get(f"period_{i}")
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

def plot_charts_and_indicators_with_entry_exits(data, chart_interval, indicator_configs, final_mask, entries, exits, **kwargs):
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
    for cfg in st.session_state.indicator_configs:
        ind = cfg["type"]
        period = cfg.get("period")
        tf = cfg.get("timeframe")
        ref_val = cfg.get("ref_val", "close")
        col_base = f"{ind}_{period}_{tf}" if period is not None else f"{ind}_{tf}"
        if ind in moving_averages:
            col_base = f"{ind}_{ref_val}_{period}_{tf}"
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

    
    # --- Entries ---
    if entries:
        entry_times = [pd.to_datetime(t) for t, _, _ in entries]
        entry_prices = [p for _, p, _ in entries]
        entry_labels = [lbl for _, _, lbl in entries]
        entry_colors = ['blue' if lbl=='LONG' else 'purple' for lbl in entry_labels]

        # 1) Marker only (always visible)
        fig.add_trace(go.Scatter(
            x=entry_times, y=entry_prices, mode='markers',
            marker=dict(symbol='triangle-up', size=10, color=entry_colors),
            name='Entry Marker'
        ), row=1, col=1)

        # 2) Marker + text (hidden by default)
        fig.add_trace(go.Scatter(
            x=entry_times, y=entry_prices, mode='markers+text',
            marker=dict(symbol='triangle-up', size=10, color=entry_colors),
            text=entry_labels, textposition='top center',
            hoverinfo='text', visible=False,  # hidden initially
            name='Entry Label'
        ), row=1, col=1)

    # --- Exits ---
    if exits:
        exit_times = [pd.to_datetime(t) for t, _, _ in exits]
        exit_prices = [p for _, p, _ in exits]
        exit_labels = [lbl for _, _, lbl in exits]
        exit_colors = ['orange' if lbl=='LONG' else 'red' for lbl in exit_labels]

        # 1) Marker only
        fig.add_trace(go.Scatter(
            x=exit_times, y=exit_prices, mode='markers',
            marker=dict(symbol='triangle-down', size=10, color=exit_colors),
            name='Exit Marker'
        ), row=1, col=1)

        # 2) Marker + text (hidden initially)
        fig.add_trace(go.Scatter(
            x=exit_times, y=exit_prices, mode='markers+text',
            marker=dict(symbol='triangle-down', size=10, color=exit_colors),
            text=exit_labels, textposition='bottom center',
            hoverinfo='text', visible=False,
            name='Exit Label'
        ), row=1, col=1)

    # --- Buttons to toggle labels ---
    
    # finalize layout
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False,
                    title=f"{kwargs.get('symbol')}  •  {kwargs.get('period_str')}  •  Base interval: {kwargs.get('interval_label')}",
                    height=900, width=1400)
    return fig



    