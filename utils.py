import numpy as np
import pandas as pd 
import streamlit as st
import talib
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go


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
    st.session_state.setdefault("computed_cols", ohlc_cols)
    st.session_state.setdefault("num_indicators", 1)
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
    for i in range(st.session_state.num_indicators):
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
    return st.session_state
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
