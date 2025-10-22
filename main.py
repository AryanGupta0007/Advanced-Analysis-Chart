import streamlit as st
import pandas as pd
import numpy as np
from utils import *
from data_breeze import fetch_historical, symbols_tuple
import traceback  
        
# ---------- App Layout ----------
st.set_page_config(layout="wide", page_title="Advanced Multi-Indicator Chart")
st.title("Advanced Multi-Indicator Chart")
set_defaults()

available_cols_cond = ["close"] if len(st.session_state.computed_cols) != 0 else st.session_state.computed_cols 
uploaded_file = st.sidebar.file_uploader("Import Conditions", type="json")
indicator_configs =  st.session_state.get("indicator_configs")

if uploaded_file is not None:
    load_file(uploaded_file=uploaded_file)    
    

# ---------- Sidebar: Chart Settings ----------
st.sidebar.header("Chart Settings")

symbol = st.sidebar.selectbox("Select Symbol", symbols_tuple, index=symbols_tuple.index("ADANIENT"))
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
# data["Price"] = data["close"]
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
    print(f"215: {indicator_configs} {st.session_state[f"type_{i}"]}")
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
        if ind in line_indicators:
            color = st.color_picker("Line Color", value=st.session_state.get(f"color_{i}"), key=f"color_{i}")
            period = st.number_input("Period", min_value=1, max_value=500, value=st.session_state.get(f"period_{i}", 20), key=f"period_{i}")
        ref_options = ['open', 'close', 'low', 'high']
        if ind in moving_averages:
            ref = st.selectbox("Ref Type", ref_options, index=ref_options.index(st.session_state.get(f"ref_{i}", "close")), key=f"ref_{i}")
            
        if ind == "BBANDS":
            period = float(st.number_input("BBANDS Period", min_value=2.0, max_value=200.0, value=float(st.session_state.get(f"bb_period_{i}", 20.0)), key=f"bb_period_{i}"))
            nbdevup = float(st.number_input("Upper Band Std Dev", min_value=0.1, max_value=5.0, value=float(st.session_state.get(f"bb_up_{i}", 2.0)), step=0.1, key=f"bb_up_{i}"))
            color_up = st.color_picker("Upper Band Color", value=st.session_state.get(f"color_bbands_nbdevup_{i}"), key=f"color_bbands_nbdevup_{i}")
            
            nbdevdn = float(st.number_input("Lower Band Std Dev", min_value=0.1, max_value=5.0, value=float(st.session_state.get(f"bb_dn_{i}", 2.0)), step=0.1, key=f"bb_dn_{i}"))
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
        if tf_value != "Same as Chart":
            if tf_value[-1] == "T" or tf_value[-1] == "H" or tf_value[-1] == "D":
                for (k, v) in interval_map.items():
                    if (v == tf_value):
                            tf_value = k
              
            print(f"gooten tf_value {interval_map[tf_value]}")
            # else:
            #     option = interval_map[tf_value]
            tf_index = options.index(tf_value)
            print(f'269. tf_index: {tf_index}') 
        
        else:
            tf_index = 0    
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
        elif tf_label in options:
            st.session_state[f"tf_{i}_index"] = options.index(tf_label)
             
        tf = interval_map[tf_label] if tf_label != "Same as Chart" else chart_interval
        params_dict = {
            "type": ind,
            "period": period,
            "timeframe": tf,
            "bb_params": bb_params,
            "macd_params": macd_params,
            "ref_val": st.session_state.get(f'ref_{i}', 'close')
        }
        if ind in line_indicators:
            params_dict["color"] = color
        print(f"params_dict: {params_dict}")
        new_configs.append(params_dict)

if (new_configs != st.session_state.indicator_configs) and new_configs != []: 
    st.session_state.indicator_configs = new_configs            
indicator_configs = st.session_state.indicator_configs                
print(f"303: {indicator_configs}")
# print(f'292: {indicator_configs}, {new_configs}')
# ---------- Compute Indicators (only for the timeframes user requested) ----------
computed_cols = st.session_state.computed_cols  # exact column names we compute; used for condition dropdowns
# which indicators will get separate subplots

all_cols, data = compute_indicators(data, indicator_config=st.session_state.indicator_configs, moving_averages=moving_averages, chart_interval=chart_interval, df=df)
for col in all_cols:
    if col not in st.session_state.computed_cols:
        st.session_state.computed_cols.append(col)
print(f"408 {st.session_state.computed_cols}")
st.markdown("## 🧩 Condition Builder")

def perform_suitable_arithematic(lhs, rhs, operator):
    if operator == "+":
        return rhs + lhs
    elif operator == "-":
        return lhs - rhs 
    elif operator == "*":
        return rhs * lhs 
    elif operator == "/":
        return lhs / rhs

def perform_operation(data, name, lhs, operator, operation_kind, rhs, **kwargs):
    col = data[lhs]
    result = None
    print("218", name, lhs, operator, operation_kind, rhs)
    if (operation_kind == 'Statistical'):
        
        if rhs == 'mean':
            result = col.mean()
            print("233:", result)
        elif rhs == 'std':
            result = col.std()
            
    elif (operation_kind == 'Arithematic'):
        rhs_kind = kwargs.get('rhs_kind')
            
        if rhs_kind == "Number":
            result = perform_suitable_arithematic(col, rhs=rhs, operator=operator)
        
        elif rhs_kind == "Column":
            rhs = data[rhs]
            result = perform_suitable_arithematic(col, rhs=rhs, operator=operator)
        

    data[name] = result
    print("237", data[name].head())
    return data 
            
with st.expander("## Create a custom column", expanded=True):
    # available_cols_cond = st.session_state.get("computed_cols", list(data.columns))
    num_operations = st.number_input("Enter the number of operands for the operation", value=st.session_state.get('num_operations'), key='num_operations')
    rhs_kind_options = ['Number', 'Column']
    operation_kind_options = ['Arithematic', 'Statistical']
    operators_options = ['+', '-', '*', '/']
    mathematical_operations_options = ['mean', 'std']
    operator_config = st.session_state.get('custom_cols_config', [])
    new_custom_cols_config = [] 
    
    for i in range(st.session_state.get('num_operations', 0)):
        st.markdown(f"## Custom Column: {i+1}")
        row_key = f'custom_col_{i}'
        row1 = st.columns(3)
        
        # a, b, c, d, e, f = st.columns([5, 3, 5, 5, 5,2])
        with row1[0]:
            col_name = st.text_input("Enter a unique custom column name ", value=st.session_state.get(f'name_{row_key}', "unique column name"), key=f'name_{row_key}')
        lhs_val = st.session_state.get(f'lhs_{row_key}', available_cols_cond[0])
        with row1[1]:
            lhs = st.selectbox(f'LHS {i+1}', options=st.session_state['computed_cols'], index=st.session_state['computed_cols'].index(lhs_val), key=f'lhs_{row_key}') 
        with row1[2]:
            operation_kind = st.selectbox(f'Operator kind {i+1}', options=operation_kind_options, key=f'operation_kind_{row_key}', index=operation_kind_options.index(st.session_state.get(f'operation_kind_{row_key}', operation_kind_options[0]))) 
        rhs_kind = None
        operator = None
        row2 = st.columns(3)
        if operation_kind == 'Arithematic':
            with row2[0]:
                operator = st.selectbox(f'Select Operator {i+1}', options=operators_options, index=operators_options.index(st.session_state.get(f'operation_{row_key}', operators_options[0])), key=f'operation_{row_key}')    
            with row2[1]:
                rhs_kind = st.selectbox(f'Select kind of RHS {i+1}', options=rhs_kind_options, index=rhs_kind_options.index(st.session_state.get(f'rhs_kind_{row_key}', rhs_kind_options[0])), key=f'rhs_kind_{row_key}')
            
            if rhs_kind == 'Number':
                with row2[2]:
                    rhs_val = st.number_input('Enter the value', key=f'rhs_num_{row_key}', value=st.session_state.get(f'rhs_num_{row_key}', 0))
    
            elif rhs_kind == 'Column':
                with row2[2]:
                    rhs_val = st.selectbox('Select the column', options=st.session_state['computed_cols'], key=f'rhs_col_{row_key}', index=st.session_state['computed_cols'].index(st.session_state.get(f'rhs_col_{row_key}', available_cols_cond[0])))
            # data = perform_operation(data, col_name, lhs_val, operator, "arithematic", rhs_val=rhs_val, rhs_kind=rhs_kind)        
            # print(data[col_name])
        elif operation_kind == 'Statistical':
            with row2[0]:
                rhs_val = st.selectbox(f'Select Operator {i+1}', options=mathematical_operations_options, index=mathematical_operations_options.index(st.session_state.get(f'mathematical_operation_{row_key}', mathematical_operations_options[0])), key=f'mathematical_operation_{row_key}') 
            # data = perform_operation(data, col_name, lhs_val, operator, "mathematical")
        params = {
            'name': col_name, 
            'lhs': lhs,
            'operator': operator,
            'rhs': rhs_val,
            'rhs_kind': rhs_kind,
            'operator_kind': operation_kind
        }
        new_custom_cols_config.append(params)        
    operate_btn = st.button('operate')

    if operate_btn:
        print('equating', data.columns)
        st.session_state['custom_cols_config'] = new_custom_cols_config

for config in st.session_state.get('custom_cols_config', []):
    data = perform_operation(
        data,
        name=config['name'],
        lhs=config['lhs'],
        operator=config['operator'],
        rhs=config['rhs'],
        operation_kind=config['operator_kind'],
        rhs_kind=config['rhs_kind']
         )
    # st.write("288 ", data[config["name"]].head())
print('291', list(data.columns))
st.session_state.computed_cols = list(data.columns)

if not "num_conditions_input" in st.session_state: 
    st.session_state.setdefault("num_conditions_input", 0)
    st.session_state.setdefault("num_conditions", 0)
st.session_state.computed_cols = list(data.columns)
available_cols_cond = st.session_state.get("computed_cols", []) 
# available_cols_cond.append( st.session_state.get("computed_cols"))
with st.expander("Create conditions (LHS op RHS). RHS can be a number or an indicator. Default RHS = Price", expanded=True):
    # build dropdown list that contains only the indicators the user added + Price
    # number of conditions
    num_conditions_input = st.number_input("Number of Conditions", min_value=0, max_value=6, value=st.session_state.get('num_conditions_input'), key="num_conditions_input", on_change=update_num_conditions)
    
    # cols_for_buttons = st.columns([1, 1, 1, 1, 2])
    new_cond_specs = []
    new_connectors = []
    for i in range(int(st.session_state.num_conditions)):
        a, b, c, d, e = st.columns([3,1,3,1,2])
        op_options = [">","<",">=","<=","=="]
        connector_options = ["AND","OR"]
        rhs_kind_options = ["Indicator", "Number"]
        rhs_kind = f'rhs_kind_{i}'
        rhs = st.session_state.get(f'rhs_{i}', available_cols_cond[0])
        lhs = st.session_state.get(f'lhs_{i}', available_cols_cond[0]) 
        if rhs not in available_cols_cond:
            rhs = available_cols_cond[0]
        elif lhs not in available_cols_cond:
            lhs = available_cols_cond[0]
        if rhs_kind: 
            r_kind = 0
        else: 
            r_kind = 1
        # print(f'lhs, rhs: {st.session_state[f"lhs_{i}"]}"], {st.session_state[f"rhs_{i}"]}"]')
        lhs = a.selectbox(f"LHS {i+1}", options=available_cols_cond, index=available_cols_cond.index(lhs), key=f"lhs_{i}")
        op = b.selectbox(f"Op {i+1}", options=op_options ,index=op_options.index(st.session_state.get(f'op_{i}', op_options[0])),  key=f"op_{i}")
        rhs_kind = c.selectbox(f"RHS kind {i+1}", options=rhs_kind_options, index=r_kind, key=f"rhs_kind_input_{i}", on_change=update_rhs_kind)
        if rhs_kind == "Number":
            rhs_val = c.number_input(f"Value {i+1}", value=st.session_state.get(f"rhs_num_{i}", 0), key=f"rhs_num_{i}")
            rhs = float(rhs_val)
            rhs_is_num = True
        else:
            rhs = c.selectbox(f"RHS {i+1}", options=available_cols_cond, index=available_cols_cond.index(rhs), key=f"rhs_{i}")
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

# ---------- Evaluate Conditions (NaN-safe, bitwise-safe) ----------
st.session_state.computed_cols = list(data.columns)

final_mask = None
if int(st.session_state.num_conditions) > 0 and evaluate_btn:
    final_mask = evaluate_conditions(data=data, computed_cols=st.session_state.computed_cols, cond_specs=st.session_state.cond_specs, connectors=st.session_state.connectors)
    # build safe arrays for all available columns (price + computed)
    st.success(f"Condition evaluated: {final_mask.sum()} matching rows found. Highlighting applied to price chart.")

# ---------- Build Plot (price + oscillator subplots) ----------
# choose oscillator cfgs for separate subplots
indicator_configs = st.session_state["indicator_configs"]
custom_cols_config = st.session_state["custom_cols_config"]
fig = plot_charts_indicators_and_cols(data=data, chart_interval=chart_interval, indicator_configs=st.session_state.indicator_configs, custom_cols_config=custom_cols_config, final_mask=final_mask, symbol=symbol, period_str=period_str, interval_label=interval_label)
st.plotly_chart(fig, use_container_width=True)

def update_num_rules(x, y):
    st.session_state[f'num_rules_{x}_{y}'] = st.session_state[f'num_rules_input_{x}_{y}']

def update_rhs_kind_for_rule(i, x, y):
    if (st.session_state[f"rhs_kind_input_{i}_{x}_{y}"] == "Indicator"):
        st.session_state[f'rhs_kind_{i}_{x}_{y}'] = True
        return    
    st.session_state[f'rhs_kind_{i}_{x}_{y}'] = False    

st.markdown(f'# YOUR BACKTESTER')
cash = st.number_input("CASH", key="cash", value=st.session_state.get('cash', 10000))    
brokerage = st.number_input("Brokerage Commission: 0.04%", key="brokerage", value=st.session_state.get('brokerage', 0.04))
slippage = st.number_input("Slippage: 0.01%", key="slippage", value=st.session_state.get('slippage', 0.01))

for x in strategy_rules.keys():
    st.markdown(f'## Define the rules for {x}')
    print("708.",  list(data.columns))
    for y in strategy_rules[x].keys():
        if y == "EXIT":
            # st.markdown(strategy_rules[x.upper()]["ENTRY"]["conditions"], len(strategy_rules[x.upper()]["ENTRY"]["conditions"]))
            if len(strategy_rules[x.upper()]["ENTRY"]["conditions"]) == 0:
                continue
                 
        st.markdown(f'### Define the {y} rules ')
        with st.expander("Create your  conditions (LHS op RHS). RHS can be a number or an indicator. Default RHS = Price", expanded=True):
        # build dropdown list that contains only the indicators the user added + Price
            x = x.lower()
            y = y.lower()
            available_cols_cond = list(data.columns) 
            
            # number of conditions
            num_rules_input = st.number_input(
                    "Number of Conditions", 
                    min_value=0, 
                    max_value=6, 
                    value=st.session_state.get(f'num_rules_input_{x}_{y}'), 
                    key=f'num_rules_input_{x}_{y}', 
                    on_change=update_num_rules, # Direct function reference
                    args=(x, y) # Explicitly pass arguments via args tuple
                )
            # cols_for_buttons = st.columns([1, 1, 1, 1, 2])
            new_cond_specs = []
            new_connectors = []
            for i in range(int(st.session_state[f'num_rules_{x}_{y}'])):
                
                a, b, c, d, e = st.columns([3,1,3,1,2])
                rhs_kind = st.session_state.get(f'rhs_kind_{i}_{x}_{y}')
                if rhs_kind: 
                    r_kind = 0
                else: 
                    r_kind = 1
                print(f'lhs, rhs: {st.session_state[f"lhs_{i}"]}"], {st.session_state[f"rhs_{i}"]}"]')
                lhs = a.selectbox(f"LHS {i+1}", options=available_cols_cond, index=available_cols_cond.index(st.session_state.get(f'lhs_{i}_{x}_{y}',available_cols_cond[0])), key=f"lhs_{i}_{x}_{y}")
                op = b.selectbox(f"Op {i+1}", options=op_options ,index=op_options.index(st.session_state.get(f'op_{i}_{x}_{y}', op_options[0])),  key=f"op_{i}_{x}_{y}")
                print(f'690: {rhs_kind_options[r_kind]}')
                rhs_kind = c.selectbox(
                    f"RHS kind {i+1}", 
                    options=rhs_kind_options, 
                    index=r_kind, 
                    key=f"rhs_kind_input_{i}_{x}_{y}", 
                    on_change=update_rhs_kind_for_rule, 
                    args=(i, x, y) # Pass i, x, and y as arguments
                        )
                if rhs_kind == "Number":
                    rhs_val = c.number_input(f"Value {i+1}", value=st.session_state.get(f"rhs_num_{i}_{x}_{y}"), key=f"rhs_num_{i}_{x}_{y}")
                    rhs = float(rhs_val)
                    rhs_is_num = True
                else:
                    rhs = c.selectbox(f"RHS {i+1}", options=available_cols_cond, index=available_cols_cond.index(st.session_state.get(f'rhs_{i}_{x}_{y}', available_cols_cond[0])), key=f"rhs_{i}_{x}_{y}")
                    rhs_is_num = False

                new_cond_specs.append((lhs, op, rhs, rhs_is_num))
                if i < int(st.session_state[f'num_rules_{x}_{y}'])-1:
                    connector = e.selectbox(f"Connector {i+1}", options=connector_options, index=connector_options.index(st.session_state.get(f"conn_{i}_{x}_{y}", connector_options[0])), key=f"conn_{i}_{x}_{y}")
                    new_connectors.append(connector)
            strategy_rules[x.upper()][y.upper()] = {
                'conditions': new_cond_specs,
                'connectors': new_connectors
            }
            
        
# if backtest_btn:
        
cond_specs = st.session_state.cond_specs
connectors = st.session_state.connectors = new_connectors

c_json = {
    "indicator_conf": st.session_state.indicator_configs,
    "computed_cols": st.session_state.computed_cols,
    "conditions_conf": {
            "conditions": cond_specs,
            "connectors": connectors
                },
    "strategy_rules": strategy_rules,
    "chart": {
        "interval": st.session_state.chart_interval,
        "period": st.session_state.chart_period
    },
    "broker": {
        "slippage": slippage,
        "brokerage": brokerage,
        "cash": cash
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
strategy_type_options = ['Single Asset', 'Multi Asset']

strategy_type = st.selectbox("Select Strategy Type", options=strategy_type_options, key='strategy_type', index=strategy_type_options.index(st.session_state.get('strategy_type', strategy_type_options[0])))
number_of_symbols = st.number_input("Number of Symbols", key="num_sym", value=st.session_state.get("num_sym", 0))

for i in range(1, number_of_symbols+1):
    if f"input_{i}" not in st.session_state:
        st.session_state[f"input_{i}"] = ""

# Build a 4x4 grid of text inputs
for row in range(4):
    cols = st.columns(4)
    for col_idx, col in enumerate(cols):
        num = row * 4 + col_idx + 1
        if num > number_of_symbols:
            break
        with col:
            st.selectbox(
                f"Symbol {num}",
                symbols_tuple,
                index=symbols_tuple.index(st.session_state.get(f"symbol_{num}_key", "ADANIENT")),
                key=f"symbol_{num}_key"
            )

backtest_btn = st.button("Backtest Strategy")
import matplotlib.pyplot as plt
from my_backtester import backtest_single_asset, backtest_multiple_assets, DynamicStrategy, MultiAssetDynamicStrategy, get_detailed_metrics, plot_strategy_metrics


def fetch_data_and_compute_indicators(symbol, chart_interval=st.session_state.chart_interval, days=days):
    data_1m = fetch_historical(symbol, days=2)
    unique_dates = np.unique(data_1m.index.date)
    last_trading_days = unique_dates[-days:]
    df = data_1m[np.isin(data_1m.index.date, last_trading_days)].copy()
    chart_interval = interval_map[chart_interval]
    # keep trading hours (if intraday) and drop zero-volume days
    try:
        df = df.between_time("09:15", "15:30")
    except Exception:
        # non intraday index - ignore
        pass
    df = df[df.groupby(df.index.date)["volume"].transform("sum") > 0]

    # resample to chart interval for display
    data = resample_candles(df, chart_interval)
    all_cols, data = compute_indicators(data, indicator_config=st.session_state.indicator_configs, moving_averages=moving_averages, chart_interval=chart_interval, df=df)
    print('532', st.session_state.get('custom_cols_config'))
    for config in st.session_state['custom_cols_config']:
        data = perform_operation(data, name=config['name'], lhs=config['lhs'], operator=config['operator'], rhs=config['rhs'], operation_kind=config['operator_kind'], rhs_kind=config['rhs_kind'])
    print('535', list(data.columns))
    return data 


if backtest_btn:
    if st.session_state['strategy_type'] == strategy_type_options[0]:
        
        symbols_to_backtest = []
        progress = st.progress(0, text="Running backtests...")
        
        for x in range(st.session_state.get("num_sym", 0)):
            num = x + 1
            symbol_to_fetch = st.session_state.get(f"symbol_{num}_key", "ADANIENT")
            data = fetch_data_and_compute_indicators(symbol_to_fetch)
            
            # final_mask = evaluate_conditions(data, computed_cols=data.columns, cond_specs=cond_specs, connectors=connectors)
            # print('547', data.columns)
            st.session_state.data_store[symbol_to_fetch] = data
            strat = backtest_single_asset(data, DynamicStrategy, cash, data.columns, strategy_rules, commission=0.01 * float(brokerage) , slippage=0.01 * float(slippage))
            st.session_state.strats[symbol_to_fetch] = strat
            progress.progress((x + 1) / number_of_symbols)
        print('448', st.session_state.strats)
        progress.empty()
        st.success("✅ Backtests completed successfully!")
    else:
        progress = st.progress(0, text="Running backtests...")
        dfs = []
        for x in range(st.session_state.get("num_sym", 0)):
            num = x + 1
            symbol_to_fetch = st.session_state.get(f"symbol_{num}_key", "ADANIENT")
            data = fetch_data_and_compute_indicators(symbol_to_fetch)
            dfs.append([data, symbol_to_fetch])
            st.session_state.data_store[symbol_to_fetch] = data
        progress.progress(0.5)
        strat, df_portfolio, df_summary, portfolio_metrics = backtest_multiple_assets(dfs, MultiAssetDynamicStrategy, cash, data.columns, strategy_rules, commission=0.01 * float(brokerage) , slippage=0.01 * float(slippage))
        st.session_state.strats['multi_assets'] = strat
        progress.progress(1)
        # print('448', st.session_state.strats)
        progress.empty()
        st.success("✅ Backtests completed successfully!")
        
        
rows = st.session_state["num_sym"] // 3 
num_rows = 1 if rows < 1 else rows  
num_cols = 2
metric_labels = ['Symbol']
metrics_list = []
for row in range(num_rows):
    cols = st.columns(num_cols)
    for col in range(num_cols):
        i = row * num_cols + col
        if (i+1 <= st.session_state["num_sym"]):
            with cols[col]:

                symbol = st.session_state[f"symbol_{i+1}_key"]
                data = st.session_state.data_store[symbol]
                final_mask = evaluate_conditions(data=data, computed_cols=st.session_state.computed_cols, cond_specs=st.session_state.cond_specs, connectors=st.session_state.connectors)
                                
                if st.session_state.get('strategy_type') == strategy_type_options[0]:
                    strat = st.session_state.strats[symbol]
                    
                    metrics, portfolio_values = get_detailed_metrics(strat)
                    metrics_keys = metrics.keys()
                    
                    if metric_labels == ['Symbol']:
                        for m_key in metrics_keys:
                            label = ''
                            for current_label in m_key.split('_'):
                                label += current_label.title() + ' '
                            metric_labels.append(label.strip())
                    # print(metric_labels)
                    values = [symbol]
                    for value in metrics.values():
                        values.append(value)
                    metrics_df = pd.DataFrame({'metric':metric_labels, 'value': values})
                    print(metrics_df.head())
                    s = metrics_df.set_index('metric').T
                    metrics_list.append(s)
                    # st.table(metrics_df)
                    # figs = plot_strategy_metrics(metrics)
                    # for fig in figs:
                    #     st.pyplot(fig)
                else:
                    strat = st.session_state["strats"]["multi_assets"]
                    # st.dataframe(df_summary)
                       
                entries = strat.entries 
                exits = strat.exits
                
                # current_conds = strat.current_conditions 
                # entry_dates, entry_prices = [entry[0] for entry in entries], [entry[1] for entry in entries] 
                # cond_dates, cond_smas, cond_closes, cond_labels = [cond[0] for cond in current_conds], [cond[2] for cond in current_conds], [cond[1] for cond in current_conds], [cond[3] for cond in current_conds]
                # entries_df = pd.DataFrame({
                #     'dt': entry_dates, 'price': entry_prices 
                # })
                # cond_df = pd.DataFrame({
                #     'dt': cond_dates, 'sma': cond_smas, 'close': cond_closes, 'label': cond_labels
                # })
                
                fig = plot_charts_and_indicators_with_entry_exits(data, chart_interval, indicator_configs, final_mask, entries, exits, symbol=symbol, period_str=period_str, interval_label=interval_label, custom_cols_config=st.session_state.get('custom_cols_config'), strategy_type=st.session_state['strategy_type']) 
                st.plotly_chart(fig, use_container_width=True)

if len(metrics_list) > 0:
    metrics_df = pd.concat(metrics_list, ignore_index=True)
    metrics_df = metrics_df.set_index('Symbol')
    st.dataframe(metrics_df, width='stretch')            

if st.session_state.strategy_type == strategy_type_options[1]:
    # st.dataframe(df_portfolio)
    # for label in portfolio_metrics.keys():
    #     pass
    
    st.dataframe(portfolio_metrics)
    df_summary = df_summary.set_index('Symbol') 
    st.dataframe(df_summary)

# for i in range(1, number_of_symbols, 4):
#     cols = st.columns(4)
#     for j, col in enumerate(cols):
#         page_num = i + j
#         if page_num <= number_of_symbols:
#             label = st.session_state.get(f"symbol_{page_num}_key", "ADANIENT")
#             if col.button(label, key=f"btn_{label}"):
#                 st.session_state["current_page"] = label


# if "current_page" in st.session_state:
#     symbol = st.session_state.get("current_page")
#     print('463', symbol)
#     print('464', st.session_state.strats[symbol])
#     strat = st.session_state.strats[symbol]
#     data = st.session_state.data_store[symbol]
#     metrics, portfolio_values = get_detailed_metrics(strat)
#     metrics_df = pd.DataFrame({'metric':metrics.keys(), 'value': metrics.values()})
#     # st.markdown(metrics_df)
#     st.table(metrics_df)
#     # figs = plot_strategy_metrics(metrics)
#     # for fig in figs:
#     #     st.pyplot(fig)
#     entries = strat.entries 
#     exits = strat.exits
#     current_conds = strat.current_conditions 
#     entry_dates, entry_prices = [entry[0] for entry in entries], [entry[1] for entry in entries] 
#     cond_dates, cond_smas, cond_closes, cond_labels = [cond[0] for cond in current_conds], [cond[2] for cond in current_conds], [cond[1] for cond in current_conds], [cond[3] for cond in current_conds]
#     entries_df = pd.DataFrame({
#         'dt': entry_dates, 'price': entry_prices 
#     })
#     cond_df = pd.DataFrame({
#         'dt': cond_dates, 'sma': cond_smas, 'close': cond_closes, 'label': cond_labels
#     })
        
#     fig = plot_charts_and_indicators_with_entry_exits(data, chart_interval, indicator_configs, final_mask, entries, exits, symbol=symbol, period_str=period_str, interval_label=interval_label, custom_cols_config=st.session_state.get('custom_cols_config')) 
#     st.plotly_chart(fig, use_container_width=True)

#     portfolio_values = portfolio_values.set_index("datetime")
#     portfolio_values.index = pd.to_datetime(portfolio_values.index)
#     # portfolio_values.drop(0)
#     plt.figure(figsize=(15, 12))
#     plt.plot(portfolio_values.index, portfolio_values["value"], label='Portfolio vs Time')
#     plt.xlabel('Datetime')
#     plt.ylabel('Value')
#     plt.title('Portfolio Value over Time')
#     plt.show()

        
                    