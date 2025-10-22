import pandas as pd
import datetime
import re
from breeze_ import *
# from angel_one_funcs import *

EXCHANGE = "NSE"

# Example symbols
symbols_tuple = tuple(all_symbols)

symbols = list(symbols_tuple)
# print(symbols, symbols_tuple)

# connectFeed(SMART_WEB, symbols=symbols)



# --- Helper function to clean numeric values ---
def clean_numeric(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return x
    # Remove all non-numeric characters except digits, dot, minus
    cleaned = re.sub(r"[^0-9.\-]", "", str(x))
    try:
        return float(cleaned)
    except:
        return None

# --- Fetch one day of data ---
def fetch_one_day(sym, target_date):
    data = fetch_today_intraday(sym, target_date)
    df = pd.DataFrame(data, columns=["datetime", "open", "high", "low", "close", "volume"])

    # --- Clean numeric columns ---
    df = df.applymap(lambda x: str(x).strip() if x is not None else x)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].apply(clean_numeric)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "close"])
    df = df.set_index("datetime").sort_index()
    # print(f"one day head: {df.head()}")
    # print(f"one day tail: {df.tail()}")
    
    return df

def fetch_historical(sym="ADANIENT", days=2):
    all_df = []
    today = datetime.datetime.now()
    lookback = 0
    
    while len(all_df) < days:
        target_date = today - datetime.timedelta(days=lookback+1)
        lookback += 1
        
        # Skip weekends
        if target_date.weekday() >= 5:
            continue
        print(f"target data: {target_date}")
        # target_date = {
        #     "date": target_date.date,
        #     "year": target_date.year,
        #     "month": target_date.month
        # }
        df_day = fetch_one_day(sym, target_date)
        if not df_day.empty:
            all_df.append(df_day)
        
        # Safety: prevent infinite loop
        if lookback > 30:
            break

    if not all_df:
        return pd.DataFrame()
    
    return pd.concat(all_df)
# print(fetch_historical(sym="ADANIENT", days=2))