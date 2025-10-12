import pandas as pd
import datetime
import re
from angel_one_funcs import *

EXCHANGE = "NSE"
obj = SMART_API_OBJ

# Example symbols
angel_one_df = pd.read_csv('ANGELFULL.csv')
symbols_tuple = tuple(angel_one_df['symbol'])
symbols = list(symbols_tuple)

connectFeed(SMART_WEB, symbols=symbols)



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
    token = get_equitytoken(sym, exch=EXCHANGE)
    if not token:
        return pd.DataFrame()

    fromdate = target_date.strftime("%Y-%m-%d 09:15")
    todate = target_date.strftime("%Y-%m-%d 15:30")

    params = {
        "exchange": EXCHANGE,
        "symboltoken": token,
        "interval": "ONE_MINUTE",
        "fromdate": fromdate,
        "todate": todate
    }

    try:
        candles = obj.getCandleData(params)
        if not candles.get("status") or not candles.get("data"):
            return pd.DataFrame()
    except:
        return pd.DataFrame()

    df = pd.DataFrame(candles["data"], columns=["datetime", "open", "high", "low", "close", "volume"])

    # --- Clean numeric columns ---
    df = df.applymap(lambda x: str(x).strip() if x is not None else x)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].apply(clean_numeric)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "close"])
    df = df.set_index("datetime").sort_index()
    print(f"one day head: {df.head()}")
    print(f"one day tail: {df.tail()}")
    
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
        
        df_day = fetch_one_day(sym, target_date)
        if not df_day.empty:
            all_df.append(df_day)
        
        # Safety: prevent infinite loop
        if lookback > 30:
            break

    if not all_df:
        return pd.DataFrame()
    
    return pd.concat(all_df)