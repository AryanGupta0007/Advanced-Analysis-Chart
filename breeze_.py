from breeze_connect import BreezeConnect
from my_config import BREEZE_API_KEY, BREEZE_SECRET, MY_API_SESSION
import sys
import pandas as pd
# Read the txt file
# Replace 'delimiter_here' with your actual delimiter, e.g., '\t' for tab, '|' for pipe, ',' for comma
symbol_df = pd.read_csv('ICICIFULL.csv')
# print(symbol_df)
eq_symbols = symbol_df[symbol_df[' "Series"'] == 'EQ']
all_symbols = eq_symbols[' "ExchangeCode"']  
# print(list(all_symbols))
# sys.exit()
breeze = BreezeConnect(api_key=BREEZE_API_KEY)
print(fr"https://api.icicidirect.com/apiuser/login?api_key={BREEZE_API_KEY}")
# Obtain your session key from https://api.icicidirect.com/apiuser/login?api_key=YOUR_API_KEY
# Incase your api-key has special characters(like +,=,!) then encode the api key before using in the url as shown below.
import urllib
# Generate Session
breeze.generate_session(api_secret=BREEZE_SECRET,
                        session_token=MY_API_SESSION)
print(breeze.get_customer_details(api_session=MY_API_SESSION))
# Generate ISO8601 Date/DateTime String
import datetime
iso_date_string = datetime.datetime.strptime("28/02/2021","%d/%m/%Y").isoformat()[:10] + 'T05:30:00.000Z'
iso_date_time_string = datetime.datetime.strptime("28/02/2021 23:59:59","%d/%m/%Y %H:%M:%S").isoformat()[:19] + '.000Z'

def fetch_stock_code(symbol):
    df = pd.read_csv('ICICIFULL.csv')
    row = df[df[' "ExchangeCode"'] == symbol]
    if not row.empty:
        return row[' "ShortName"'].iloc[0]  # get first value
    else:
        return None  # or raise error if symbol not found
import datetime

def fetch_today_intraday(symbol, target_date):
    # Market start and end times
    start_time = datetime.datetime(target_date.year, target_date.month, target_date.day, 9, 15)
    end_time   = datetime.datetime(target_date.year, target_date.month, target_date.day, 15, 30)

    from_date = start_time.isoformat()[:19] + '.000Z'
    to_date   = end_time.isoformat()[:19] + '.000Z'

    stock_code = fetch_stock_code(symbol)
    print(f"stock_code: {stock_code}")
    if stock_code is None:
        print("Stock code not found")
        return None
    print(from_date, to_date)
    # Fetch intraday data at 1-minute interval
    data = breeze.get_historical_data_v2(
        interval="1minute",
        from_date=from_date,
        to_date=to_date,
        stock_code=stock_code,
        exchange_code="NSE",
        product_type="cash"
    )

    if not data or not data.get('Success'):
        print("No data returned from API:", data)
        return None

    df = pd.DataFrame(data['Success'])
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

    return df



def fetch_historical_data(symbol: str, start_date: list, end_date: list, interval="1s"):
    intervals = {
        "1s": "1second",
        "1m": "1minute",
        "5m": "5minute",
        "30m": "30minute",
        "1d": "1day"
    }
    (start_year, start_month,  start_day) = (start_date[0], start_date[1], start_date[2])
    (end_year, end_month,  end_day) = (end_date[0], end_date[1], end_date[2])
    from_date = datetime.datetime(start_year, start_month,  start_day, 9, 15).isoformat()[:19] + '.000Z'
    to_date   = datetime.datetime(end_year, end_month,  end_day, 15, 30).isoformat()[:19] + '.000Z'
    stock_code = fetch_stock_code(symbol)
    # --- Fetch Historical Data ---
    data = breeze.get_historical_data_v2(interval=intervals[interval],
                        from_date= from_date,
                        to_date= to_date,
                        stock_code=stock_code,
                        exchange_code="NSE",
                        product_type="cash")

    # print(data)
    # --- Convert to DataFrame ---
    print(data)
    df = pd.DataFrame(data['Success'])
    # print(df.head())
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # print(df.head())
    return df

# df  = fetch_historical_data("ETERNAL", [2025, 10, 5], [2025, 10, 5], "1s")
symbol = "ETERNAL"
start_date = {
    "year": 2025,
    "month": 10,
    "day": 6
}
date = {
    "year": 2025,
    "month": 10,
    "day": 6
}
interval   = "1m"            # 1-second candles

# Fetch data
# df = fetch_today_intraday(symbol, date)
# if df is not None:
#     print(df.head(10))
#     print(df.tail(2))
#     print(df.shape)
# else:
#     print("NO Data Found")