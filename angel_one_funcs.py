from SmartApi import SmartConnect         
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from my_config import *
import logging,threading,time,sys
import pandas as pd
import datetime
from dotenv import load_dotenv

token_dict = {}
def login():
    """This fx enables user to login and create two objects ie smartApi and sws(for websocket)"""

    smartApi =SmartConnect(api_key=API_KEY)      #smartApi is the object created
    data = smartApi.generateSession(USERNAME,PIN,TOKEN)
    s=data['status']
    if s==True:
        #print(data)
        name=data['data']['name']
        # print("LOGIN SUCCESSFUL FOR",name)                      #PRINT LOGIN SUCCESSFUL MESSAGE
        logging.info(f"LOGIN SUCCESSFUL FOR {name}")
        cash=smartApi.rmsLimit()['data'] 
        cash=cash['availablecash']                   
        # print("AVAILABLE CASH LIMIT IS",cash)                   #PRINT AVAILABLE CASH LIMIT
        logging.info(f"AVAILABLE CASH LIMIT IS {cash}")
    else:
        print("LOGIN UNSUCCESSFUL")
        print("INVALID CREDENTIALS")
        logging.info("LOGIN UNSUCCESSFUL")
    # print(data)
    authToken = data['data']['jwtToken']
    refreshToken = data['data']['refreshToken']
    feed_Token = smartApi.getfeedToken()                     # fetch the feedtoken
    res = smartApi.getProfile(refreshToken)
    #logging.info(f'{res["data"]["products"]}')
    logging.info(f'{res["data"]["exchanges"]}')             #exchanged subscribed
    sws = SmartWebSocketV2(authToken, API_KEY, USERNAME, feed_Token ,max_retry_attempt=5) 
    return smartApi,sws

def get_equitytoken(symname:str,exch="NSE"):
    
    if exch=="NSE":
        symname=symname+"-"+"EQ"
    # print(os.getcwd())
    df=pd.read_csv("ANGELFULL.csv", low_memory=False)
    df=df[(df['exch_seg']==exch)&(df['symbol']==symname)]
    df=df[['token','symbol']]
    
    if df.empty!=True:
    
        token=df.iloc[0,0]
        return token

def create_log_file(a:str):
    t=datetime.now()
    t1=t.strftime("%d-%b-%Y %H:%M:%S")
    t2=t.strftime("%d%b%Y %H%M%S")
    filename=a + t2 + ".txt"                                   #logfile name found using tradingdate1
    logging.basicConfig(filename=filename,level=logging.INFO,format="%(asctime)s-%(message)s",datefmt="%d-%b-%Y %H:%M:%S")

def on_data(wsapp, message):
    global token_dict
    try:
        # print(message)
        token = message['token']
        ltp = message['last_traded_price'] / 100
        vol = message.get("last_traded_quantity", 0)
        # print(ltp, vol)
        if token not in token_dict:
            token_dict[token] = {"ltp": ltp, "volume": vol}
        else:
            token_dict[token]["ltp"] = ltp
            token_dict[token]["volume"] += vol  # accumulate volume
    except Exception as e:
        print("on_data error:", e)

def on_error(wsapp,error):
    print("error")
    logging.info(f"---------Connection Error {error}-----------")

def on_close(wsapp):
    print("Connection Closed")
    logging.info(f"---------Connection closed-----------")

def subscribeSymbol(token_list,sws):
    logging.info(f'Subscribed to new tokens -------  {token_list}')
    sws.subscribe(CORRELATION_ID, FEED_MODE, token_list)


def subscribe_symbol(sym):
    token_list = []
    securityid = get_equitytoken(sym)
    token_list.append(securityid)
    subscribeList = [{"exchangeType": 1, "tokens": token_list}]
    subscribeSymbol(subscribeList, SMART_WEB)
    return securityid

def connectFeed(sws, symbols):
    def on_open(wsapp):
        print("Connected to WebSocket.")
        token_list = []
        for sym in symbols:
            token = get_equitytoken(sym)
            token_list.append(token)
        subscribeList = [{"exchangeType": 1, "tokens": token_list}]
        sws.subscribe(CORRELATION_ID, FEED_MODE, subscribeList)

    sws.on_open = on_open
    sws.on_data = on_data
    sws.on_error = on_error
    sws.on_close = on_close

    threading.Thread(target=sws.connect, daemon=True).start()

login_result=login()
SMART_API_OBJ=login_result[0]                                    # create smart api object for trading 
SMART_WEB=login_result[1]           
