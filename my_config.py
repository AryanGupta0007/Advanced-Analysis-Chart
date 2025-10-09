import pytz,pyotp
from datetime import datetime
import  streamlit as st 
TIME_ZONE = pytz.timezone('Asia/Kolkata')
TZ_INFO = datetime.now(TIME_ZONE).tzinfo
API_KEY =    st.secrets["api"]["API_KEY"]
USERNAME =  st.secrets["api"]["USERNAME"]
PIN =       st.secrets["api"]["PIN"]

TOKEN =    pyotp.TOTP(st.secrets["api"]["TOKEN"]).now()   #HERE we get totp using pyotp .

SMART_API_OBJ =None
LIVE_FEED_JSON= {}
SMART_WEB = None
CORRELATION_ID = st.secrets["api"]["CORRELATION_ID"]                           # ANY random string
FEED_MODE = 2                                               #MODE FOR LTP,2 for quote
BREEZE_API_KEY = st.secrets["api"]["BREEZE_API_KEY"]
BREEZE_SECRET =  st.secrets["api"]["BREEZE_SECRET"]
MY_API_SESSION = st.secrets["api"]["MY_API_SESSION"]
