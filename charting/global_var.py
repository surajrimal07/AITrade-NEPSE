import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import copy

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

INITIAL_JWT_TOKEN = ""
INITIAL_CSRF_TOKEN = ""

JWT_TOKEN = copy.deepcopy(INITIAL_JWT_TOKEN)
CSRF_TOKEN = copy.deepcopy(INITIAL_CSRF_TOKEN)


headers ={'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.28 Safari/537.36', 'Authorization': 'Bearer ' + JWT_TOKEN, 'xsrf-token': CSRF_TOKEN}

session = requests.Session()
session.headers = headers

portfolio_id = None

baseUrl = 'https://api.surajr.com.np/api/'
#baseUrl = 'https://localhost:4000/api/'

websocket_url = 'wss://socket.surajr.com.np/?room=portfolio&jwt='

defaultSymbol = 'NEPSE'
defaultTimeFrame = '1D'

SubchartSecurity = 'NEPSE'
SubchartTimeFrame = '1'

default_quantity = 1 #stock quantity to buy

autorefresh = False
auto_refresh_task = None
INITIAL_USER_DATA = {'name': 'No User', 'email': 'No User', 'userAmount': 0}

INITIAL_USER_PORTFOLIO = {'recommendation': 'None','stocks': [
        {'name': 'None','ltp': 0,'symbol': 'None', 'quantity': 0, 'wacc': 0, 'costprice': 0, 'currentprice': 0, 'netgainloss': 0,'time': 0}
    ], 'totalunits': 0, 'portfoliocost': 0, 'portfoliovalue': 0, 'portgainloss': 0, 'portfolioPercentage': 0, 'totalStocks': 0,'totalunits': 0,}

isLoggedin = False

basic_prediction_result = [
    {'prediction': 'No Data Available', 'strength': 0}
]

userData = copy.deepcopy(INITIAL_USER_DATA)
userPortfolio = copy.deepcopy(INITIAL_USER_PORTFOLIO)

algo_names = ['Regression', 'TimeSeries', 'LSMT', 'Deep_learning', 'custom_algorithm']


fetchLivePortfolio = True


FULLSCREEN = '■'
CLOSE = '×'

userEmail = "ikalpana74@gmail.com"
userPassword = "Nightiee@0014"