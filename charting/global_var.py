import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

JWT_TOKEN = None
CSRF_TOKEN = None

session = requests.Session()
portfolio_id = None

#baseUrl = 'https://api.zorsha.com.np/api/'
baseUrl = 'https://localhost:4000/api/'

websocket_url = 'wss://localhost:8081/?room=portfolio&jwt='

defaultSymbol = 'NEPSE'
defaultTimeFrame = '1D'

SubchartSecurity = 'NEPSE'
SubchartTimeFrame = '1'

default_quantity = 1 #stock quantity to buy

autorefresh = False
auto_refresh_task = None
userData = [
    {'name': 'No User', 'email': 'No User', 'userAmount': 0}
]
userPortfolio = [
    {'recommendation': 'None','stocks': [
        {'name': 'None','ltp': 0,'symbol': 'None', 'quantity': 0, 'wacc': 0, 'costprice': 0, 'currentprice': 0, 'netgainloss': 0,'time': 0}
    ], 'totalunits': 0, 'portfoliocost': 0, 'portfoliovalue': 0, 'portgainloss': 0, 'portfolioPercentage': 0, 'totalStocks': 0,'totalunits': 0,}
]
isLoggedin = False
basic_prediction_result = [
    {'prediction': 'No Data Available', 'strength': 0}
]

fetchLivePortfolio = True