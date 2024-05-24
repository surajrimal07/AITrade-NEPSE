import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

JWT_TOKEN = None
session = requests.Session()
portfolio_id = None


#baseUrl = 'https://api.zorsha.com.np/api/'
baseUrl = 'https://localhost:4000/api/'