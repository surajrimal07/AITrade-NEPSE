import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import json

from global_var import JWT_TOKEN, session

async def fetch_prediction():
    url = 'https://api.zorsha.com.np/api/heavyStocks'
    heavyStockResponse = requests.get(url, verify=False)
    if heavyStockResponse.status_code == 200:
        data = heavyStockResponse.json()
        prediction = data.get('prediction')
        strength = data.get('strength')

        if prediction is not None and strength is not None:
            return prediction, strength
        else:
            return 'Prediction or strength not found in the response!', None
    else:
        return 'Failed to fetch prediction!', None
