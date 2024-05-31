import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import json
import time
from toastify import notify

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from global_var import *
from data_process import get_last_price

# def show_toast(message):
#     app = QApplication.instance() or QApplication(sys.argv)
#     QMessageBox.information(None, 'Error Occurred', f"Message: {message}")

def show_error_dialog(message):
    app = QApplication.instance() or QApplication(sys.argv)
    QMessageBox.warning(None, 'Error Occurred', f"Message: {message}")

def add_stock_to_portfolio(symbol, quantity, time = int(time.time())):
    global JWT_TOKEN, session, portfolio_id
    last_price = get_last_price()

    url = baseUrl+'user/addstock'
    payload = {'symboll': symbol, 'quantityy': quantity, 'price': last_price,'id': portfolio_id, time : time}
    headers = {'Authorization': 'Bearer ' + JWT_TOKEN}
    response = session.post(url, data=payload, headers=headers, verify=False)
    response_data = response.json()
    if response.status_code == 200:
        trading_portfolio = next((p for p in response_data['data']['portfolio'] if p['name'] == 'tradingdashboard'), None)
        #trading_portfolio = next((p for p in response_data['data'] if p['name'] == 'tradingdashboard'), None)
        if trading_portfolio:
           return trading_portfolio
        else:
            show_error_dialog(response_data.get('message', ''))
            return None
    else:
        #show_toast('Error adding stock to portfolio!')
        show_error_dialog(response_data.get('message', ''))
        return None

def remove_stock_from_portfolio(email, symbol, quantity):
    global JWT_TOKEN
    global session
    global portfolio_id

    url = baseUrl+'user/removestock'
    payload = {'email': email, 'symbol': symbol, 'quantity': quantity,'id': portfolio_id}
    headers = {'Authorization': 'Bearer ' + JWT_TOKEN}
    response = session.post(url, data=payload, headers=headers, verify=False)
    trading_portfolio = next((p for p in portfolioResponse.json()['data']['portfolio'] if p['name'] == 'tradingdashboard'), None)
    #trading_portfolio = next((p for p in response.json()['data']['portfolio'] if p['name'] == 'tradingdashboard'), None)
    if response.status_code == 200:
        if trading_portfolio:
           return trading_portfolio
        else:
            show_error_dialog('Error removing stock from portfolio!')
            return None

def fetch_trading_portfolio():
    global JWT_TOKEN
    global session
    global portfolio_id

    url = baseUrl+'user/getallportforuser'
    headers = {'Authorization': 'Bearer ' + JWT_TOKEN}
    portfolioResponse = session.get(url, headers=headers, verify=False)
    if portfolioResponse.status_code == 200:
        trading_portfolio = next((p for p in portfolioResponse.json()['data']['portfolio'] if p['name'] == 'tradingdashboard'), None)
        #trading_portfolio = next((p for p in portfolioResponse.json()['data'] if p['name'] == 'tradingdashboard'), None)
        if trading_portfolio:
           portfolio_id = trading_portfolio['_id']
           return trading_portfolio
        else:
            show_error_dialog('No trading portfolio found!')
            return None
    else:
        show_error_dialog('Failed to fetch trading portfolio!')
        return None


def login(email, password):
    global JWT_TOKEN
    global session

    url = baseUrl+'user/login'
    portfolio_payload = {'email': email}
    payload = {'email': email, 'password': password}
    try:
        response = session.post(url, data=payload, verify=False)
        if response.status_code == 200:

            userdata = {
                'email': response.json()['data']['email'],
                'name': response.json()['data']['name'],
                'userAmount': response.json()['data']['userAmount']
            }

            JWT_TOKEN = response.json()['data']['token']
            return userdata, None
        else:
           error_message = response.json().get('message', 'Login failed with status code {}'.format(response.status_code))
           return None, error_message
    except Exception as e:
        return None, str(e)

class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.user_data = None
        self.setWindowTitle('Login to AI Trader')
        self.layout = QVBoxLayout()

        self.name_label = QLabel('Email:')
        self.name_label.setStyleSheet("font-size: 18px;")
        self.layout.addWidget(self.name_label)

        self.name_input = QLineEdit()
        self.name_input.setFixedHeight(50)
        self.name_input.setStyleSheet("font-size: 18px;")
        self.name_input.setText("ikalpana74@gmail.com")
        self.layout.addWidget(self.name_input)

        self.password_label = QLabel('Password:')
        self.password_label.setStyleSheet("font-size: 18px;")
        self.layout.addWidget(self.password_label)

        self.password_input = QLineEdit()
        self.password_input.setFixedHeight(50)
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet("font-size: 18px;")
        self.password_input.setText("Nightiee@0014")
        self.layout.addWidget(self.password_input)

        self.login_button = QPushButton('Login')
        self.login_button.setFixedHeight(60)
        self.login_button.setStyleSheet("font-size: 18px;")
        self.login_button.clicked.connect(self.on_login)
        self.layout.addWidget(self.login_button)

        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.setFixedHeight(60)
        self.cancel_button.setStyleSheet("font-size: 18px;")
        self.cancel_button.clicked.connect(self.reject)
        self.layout.addWidget(self.cancel_button)

        self.setLayout(self.layout)

        self.setFixedWidth(400)

    def on_login(self):
        name = self.name_input.text()
        password = self.password_input.text()
        user_data, error_message = login(name, password)
        if user_data:
            self.user_data = user_data
            QMessageBox.information(self, 'Login Success', 'Welcome back {}!'.format(user_data['name']))
            self.accept()
        else:
            error_message = error_message or 'Invalid credentials!'
            QMessageBox.warning(self, 'Login Error', error_message)


class LogoutDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Logout from AI Trader')
        self.layout = QVBoxLayout()

        self.logout_button = QPushButton('Logout')
        self.logout_button.setFixedHeight(60)
        self.logout_button.setStyleSheet("font-size: 18px;")
        self.logout_button.clicked.connect(self.accept)
        self.layout.addWidget(self.logout_button)

        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.setFixedHeight(60)
        self.cancel_button.setStyleSheet("font-size: 18px;")
        self.cancel_button.clicked.connect(self.reject)
        self.layout.addWidget(self.cancel_button)

        self.setLayout(self.layout)
        self.setFixedWidth(400)

    def on_logout(self):
        result = QMessageBox.question(self, 'Logout Confirmation', 'Are you sure you want to logout?', QMessageBox.Yes | QMessageBox.No)
        if result == QMessageBox.Yes:
            QMessageBox.information(self, 'Logout', 'Logout successful!')
            self.accept()

def show_login_dialog():
    app = QApplication(sys.argv)
    login_dialog = LoginDialog()
    login_dialog.setModal(True)

    result = login_dialog.exec_()
    if result == QDialog.Accepted:
        return login_dialog.user_data
    else:
        print('Login canceled')
        return None

def show_logout_dialog():
    app = QApplication(sys.argv)
    login_dialog = LogoutDialog()
    login_dialog.setWindowTitle('Logout from AutoTrader')
    login_dialog.setModal(True)

    result = login_dialog.exec_()
    if result == QDialog.Accepted:
        return True
    else:
        print('Logout canceled')
        return False
