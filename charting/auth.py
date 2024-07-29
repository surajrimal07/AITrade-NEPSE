import sys
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import json
import time
from toastify import notify
import copy

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from global_var import *
from data_process import get_last_price

def show_toast(message):
    app = QApplication.instance() or QApplication(sys.argv)
    QMessageBox.information(None, 'Information', f"Message: {message}")

def fetch_csrf_token():
    global CSRF_TOKEN
    global session

    url = baseUrl+'/user/csrf-token'
    response = session.get(baseUrl+'/user/csrf-token', verify=False)
    if response.status_code == 200:
        CSRF_TOKEN = response.json().get('token', None)
#        CSRF_TOKEN = response.json().get('token', None)
        session.headers.update({'xsrf-token': CSRF_TOKEN})
        return CSRF_TOKEN
    else:
        return None

def show_error_dialog(message):
    app = QApplication.instance() or QApplication(sys.argv)
    QMessageBox.warning(None, 'Error Occurred', f"Message: {message}")

def add_stock_to_portfolio(symbol, quantity, time = int(time.time())):
    global JWT_TOKEN, session, portfolio_id, CSRF_TOKEN
    last_price = get_last_price()

    url = baseUrl+'user/addstock'
    payload = {'symboll': symbol, 'quantityy': quantity, 'price': last_price,'id': portfolio_id, time : time}

    response = session.post(url, data=payload, verify=False)
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
        show_error_dialog(response_data.get('message', ''))
        return None

def remove_stock_from_portfolio(symbol, quantity):
    global JWT_TOKEN, session, portfolio_id, CSRF_TOKEN
    last_price = get_last_price()


    url = baseUrl+'user/remstock'
    payload = {'symbol': symbol, 'quantity': quantity,'id': portfolio_id,'price': last_price}

    response = session.post(url, data=payload, verify=False)
    response_data = response.json()
    trading_portfolio = next((p for p in response_data['data']['portfolio'] if p['name'] == 'tradingdashboard'), None)
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
    global CSRF_TOKEN

    url = baseUrl+'user/getallportforuser'
    headers = {'Authorization': 'Bearer ' + JWT_TOKEN,
    'xsrf-token': CSRF_TOKEN}
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


def fetch_user_data_api():
    global JWT_TOKEN
    global session
    global CSRF_TOKEN

    url = baseUrl+'user/verify'
    headers = {'Authorization': 'Bearer ' + JWT_TOKEN,'xsrf-token': CSRF_TOKEN}
    userDataResponse = session.get(url, headers=headers, verify=False)
    if userDataResponse.status_code == 200:
        return userDataResponse.json()['data']
    else:
        show_error_dialog('Failed to fetch user data!')
        return None


def logout():
    global JWT_TOKEN, session,INITIAL_JWT_TOKEN,INITIAL_CSRF_TOKEN, CSRF_TOKEN

    url = baseUrl+'user/logout'
    response = session.get(url, verify=False)
    response_data = response.json()
    if response.status_code == 200:

        session.cookies.clear()
        session.headers.clear()

        JWT_TOKEN = copy.deepcopy(INITIAL_JWT_TOKEN)
        CSRF_TOKEN = copy.deepcopy(INITIAL_CSRF_TOKEN)

        show_error_dialog(response_data.get('message', ''))
        return True
    else:
        show_error_dialog(response_data.get('message', ''))
        return False


def login(email, password):
    global JWT_TOKEN, session, CSRF_TOKEN, userEmail, userPassword
    if email != userEmail and password != userPassword:
        userEmail = email
        userPassword = password

    url = baseUrl+'user/login'
    portfolio_payload = {'email': email}
    payload = {'email': email, 'password': password}
    session.headers.update({'Accept': 'application/json'})
    session.headers.update({'Access-Control-Allow-Origin': '*'})
    session.headers.update({'Access-Control-Allow-Credentials': 'true'})

    try:
        csrf_token = session.get(baseUrl+'/user/csrf-token', verify=False)
        if csrf_token.status_code == 200:
            CSRF_TOKEN = csrf_token.json().get('token', None)
            session.headers.update({'xsrf-token': CSRF_TOKEN})

            response = session.post(url, data=payload, verify=False)

            if response.status_code == 200:

                userdata = {
                'email': response.json()['data']['email'],
                'name': response.json()['data']['name'],
                'userAmount': response.json()['data']['userAmount']
                }

                JWT_TOKEN = response.json()['data']['token']
                session.headers.update({'Authorization': 'Bearer ' + JWT_TOKEN})
                return userdata, None

            else:
                error_message = response.json().get('message', 'Login failed with status code {}'.format(response.status_code))
                return None, error_message

        else:
            return None, 'Failed to fetch CSRF token!'

    except Exception as e:
        return None, str(e)

def signup(name, email, phone, password):
    global JWT_TOKEN, session, CSRF_TOKEN

    fetch_csrf_token()

    url = baseUrl+'user/create'
    payload = {'name': name, 'email': email, 'phone': phone, 'password': password}
    response = session.post(url, data=payload, verify=False)
    if response.status_code == 200:
        return response.json()['data'], None
    else:
        error_message = response.json().get('message', 'Signup failed with status code {}'.format(response.status_code))
        return None, error_message

class SignupDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Signup to AI Trader')
        self.layout = QVBoxLayout()

        self.name_label = QLabel('Name:')
        self.name_label.setStyleSheet("font-size: 18px;")
        self.layout.addWidget(self.name_label)

        self.name_input = QLineEdit()
        self.name_input.setFixedHeight(50)
        self.name_input.setStyleSheet("font-size: 18px;")
        self.layout.addWidget(self.name_input)

        self.phone_label = QLabel('Phone:')
        self.phone_label.setStyleSheet("font-size: 18px;")
        self.layout.addWidget(self.phone_label)

        self.phone_input = QLineEdit()
        self.phone_input.setFixedHeight(50)
        self.phone_input.setStyleSheet("font-size: 18px;")
        self.layout.addWidget(self.phone_input)

        self.email_label = QLabel('Email:')
        self.email_label.setStyleSheet("font-size: 18px;")
        self.layout.addWidget(self.email_label)

        self.email_input = QLineEdit()
        self.email_input.setFixedHeight(50)
        self.email_input.setStyleSheet("font-size: 18px;")
        self.layout.addWidget(self.email_input)

        self.password_label = QLabel('Password:')
        self.password_label.setStyleSheet("font-size: 18px;")
        self.layout.addWidget(self.password_label)

        self.password_input = QLineEdit()
        self.password_input.setFixedHeight(50)
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet("font-size: 18px;")
        self.layout.addWidget(self.password_input)

        self.signup_button = QPushButton('Signup')
        self.signup_button.setFixedHeight(60)
        self.signup_button.setStyleSheet("font-size: 18px;")
        self.signup_button.clicked.connect(self.on_signup)
        self.layout.addWidget(self.signup_button)

        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.setFixedHeight(60)
        self.cancel_button.setStyleSheet("font-size: 18px;")
        self.cancel_button.clicked.connect(self.reject)
        self.layout.addWidget(self.cancel_button)

        self.setLayout(self.layout)
        self.setFixedWidth(400)

    def on_signup(self):
        name = self.name_input.text()
        password = self.password_input.text()
        email = self.email_input.text()
        phone = self.phone_input.text()

        user_data, error_message = signup(name, password,email, phone)
        if user_data:
            self.user_data = user_data
            QMessageBox.information(self, 'Login Success', 'Welcome back {}!'.format(user_data['name']))
            self.accept()
        else:
            error_message = error_message or 'Invalid credentials!'
            QMessageBox.warning(self, 'Login Error', error_message)

class LoginDialog(QDialog):
    global userEmail, userPassword

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
        self.name_input.setText(userEmail)
        self.layout.addWidget(self.name_input)

        self.password_label = QLabel('Password:')
        self.password_label.setStyleSheet("font-size: 18px;")
        self.layout.addWidget(self.password_label)

        self.password_input = QLineEdit()
        self.password_input.setFixedHeight(50)
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet("font-size: 18px;")
        self.password_input.setText(userPassword)
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

        self.signup_button = QPushButton("Don't have an account? Signup")
        self.signup_button.setFixedHeight(60)
        self.signup_button.setStyleSheet("font-size: 18px;")
        self.signup_button.clicked.connect(self.show_signup_dialog)
        self.layout.addWidget(self.signup_button)

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

    def show_signup_dialog(self):
        signup_dialog = SignupDialog()
        signup_dialog.exec_()


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

class ModelNotTrainedDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Model Not Trained')
        self.layout = QVBoxLayout()

        self.message_label = QLabel('Model not trained for the selected stock!')
        self.message_label.setStyleSheet("font-size: 18px;")
        self.layout.addWidget(self.message_label)

        self.ok_button = QPushButton('OK')
        self.ok_button.setFixedHeight(60)
        self.ok_button.setStyleSheet("font-size: 18px;")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

        self.setLayout(self.layout)
        self.setFixedWidth(400)

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
        isloggedout = logout()
        if isloggedout:
            return True
        else:
            return False
    else:
        print('Logout canceled')
        return False

def show_model_not_trained_dialog():
    app = QApplication(sys.argv)
    dialog = ModelNotTrainedDialog()
    dialog.setModal(True)
    dialog.exec_()