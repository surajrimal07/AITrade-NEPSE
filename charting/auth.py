import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def login():
    url = 'https://localhost:4000/api/login'
    payload = {'email': 'suraj@rimal.com', 'password': '111111'}
    try:
        response = requests.post(url, data=payload, verify=False)
        if response.status_code == 200:
            userdata = {
                'email': response.json()['data']['email'],
                'name': response.json()['data']['name'],
                'userAmount': response.json()['data']['userAmount']
            }
            print('Login successful!')
            return userdata
        else:
            print(f'Login failed with status code {response.status_code}')
            return None
    except Exception as e:
        print(f'Error during login: {e}')
        return None

# # Example usage
# login_data = login()
# if login_data:
#     print(f'Name: {login_data["name"]}, Amount: {login_data["userAmount"]}')
# else:
#     print('Login failed.')
