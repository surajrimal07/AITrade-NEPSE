from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import time
import schedule
from bs4 import BeautifulSoup
from datetime import datetime
import csv

app = Flask(__name__)
CORS(app)

#root route
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('name')
        if name:
            return jsonify({'message': f'Hello, {name}!'})
        else:
            return jsonify({'message': 'Hello, World!'})
    else:
        return jsonify({'message': 'Hello, World!'})

def remove_commas(value):
    return int(value.replace(',', ''))

def parse_change(change_str):
    return float(change_str.replace('%', ''))

#function to extract prices
row_number = 1

def extract_prices():
    url = 'https://www.sharesansar.com/live-trading'
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        prices_data = {}

        # Extract time
        time_element = soup.find('span', id='dDate')
        time_str = time_element.text.strip() if time_element else None
        time_value = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S') if time_str else None

        mu_list = soup.find_all('div', class_='mu-list')
        for mu_item in mu_list:
            name = mu_item.find('h4').text.strip()
            price = mu_item.find('p', class_='mu-price').text.strip()
            index_value = mu_item.find('span', class_='mu-value').text.strip()
            index_value = float(index_value.replace(',', '')) if index_value else None
            change = mu_item.find('span', class_='mu-percent').text.strip()
            change_value = parse_change(change)
            turnover = remove_commas(price)
            item_data = {
                'Name': name,
                'Index': index_value,
                'Turnover': turnover,
                'Change': change_value
            }
            prices_data[name] = item_data

        #also write to csv
        with open('prices_data.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            rows = list(csvreader)
            if len(rows) > 1:
                last_row = rows[-1]
                if last_row[0]:
                    last_row_number = int(last_row[0])
                    row_number = last_row_number + 1
                else:
                    row_number = 1
            else:
                row_number = 1

        nepse_data = prices_data.get('NEPSE Index')
        with open('prices_data.csv', 'a', newline='') as csvfile:
            fieldnames = ['Row','Time', 'Index', 'Turnover', 'Change']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()
            if nepse_data:
                writer.writerow({
                    'Row': row_number,
                    'Time': time_value.strftime('%Y-%m-%d %H:%M:%S'),
                    'Index': nepse_data.get('Index'),
                    'Turnover': nepse_data.get('Turnover'),
                    'Change': nepse_data.get('Change')
                })
                row_number += 1

        return prices_data, time_value
    else:
        return {'error': 'Failed to fetch data'}

prices_data = extract_prices()


#route to get prices
@app.route('/getIndex', methods=['GET'])
def get_index_data():
    global prices_data
    prices_data, time_value = extract_prices()
    return jsonify({
        'Time': time_value.strftime('%Y-%m-%d %H:%M:%S') if time_value else None,
        'PricesData': list(prices_data.values())
    })

schedule.every(5).seconds.do(extract_prices)

def start_price_extraction():
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    import threading
    price_extraction_thread = threading.Thread(target=start_price_extraction)
    price_extraction_thread.start()

    app.run(debug=True)
