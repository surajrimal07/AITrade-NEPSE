import json
import pandas as pd

# Initialize lists to store data
timestamps = []
open_prices = []
high_prices = []
low_prices = []
close_prices = []
volumes = []

# Load the raw JSON data line by line
with open('daily.json', 'r') as file:
    for line in file:
        json_data = json.loads(line)
        timestamps.extend(json_data['t'])
        open_prices.extend(json_data['o'])
        high_prices.extend(json_data['h'])
        low_prices.extend(json_data['l'])
        close_prices.extend(json_data['c'])
        volumes.extend(json_data['v'])

# Convert Unix epoch timestamps to proper datetime format
date_times = pd.to_datetime(timestamps, unit='s')

# Create a list of dictionaries for each record
data = []
for date, open_price, high, low, close, volume in zip(
    date_times.strftime('%Y-%m-%d'),
    open_prices,
    high_prices,
    low_prices,
    close_prices,
    volumes
):
    data.append({
        'date': date,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

# Save the data to JSON with the desired structure
with open('dailytest1.json', 'w') as json_file:
    json_file.write('[')
    for idx, record in enumerate(data):
        if idx > 0:
            json_file.write(',')
        json.dump(record, json_file)
    json_file.write(']')


import pandas as pd
import requests
from lightweight_charts import Chart
from io import StringIO
from time import sleep
import threading
import time, datetime
from data_process import process_json_data
from requests.packages.urllib3.exceptions import InsecureRequestWarning

SecurityName = 'NEPSE'
timeFrame = '1D'
autorefresh = False

# Disable SSL warnings on localhost with self signed certificate
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def calculate_sma(df, period: int = 50):
    return pd.DataFrame({
        'time': df['date'],
        f'SMA {period}': df['close'].rolling(window=period).mean()
    }).dropna()

def time_frame_manipulation(timeFrame):
    if timeFrame in ['1Min', '5Min', '10Min', '15Min', '1H', '2H']:
        return '1'
    elif timeFrame in ['1D', '1W', '2D']:
        return '1D'
    else:
        return '1D'


def fetch_data(SecurityName, timeFrame):
    print(f"getting bar data for {SecurityName} {timeFrame}")
    #Chart.spinner(True)
    manipulatedTimeFrame = time_frame_manipulation(timeFrame)

    available_symbols = fetch_available_symbols()
    if available_symbols is None:
        #Chart.spinner(False)
        return None

    if SecurityName not in available_symbols:
        print(f"Symbol {SecurityName} not found in available symbols")
        #Chart.spinner(False)
        return None

    url = 'https://localhost:4000/api/getcompanyohlc?symbol=' + SecurityName + '&timeFrame=' + manipulatedTimeFrame
    response = requests.get(url,verify=False)
    if response.status_code == 200:
        #Chart.spinner(False)
        return process_json_data(response.json(),timeFrame,manipulatedTimeFrame)
    else:
        #Chart.spinner(False)
        print(f"Failed to fetch data from {url}")
        return None

def fetch_available_symbols():
    url = 'https://localhost:4000/api/availablenepsecompanies'
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        json_data = response.json()
        if 'data' in json_data:
            return json_data['data']
        else:
            print("Data field not found in JSON response.")
            return None
    else:
        print(f"Failed to fetch data from {url}")
        return None

def on_search(chart, searched_string):
    new_data = fetch_data(searched_string, chart.topbar['timeframe'].value)
    if new_data is None or new_data.empty:
        return
    chart.topbar['symbol'].set(searched_string)
    chart.watermark(searched_string + ' ' + chart.topbar['timeframe'].value)
    chart.set(new_data)

def on_timeframe_selection(chart):
    new_data = fetch_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
    if new_data is None or new_data.empty:
        return
    chart.watermark(chart.topbar['symbol'].value + ' ' + chart.topbar['timeframe'].value)
    chart.set(new_data, True)

def take_screenshot(key):
    img = Chart.screenshot()
    t = time.time()
    with open(f"screenshot-{t}.png", 'wb') as f:
        f.write(img)

async def fetch_others_data():
    print("fetching other data")

def refresh_data(chart):
    new_data = fetch_data(Chart.topbar['symbol'].value, Chart.topbar['timeframe'].value)
    if new_data is None or new_data.empty:
        return
    chart.set(new_data, True)

def auto_refresh(chart): #bugged, use update chart from tick value
    def refresh_loop():
        while chart.topbar['autorefresh'].value == 'True':
            print("Auto Refreshing Data...")
            refresh_data(chart)
            sleep(10)
        print("Auto Refresh stopped.")

    refresh_thread = threading.Thread(target=refresh_loop)
    refresh_thread.start()

    while True:
        if chart.is_alive:
            print ("Chart is still open")
            sleep(1)
        else:
            print("Chart window closed. Stopping auto refresh.")
            refresh_thread.join()
            break

if __name__ == '__main__':
    global Chart
    Chart = Chart(toolbox=False, maximize=True, inner_width=0.65, inner_height=1, title='Nepse Chart')
    Chart.legend(visible = True, font_family = 'Trebuchet MS', ohlc = True, percent = True)
    line = Chart.create_line('SMA 50')
    Chart.grid(vert_enabled = True, horz_enabled = True)
    Chart.watermark(SecurityName + ' ' + timeFrame)
    Chart.topbar.menu('menu', ('File', 'Edit', 'View', 'Help'), default='File')
    Chart.topbar.textbox('symbol', SecurityName)
    Chart.topbar.switcher('timeframe', ('1Min', '5Min', '10Min', '15Min','1D', '2D', '1W'), default=timeFrame, func=on_timeframe_selection)
    Chart.events.search += on_search
    Chart.topbar.button('screenshot', 'Screenshot', func=take_screenshot)
    Chart.topbar.button('refresh', 'Manual Refresh', func=refresh_data)
    Chart.topbar.textbox('autorefreshtext', 'Autoreresh')
    #Chart.topbar.switcher('autorefresh', ('True', 'False'), default='False', func=auto_refresh)

    df = fetch_data(SecurityName, timeFrame)
    if df is None:
        print('No data to display')
        exit(1)

    sma_data = calculate_sma(df, period=50)
    line.set(sma_data)
    Chart.set(df)

    Chart.show(block=True)