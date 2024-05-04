import pandas as pd
import requests
from lightweight_charts import Chart
from io import StringIO
from time import sleep
import time, datetime
import asyncio
import threading
from data_process import process_json_data

SecurityName = 'NEPSE'
timeFrame = '1D'

def calculate_sma(df, period: int = 50):
    return pd.DataFrame({
        'time': df['date'],
        f'SMA {period}': df['close'].rolling(window=period).mean()
    }).dropna()

def fetch_data():
    print(f"getting bar data for {SecurityName} {timeFrame}")
    Chart.spinner(True)

    available_symbols = fetch_available_symbols()
    if available_symbols is None:
        Chart.spinner(False)
        return None

    if SecurityName not in available_symbols:
        print(f"Symbol {symbol} not found in available symbols")
        Chart.spinner(False)
        return None

    url = 'https://localhost:4000/api/getcompanyohlc?symbol=' + SecurityName + '&timeFrame=' + timeFrame
    response = requests.get(url,verify=False)
    if response.status_code == 200:
        Chart.spinner(False)
        return process_json_data(response.json())
    else:
        Chart.spinner(False)
        print(f"Failed to fetch data from {url}")
        return None

    # url = 'https://localhost:4000/api/nepsedailyindex?format=csv'
    # response = requests.get(url,verify=False)
    # if response.status_code == 200:
    #     Chart.spinner(False)
    #     return pd.read_csv(StringIO(response.text))
    # else:
    #     Chart.spinner(False)
    #     print(f"Failed to fetch data from {url}")
    #     return None

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
    get_bar_data(searched_string, chart.topbar['timeframe'].value)
    chart.topbar['symbol'].set(searched_string)

def on_timeframe_selection(chart):
    print("selected timeframe")
    print(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
    get_bar_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)

def on_timeframe_selection(timeframe):
    print(f"Timeframe: {timeframe}")

def take_screenshot(key):
    img = Chart.screenshot()
    t = time.time()
    with open(f"screenshot-{t}.png", 'wb') as f:
        f.write(img)

async def fetch_others_data():
    print("fetching other data")
    await asyncio.sleep(1)
    print("fetching other data done")

def run_main():
    global Chart
    Chart = Chart(toolbox=False, maximize=True, inner_width=0.65, inner_height=1, title='Nepse Chart')
    Chart.legend(visible = True, font_family = 'Trebuchet MS', ohlc = True, percent = True)
    line = Chart.create_line('SMA 50')

    Chart.grid(vert_enabled = True, horz_enabled = True)
    Chart.watermark(SecurityName + timeFrame)
    Chart.topbar.menu('menu', ('File', 'Edit', 'View', 'Help'), default='File')
    Chart.topbar.textbox('symbol', SecurityName)
    Chart.topbar.switcher('timeframe', ('1 mins','5 mins', '15 mins', '1 hour', '1D'), default=timeFrame, func=on_timeframe_selection)
    Chart.events.search += on_search
    Chart.topbar.button('screenshot', 'Screenshot', func=take_screenshot)
    Chart.topbar.button('refresh', 'Refresh', func=fetch_data)

    df = fetch_data()
    if df is None:
        print('No data to display')
        exit(1)

    sma_data = calculate_sma(df, period=50)
    line.set(sma_data)
    Chart.set(df)

    Chart.show(block=True)


async def main():
    threading.Thread(target=run_main).start()
    await fetch_others_data()

if __name__ == '__main__':
    asyncio.run(main())
