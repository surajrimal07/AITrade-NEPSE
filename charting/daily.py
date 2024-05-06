import pandas as pd
import requests
from lightweight_charts import Chart
from io import StringIO
from time import sleep
import time
import datetime
import asyncio
import threading
from data_process import process_json_data
from requests.packages.urllib3.exceptions import InsecureRequestWarning

SecurityName = 'NEPSE'
timeFrame = '1D'
autorefresh = False

# Disable SSL warnings on localhost with self signed certificate
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

async def calculate_sma(df, period: int = 50):
    return pd.DataFrame({
        'time': df['date'],
        f'SMA {period}': df['close'].rolling(window=period).mean()
    }).dropna()

def time_frame_manipulation(timeFrame):
    if timeFrame.endswith('Min'):
        return (timeFrame[:-3])
    else:
        return '1D'


async def fetch_data(SecurityName, timeFrame):
    print(f"getting bar data for {SecurityName} {timeFrame}")
    manipulatedTimeFrame = time_frame_manipulation(timeFrame)

    # available_symbols = fetch_available_symbols()
    # if available_symbols is None:
    #     return None

    # if SecurityName not in available_symbols:
    #     print(f"Symbol {SecurityName} not found in available symbols")
    #     return None

    url = 'https://localhost:4000/api/getcompanyohlc?symbol=' + SecurityName + '&timeFrame=' + manipulatedTimeFrame
    response = requests.get(url,verify=False)
    if response.status_code == 200:
        return process_json_data(response.json(),manipulatedTimeFrame)
    else:
        print(f"Failed to fetch data from {url}, probably the symbol is not available.")
        return None

async def fetch_available_symbols():
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

async def on_search(chart, searched_string):
    new_data = await fetch_data(searched_string, chart.topbar['timeframe'].value)
    if new_data is None or new_data.empty:
        return
    chart.topbar['symbol'].set(searched_string)
    chart.watermark(searched_string + ' ' + chart.topbar['timeframe'].value)
    chart.set(new_data)

async def on_timeframe_selection(chart):
    new_data = await fetch_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
    if new_data is None or new_data.empty:
        return
    chart.watermark(chart.topbar['symbol'].value + ' ' + chart.topbar['timeframe'].value)
    chart.set(new_data, True)

async def take_screenshot(key):
    img = Chart.screenshot()
    t = time.time()
    with open(f"screenshot-{t}.png", 'wb') as f:
        f.write(img)

async def fetch_others_data():
    print("fetching other data")

async def refresh_data(chart):
    new_data = await fetch_data(Chart.topbar['symbol'].value, Chart.topbar['timeframe'].value)
    if new_data is None or new_data.empty:
        return
    chart.set(new_data, True)

auto_refresh_task = None

async def auto_refresh(chart):
    global auto_refresh_task

    if auto_refresh_task and not auto_refresh_task.done():
        auto_refresh_task.cancel()

    async def refresh():
        while chart.is_alive and chart.topbar['autorefresh'].value == 'True':
            await asyncio.sleep(5 - (datetime.datetime.now().microsecond / 1_000_000))
            new_data = await fetch_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
            if new_data is not None and not new_data.empty:
                chart.set(new_data, True)

    auto_refresh_task = asyncio.create_task(refresh())

async def main():
    global Chart
    Chart = Chart(toolbox=False, maximize=True, inner_width=0.65, inner_height=1, title='Nepse Chart')
    Chart.legend(visible = True, font_family = 'Trebuchet MS', ohlc = True, percent = True)
    line = Chart.create_line('SMA 50')
    Chart.grid(vert_enabled = True, horz_enabled = True)
    Chart.watermark(SecurityName + ' ' + timeFrame)
    Chart.topbar.menu('menu', ('File', 'Edit', 'View', 'Help'), default='File')
    Chart.topbar.textbox('symbol', SecurityName)
    Chart.topbar.switcher('timeframe', ('1Min', '5Min', '10Min', '15Min','1D', '1W'), default=timeFrame, func= on_timeframe_selection)
    Chart.events.search += on_search
    Chart.topbar.button('screenshot', 'Screenshot', func= take_screenshot)
    Chart.topbar.button('refresh', 'Manual Refresh', func= refresh_data)
    Chart.topbar.textbox('autorefreshtext', 'Autoreresh')
    Chart.topbar.switcher('autorefresh', ('True', 'False'), default='False', func= auto_refresh)

    df = await fetch_data(SecurityName, timeFrame)
    if df is None:
        print('No data to display')
        exit(1)

    sma_data = await calculate_sma(df, period=50)
    line.set(sma_data)

    Chart.set(df)
    await asyncio.gather(Chart.show_async(block=True), auto_refresh(Chart))

if __name__ == '__main__':
    asyncio.run(main())
