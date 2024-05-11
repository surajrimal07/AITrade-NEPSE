import pandas as pd
import requests
from lightweight_charts import Chart
from io import StringIO
from time import sleep
import time
import datetime
from datetime import date
import asyncio
import threading
import numpy as np
from sklearn.linear_model import LinearRegression
from data_process import process_json_data
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from auth import login

SecurityName = 'NEPSE'
timeFrame = '1D'
autorefresh = False
auto_refresh_task = None

SubchartSecurity = 'NEPSE'
SubchartTimeFrame = '1'
baseUrl = 'https://localhost:4000/api/'

# Disable SSL warnings on localhost with self signed certificate
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

async def calculate_sma(df, period: int = 50):
    return pd.DataFrame({
        'time': df['date'],
        f'SMA {period}': df['close'].rolling(window=period).mean()
    }).dropna()

async def login_user():
    login_data = login()
    if login_data:
        return login_data


def time_frame_manipulation(timeFrame):
    time_frame_mapping = {
        '1Min': '1',
        '5Min': '5',
        '10Min': '10',
        '15Min': '15',
        '1': '1',
        '1W': '1W',
    }
    return time_frame_mapping.get(timeFrame, '1D')

async def perform_regression_analysis(df):
    y = np.array(df["close"])
    x = np.linspace(1, len(y), len(y)).reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(x, y)
    next_index = reg.predict(np.array(len(y) + 1).reshape(1, -1))[0]
    next_index = round(next_index, 2)

    today = date.today()
    date_str = date.isoformat(today)

    regression_equation = f"{reg.coef_[0]:.2f}x + {reg.intercept_:.2f}"
    return next_index, date_str, regression_equation, reg.coef_[0], reg.intercept_

async def fetch_data(SecurityName, timeFrame):
    print(f"getting bar data for {SecurityName} {timeFrame}")
    manipulatedTimeFrame = time_frame_manipulation(timeFrame)

    #url = 'https://localhost:4000/api/getcompanyohlc?symbol=' + SecurityName + '&timeFrame=' + manipulatedTimeFrame
    url = 'https://api.zorsha.com.np/api/getcompanyohlc?symbol=' + SecurityName + '&timeFrame=' + manipulatedTimeFrame

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
    print(f'Search Text: "{searched_string}" | Chart/SubChart ID: "{chart.id}"')
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
    new_data = await fetch_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
    if new_data is None or new_data.empty:
        return
    chart.set(new_data, True)

async def auto_refresh(chart):
    global auto_refresh_task

    # if auto_refresh_task and not auto_refresh_task.done():
    #     auto_refresh_task.cancel()

    async def refresh(chart): #chart.is_alive and chart.topbar['autorefresh'].value == 'True':
        while chart.topbar['autorefresh'].value == 'True':
            await asyncio.sleep(10 - (datetime.datetime.now().microsecond / 1_000_000))
            new_data = await fetch_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
            if new_data is not None and not new_data.empty:
                chart.set(new_data, True)

    auto_refresh_task = asyncio.create_task(refresh(chart))

def on_row_click(row):
    row['PL'] = round(row['PL']+1, 2)
    row.background_color('PL', 'green' if row['PL'] > 0 else 'red')

    table.footer[1] = row['Ticker']

async def main():
    global Chart
    user = await login_user()
    Chart = Chart(toolbox=False, maximize=True, inner_width=0.7, inner_height=1, title='Nepse Chart')
    Chart.legend(visible = True, font_family = 'Trebuchet MS', ohlc = True, percent = True)
    line = Chart.create_line('SMA 50')
    Chart.grid(vert_enabled = True, horz_enabled = True)
    Chart.watermark(SecurityName + ' ' + timeFrame)
    #Chart.topbar.menu('menu', ('Menu','File' 'Edit', 'View', 'Help'), default='Menu')
    Chart.topbar.button('login', 'login', func= login_user())
    Chart.topbar.textbox('username', user['name'])
    Chart.topbar.textbox('symbol', SecurityName)

    #Chart.topbar.menu('timeframe', ('Timeframe','1Min', '5Min', '10Min', '15Min','1D'), default='Timeframe',func= on_timeframe_selection)
    Chart.topbar.switcher('timeframe', ('1Min', '5Min', '10Min', '15Min','1D'), default=timeFrame, func= on_timeframe_selection)
    Chart.events.search += on_search
    Chart.topbar.button('screenshot', 'Screenshot', func= take_screenshot)
   # Chart.topbar.button('refresh', 'Refresh', func= refresh_data)
    Chart.topbar.textbox('autorefreshtext', 'autorefresh')
    Chart.topbar.switcher('autorefresh', ('True', 'False'), default='False', func= auto_refresh)
    Chart.topbar.textbox('autoretradetext', 'autortrade')
    Chart.topbar.switcher('autotrade', ('True', 'False'), default='False', func= auto_refresh)


    df = await fetch_data(SecurityName, timeFrame)
    if df is None:
        print('No data to display')
        exit(1)

    sma_data = await calculate_sma(df, period=50)
    line.set(sma_data)

#subchart
    subchart = Chart.create_subchart(width=0.3, height=0.4)
    subchart.watermark(SubchartSecurity + ' ' + SubchartTimeFrame)
    subchart.topbar.textbox('symbol', SubchartSecurity)
    subchart.topbar.switcher('timeframe', ('1', '5', '10'), default=SubchartTimeFrame, func= on_timeframe_selection)
    subchart.events.search += on_search
    #subchart.topbar.button('refresh', 'Refresh', func= refresh_data)
    subchart.topbar.textbox('autorefreshtext', 'autorefresh')
    subchart.topbar.switcher('autorefresh', ('True', 'False'), default='False', func= auto_refresh)

    df2 = await fetch_data(SubchartSecurity, SubchartTimeFrame)

    if df2 is None:
        print('No data to display')
        exit(1)

    # sub_line = subchart.create_line('SMA 50')
    # sma_subchart = await calculate_sma(df2, period=50)
    # sub_line.set(sma_subchart)

##end of subchart

#table
    table = Chart.create_table(width=0.3, height=0.2,
                  headings=('Ticker', 'Quantity', 'Status', '%', 'PL'),
                  widths=(0.2, 0.1, 0.2, 0.2, 0.3),
                  alignments=('center', 'center', 'right', 'right', 'right'),
                  position='left', func=on_row_click)

    table.format('PL', f'£ {table.VALUE}')
    table.format('%', f'{table.VALUE} %')

    table.new_row('SPY', 3, 'Submitted', 0, 0)
    table.new_row('AMD', 1, 'Filled', 25.5, 105.24)
    table.new_row('NVDA', 2, 'Filled', -0.5, -8.24)

    table.footer(2)
    table.footer[0] = 'Selected:'

#end of table

#perform regression analysis
    next_index, date_str, regression_equation, slope, intercept = await perform_regression_analysis(df)
    x_values = np.array(range(1, len(df) + 2))
    y_values = slope * x_values + intercept
    regressionline = Chart.create_line('Regression Line', color='blue')
    #regressionline.set(pd.DataFrame({'time': df['date'], 'value': y_values}).dropna())
#    Chart.marker('Predicted Next Index Value')



#end of regression analysis

    Chart.set(df)
    subchart.set(df2)

    Chart.marker(text=f"Predicted Next Index Value")

    await asyncio.gather(Chart.show_async(block=True), auto_refresh(Chart))

if __name__ == '__main__':
    asyncio.run(main())
