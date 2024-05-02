import pandas as pd
import requests
from lightweight_charts import Chart
from io import StringIO
from time import sleep
import time, datetime
import asyncio
import nest_asyncio
nest_asyncio.apply()

def calculate_sma(df, period: int = 50):
    return pd.DataFrame({
        'time': df['date'],
        f'SMA {period}': df['close'].rolling(window=period).mean()
    }).dropna()

def fetch_data():
    ##print(f"getting bar data for {symbol} {timeframe}")
    Chart.spinner(True)

    url = 'https://localhost:4000/api/nepsedailyindex?format=csv'
    response = requests.get(url,verify=False)
    if response.status_code == 200:
        Chart.spinner(False)
        return pd.read_csv(StringIO(response.text))
    else:
        Chart.spinner(False)
        print(f"Failed to fetch data from {url}")
        return None

# def fetch_tick_data():
#     url = 'https://localhost:4000/api/intradayindexgraph'
#     response = requests.get(url,verify=False)
#     if response.status_code == 200:
#         return pd.read_csv(StringIO(response.text))
#     else:
#         print(f"Failed to fetch data from {url}")
#         return None

def on_timeframe_selection(chart):
    print("selected timeframe")
    print(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
    get_bar_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)

def on_search(chart, searched_string):
    get_bar_data(searched_string, chart.topbar['timeframe'].value)
    chart.topbar['symbol'].set(searched_string)

def take_screenshot(key):
    img = Chart.screenshot()
    t = time.time()
    with open(f"screenshot-{t}.png", 'wb') as f:
        f.write(img)

##not used now
# def restructure_data(data):
#     stock_data = data.get('stock_data', [])
#     df = pd.DataFrame(stock_data)
#     df['date'] = pd.to_datetime(df['date'])
#     df['open'] = df['open'].astype(float)
#     df['high'] = df['high'].astype(float)
#     df['low'] = df['low'].astype(float)
#     df['close'] = df['close'].astype(float)
#     df['volume'] = df['volume'].astype(float)
#     df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

#     df = df.drop_duplicates(subset=['date'])

#     df.insert(0, '', range(len(df)))

#     df.to_csv('stock_data.csv', index=False)
#     return df
    #df = pd.read_csv('minutebest.csv')


async def main():
    SecurityName = 'NEPSE'
    timeFrame = '1D'

    # df = fetch_data()
    # if df is None:
    #     print('No data to display')
    #     exit(1)

    #time.sleep(1)

    #Chart = Chart(toolbox=False, maximize=True, title='Nepse Chart'

    Chart = Chart(toolbox=False, maximize=True, inner_width=0.65, inner_height=1, title='Nepse Chart')
    Chart.legend(visible = True, font_family = 'Trebuchet MS', ohlc = True, percent = True)
    line = Chart.create_line('SMA 50')

    Chart.grid(vert_enabled = True, horz_enabled = True)
    Chart.watermark(SecurityName + timeFrame)
    Chart.topbar.menu('menu', ('File', 'Edit', 'View', 'Help'), default='File')
    Chart.topbar.textbox('symbol', SecurityName)
    Chart.topbar.switcher('timeframe', ('5 mins', '15 mins', '1 hour', '1D'), default=timeFrame, func=on_timeframe_selection)
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

    #Chart.show(block=True)
    await Chart.show_async(block = True)



if __name__ == '__main__':
    asyncio.run(main())