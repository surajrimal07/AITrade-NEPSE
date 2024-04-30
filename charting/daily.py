import pandas as pd
import requests
from lightweight_charts import Chart
from io import StringIO
from time import sleep


def calculate_sma(df, period: int = 50):
    return pd.DataFrame({
        'time': df['date'],
        f'SMA {period}': df['close'].rolling(window=period).mean()
    }).dropna()

def fetch_data():
    url = 'https://localhost:4000/api/nepsedailyindex?format=csv'
    response = requests.get(url,verify=False)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        print(f"Failed to fetch data from {url}")
        return None

def fetch_tick_data():
    url = 'https://localhost:4000/api/intradayindexgraph'
    response = requests.get(url,verify=False)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        print(f"Failed to fetch data from {url}")
        return None

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


if __name__ == '__main__':
    df = fetch_data()

    chart = Chart(toolbox=False, maximize=True, title='Nepse Chart')
    chart.legend(visible=True)
    chart.set(df)
    line = chart.create_line('SMA 50')
    sma_data = calculate_sma(df, period=50)
    line.set(sma_data)
    chart.show(block=True)

