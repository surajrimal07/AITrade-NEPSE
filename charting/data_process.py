import pandas as pd
import json
import logging
import pandas as pd

last_price = None

logging.basicConfig(level=logging.INFO)

def process_json_data(raw_data,timeFrame):
    global last_price

    timestamps = raw_data['t']
    open_prices = raw_data['o']
    high_prices = raw_data['h']
    low_prices = raw_data['l']
    close_prices = raw_data['c']
    volumes = raw_data['v']

    date_times = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert('Asia/Kathmandu')

    df = pd.DataFrame({
    'date': pd.to_datetime(date_times.strftime('%Y-%m-%d' if timeFrame == '1D' else '%Y-%m-%d %H:%M:%S')),
    'open': open_prices,
    'high': high_prices,
    'low': low_prices,
    'close': close_prices,
    'volume': volumes
})

    df = df.drop_duplicates(subset=['date'])

    last_price = df['close'].iloc[-1]

    return df


def get_last_price():
    global last_price
    return last_price