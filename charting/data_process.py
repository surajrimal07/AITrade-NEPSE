import pandas as pd
import json
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

def process_json_data(raw_data,timeFrame):
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
    print(timeFrame +' Data is available from date ' + str(df['date'].min()) + ' to ' + str(df['date'].max()))

    df = df.drop_duplicates(subset=['date'])

    return df