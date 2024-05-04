import pandas as pd
import json

def process_json_data(raw_data):
    # Initialize lists to store data
    timestamps = raw_data['t']
    open_prices = raw_data['o']
    high_prices = raw_data['h']
    low_prices = raw_data['l']
    close_prices = raw_data['c']
    volumes = raw_data['v']

    # Convert Unix epoch timestamps to proper datetime format
    date_times = pd.to_datetime(timestamps, unit='s')

    # Create a list of dictionaries for each record
    data = []
    seen_dates = set()  # Keep track of seen dates to remove duplicates
    for date, open_price, high, low, close, volume in zip(
        date_times.strftime('%Y-%m-%d'),
        open_prices,
        high_prices,
        low_prices,
        close_prices,
        volumes
    ):
        if date not in seen_dates:
            data.append({
                'date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
            seen_dates.add(date)

    return pd.DataFrame(data)
