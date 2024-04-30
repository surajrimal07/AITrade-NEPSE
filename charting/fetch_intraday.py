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
