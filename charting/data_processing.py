import json
import pandas as pd

# Load the raw JSON data line by line
timestamps = []
open_prices = []
high_prices = []
low_prices = []
close_prices = []
volumes = []


with open('minute.json', 'r') as file:
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

df = pd.DataFrame({'Timestamp': date_times})

df = pd.DataFrame({
    'date': date_times.strftime('%Y-%m-%d %H:%M:%S'),  # Format date without time part
    'open': open_prices,
    'high': high_prices,
    'low': low_prices,
    'close': close_prices,
    'volume': volumes
})

df = df.drop_duplicates(subset=['date'])

# Save the DataFrame with the formatted date as a CSV file
df.to_csv('minutebest.csv', index_label='')
#df.to_json('dailytest1.json', orient='records', lines=True)

# Print the top content of the DataFrame
print(df.head())
