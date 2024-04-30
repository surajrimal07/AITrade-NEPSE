import json
import pandas as pd
import requests
from io import StringIO

def fetch_data():
    url = 'https://www.nepsealpha.com/trading/1/history?force_key=fhgjgjgjggfert&symbol=NEPSE&from=766368000&to=1713225600&resolution=1D&pass=ok&fs=fhgjgjgjggfert&shouldCache=1'
    response = requests.get(url)
    if response.status_code == 200:
        json_data = json.loads(response.text)

        # Save the JSON data to a file
        with open('response_data.json', 'w') as json_file:
            json.dump(json_data, json_file)

        return json_data
    else:
        print(f"Failed to fetch data from {url}")
        return None

# Fetch data from the API
data = fetch_data()