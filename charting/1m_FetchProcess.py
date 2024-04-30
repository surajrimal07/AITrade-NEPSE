import pandas as pd
import requests
from io import StringIO
import pytz

def fetch_tick_data():
    url = 'https://localhost:4000/api/intradayindexgraph'
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        json_data = response.json()
        if "data" in json_data:
            data = json_data["data"]
            df = pd.DataFrame(data, columns=["time", "index"])
            df.rename(columns={"index": "price"}, inplace=True)
            df["time"] = pd.to_datetime(df["time"])


            # Convert time to Kathmandu timezone
            df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f") + "+05:45"


            df.index.name = ""
            df.to_csv("time_price_data.csv")
            return df
        else:
            print(f"No 'data' field found in JSON response from {url}")
            return None
    else:
        print(f"Failed to fetch data from {url}")
        return None

# Example usage:
fetch_tick_data()