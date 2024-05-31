import requests
from global_var import *
import os
from data_process import process_json_data

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

async def fetch_data(SecurityName, timeFrame):
    global baseUrl, time_frame_manipulation
    print(f"getting bar data for {SecurityName} {timeFrame}")
    manipulatedTimeFrame = time_frame_manipulation(timeFrame)
    url = baseUrl+'/getcompanyohlc?symbol=' + SecurityName + '&timeFrame=' + manipulatedTimeFrame

    response = requests.get(url,verify=False)
    if response.status_code == 200:
        symbol_data = process_json_data(response.json(),manipulatedTimeFrame)

        folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", f"{SecurityName}", timeFrame)
        os.makedirs(folder_path, exist_ok=True)

        csv_path = os.path.join(folder_path, f"{SecurityName}_{timeFrame}.csv")
        symbol_data.to_csv(csv_path, index=False)

        return symbol_data
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