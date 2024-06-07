import requests
from global_var import *
import os
import json
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
    manipulatedTimeFrame = time_frame_manipulation(timeFrame)
    url = baseUrl+'/getcompanyohlc?symbol=' + SecurityName + '&timeFrame=' + manipulatedTimeFrame

    response = requests.get(url,verify=False)
    if response.status_code == 200:
        symbol_data = process_json_data(response.json(),manipulatedTimeFrame)

        folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", f"{SecurityName}", manipulatedTimeFrame)
        os.makedirs(folder_path, exist_ok=True)

        csv_path = os.path.join(folder_path, f"{SecurityName}_{manipulatedTimeFrame}.csv")
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

async def fetch_tick_data(SecurityName, timeFrame):
    global time_frame_manipulation, baseUrl
    print(f"getting tick bar data for {SecurityName} {timeFrame}")
    manipulatedTimeFrame = time_frame_manipulation(timeFrame)
    url = baseUrl+'/getcompanyohlc?symbol=' + SecurityName + '&timeFrame=' + manipulatedTimeFrame + '&intradayupdate=true'

    response = requests.get(url,verify=False)
    if response.status_code == 200:
        symbol_data = process_json_data(response.json(),manipulatedTimeFrame)
        folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", f"{SecurityName}", manipulatedTimeFrame)
        os.makedirs(folder_path, exist_ok=True)

        csv_path = os.path.join(folder_path, f"{SecurityName}_{manipulatedTimeFrame}.csv")
        symbol_data.to_csv(csv_path, index=False)

        return symbol_data
    else:
        return None

def save_symbol_model_value(symbol, timeframe, model, accuracy):
    global algo_names
    folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(folder_path, exist_ok=True)
    json_path = os.path.join(folder_path, f"symbol_tf_accuracy.json")

    with open(json_path, 'r') as f:
        data = json.load(f)

    if symbol in data:
        if timeframe in data[symbol]:
            data[symbol][timeframe][model] = accuracy
        else:
            data[symbol][timeframe] = {model: accuracy}
    else:
        data[symbol] = {timeframe: {model: accuracy}}

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def fetch_symbol_model_value(symbol, timeframe):
    global algo_names
    folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(folder_path, exist_ok=True)
    json_path = os.path.join(folder_path, f"symbol_tf_accuracy.json")

    with open(json_path, 'r') as f:
        data = json.load(f)

        if symbol in data:
            if timeframe in data[symbol]:
                available_models = data[symbol][timeframe]
                trained_models = list(available_models.keys())
                missing_models = [model for model in algo_names if model not in trained_models]

                for model in missing_models:
                    available_models[model] = "N/A"

                return available_models
            else:
                return {model: "N/A" for model in algo_names}
        else:
            return {model: "N/A" for model in algo_names}


def checkIfModelExists(model_name, symbol_name, time_frame):
    model_data = fetch_symbol_model_value(symbol_name, time_frame)
    if model_name in model_data and model_data[model_name] != 'N/A':
        return True
    return False

# print(checkIfModelExists('LSMT','NEPSE','1D'))

#print(fetch_symbol_model_value('NEPSE','5'))