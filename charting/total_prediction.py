
from other_data import fetch_prediction
from regression import regression_plot
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
from custom_algorithm import calculate_Cus

async def buy_sell_tradingBot(SecurityName="NEPSE", timeFrame='1D'):
    overallPrediction = ''

    folder_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", f"{SecurityName}", f"{timeFrame}")
    csv_path = os.path.join(folder_name, f"{SecurityName}_{timeFrame}.csv")

    if SecurityName is None or timeFrame is None:
        print("Error: SecurityName and timeFrame must be provided if df is None.")
        return

    df = pd.read_csv(csv_path, parse_dates=['date'])
    df.drop(['high', 'open', 'low', 'volume'], axis=1, inplace=True)

    cusResult = calculate_Cus(df)

    regressionPrediction = regression_plot(SecurityName, timeFrame, only_data=True)
    accuracy_percent, next_index = regressionPrediction

    prediction_data = await fetch_prediction()
    if prediction_data:
        overallPrediction = prediction_data[1]

    if len(cusResult) < 2:
        return 'Wait and watch recommended'

    if 30 < cusResult.iloc[-1] < 50 and cusResult.iloc[-1] > cusResult.iloc[-2]:
        if overallPrediction > 1:
            if accuracy_percent > 60 and next_index > df['close'].iloc[-1]:
                return 'Buy'
        else:
            return 'Wait and watch recommended'
    elif 70 < cusResult.iloc[-1] < 100 and cusResult.iloc[-1] < cusResult.iloc[-2]:
        if overallPrediction < 1:
            if accuracy_percent > 60 and next_index < df['close'].iloc[-1]:
                return 'Sell'
        else:
            return 'Wait and watch recommended'

    return 'Wait and watch recommended'

