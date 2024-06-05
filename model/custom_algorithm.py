import pandas as pd
import os

def custom_algorithm(symbol, timeframe, trailing_window=14):
    symbol_timeframe = f"{symbol}_{timeframe}"
    folder_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", f"{symbol}", f"{timeframe}")
    csv_path = os.path.join(folder_name, f"{symbol_timeframe}.csv")
    data = pd.read_csv(csv_path).dropna()

    data['diff'] = data['close'].diff()
    data['gain'] = data['diff'].where(data['diff'] > 0, 0)
    data['loss'] = -data['diff'].where(data['diff'] < 0, 0)

    avg_gain = data['gain'].rolling(window=trailing_window).mean()
    avg_loss = data['loss'].rolling(window=trailing_window).mean()

    rs = avg_gain / avg_loss
    prediction = 100 - (100 / (1 + rs))
    prediction_rounded = prediction.round()

    prediction_rounded = prediction_rounded.fillna(0).astype(int)

    result = pd.DataFrame({
        'date': data['date'],
        'close': data['close'],
        'value': prediction_rounded,
        'recommended': 'Hold'  # Initialize the 'recommended' column with 'Hold'
    })

    result = result[result['value'] != 0]

    # Sell condition
    sell_condition = (result['value'].shift(1) >= 75) & (result['value'] < 65)
    sell_condition = sell_condition.fillna(False)
    result.loc[sell_condition, 'recommended'] = 'Sell'

    # After a sell signal, set the next row's recommended value to 'Hold'
    result.loc[sell_condition.shift(-1).fillna(False), 'recommended'] = 'Hold'

    # Buy condition
    buy_condition = (result['value'].shift(1) < 35) & (result['value'] >= 45)
    result.loc[buy_condition, 'recommended'] = 'Buy'

    return result

data = custom_algorithm('NEPSE', 1, 30)
print(data)