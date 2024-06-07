import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_Cus(data, period_Cus=14):
    differencePrice = data['close'].diff()
    differencePriceValues = differencePrice.values

    current_average_positive = 0
    current_average_negative = 0
    Cus = []
    price_index = 0

    for difference in differencePriceValues[1:]:
        if difference > 0:
            positive_difference = difference
            negative_difference = 0
        elif difference < 0:
            negative_difference = np.abs(difference)
            positive_difference = 0
        else:
            negative_difference = 0
            positive_difference = 0

        if price_index < period_Cus:
            current_average_positive += (1 / period_Cus) * positive_difference
            current_average_negative += (1 / period_Cus) * negative_difference

            if price_index == (period_Cus - 1):
                if current_average_negative != 0:
                    Cus.append(100 - 100 / (1 + (current_average_positive / current_average_negative)))
                else:
                    Cus.append(100)
        else:
            current_average_positive = ((period_Cus - 1) * current_average_positive + positive_difference) / period_Cus
            current_average_negative = ((period_Cus - 1) * current_average_negative + negative_difference) / period_Cus

            if current_average_negative != 0:
                Cus.append(100 - 100 / (1 + (current_average_positive / current_average_negative)))
            else:
                Cus.append(100)

        price_index += 1

    return pd.Series(data=Cus, index=data.index[period_Cus:])

def show_custom_Chart(SecurityName, timeFrame, period_Cus=14, max_points=500):

    folder_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", f"{SecurityName}", f"{timeFrame}")
    csv_path = os.path.join(folder_name, f"{SecurityName}_{timeFrame}.csv")

    if SecurityName is None or timeFrame is None:
        print("Error: SecurityName and timeFrame must be provided if df is None.")
        return

    # symbol_timeframe = f"{symbol}_{timeframe}"
    # folder_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", f"{symbol}", f"{timeframe}")
    # csv_path = os.path.join(folder_name, f"{symbol_timeframe}.csv")

    # Load data without parsing dates
    data = pd.read_csv(csv_path, parse_dates=['date'])

    # Trimming the data
    data = data.iloc[-max_points:].reset_index(drop=True)
    Cus_series = calculate_Cus(data, period_Cus=period_Cus)

    num_entries = Cus_series.values.size
    line30 = num_entries * [30]
    line75 = num_entries * [75]
    lines30_75 = pd.DataFrame({'30 limit': line30, '75 limit': line75}, index=Cus_series.index)

    # Plotting
    with plt.style.context('ggplot'):
        fig, ax1 = plt.subplots(figsize=(16, 8))

        # Plot Close Price
        ax1.plot(data['date'], data['close'], label='Close price')
        ax1.set_ylabel(SecurityName + ' price NRS', fontsize=12)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.set_ylim(0, 100)
        ax2.plot(data['date'][period_Cus:], [30]*len(data['date'][period_Cus:]), lw=1, linestyle='--', color='green', label='Buy Signal Line')
        ax2.plot(data['date'][period_Cus:], [75]*len(data['date'][period_Cus:]), lw=1, linestyle='--', color='red', label='Sell Signal Line')
        ax2.set_ylabel('Custom Algo Value Line')
        ax2.legend(loc='upper right')

        # Annotations
        for i in range(len(Cus_series)):
            if Cus_series.iloc[i] >= 75:
                ax1.annotate('', xy=(data['date'].iloc[Cus_series.index[i]], data['close'].iloc[Cus_series.index[i]]),
                            xytext=(0, 10), textcoords='offset points', arrowprops=dict(arrowstyle='-|>', color='red', mutation_scale=10, linewidth=2, linestyle='solid'))
            elif Cus_series.iloc[i] <= 30:
                ax1.annotate('', xy=(data['date'].iloc[Cus_series.index[i]], data['close'].iloc[Cus_series.index[i]]),
                            xytext=(0, -10), textcoords='offset points', arrowprops=dict(arrowstyle='-|>', color='green', mutation_scale=10, linewidth=2, linestyle='solid'))

        # Customize x-axis
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.show()

# def generate_signals(data, period_Cus=14, overbought=75, oversold=30):
#     Cus_series = calculate_Cus(data, period_Cus=period_Cus)
#     signals = pd.DataFrame(index=data.index)
#     signals['signal'] = 0.0

#     signals['signal'][period_Cus:] = np.where((Cus_series >= overbought) & (Cus_series.shift(1) < overbought), -1, signals['signal'])
#     signals['signal'][period_Cus:] = np.where((Cus_series <= oversold) & (Cus_series.shift(1) > oversold), 1, signals['signal'])

#     return signals

# def calculate_accuracy(data, signals):
#     accuracy = []
#     for i in range(1, len(signals)):
#         if signals['signal'][i] == 1:  # Buy signal
#             if data['close'][i] > data['close'][i - 1]:
#                 accuracy.append(True)
#             else:
#                 accuracy.append(False)
#         elif signals['signal'][i] == -1:  # Sell signal
#             if data['close'][i] < data['close'][i - 1]:
#                 accuracy.append(True)
#             else:
#                 accuracy.append(False)

#     return sum(accuracy) / len(accuracy) * 100


#show_custom_Chart('NEPSE', '1D', 14, max_points=400)

def save_symbol_model_value(symbol, timeframe, model, accuracy):
    global algo_names
    folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)))
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