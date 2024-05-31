# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import joblib
# import mpld3
# import math
# import matplotlib.pyplot as plt
# from mpld3 import plugins
# from matplotlib.dates import DayLocator, DateFormatter
# from sklearn.model_selection import train_test_split

# symbol_timeframe = 'NEPSE_1'  # Example: 'NEPSE_1D', 'NEPSE_1M', etc.
# symbol, timeframe = symbol_timeframe.split('_')
# folder_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", f"{symbol}", f"{timeframe}")

# # Load the dataset
# csv_path = os.path.join(folder_name, f"{symbol_timeframe}.csv")
# df = pd.read_csv(csv_path).dropna()
# df.columns = df.columns.str.capitalize()
# df = df.set_index('Date')
# data_close = df.filter(['Close'])
# dataset = data_close.values

# training_data_len = math.ceil(len(dataset) * .8)
# train_dates = df.index[10:training_data_len]
# test_dates = df.index[training_data_len:]

# scaler = joblib.load(os.path.join(folder_name, f"{symbol}_scaler.pkl"))
# x_train = np.load(os.path.join(folder_name, f"{symbol}_x_train.npy"))
# y_train = np.load(os.path.join(folder_name, f"{symbol}_y_train.npy"))
# x_test = np.load(os.path.join(folder_name, f"{symbol}_x_test.npy"))
# y_test = np.load(os.path.join(folder_name, f"{symbol}_y_test.npy"))

# # Load the trained model
# model = tf.keras.models.load_model(os.path.join(folder_name, f"{symbol}_model.keras"))

# # Generate predictions
# train_predictions = model.predict(x_train)
# train_predictions = scaler.inverse_transform(train_predictions)

# y_train_original = y_train.reshape(-1, 1)
# y_train_original = scaler.inverse_transform(y_train_original)

# test_predictions = model.predict(x_test)
# test_predictions = scaler.inverse_transform(test_predictions)

# y_test_original = y_test.reshape(-1, 1)

# # Calculate accuracy metrics
# train_mse = np.mean((train_predictions - y_train_original) ** 2)
# test_mse = np.mean((test_predictions - y_test_original) ** 2)
# print(f"Training MSE: {train_mse:.4f}, Testing MSE: {test_mse:.4f}")

# window_size = 80

# # Select the last 80 days of data
# last_80_days = dataset[-window_size:]

# # Scale the data
# last_80_days_scaled = scaler.transform(last_80_days)

# # Reshape the data for model prediction
# X_test = []
# X_test.append(last_80_days_scaled)
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# # Predict the next 10 days' stock prices
# predicted_prices = []

# for i in range(10):
#     predicted_price = model.predict(X_test)
#     predicted_prices.append(predicted_price[0][0])
#     # Update X_test for the next prediction
#     X_test = np.concatenate((X_test[:, 1:, :], predicted_price.reshape(-1, 1, 1)), axis=1)

# # Invert the scaling
# predicted_prices = np.array(predicted_prices).reshape(-1, 1)
# predicted_prices = scaler.inverse_transform(predicted_prices)

# print("Predicted Prices for Next 10 Days:", predicted_prices)

# latest_dates = df.index[-window_size:]
# latest_data = df['Close'].tail(window_size)

# x_labels = [f"Day {i + 1}" for i in range(len(predicted_prices))]

# # data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S').dt.date
# # data['date_label'] = data['date'].dt.strftime('%m-%d')



# fig1, ax1 = plt.subplots(figsize=(10, 6))
# ax1.plot(train_dates, y_train_original, label='Actual train', color='blue')
# ax1.plot(train_dates, train_predictions, label='Train Prediction', color='orange')
# ax1.plot(test_dates, y_test_original, label='Actual test data', color='green')
# ax1.plot(test_dates, test_predictions, label='Test Prediction', color='red')
# ax1.set_title('Actual Test vs Test Prediction')
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Price')
# plt.xticks(rotation=30,fontsize=10)

# plt.xticks(ticks=np.arange(0, len(test_dates), step=20),  # Show labels every 20th date
#            labels=[str(x.split(' ')[0])[:4] for x in test_dates[::20]])  # Extract year for every 20th date

# ax1.legend()
# plt.show()

# fig2, ax2 = plt.subplots(figsize=(10, 6))
# ax2.plot(latest_dates, latest_data, label='Latest 80 days data')
# ax2.plot(x_labels, predicted_prices, label='Prediction for 10 days')

# ax2.set_title('Recent Data and Prediction for 10 Days')
# ax2.set_xlabel('Date')
# ax2.set_ylabel('Close Price')
# plt.xticks(rotation=90, ha='right')
# ax2.legend()
# interactive_plot2 = mpld3.fig_to_html(fig2)
# plt.show()

# fig3, ax3 = plt.subplots(figsize=(10, 6))
# ax3.plot(x_labels, predicted_prices, label='Predicted Prices', color='red')

# ax3.set_title('Predicted Prices for Next 10 Days')
# ax3.set_xlabel('Prediction Days')
# ax3.set_ylabel('Price')
# plt.xticks(rotation=90, ha='right')
# ax3.legend()
# plt.show()



#good but date label issue
# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import joblib
# import mpld3
# import math
# import matplotlib.pyplot as plt
# from mpld3 import plugins
# from matplotlib.dates import DayLocator, DateFormatter
# from sklearn.model_selection import train_test_split

# symbol_timeframe = 'NEPSE_1'  # Example: 'NEPSE_1D', 'NEPSE_1M', etc.
# symbol, timeframe = symbol_timeframe.split('_')
# folder_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", f"{symbol}", f"{timeframe}")

# # Load the dataset
# csv_path = os.path.join(folder_name, f"{symbol_timeframe}.csv")
# df = pd.read_csv(csv_path).dropna()
# df.columns = df.columns.str.capitalize()
# df = df.set_index('Date')
# data_close = df.filter(['Close'])
# dataset = data_close.values

# training_data_len = math.ceil(len(dataset) * .8)
# train_dates = df.index[10:training_data_len]
# test_dates = df.index[training_data_len:]

# scaler = joblib.load(os.path.join(folder_name, f"{symbol}_scaler.pkl"))
# x_train = np.load(os.path.join(folder_name, f"{symbol}_x_train.npy"))
# y_train = np.load(os.path.join(folder_name, f"{symbol}_y_train.npy"))
# x_test = np.load(os.path.join(folder_name, f"{symbol}_x_test.npy"))
# y_test = np.load(os.path.join(folder_name, f"{symbol}_y_test.npy"))

# # Load the trained model
# model = tf.keras.models.load_model(os.path.join(folder_name, f"{symbol}_model.keras"))

# # Generate predictions
# train_predictions = model.predict(x_train)
# train_predictions = scaler.inverse_transform(train_predictions)

# y_train_original = y_train.reshape(-1, 1)
# y_train_original = scaler.inverse_transform(y_train_original)

# test_predictions = model.predict(x_test)
# test_predictions = scaler.inverse_transform(test_predictions)

# y_test_original = y_test.reshape(-1, 1)

# # Calculate accuracy metrics
# train_mse = np.mean((train_predictions - y_train_original) ** 2)
# test_mse = np.mean((test_predictions - y_test_original) ** 2)
# print(f"Training MSE: {train_mse:.4f}, Testing MSE: {test_mse:.4f}")

# window_size = 80

# # Select the last 80 data points
# last_80_points = dataset[-window_size:]

# # Scale the data
# last_80_points_scaled = scaler.transform(last_80_points)

# # Reshape the data for model prediction
# X_test = []
# X_test.append(last_80_points_scaled)
# X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# # Determine the prediction duration based on the timeframe
# if timeframe == '1D':
#     prediction_duration = 15  # Predict next 15 days
#     prediction_label = 'Days'
# else:
#     prediction_duration = 4 * 60  # Predict next 4 hours in minutes
#     prediction_label = 'Minutes'

# # Predict the next prices
# predicted_prices = []

# for i in range(prediction_duration):
#     predicted_price = model.predict(X_test)
#     predicted_prices.append(predicted_price[0][0])
#     # Update X_test for the next prediction
#     X_test = np.concatenate((X_test[:, 1:, :], predicted_price.reshape(-1, 1, 1)), axis=1)

# # Invert the scaling
# predicted_prices = np.array(predicted_prices).reshape(-1, 1)
# predicted_prices = scaler.inverse_transform(predicted_prices)

# print(f"Predicted Prices for Next {prediction_duration} {prediction_label}:", predicted_prices)

# latest_dates = df.index[-window_size:]
# latest_data = df['Close'].tail(window_size)

# x_labels = [f"{prediction_label[:-1]} {i + 1}" for i in range(len(predicted_prices))]

# fig1, ax1 = plt.subplots(figsize=(10, 6))
# ax1.plot(train_dates, y_train_original, label='Actual train', color='blue')
# ax1.plot(train_dates, train_predictions, label='Train Prediction', color='orange')
# ax1.plot(test_dates, y_test_original, label='Actual test data', color='green')
# ax1.plot(test_dates, test_predictions, label='Test Prediction', color='red')
# ax1.set_title('Actual Test vs Test Prediction')
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Price')
# plt.xticks(rotation=30, fontsize=10)

# plt.xticks(ticks=np.arange(0, len(test_dates), step=20),  # Show labels every 20th date
#            labels=[str(x.split(' ')[0])[:4] for x in test_dates[::20]])  # Extract year for every 20th date

# ax1.legend()
# plt.show()

# fig2, ax2 = plt.subplots(figsize=(10, 6))
# ax2.plot(latest_dates, latest_data, label='Latest 80 data points')
# ax2.plot(x_labels, predicted_prices, label=f'Prediction for {prediction_duration} {prediction_label}')

# ax2.set_title(f'Recent Data and Prediction for {prediction_duration} {prediction_label}')
# ax2.set_xlabel('Date' if timeframe == '1D' else 'Time')
# ax2.set_ylabel('Close Price')
# plt.xticks(rotation=90, ha='right')
# ax2.legend()
# interactive_plot2 = mpld3.fig_to_html(fig2)
# plt.show()

# fig3, ax3 = plt.subplots(figsize=(10, 6))
# ax3.plot(x_labels, predicted_prices, label='Predicted Prices', color='red')

# ax3.set_title(f'Predicted Prices for Next {prediction_duration} {prediction_label}')
# ax3.set_xlabel(f'Prediction {prediction_label}')
# ax3.set_ylabel('Price')
# plt.xticks(rotation=90, ha='right')
# ax3.legend()
# plt.show()

import calendar
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.dates as mdates
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

def load_and_prepare_data(symbol_timeframe):
    symbol, timeframe = symbol_timeframe.split('_')
    folder_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", f"{symbol}", f"{timeframe}")

    # Load the dataset
    csv_path = os.path.join(folder_name, f"{symbol_timeframe}.csv")
    df = pd.read_csv(csv_path).dropna()
    df.columns = df.columns.str.capitalize()
    df = df.set_index('Date')
    data_close = df.filter(['Open'])
    dataset = data_close.values

    training_data_len = math.ceil(len(dataset) * .8)
    train_dates = df.index[10:training_data_len]
    test_dates = df.index[training_data_len:]

    scaler = joblib.load(os.path.join(folder_name, f"{symbol}_scaler.pkl"))
    x_train = np.load(os.path.join(folder_name, f"{symbol}_x_train.npy"))
    y_train = np.load(os.path.join(folder_name, f"{symbol}_y_train.npy"))
    x_test = np.load(os.path.join(folder_name, f"{symbol}_x_test.npy"))
    y_test = np.load(os.path.join(folder_name, f"{symbol}_y_test.npy"))

    # Load the trained model
    model = tf.keras.models.load_model(os.path.join(folder_name, f"{symbol}_model.keras"))

    return df, dataset, training_data_len, train_dates, test_dates, scaler, x_train, y_train, x_test, y_test, model

def generate_predictions(x_train, y_train, x_test, y_test, scaler, model):
    # Generate predictions
    train_predictions = model.predict(x_train)
    train_predictions = scaler.inverse_transform(train_predictions)

    y_train_original = y_train.reshape(-1, 1)
    y_train_original = scaler.inverse_transform(y_train_original)

    test_predictions = model.predict(x_test)
    test_predictions = scaler.inverse_transform(test_predictions)

    y_test_original = y_test.reshape(-1, 1)

    # Calculate accuracy metrics
    train_mse = np.mean((train_predictions - y_train_original) ** 2)
    test_mse = np.mean((test_predictions - y_test_original) ** 2)
    print(f"Training MSE: {train_mse:.4f}, Testing MSE: {test_mse:.4f}")

    return train_predictions, y_train_original, test_predictions, y_test_original

def predict_future_prices(dataset, scaler, model, timeframe):
    window_size = 80

    # Select the last 80 data points
    last_80_points = dataset[-window_size:]

    # Scale the data
    last_80_points_scaled = scaler.transform(last_80_points)

    # Reshape the data for model prediction
    X_test = []
    X_test.append(last_80_points_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Determine the prediction duration based on the timeframe
    if timeframe == '1D':
        prediction_duration = 7  # Predict next 15 days
        prediction_label = 'Days'
    else:
        prediction_duration = 4 * 60  # Predict next 4 hours in minutes
        prediction_label = 'Minutes'

    # Predict the next prices
    predicted_prices = []

    for i in range(prediction_duration):
        predicted_price = model.predict(X_test)
        predicted_prices.append(predicted_price[0][0])
        # Update X_test for the next prediction
        X_test = np.concatenate((X_test[:, 1:, :], predicted_price.reshape(-1, 1, 1)), axis=1)

    # Invert the scaling
    predicted_prices = np.array(predicted_prices).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    #print(f"Predicted Prices for Next {prediction_duration} {prediction_label}:", predicted_prices)

    return predicted_prices, prediction_label, prediction_duration

def format_dates(dates, timeframe):
    if timeframe == '1D':
         return dates
    else:
        return [datetime.strptime(date, '%Y-%m-%d %H:%M:%S').strftime('%H:%M') for date in dates]

def plot_charts(train_dates, y_train_original, train_predictions, test_dates, y_test_original, test_predictions, latest_dates, latest_data, predicted_prices, prediction_label, prediction_duration, timeframe):
    formatted_train_dates = format_dates(train_dates, timeframe)
    formatted_test_dates = format_dates(test_dates, timeframe)
    formatted_latest_dates = format_dates(latest_dates, timeframe)

    x_labels = [f"{prediction_label[:-1]} {i + 1}" for i in range(len(predicted_prices))]

    fig1, ax1 = plt.subplots(figsize=(16, 10))
    ax1.plot(formatted_train_dates, y_train_original, label='Actual train', color='blue')
    ax1.plot(formatted_train_dates, train_predictions, label='Train Prediction', color='orange')
    ax1.plot(formatted_test_dates, y_test_original, label='Actual test data', color='green')
    ax1.plot(formatted_test_dates, test_predictions, label='Test Prediction', color='red')
    ax1.set_title('Actual Test vs Test Prediction')
    ax1.set_xlabel('Date' if timeframe == '1D' else 'Time')
    ax1.set_ylabel('Price')
    plt.xticks(rotation=30, fontsize=10)

    # Reduce the number of x-axis labels for better readability
    if timeframe == '1D':
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))

    else:
        ax1.set_xticks(np.arange(0, len(formatted_test_dates), max(1, len(formatted_test_dates) // 10)))
        ax1.set_xticklabels([formatted_test_dates[i] for i in range(0, len(formatted_test_dates), max(1, len(formatted_test_dates) // 10))])

    plt.subplots_adjust(bottom=0.2)
    ax1.legend()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(16, 10))
    ax2.plot(formatted_latest_dates, latest_data, label='Latest 80 data points')

    day_locator = mdates.DayLocator(interval=2)
    ax2.xaxis.set_major_locator(day_locator)
    ax2.plot(x_labels, predicted_prices, label=f'Prediction for {prediction_duration} {prediction_label}')

    ax2.set_title(f'Recent Data and Prediction for {prediction_duration} {prediction_label}')
    ax2.set_xlabel('Date' if prediction_label == 'Days' else 'Time')
    ax2.set_ylabel('Close Price')
    plt.xticks(rotation=90, ha='right')
    #plt.get_current_fig_manager().window.state('zoomed')
    ax2.legend()
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(16, 10))
    ax3.plot(x_labels, predicted_prices, label='Predicted Prices', color='red')

    ax3.set_title(f'Predicted Prices for Next {prediction_duration} {prediction_label}')
    ax3.set_xlabel(f'Prediction {prediction_label}')
    ax3.set_ylabel('Price')
    plt.xticks(rotation=90, ha='right')
    ax3.legend()
    plt.show()

def main():
    symbol_timeframe = 'NEPSE_1D'  # Example: 'NEPSE_1D', 'NEPSE_1M', etc.
    df, dataset, training_data_len, train_dates, test_dates, scaler, x_train, y_train, x_test, y_test, model = load_and_prepare_data(symbol_timeframe)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_predictions = executor.submit(generate_predictions, x_train, y_train, x_test, y_test, scaler, model)
        future_prices = executor.submit(predict_future_prices, dataset, scaler, model, symbol_timeframe.split('_')[1])

        for future in as_completed([future_predictions, future_prices]):
            if future == future_predictions:
                train_predictions, y_train_original, test_predictions, y_test_original = future.result()
            else:
                predicted_prices, prediction_label, prediction_duration = future.result()

        latest_dates = df.index[-80:]
        latest_data = df['Close'].tail(80)

        plot_charts(train_dates, y_train_original, train_predictions, test_dates, y_test_original, test_predictions, latest_dates, latest_data, predicted_prices, prediction_label, prediction_duration, symbol_timeframe.split('_')[1])

if __name__ == '__main__':
    main()

