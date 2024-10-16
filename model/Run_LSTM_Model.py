
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.dates as mdates
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from custom_algorithm import save_symbol_model_value
from sklearn.metrics import mean_absolute_percentage_error, r2_score


def load_and_prepare_data(symbol_timeframe):
    symbol, timeframe = symbol_timeframe.split('_')
    folder_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", f"{symbol}", f"{timeframe}")

    # Load the dataset
    csv_path = os.path.join(folder_name, f"{symbol_timeframe}.csv")
    df = pd.read_csv(csv_path).dropna()
    df.columns = df.columns.str.capitalize()
    df = df.set_index('Date')
    data_close = df.filter(['Close'])
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

def generate_predictions(x_train, y_train, x_test, y_test, scaler, model, symbol, timeframe):
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

    # Calculate R-squared score
    r2 = r2_score(y_test_original, test_predictions)
    r2_percentage = round(r2 * 100)
    print(f"LSMT Model Accuracy is: {r2_percentage:.2f}%")
    save_symbol_model_value(symbol, timeframe, "LSMT", r2_percentage)

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

    plt.switch_backend('TkAgg')

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

    # plt.subplots.adjust(bottom=0.2)
    ax1.legend()
    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.plot(formatted_latest_dates, latest_data, label='Latest 80 data points')

    day_locator = mdates.DayLocator(interval=2)
    ax2.xaxis.set_major_locator(day_locator)
    ax2.plot(x_labels, predicted_prices, label=f'Prediction for {prediction_duration} {prediction_label}')

    ax2.set_title(f'Recent Data and Prediction for {prediction_duration} {prediction_label}')
    ax2.set_xlabel('Date' if prediction_label == 'Days' else 'Time')
    ax2.set_ylabel('Close Price')
    plt.xticks(rotation=90, ha='right')
    ax2.legend()
    plt.tight_layout()
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.plot(x_labels, predicted_prices, label='Predicted Prices', color='red')

    ax3.set_title(f'Predicted Prices for Next {prediction_duration} {prediction_label}')
    ax3.set_xlabel(f'Prediction {prediction_label}')
    ax3.set_ylabel('Close Price')
    plt.xticks(rotation=30, fontsize=10)
    ax3.legend()
    plt.tight_layout()
    plt.show()

    plt.close('all')

def showLSMT(symbol, timeframe):
    df, dataset, training_data_len, train_dates, test_dates, scaler, x_train, y_train, x_test, y_test, model = load_and_prepare_data(f"{symbol}_{timeframe}")

    train_predictions, y_train_original, test_predictions, y_test_original = generate_predictions(x_train, y_train, x_test, y_test, scaler, model, symbol, timeframe)

    latest_dates = df.index[-80:]
    latest_data = df['Open'].values[-80:]

    predicted_prices, prediction_label, prediction_duration = predict_future_prices(dataset, scaler, model, timeframe)

    plot_charts(train_dates, y_train_original, train_predictions, test_dates, y_test_original, test_predictions, latest_dates, latest_data, predicted_prices, prediction_label, prediction_duration, timeframe)


def run_model_and_show_dialog(symbol, timeframe):
    showLSMT(symbol, timeframe)

#run_model_and_show_dialog('NEPSE', '1D')
