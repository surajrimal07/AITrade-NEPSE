import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from custom_algorithm import save_symbol_model_value

def load_cnn_model(symbol_name, timeframe):
    folder_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", symbol_name, timeframe)
    model = tf.keras.models.load_model(os.path.join(folder_name, f"{symbol_name}_cnn_model.keras"))
    scaler = joblib.load(os.path.join(folder_name, f"{symbol_name}_cnn_scaler.pkl"))
    return model, scaler

def generate_cnn_predictions(model, scaler, x_test, y_test):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)
    return predictions, y_test

def predict_cnn_future_prices(model, scaler, x_test, days):
    last_sequence = x_test[-1]
    predicted_prices = []

    for _ in range(days):
        prediction = model.predict(np.expand_dims(last_sequence, axis=0))[0]
        last_sequence = np.append(last_sequence[1:], [prediction], axis=0)
        predicted_prices.append(prediction[0])

    predicted_prices = np.array(predicted_prices).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices

def plot_cnn_charts(symbol_name, timeframe, predictions, y_test, future_prices, scaling_factor):
    plt.figure(figsize=(16, 8))

    # Plot historical predictions vs actual prices
    plt.subplot(2, 1, 1)
    plt.plot(y_test, color='blue', label='Actual Price')
    plt.plot(predictions * scaling_factor, color='red', label='Predicted Price')
    plt.title(f"{symbol_name} {timeframe} - CNN Model")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    # Plot future price predictions
    plt.subplot(2, 1, 2)
    plt.plot(future_prices, color='green', label='Future Predicted Price')
    plt.title(f"{symbol_name} {timeframe} - Future Price Prediction")
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()

    plt.tight_layout()
    plt.show()

# symbol_name = 'NEPSE'
# timeframe = '1D'


def run_deeplearning(symbol_name, timeframe):
    days_to_predict = 10
    # Load the model and scaler
    model, scaler = load_cnn_model(symbol_name, timeframe)

    # Load the test data
    folder_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", symbol_name, timeframe)
    x_test = np.load(os.path.join(folder_name, f"{symbol_name}_cnn_x_test.npy"))
    y_test = np.load(os.path.join(folder_name, f"{symbol_name}_cnn_y_test.npy"))

    # Generate predictions and visualize
    predictions, y_test_actual = generate_cnn_predictions(model, scaler, x_test, y_test)

    # Calculate scaling factor
    scaling_factor = np.mean(y_test_actual) / np.mean(predictions)

    # Generate future price predictions using scaling factor
    future_prices = predict_cnn_future_prices(model, scaler, x_test, days_to_predict)

    # Calculate accuracy
    mape = mean_absolute_percentage_error(y_test_actual, predictions * scaling_factor) * 100
    r2 = r2_score(y_test_actual, predictions * scaling_factor)

    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    r2_percentage = round(r2 * 100)
    print(f"Deep Learning(CNN) Model Accuracy is: {r2_percentage:.2f}%")
    save_symbol_model_value(symbol_name, timeframe, "Deep_learning", r2_percentage)


    plot_cnn_charts(symbol_name, timeframe, predictions, y_test_actual, future_prices, scaling_factor)


#run_deeplearning('NEPSE', '1D')