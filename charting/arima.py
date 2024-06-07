import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def arima_stock_prediction(SecurityName="NEPSE", timeFrame='1D', forecast_steps=50, save_plot=True):

    folder_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", f"{SecurityName}", f"{timeFrame}")
    csv_path = os.path.join(folder_name, f"{SecurityName}_{timeFrame}.csv")

    if SecurityName is None or timeFrame is None:
        print("Error: SecurityName and timeFrame must be provided if df is None.")
        return

    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.capitalize()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data.index = data.index.to_period('D')

    stock_prices = data['Close']

    # Data Preprocessing (Differencing)
    diffed_prices = stock_prices.diff().dropna()

    # Train-test split
    train_size = int(len(diffed_prices) * 0.8)
    train_data, test_data = diffed_prices[:train_size], diffed_prices[train_size:]

    # Fit ARIMA model with adjusted hyperparameters
    try:
        model = ARIMA(train_data, order=(5,1,1), freq='D')  # Adjust order as needed
        fitted_model = model.fit()

        # Forecast
        forecast = fitted_model.forecast(steps=forecast_steps)

        # Inverse difference to get actual values
        forecast_values = np.cumsum(forecast)
        actual_values = stock_prices.iloc[-1] + forecast_values   # Assuming last actual value as starting point

        # Calculate accuracy
        absolute_errors = np.abs(actual_values - stock_prices[-1:])  # Using last actual value as reference
        mean_absolute_error_value = np.mean(absolute_errors)

        # Print accuracy
        print(f"The Mean Absolute Error (MAE) of ARIMA model is approximately {mean_absolute_error_value:.2f}.")

        # Visualize results
        plt.figure(figsize=(10, 6))
        plt.plot(test_data.index[-forecast_steps:].to_timestamp(), actual_values, label='Actual Prices')
        plt.plot(test_data.index[-forecast_steps:].to_timestamp(), forecast_values, label='Predicted Prices', color='red')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.title('ARIMA Stock Price Prediction')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_plot:
            plt.savefig("arima_stock_prediction.png")

        plt.show()

        return mean_absolute_error_value

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Example usage
arima_stock_prediction(SecurityName="NEPSE", timeFrame='1D', forecast_steps=50, save_plot=True)
