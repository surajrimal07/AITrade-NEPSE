import pandas as pd
import numpy as np
import requests
from statsmodels.tsa.seasonal import seasonal_decompose
from data_process import process_json_data
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def time_series_analysis(SecurityName="NEPSE", timeFrame='1D', forecast_steps=50):
    if SecurityName is None or timeFrame is None:
        print("Error: SecurityName and timeFrame must be provided if df is None.")
        return

    url = f"https://api.zorsha.com.np/api/getcompanyohlc?symbol={SecurityName}&timeFrame={timeFrame}"
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        data = process_json_data(response.json(), timeFrame)
    else:
        print(f"Failed to fetch data from {url}, probably the symbol is not available.")
        return

    df = pd.DataFrame(data)
    df.rename(columns={'close': 'Index Value', 'date': 'Date (AD)'}, inplace=True)
    df.drop(['high', 'open', 'low', 'volume'], axis=1, inplace=True)


    df['Date (AD)'] = pd.to_datetime(df['Date (AD)'])
    df.set_index('Date (AD)', inplace=True)

    decomposition = seasonal_decompose(df['Index Value'], model='multiplicative', period=12)

    # Plotting
    plt.figure(figsize=(12, 10))

    plt.subplot(311)
    plt.plot(df.index, df['Index Value'], label='Original')
    plt.legend()

    plt.subplot(312)
    plt.plot(df.index, decomposition.trend, label='Trend')
    plt.legend()

    # Generate forecasted values based on the trend component
    trend_values = decomposition.trend.dropna()
    last_trend_value = trend_values.iloc[-1]  # Use iloc for position-based indexing
    forecasted_values = [last_trend_value * (1 + i/100) for i in range(1, forecast_steps + 1)]
    forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='D')[1:]

    plt.subplot(313)
    plt.plot(df.index, df['Index Value'], label='Original')
    plt.plot(forecast_index, forecasted_values, label='Forecast', color='red')
    plt.legend()

    plt.suptitle('Time Series Decomposition with Forecast')
    plt.tight_layout()

    plt.show()


# Example usage
time_series_analysis()
