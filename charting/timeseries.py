import pandas as pd
import numpy as np
import os
from data_fetch import save_symbol_model_value

from statsmodels.tsa.seasonal import seasonal_decompose
from data_process import process_json_data
import matplotlib.pyplot as plt

def time_series_analysis(SecurityName="NEPSE", timeFrame='1D', forecast_steps=50, save_plot=True):

    folder_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", f"{SecurityName}", f"{timeFrame}")
    csv_path = os.path.join(folder_name, f"{SecurityName}_{timeFrame}.csv")

    if SecurityName is None or timeFrame is None:
        print("Error: SecurityName and timeFrame must be provided if df is None.")
        return

    df = pd.read_csv(csv_path, parse_dates=['date'])
    df.rename(columns={'close': 'Index Value', 'date': 'Date (AD)'}, inplace=True)
    df.drop(['high', 'open', 'low', 'volume'], axis=1, inplace=True)

    df['Date (AD)'] = pd.to_datetime(df['Date (AD)'])
    df.set_index('Date (AD)', inplace=True)

    decomposition = seasonal_decompose(df['Index Value'], model='multiplicative', period=12)

    plt.switch_backend('TkAgg')
    plt.figure(num= SecurityName+" Time Series Analysis", figsize=(16, 8))
    plt.subplot(311)
    plt.plot(df.index, df['Index Value'], label='Original')
    plt.legend()

    plt.subplot(312)
    plt.plot(df.index, decomposition.trend, label='Trend')
    plt.legend()

    trend_values = decomposition.trend.dropna()
    last_trend_value = trend_values.iloc[-1]
    forecasted_values = [last_trend_value * (1 + i/100) for i in range(1, forecast_steps + 1)]
    forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='D')[1:]

    plt.subplot(313)
    plt.plot(df.index, df['Index Value'], label='Original')
    plt.plot(forecast_index, forecasted_values, label='Forecast', color='red')
    plt.legend()

    plt.suptitle('Time Series Decomposition with Forecast')
    plt.tight_layout()

    # Calculate accuracy metrics
    actual_values = df['Index Value'][-forecast_steps:]
    absolute_errors = [abs(actual - forecasted) for actual, forecasted in zip(actual_values, forecasted_values)]
    mean_absolute_error_value = sum(absolute_errors) / len(absolute_errors)

    mean_actual_value = np.mean(actual_values)
    accuracy_percent = round(100 * (1 - mean_absolute_error_value / mean_actual_value), 2)

    save_symbol_model_value(SecurityName, timeFrame, "TimeSeries", accuracy_percent)

    if save_plot:
        plt.savefig("time_series.png")

    plt.tight_layout()
    plt.show()