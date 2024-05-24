import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import date
from data_process import process_json_data
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from global_var import baseUrl


requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def regression_plot(SecurityName, timeFrame,forecast_periods=30, save_plot=True):
    url = f"{baseUrl}getcompanyohlc?symbol={SecurityName}&timeFrame={timeFrame}"
    #url = f'https://api.zorsha.com.np/api/getcompanyohlc?symbol={SecurityName}&timeFrame={timeFrame}'
    response = requests.get(url, verify=False)

    if response.status_code != 200:
        print(f"Failed to fetch data from {url}, probably the symbol is not available.")
        return None

    data = process_json_data(response.json(), timeFrame)

    df = pd.DataFrame(data)
    df.rename(columns={'close': 'Index Value', 'date': 'Date (AD)'}, inplace=True)
    df.drop(['high', 'open', 'low', 'volume'], axis=1, inplace=True)

    # Perform linear regression
    y = np.array(df["Index Value"])
    x = np.linspace(1, len(y), len(y)).reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(x, y)
    next_index = reg.predict(np.array(len(y) + 1).reshape(1, -1))[0]

    # Calculate regression metrics
    y_pred = reg.predict(x)
    r_squared = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    accuracy_percent = r_squared * 100

    # Extend data for forecasting
    future_dates = pd.date_range(start=df["Date (AD)"].max(), periods=forecast_periods + 1, freq='B')[1:]
    future_x = np.arange(len(y), len(y) + forecast_periods).reshape(-1, 1)
    future_y = reg.predict(future_x)

    # Plot data and regression line
    plt.figure(num="Nepse Regression Analysis", figsize=(16, 8))
    plt.plot(df["Date (AD)"], df["Index Value"], "bo-", label="Index Value")
    plt.plot(df.iloc[-1]["Date (AD)"], next_index, "ro", label="Predicted Next Index Value")

    plt.annotate(
        f"Predicted: {next_index:.2f}",
        xy=(df.iloc[-1]["Date (AD)"], next_index),
        xytext=(-50, 30),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    plt.annotate(
        f"Regression Line: {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}\n"
        f"Forecast Periods: {forecast_periods}\n"
        f"R-squared: {r_squared:.2f}\n"
        f"MAE: {mae:.2f}\n"
        f"MSE: {mse:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"Accuracy: {accuracy_percent:.2f}%",
        xy=(0.95, 0.05),
        xycoords="axes fraction",
        fontsize=10,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1),
    )

    plt.plot(df["Date (AD)"], reg.predict(x), "g--", label="Regression Line")
    plt.xticks(rotation=45, ha="right")
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.title("Nepal Stock Exchange NEPSE Index Value")
    plt.legend()
    plt.grid()

    if save_plot:
        plt.savefig("regression_graph.png")

    plt.show()