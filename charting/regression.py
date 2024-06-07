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
import os
from data_fetch import save_symbol_model_value

def regression_plot(SecurityName, timeFrame,forecast_periods=30, save_plot=True):
    folder_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", f"{SecurityName}", f"{timeFrame}")
    csv_path = os.path.join(folder_name, f"{SecurityName}_{timeFrame}.csv")

    if SecurityName is None or timeFrame is None:
        print("Error: SecurityName and timeFrame must be provided if df is None.")
        return

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    df.dropna(subset=['close'], inplace=True)

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
    #accuracy_percent = r_squared * 100
    accuracy_percent = round(r_squared * 100, 2)

    save_symbol_model_value(SecurityName, timeFrame, "Regression", accuracy_percent)

    # Extend data for forecasting
    future_dates = pd.date_range(start=df["Date (AD)"].max(), periods=forecast_periods + 1, freq='B')[1:]
    future_x = np.arange(len(y), len(y) + forecast_periods).reshape(-1, 1)
    future_y = reg.predict(future_x)

    # Plot data and regression line
    plt.figure(num= SecurityName+" Regression Analysis", figsize=(16, 8))
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
    plt.title(SecurityName+ " Price value")
    plt.legend()
    plt.grid()

    if save_plot:
        plt.savefig("regression_graph.png")

    plt.tight_layout()
    plt.show()