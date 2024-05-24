# import requests
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# from datetime import date
# from data_process import process_json_data
# from requests.packages.urllib3.exceptions import InsecureRequestWarning

# requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# def regression_plot(SecurityName, timeFrame, save_plot=True):
#     url = f'https://api.zorsha.com.np/api/getcompanyohlc?symbol={SecurityName}&timeFrame={timeFrame}'
#     response = requests.get(url, verify=False)

#     if response.status_code != 200:
#         print(f"Failed to fetch data from {url}, probably the symbol is not available.")
#         return None

#     data = process_json_data(response.json(), timeFrame)

#     df = pd.DataFrame(data)
#     df.rename(columns={'close': 'Index Value', 'date': 'Date (AD)'}, inplace=True)
#     df.drop(['high', 'open', 'low', 'volume'], axis=1, inplace=True)

#     # Perform linear regression
#     y = np.array(df["Index Value"])
#     x = np.linspace(1, len(y), len(y)).reshape(-1, 1)
#     reg = LinearRegression()
#     reg.fit(x, y)
#     next_index = reg.predict(np.array(len(y) + 1).reshape(1, -1))[0]

#     # Plot data and regression line
#     plt.figure(num="Nepse Regression Analysis", figsize=(16, 8))
#     plt.plot(df["Date (AD)"], df["Index Value"], "bo-", label="Index Value")
#     plt.plot(df.iloc[-1]["Date (AD)"], next_index, "ro", label="Predicted Next Index Value")

#     plt.annotate(
#         f"Predicted: {next_index:.2f}",
#         xy=(df.iloc[-1]["Date (AD)"], next_index),
#         xytext=(-50, 30),
#         textcoords="offset points",
#         arrowprops=dict(arrowstyle="->", color="red"),
#     )

#     plt.annotate(
#         f"Regression Line: {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}",
#         xy=(0.05, 0.95),
#         xycoords="axes fraction",
#         fontsize=12,
#         ha="left",
#         va="top",
#     )

#     plt.plot(df["Date (AD)"], reg.predict(x), "g--", label="Regression Line")
#     plt.xticks(rotation=45, ha="right")
#     plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
#     plt.xlabel("Date")
#     plt.ylabel("Index Value")
#     plt.title("Nepal Stock Exchange NEPSE Index Value")
#     plt.legend()
#     plt.grid()

#     if save_plot:
#         plt.savefig("graph.png")

#     plt.show()

# # Example usage:
# regression_plot("NEPSE", "1")


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import date
import matplotlib.pyplot as plt
import requests
from data_process import process_json_data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def regression_plot(SecurityName="NEPSE", timeFrame="1D", df=None):
    if df is None:
        url = f"https://api.zorsha.com.np/api/getcompanyohlc?symbol={SecurityName}&timeFrame={timeFrame}"
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            data = process_json_data(response.json(), timeFrame)
        else:
            print(f"Failed to fetch data from {url}, probably the symbol is not available.")
            return
    else:
        data = df

    data = pd.DataFrame(data)
    data.rename(columns={'close': 'Index Value', 'date': 'Date (AD)'}, inplace=True)
    data.drop(['high', 'open','low', 'volume'], axis=1, inplace=True)

    # Create X and Y arrays for regression analysis
    y = np.array(data["Index Value"])
    x = np.linspace(1, len(y), len(y)).reshape(-1, 1)

    # Perform linear regression on the data
    reg = LinearRegression()
    reg.fit(x, y)

    # Predict the next data point using the linear regression model
    next_index = reg.predict(np.array(len(y) + 1).reshape(1, -1))[0]

    # Calculate regression metrics
    y_pred = reg.predict(x)
    r_squared = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    accuracy_percent = r_squared * 100

    print(f"R-squared: {r_squared:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Accuracy: {accuracy_percent:.2f}%")

    # Plot the data and regression line
    plt.figure(num="Nepse Regression Analysis", figsize=(16, 8))
    plt.plot(data["Date (AD)"], data["Index Value"], "bo-", label="Index Value")

    last_date = data.iloc[-1]["Date (AD)"]
    plt.plot(last_date, next_index, "ro", label="Predicted Next Index Value")

    # Add annotations
    plt.annotate(
        f"Predicted: {next_index:.2f}",
        xy=(last_date, next_index),
        xytext=(-50, 30),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    # Add annotations
    plt.annotate(
        f"Regression Line: {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        ha="left",
        va="top",
    )

    # Add regression line
    plt.plot(data["Date (AD)"], reg.predict(x), "g--", label="Regression Line")

    # Format x-axis
    plt.xticks(rotation=45, ha="right")
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))

    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.title("Nepal Stock Exchange NEPSE Index Value")
    plt.legend()
    plt.grid()

    plt.savefig("graph.png")
    plt.show()


def time_series_analysis(df=None, SecurityName=None, timeFrame=None):
    if df is None:
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
    else:
        data = df

    df = pd.DataFrame(data)
    df.rename(columns={'close': 'Index Value', 'date': 'Date (AD)'}, inplace=True)
    df.drop(['high', 'open', 'low', 'volume'], axis=1, inplace=True)

    # Convert the 'Date (AD)' column to datetime format if needed
    df['Date (AD)'] = pd.to_datetime(df['Date (AD)'])

    # Set the 'Date (AD)' column as the index
    df.set_index('Date (AD)', inplace=True)

    # Perform seasonal decomposition
    decomposition = seasonal_decompose(df['Index Value'], model='multiplicative', period=12)

    # Plot the decomposition components
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(df.index, df['Index Value'], label='Original')
    plt.legend()

    plt.subplot(412)
    plt.plot(df.index, decomposition.trend, label='Trend')
    plt.legend()

    plt.subplot(413)
    plt.plot(df.index, decomposition.seasonal, label='Seasonal')
    plt.legend()

    plt.subplot(414)
    plt.plot(df.index, decomposition.resid, label='Residual')
    plt.legend()

    plt.suptitle('Time Series Decomposition')
    plt.tight_layout()
    plt.show()


time_series_analysis()



