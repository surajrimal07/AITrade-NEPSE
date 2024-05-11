# import pandas as pd
# from lightweight_charts import Chart

# def on_row_click(row):
#     row['PL'] = round(row['PL']+1, 2)
#     row.background_color('PL', 'green' if row['PL'] > 0 else 'red')

#     table.footer[1] = row['Ticker']

# def on_footer_click(table, box_index):
#     print(f'Box number {box_index+1} was pressed.')

# if __name__ == '__main__':
#     chart = Chart(width=1000, inner_width=0.7, inner_height=1)
#     subchart = chart.create_subchart(width=0.3, height=0.5)
#     df = pd.read_csv('ohlcv.csv')
#     chart.set(df)
#     subchart.set(df)

#     table = chart.create_table(width=0.3, height=0.2,
#                   headings=('Ticker', 'Quantity', 'Status', '%', 'PL'),
#                   widths=(0.2, 0.1, 0.2, 0.2, 0.3),
#                   alignments=('center', 'center', 'right', 'right', 'right'),
#                   position='left', func=on_row_click)

#     table.format('PL', f'£ {table.VALUE}')
#     table.format('%', f'{table.VALUE} %')

#     table.new_row('SPY', 3, 'Submitted', 0, 0)
#     table.new_row('AMD', 1, 'Filled', 25.5, 105.24)
#     table.new_row('NVDA', 2, 'Filled', -0.5, -8.24)

#     table.footer(2)
#     table.footer[0] = 'Selected:'

#     table.footer(3, func=on_footer_click)

#     chart.show(block=True)

import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import date

# import openpyxl
import matplotlib.pyplot as plt
import bs4
import html5lib

# import telegram
# from telegram import InputFile


# Load data from the URL
url = "https://merolagani.com/Indices.aspx"
df = pd.read_html(url)[0]
df = df.iloc[::-1].reset_index(drop=True)

print(df)

# Create X and Y arrays for regression analysis
y = np.array(df["Index Value"])
x = np.linspace(1, len(y), len(y)).reshape(-1, 1)

# Perform linear regression on the data
reg = LinearRegression()
reg.fit(x, y)

# Predict the next data point using the linear regression model
next_index = reg.predict(np.array(len(y) + 1).reshape(1, -1))[0]

# Get today's date
today = date.today()
date_str = date.isoformat(today)

# Plot the data and regression line
plt.figure(figsize=(16, 8))
plt.plot(df["Date (AD)"], df["Index Value"], "bo-", label="Index Value")
plt.plot(date_str, next_index, "ro", label="Predicted Next Index Value")

# Add annotations
plt.annotate(
    f"Predicted: {next_index:.2f}",
    xy=(date_str, next_index),
    xytext=(-50, 30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="red"),
)
plt.annotate(
    f"Regression Line: {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}",
    xy=(0.05, 0.95),
    xycoords="axes fraction",
    fontsize=12,
    ha="left",
    va="top",
)

# Add regression line
plt.plot(df["Date (AD)"], reg.predict(x), "g--", label="Regression Line")

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