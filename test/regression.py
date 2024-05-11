import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import date
import matplotlib.pyplot as plt
import bs4
import html5lib

url = "https://merolagani.com/Indices.aspx"
df = pd.read_html(url)[0]
df = df.iloc[::-1].reset_index(drop=True)

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