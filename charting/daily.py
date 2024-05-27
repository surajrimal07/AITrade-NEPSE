import pandas as pd
import requests
from lightweight_charts import Chart
from io import StringIO
from time import sleep
import time
import datetime
from datetime import date
import asyncio
import threading
import numpy as np
from sklearn.linear_model import LinearRegression
from data_process import process_json_data
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from auth import show_login_dialog, show_logout_dialog, fetch_trading_portfolio,add_stock_to_portfolio
import websockets
import json
from regression import regression_plot
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from other_data import fetch_prediction
from timeseries import time_series_analysis
from global_var import *


#convert all these to settings later
SecurityName = 'NEPSE'
timeFrame = '1D'
autorefresh = False
auto_refresh_task = None
userData = [
    {'name': 'No User', 'email': 'No User', 'userAmount': 0}
]
userPortfolio = [
    {'recommendation': 'None','stocks': [
        {'name': 'None','ltp': 0,'symbol': 'None', 'quantity': 0, 'wacc': 0, 'costprice': 0, 'currentprice': 0, 'netgainloss': 0,'time': 0}
    ], 'totalunits': 0, 'portfoliocost': 0, 'portfoliovalue': 0, 'portgainloss': 0, 'portfolioPercentage': 0, 'totalStocks': 0,'totalunits': 0,}
]
isLoggedin = False
basic_prediction_result = [
    {'prediction': 'No Data Available', 'strength': 0}
]

SubchartSecurity = 'NEPSE'
SubchartTimeFrame = '1'
default_quantity = 1 #stock quantity to buy
fetchLivePortfolio = True

# Disable SSL warnings on localhost with self signed certificate
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def calculate_sma(df, period: int = 50):
    sma_data = pd.DataFrame({
        'time': df['date'],
        f'SMA {period}': df['close'].rolling(window=period).mean()
    }).dropna()


    # Chart.sma_line = Chart.create_line('SMA 50')
    # Chart.sma_line.set(sma_data)

    if hasattr(Chart, 'sma_line'):
        Chart.sma_line.set(sma_data)
    else:
        Chart.sma_line = Chart.create_line('SMA 50', color='WHITE',price_line = False,price_label = False,width=1)
        Chart.sma_line.set(sma_data)

async def fetch_prediction_data():
    prediction_data = await fetch_prediction()
    if prediction_data:
        basic_prediction_result[0]['prediction'] = prediction_data[0]
        basic_prediction_result[0]['strength'] = prediction_data[1]
    else:
        print("Failed to fetch prediction data")

async def buy_stock(df):
    global isLoggedin
    global userPortfolio
    if isLoggedin:
        userPortfolio = add_stock_to_portfolio(Chart.topbar['symbol'].value, default_quantity)
        if userPortfolio:
            update_portfolio_table(Chart)
    else:
        await login_user(Chart)

async def fetch_user_portfolio():
    global userPortfolio
    userPortfolio = fetch_trading_portfolio()
    if userPortfolio:
        update_portfolio_table(Chart)
    else:
        print("Failed to fetch user portfolio")

async def login_user(chart):
    global isLoggedin
    global userData
    if isLoggedin:
        logout_data = show_logout_dialog()
        if logout_data:
            chart.topbar['login'].set(user['name'])
            isLoggedin = False
    else :
        userData = show_login_dialog()
        if userData:
            chart.topbar['login'].set(userData['name'])
            await fetch_user_portfolio()
            isLoggedin = True


def time_frame_manipulation(timeFrame):
    time_frame_mapping = {
        '1Min': '1',
        '5Min': '5',
        '10Min': '10',
        '15Min': '15',
        '1': '1',
        '1W': '1W',
    }
    return time_frame_mapping.get(timeFrame, '1D')

def showAlgorithmGUI(chart):
    if chart.topbar['algo'].value == 'Regression':
        regression_plot(chart.topbar['symbol'].value, time_frame_manipulation(chart.topbar['timeframe'].value))
    elif chart.topbar['algo'].value == 'TimeSeries':
        time_series_analysis(chart.topbar['symbol'].value, time_frame_manipulation(chart.topbar['timeframe'].value))


async def perform_regression_analysis(df):
    y = np.array(df["close"])
    x = np.linspace(1, len(y), len(y)).reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(x, y)

    next_index = reg.predict(np.array([[len(y) + 1]]))[0]
    next_index = round(next_index, 2)

    slope = reg.coef_[0]
    intercept = reg.intercept_
    regression_equation = f"{slope:.2f}x + {intercept:.2f}"

    today = date.today()
    date_str = date.isoformat(today)

    # Calculate regression metrics
    y_pred = reg.predict(x)
    r_squared = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    accuracy_percent = r_squared * 100

    # print(f"R-squared: {r_squared:.2f}")
    # print(f"Mean Absolute Error (MAE): {mae:.2f}")
    # print(f"Mean Squared Error (MSE): {mse:.2f}")
    # print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    # print(f"Accuracy: {accuracy_percent:.2f}%")

    # Update or create the regression line
    x_values = np.array(range(1, len(df) + 2))
    y_values = slope * x_values + intercept

    # Ensure lengths are the same
    min_length = min(len(df['date']), len(y_values))
    df_truncated = df.iloc[:min_length]
    y_values_truncated = y_values[:min_length]

    if hasattr(Chart, 'regression_line'):
        Chart.regression_line.set(pd.DataFrame({'time': df_truncated['date'], 'RegressionLine': y_values_truncated}))
    else:
        Chart.regression_line = Chart.create_line('RegressionLine', color='WHITE',price_line = False,price_label = False,width=1)
        Chart.regression_line.set(pd.DataFrame({'time': df_truncated['date'], 'RegressionLine': y_values_truncated}))

async def fetch_data(SecurityName, timeFrame):
    print(f"getting bar data for {SecurityName} {timeFrame}")
    manipulatedTimeFrame = time_frame_manipulation(timeFrame)
    url = baseUrl+'/getcompanyohlc?symbol=' + SecurityName + '&timeFrame=' + manipulatedTimeFrame

    response = requests.get(url,verify=False)
    if response.status_code == 200:
        return process_json_data(response.json(),manipulatedTimeFrame)
    else:
        print(f"Failed to fetch data from {url}, probably the symbol is not available.")
        return None

async def fetch_tick_data(SecurityName, timeFrame):
    print(f"getting tick bar data for {SecurityName} {timeFrame}")
    manipulatedTimeFrame = time_frame_manipulation(timeFrame)
    url = baseUrl+'/getcompanyohlc?symbol=' + SecurityName + '&timeFrame=' + manipulatedTimeFrame + '&intradayupdate=true'

    response = requests.get(url,verify=False)
    if response.status_code == 200:
        return process_json_data(response.json(),manipulatedTimeFrame)
    else:
        return None


async def fetch_available_symbols():
    url = 'https://localhost:4000/api/availablenepsecompanies'
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        json_data = response.json()
        if 'data' in json_data:
            return json_data['data']
        else:
            print("Data field not found in JSON response.")
            return None
    else:
        print(f"Failed to fetch data from {url}")
        return None

async def recalculate_dataset(df):
    await perform_regression_analysis(df)
    calculate_sma(df)

async def on_search(chart, searched_string):
    new_data = await fetch_data(searched_string, chart.topbar['timeframe'].value)
    if new_data is None or new_data.empty:
        return
    chart.topbar['symbol'].set(searched_string)
    chart.watermark(searched_string + ' ' + chart.topbar['timeframe'].value)
    await recalculate_dataset(new_data)
    chart.set(new_data)

async def on_timeframe_selection(chart):
    new_data = await fetch_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
    if new_data is None or new_data.empty:
        return
    chart.watermark(chart.topbar['symbol'].value + ' ' + chart.topbar['timeframe'].value)
    await recalculate_dataset(new_data)
    chart.set(new_data, True)

async def take_screenshot(key):
    img = Chart.screenshot()
    t = time.time()
    with open(f"screenshot-{t}.png", 'wb') as f:
        f.write(img)

# async def fetch_others_data():
#     print("fetching other data")

# async def refresh_data(chart):
#     new_data = await fetch_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
#     if new_data is None or new_data.empty:
#         return
#     chart.set(new_data, True)

async def auto_trade(chart):
    new_button_value = 'AutoTrade On' if chart.topbar['autoretrade_button'].value == 'AutoTrade Off' else 'AutoTrade Off'
    chart.topbar['autoretrade_button'].set(new_button_value)

    # global auto_refresh_task

    # async def trade(chart):
    #     while chart.topbar['autotrade_button'].value == 'AutoTrade On':
    #         await asyncio.sleep(10 - (datetime.datetime.now().microsecond / 1_000_000))
    #         new_data = await fetch_tick_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
    #         if new_data is not None and not new_data.empty:
    #             chart.set(new_data, True)

    # auto_refresh_task = asyncio.create_task(trade(chart))

async def auto_refresh(chart):
    new_button_value = 'AutoRefresh Off' if chart.topbar['autorefresh_button'].value == 'AutoRefresh On' else 'AutoRefresh On'
    chart.topbar['autorefresh_button'].set(new_button_value)

    global auto_refresh_task

    async def refresh(chart):
        while chart.topbar['autorefresh_button'].value == 'AutoRefresh On':
            await asyncio.sleep(10 - (datetime.datetime.now().microsecond / 1_000_000))
            new_data = await fetch_tick_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
            if new_data is not None and not new_data.empty:
                chart.set(new_data, True)

    auto_refresh_task = asyncio.create_task(refresh(chart))


def on_row_click(row):
    print(f'Row Clicked: {row}')

def create_portfolio_table(chart):
    table = chart.create_table(width=0.3, height=0.3,
                               headings=('Symbol', 'Quantity', 'WACC', 'Total Cost', 'Current Value', 'P&L','%'),
                               widths=(0.2, 0.1, 0.1, 0.2, 0.2, 0.2,0.2),alignments=('center', 'center', 'center', 'center', 'center','center'),func=on_portfolio_row_click)
    table.format('WACC', f'Rs  {table.VALUE}')
    table.format('Total Cost', f'Rs  {table.VALUE}')
    table.format('Current Value', f'Rs  {table.VALUE}')
    table.format('P&L', f'Rs  {table.VALUE}')
    table.format('%', f'{table.VALUE}%')

    table.new_row('No Stocks', 0, 0, 0, 0, 0,0)
    table.header(1)
    table.header[0] = userData[0]['name'] + ' Portfolio'

    table.footer(7)
    table.footer[0] = 'Total:'

    chart.portfolio_table = table

def update_portfolio_table(chart):
    if hasattr(chart, 'portfolio_table'):

        table = chart.portfolio_table
        table.clear()
        table.header[0] = userData['name'] + ' Portfolio '+ ' (Balance Rs '+ str(userData['userAmount']) + ')'

        for stock in userPortfolio['stocks']:
            percentage = round((stock['currentprice'] - stock['costprice']) / stock['costprice'] * 100, 0)
            row = table.new_row(stock['symbol'], stock['quantity'], stock['wacc'], stock['costprice'],stock['currentprice'], stock['netgainloss'],percentage)
            row.background_color('P&L', 'green' if row['P&L'] > 0 else 'red')

        table.footer[1] = str(userPortfolio['totalunits'])+" Unit"
        table.footer[2] = ' '
        table.footer[3] = "Rs "+ str(userPortfolio['portfoliocost'])
        table.footer[4] = "Rs "+ str(userPortfolio['portfoliovalue'])
        table.footer[5] = "Rs "+ str(userPortfolio['portgainloss'])
        table.footer[6] = str(userPortfolio['portfolioPercentage']) + "%"

    else:
        print("Portfolio table does not exist. Creating a new one.")
        create_portfolio_table(chart)

async def on_portfolio_row_click(row):
    if isLoggedin:
        new_data = await fetch_data(row['Symbol'], Chart.topbar['timeframe'].value)
        if new_data is None or new_data.empty:
            return
        Chart.topbar['symbol'].set(row['Symbol'])
        Chart.watermark(row['Symbol'] + ' ' + Chart.topbar['timeframe'].value)
        await recalculate_dataset(new_data)
        Chart.set(new_data)

# def buy_stock(chart):
#     print ("Buying stock")

def sell_stock(chart):
    print ("Selling stock")


async def main():
    global Chart

    df = await fetch_data(SecurityName, timeFrame)
    if df is None:
        print('No data to display')
        exit(1)

    Chart = Chart(toolbox=False, maximize=True, inner_width=0.7, inner_height=1, title='10Paisa AI Dashboard')
    Chart.legend(visible = True, font_family = 'Trebuchet MS', ohlc = True, percent = True)
    Chart.grid(vert_enabled = True, horz_enabled = True)
    Chart.watermark(SecurityName + ' ' + timeFrame)
    Chart.topbar.button('login', 'Login', func= login_user)
    Chart.topbar.textbox('symbol', SecurityName)
    #Chart.toolbar.button('connect', 'Connect', func= connect_websocket)
    Chart.topbar.menu('algo', ('Algorithms','Regression','TimeSeries', 'Algorithm1', 'Algorithm2'), default='Algorithms', func=showAlgorithmGUI)
    Chart.topbar.switcher('timeframe', ('1Min', '5Min', '10Min', '15Min','1D'), default=timeFrame, func= on_timeframe_selection)
    Chart.events.search += on_search
    Chart.topbar.button('screenshot', 'Screenshot', func= take_screenshot)
    Chart.topbar.button('autorefresh_button', 'AutoRefresh On', func=auto_refresh)
    # Chart.topbar.textbox('autorefreshtext', 'autorefresh')
    # Chart.topbar.switcher('autorefresh', ('True', 'False'), default='False', func= auto_refresh)
    Chart.topbar.button('autoretrade_button', 'AutoTrade Off', func= auto_trade)
    Chart.topbar.button('buy_buttton', 'Buy', func=buy_stock )
    Chart.topbar.button('sell', 'Sell', func= sell_stock)


#subchart
    subchart = Chart.create_subchart(width=0.3, height=0.4)
    subchart.watermark(SubchartSecurity + ' ' + SubchartTimeFrame)
    subchart.topbar.textbox('symbol', SubchartSecurity)
    subchart.topbar.switcher('timeframe', ('1Min', '5Min', '10Min'), default=SubchartTimeFrame, func= on_timeframe_selection)
    subchart.events.search += on_search
    subchart.topbar.button('autorefresh_button', 'AutoRefresh Off', func=auto_refresh)

    df2 = await fetch_data(SubchartSecurity, SubchartTimeFrame)

    if df2 is None:
        print('No data to display')
        exit(1)

    # sub_line = subchart.create_line('SMA 50')
    # sma_subchart = await calculate_sma(df2, period=50)
    # sub_line.set(sma_subchart)

##end of subchart

#sma
    calculate_sma(df)

#table2 #algorithm table
    table2 = Chart.create_table(width=0.3, height=0.1,
                    headings=('Model', 'R-squared', 'Accuracy:'),
                    widths=(0.2, 0.1, 0.2),
                    alignments=('center', 'center', 'right'),
                    position='left', func=on_row_click)

    table2.header(1)
    table2.header[0] = 'Algorithm Table'
    table2.format('PL (rs)', f'£ {table2.VALUE}')
    table2.format('Change', f'{table2.VALUE} %')

    table2.new_row(Chart.topbar['algo'].value, 3, 0)

    table2.footer(2)
    table2.footer[0] = 'Selected:'

#fetch prediction data
    await fetch_prediction_data()
    #table2.footer[1] = basic_prediction_result[0]['prediction']

#table3 #Total Accuracy table
    table3 = Chart.create_table(width=0.3, height=0.1,
                    headings=('Model', 'R-squared', 'Accuracy:'),
                    widths=(0.2, 0.1, 0.2),
                    alignments=('center', 'center', 'right'),
                    position='right', func=on_row_click)
    table3.header(1)
    table3.header[0] = 'Prediction Table'
    table3.format('PL (rs)', f'£ {table3.VALUE}')
    table3.format('Change', f'{table3.VALUE} %')

    table3.new_row(Chart.topbar['algo'].value, 3, 0)

    table3.footer(2)
    table3.footer[0] = 'Probability:'
    table3.footer[1] = basic_prediction_result[0]['prediction']


#table #portfolio table
    create_portfolio_table(Chart)
#end of table

#perform regression analysis
    await perform_regression_analysis(df)
    # next_index, date_str, regression_equation, slope, intercept = await perform_regression_analysis(df)
    # x_values = np.array(range(1, len(df) + 2))
    # y_values = slope * x_values + intercept

    # regressionline = Chart.create_line('RegressionLine', color='white')

    # # Ensure lengths are the same
    # min_length = min(len(df['date']), len(y_values))
    # df_truncated = df.iloc[:min_length]
    # y_values_truncated = y_values[:min_length]

    # new_df = pd.DataFrame({'time': df_truncated['date'], 'RegressionLine': y_values_truncated})
    # regressionline.set(new_df)

#end of regression analysis

    Chart.set(df)
    subchart.set(df2)

   # Chart.marker(text=f"Predicted Next Index Value")

    await asyncio.gather(Chart.show_async(block=True), auto_refresh(Chart))

if __name__ == '__main__':
    asyncio.run(main())
