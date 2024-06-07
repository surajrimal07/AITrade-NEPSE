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
from auth import show_login_dialog, show_logout_dialog, fetch_trading_portfolio,add_stock_to_portfolio,fetch_user_data_api,remove_stock_from_portfolio,show_model_not_trained_dialog
import websockets
import json
from regression import regression_plot
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from other_data import fetch_prediction
from timeseries import time_series_analysis
from global_var import *
from ws_socket import fetch_live_portfolio
from data_fetch import fetch_data, time_frame_manipulation,fetch_symbol_model_value,fetch_tick_data, checkIfModelExists
import signal

#from model.Run_LSTM_Model import WaitDialog, ModelThread

#convert all these to settings later
# SecurityName = 'NEPSE'
# timeFrame = '1D'


# Disable SSL warnings on localhost with self signed certificate
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

#testing
async def fetch_portfolio_wss():
    if fetchLivePortfolio and isLoggedin:
        print("Fetching live portfolio data")
        portfolio = await fetch_live_portfolio()
        if portfolio:
            print(portfolio)
        else:
            print("Failed to fetch portfolio data")


def calculate_sma(df, period: int = 50):

    sma_data = pd.DataFrame({
        'time': df['date'],
        f'SMA {period}': df['close'].rolling(window=period).mean()
    }).dropna()


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
        await fetch_user_data()

        if userPortfolio:
            update_portfolio_table(Chart)
    else:
        await login_user(Chart)

async def sell_stock(df):
    global isLoggedin
    global userPortfolio
    if isLoggedin:
        userPortfolio = remove_stock_from_portfolio(Chart.topbar['symbol'].value, default_quantity)
        await fetch_user_data()

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

async def fetch_user_data(): #for later refresh calls to get updated user data after buy or sell occurs
    global userData
    userData = fetch_user_data_api()

async def login_user(chart):
    global isLoggedin,userData,userPortfolio, INITIAL_USER_PORTFOLIO,INITIAL_USER_DATA
    if isLoggedin:
        logout_data = show_logout_dialog()
        if logout_data:
            userPortfolio = copy.deepcopy(INITIAL_USER_PORTFOLIO)
            userData = copy.deepcopy(INITIAL_USER_DATA)

            chart.topbar['login'].set(userData['name'])
            clear_portfolio_table(chart)
            isLoggedin = False

    else :
        userData = show_login_dialog()
        if userData:
            chart.topbar['login'].set(userData['name'])
            await fetch_user_portfolio()
            isLoggedin = True


def showAlgorithmGUI(chart):
    global time_frame_manipulation
    if chart.topbar['algo'].value == 'Regression':
        if checkIfModelExists('Regression',chart.topbar['symbol'].value,time_frame_manipulation(chart.topbar['timeframe'].value)) == False:
            show_model_not_trained_dialog()
        else:
            regression_plot(chart.topbar['symbol'].value, time_frame_manipulation(chart.topbar['timeframe'].value))
    elif chart.topbar['algo'].value == 'TimeSeries':
        time_series_analysis(chart.topbar['symbol'].value, time_frame_manipulation(chart.topbar['timeframe'].value))
    elif chart.topbar['algo'].value == 'LSMT':
        if checkIfModelExists('LSMT',chart.topbar['symbol'].value,time_frame_manipulation(chart.topbar['timeframe'].value)) == False:
            show_model_not_trained_dialog()
        else:
            import sys
            import os
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
            from Run_LSTM_Model import run_model_and_show_dialog
            run_model_and_show_dialog(chart.topbar['symbol'].value, time_frame_manipulation(chart.topbar['timeframe'].value))
    elif chart.topbar['algo'].value == 'custom_algorithm':
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
        from custom_algorithm import show_custom_Chart
        show_custom_Chart(chart.topbar['symbol'].value, time_frame_manipulation(chart.topbar['timeframe'].value, 14, max_points=400))
    elif chart.topbar['algo'].value == 'Deep_learning':
        if checkIfModelExists('Deep_learning',chart.topbar['symbol'].value,time_frame_manipulation(chart.topbar['timeframe'].value)) == False:
            show_model_not_trained_dialog()


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


async def recalculate_dataset(df):
    await perform_regression_analysis(df)
    calculate_sma(df)

async def on_search(chart, searched_string):
    global fetch_data
    new_data = await fetch_data(searched_string, chart.topbar['timeframe'].value)
    if new_data is None or new_data.empty:
        return

    chart.topbar['symbol'].set(searched_string)
    update_algo_table(chart)
    chart.watermark(searched_string + ' ' + chart.topbar['timeframe'].value)

    if hasattr(chart, 'regression_line'):
        await recalculate_dataset(new_data)

    chart.set(new_data)

async def on_timeframe_selection(chart):
    global fetch_data
    new_data = await fetch_data(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
    if new_data is None or new_data.empty:
        return

    chart.watermark(chart.topbar['symbol'].value + ' ' + chart.topbar['timeframe'].value)
    update_algo_table(chart)
    if hasattr(chart, 'regression_line'):
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
            if isLoggedin:
                await fetch_user_portfolio()
            if hasattr(chart, 'regression_line'):
                await recalculate_dataset(new_data)

    auto_refresh_task = asyncio.create_task(refresh(chart))


def on_row_click(row):
    if row['Model'] == 'Regression':
        regression_plot(Chart.topbar['symbol'].value, time_frame_manipulation(Chart.topbar['timeframe'].value))
    elif row['Model'] == 'TimeSeries':
        time_series_analysis(Chart.topbar['symbol'].value, time_frame_manipulation(Chart.topbar['timeframe'].value))
    elif row['Model'] == 'LSMT':
        if checkIfModelExists('LSMT',Chart.topbar['symbol'].value,time_frame_manipulation(Chart.topbar['timeframe'].value)) == False:
            show_model_not_trained_dialog()
        else:
            import sys
            import os
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
            from Run_LSTM_Model import run_model_and_show_dialog
            run_model_and_show_dialog(Chart.topbar['symbol'].value, time_frame_manipulation(Chart.topbar['timeframe'].value))
    elif row['Model'] == 'custom_algorithm':
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
        from custom_algorithm import show_custom_Chart
        show_custom_Chart(Chart.topbar['symbol'].value, time_frame_manipulation(Chart.topbar['timeframe'].value))
    elif row['Model'] == 'Deep_learning':
        if checkIfModelExists('Deep_learning',Chart.topbar['symbol'].value,time_frame_manipulation(Chart.topbar['timeframe'].value)) == False:
            show_model_not_trained_dialog()

def create_algo_table(chart):
    global algo_names
    table = Chart.create_table(width=0.3, height=0.15,
                headings=('Model', 'Accuracy'),
                widths=(0.2, 0.2),
                alignments=('center', 'center'),
                position='right', func=on_row_click)

    table.header(1)
    table.header[0] = 'Prediction Table'
    table.format('Accuracy', f'{table.VALUE} %')

    data = fetch_symbol_model_value(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
    if isinstance(data, dict):
        for model, accuracy in data.items():
            table.new_row(model, accuracy)

    table.footer(2)
    table.footer[0] = 'Nepse Probability:'
    table.footer[1] = basic_prediction_result[0]['prediction']

    chart.algo_table = table

def update_algo_table(chart):
    global algo_names
    if hasattr(chart, 'algo_table'):
        table = chart.algo_table
        table.clear()

        data = fetch_symbol_model_value(chart.topbar['symbol'].value, chart.topbar['timeframe'].value)
        if isinstance(data, dict):
            for model, accuracy in data.items():
                table.new_row(model, accuracy)

        table.footer[1] = basic_prediction_result[0]['prediction']

def create_portfolio_table(chart):
    table = chart.create_table(width=0.3, height=0.2,
                               headings=('Symbol', 'Quantity', 'WACC', 'Total Cost', 'Current Value', 'P&L','%'),
                               widths=(0.2, 0.1, 0.1, 0.2, 0.2, 0.2,0.2),alignments=('center', 'center', 'center', 'center', 'center','center'),func=on_portfolio_row_click)
    table.format('WACC', f'Rs  {table.VALUE}')
    table.format('Total Cost', f'Rs  {table.VALUE}')
    table.format('Current Value', f'Rs  {table.VALUE}')
    table.format('P&L', f'Rs  {table.VALUE}')
    table.format('%', f'{table.VALUE}%')

    table.new_row('No Stocks', 0, 0, 0, 0, 0,0)
    table.header(1)
    table.header[0] = userData['name'] + ' Portfolio'

    table.footer(7)
    table.footer[0] = 'Total:'
    table.footer[1] = str(userPortfolio['totalunits'])+" Unit"
    table.footer[2] = ' '
    table.footer[3] = "Rs "+ str(userPortfolio['portfoliocost'])
    table.footer[4] = "Rs "+ str(userPortfolio['portfoliovalue'])
    table.footer[5] = "Rs "+ str(userPortfolio['portgainloss'])
    table.footer[6] = str(userPortfolio['portfolioPercentage']) + "%"

    chart.portfolio_table = table

def clear_portfolio_table(chart):
    if hasattr(chart, 'portfolio_table'):
        table = chart.portfolio_table
        table.clear()

        table.header[0] = userData['name'] + ' Portfolio'
        table.new_row('No Stocks', 0, 0, 0, 0, 0,0)
        table.footer[0] = 'Total:'
        table.footer[1] = str(userPortfolio['totalunits'])+" Unit"
        table.footer[2] = ' '
        table.footer[3] = "Rs "+ str(userPortfolio['portfoliocost'])
        table.footer[4] = "Rs "+ str(userPortfolio['portfoliovalue'])
        table.footer[5] = "Rs "+ str(userPortfolio['portgainloss'])
        table.footer[6] = str(userPortfolio['portfolioPercentage']) + "%"

def update_portfolio_table(chart):
    if hasattr(chart, 'portfolio_table'):

        table = chart.portfolio_table
        table.clear()
        table.header[0] = userData['name'] + ' Portfolio '+ ' (Balance Rs '+ str(userData['userAmount']) + ')'

        for stock in userPortfolio['stocks']:
            row = table.new_row(stock['symbol'], stock['quantity'], stock['wacc'], stock['costprice'],stock['currentprice'], stock['netgainloss'],stock['netgainlossPercent'])
            row.background_color('P&L', 'green' if row['P&L'] > 0 else ('grey' if row['P&L'] == 0 else 'red'))

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
    global isLoggedin, fetch_data, Chart
    if isLoggedin:
        new_data = await fetch_data(row['Symbol'], Chart.topbar['timeframe'].value)
        if new_data is None or new_data.empty:
            return
        Chart.topbar['symbol'].set(row['Symbol'])
        update_algo_table(Chart)
        Chart.watermark(row['Symbol'] + ' ' + Chart.topbar['timeframe'].value)
        if hasattr(Chart, 'regression_line'):
            await recalculate_dataset(new_data)

        Chart.set(new_data)

async def main():
    global Chart, fetch_data,algo_names

    df = await fetch_data(defaultSymbol, defaultTimeFrame)
    if df is None:
        print('No data to display')
        exit(1)

    Chart = Chart(toolbox=False, maximize=True, inner_width=0.7, inner_height=1, title='10Paisa AI Dashboard')
    Chart.legend(visible = True, font_family = 'Trebuchet MS', ohlc = True, percent = True)
    Chart.grid(vert_enabled = True, horz_enabled = True)
    Chart.watermark(defaultSymbol + ' ' + defaultTimeFrame)
    Chart.topbar.button('login', 'Login', func= login_user)
    Chart.topbar.textbox('symbol', defaultSymbol)
    Chart.topbar.menu('algo', ('Algorithms', *algo_names), default='Algorithms', func=showAlgorithmGUI)
    Chart.topbar.switcher('timeframe', ('1Min', '5Min', '10Min', '15Min','1D'), default=defaultTimeFrame, func= on_timeframe_selection)
    Chart.events.search += on_search
    Chart.topbar.button('screenshot', 'Screenshot', func= take_screenshot)
    Chart.topbar.button('autorefresh_button', 'AutoRefresh On', func=auto_refresh)
    Chart.topbar.button('autoretrade_button', 'AutoTrade Off', func= auto_trade)
    Chart.topbar.button('buy_buttton', 'Buy', func=buy_stock )
    Chart.topbar.button('sell', 'Sell', func= sell_stock)
    Chart.topbar.textbox('Settings')


#subchart
    subchart = Chart.create_subchart(width=0.3, height=0.4)
    subchart.watermark(SubchartSecurity + ' ' + SubchartTimeFrame)
    subchart.topbar.textbox('symbol', SubchartSecurity)
    subchart.topbar.switcher('timeframe', ('1Min', '5Min', '10Min'), default=SubchartTimeFrame, func= on_timeframe_selection)
    subchart.events.search += on_search
    subchart.topbar.button('autorefresh_button', 'AutoRefresh Off', func=auto_refresh)
    #subchart.topbar.button('max', FULLSCREEN, False, align='right', func=on_max)

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

#fetch prediction data
    await fetch_prediction_data()
    #table2.footer[1] = basic_prediction_result[0]['prediction']

#table3 #Total Accuracy table
    create_algo_table(Chart)
#table #portfolio table
    create_portfolio_table(Chart)
#end of table

#perform regression analysis
    await perform_regression_analysis(df)

    Chart.set(df)
    subchart.set(df2)

   # Chart.marker(text=f"Predicted Next Index Value")

    await asyncio.gather(Chart.show_async(block=True), auto_refresh(Chart))
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

if __name__ == '__main__':
    asyncio.run(main())
