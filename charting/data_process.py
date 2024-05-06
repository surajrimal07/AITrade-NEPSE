import pandas as pd
import json
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

def process_json_data(raw_data,manipulatedTimeFrame,orginaltimeFrame):
    timestamps = raw_data['t']
    open_prices = raw_data['o']
    high_prices = raw_data['h']
    low_prices = raw_data['l']
    close_prices = raw_data['c']
    volumes = raw_data['v']

    date_times = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert('Asia/Kathmandu')
    #df = pd.DataFrame({'Timestamp': date_times})

    df = pd.DataFrame({
    #'date': date_times,
    #'date': date_times.strftime('%Y-%m-%d' if orginaltimeFrame == '1D' else '%Y-%m-%d %H:%M:%S'),
    'date': pd.to_datetime(date_times.strftime('%Y-%m-%d' if orginaltimeFrame == '1D' else '%Y-%m-%d %H:%M:%S')),
    'open': open_prices,
    'high': high_prices,
    'low': low_prices,
    'close': close_prices,
    'volume': volumes
})

    print(df['date'].min())
    print(df['date'].max())

    df = df.drop_duplicates(subset=['date'])
    #resampling data for other timeframes
    if manipulatedTimeFrame == '1Min':
        df = df
    elif manipulatedTimeFrame == '5Min':
        df = df.iloc[::5]

        df['date'] = df['date'].values
        df['open'] = df['open'].groupby(df.index // 5).first()
        df['high'] = df['high'].groupby(df.index // 5).max()
        df['low'] = df['low'].groupby(df.index // 5).min()
        df['close'] = df['close'].groupby(df.index // 5).last()
        df['volume'] = df['volume'].groupby(df.index // 5).sum()

    elif manipulatedTimeFrame == '10Min':
        df_resampled = df.resample('10T').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    elif manipulatedTimeFrame == '15Min':
        df_resampled = df.resample('15T').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    elif manipulatedTimeFrame == '1H':
        df_resampled = df.resample('1H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    elif manipulatedTimeFrame == '2H':
        df_resampled = df.resample('2H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    elif manipulatedTimeFrame == '1D':
        df_resampled = df

    elif manipulatedTimeFrame == '2D':
        df_resampled = df.resample('2D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    elif orginaltimeFrame == '1W':
        manipulatedTimeFrame = df.resample('1W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    else:
        raise ValueError("Invalid time frame specified.")

    return df