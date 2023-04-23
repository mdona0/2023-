import yfinance as yf
import datetime
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.dates as mdates

def get_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def get_interest_rate_data(start_date, end_date):
    try:
        interest_rate_data = pdr.get_data_fred('GS10', start_date, end_date)  # 10-year government bond interest rate
        interest_rate_data.reset_index(inplace=True)
        interest_rate_data.rename(columns={'DATE': 'Date'}, inplace=True)
        return interest_rate_data
    except Exception as e:
        print(f"Error fetching interest rate data: {e}")
        return None

def plot_stock_data_3d(stock_data, interest_rate_data, ticker, ax):
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    interest_rate_data['DATE'] = pd.to_datetime(interest_rate_data['DATE'])


def plot_stock_data_3d(stock_data, interest_rate_data, ticker, ax):
    # Check if 'Date' column exists in stock_data
    if 'Date' not in stock_data.columns:
        stock_data = stock_data.rename(columns={'DATE': 'Date'})
    
    # Check if 'Date' column exists in interest_rate_data
    if 'Date' not in interest_rate_data.columns:
        interest_rate_data = interest_rate_data.rename(columns={'DATE': 'Date'})

    # Try converting Date columns to datetime type
    try:
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    except KeyError:
        pass

    try:
        interest_rate_data['Date'] = pd.to_datetime(interest_rate_data['Date'])
    except KeyError:
        pass

    # Merge the stock and interest rate data
    merged_data = pd.merge(stock_data, interest_rate_data, on='Date', how='inner')

    # Convert dates to numerical values
    date_numeric = mdates.date2num(merged_data['Date'].to_numpy())

    # Plot the 3D graph
    ax.plot(date_numeric, merged_data['Close'].to_numpy(), merged_data['GS10'].to_numpy(), label=ticker)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close')
    ax.set_zlabel('GS10')
    ax.legend()

if __name__ == "__main__":
    tickers = ['7203.T', '9984.T', '6758.T']  # 日本の銘柄のティッカーシンボルのリスト
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2023, 4, 20)

    interest_rate_data = get_interest_rate_data(start_date, end_date)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')

    all_stock_data = []
    for ticker in tickers:
        stock_data = get_stock_data(ticker, start_date, end_date)
        all_stock_data.append(stock_data)
        plot_stock_data_3d(stock_data, interest_rate_data, ticker, ax)

    # Calculate arithmetic mean and geometric mean
    merged_data = pd.concat(all_stock_data, keys=tickers)
    merged_data = merged_data[['Close']].reset_index()
    stock_data1 = yf.download(tickers[0], start=start_date, end=end_date).reset_index()
    stock_data2 = yf.download(tickers[1], start=start_date, end=end_date).reset_index()
    stock_data3 = yf.download(tickers[2], start=start_date, end=end_date).reset_index()

    arithmetic_mean = pd.DataFrame({'Date': stock_data1['Date'], 'Close': (stock_data1['Close'] + stock_data2['Close'] + stock_data3['Close']) / 3})
    geometric_mean = pd.DataFrame({'Date': stock_data1['Date'], 'Close': np.cbrt(stock_data1['Close'] * stock_data2['Close'] * stock_data3['Close'])})

    plot_stock_data_3d(arithmetic_mean, interest_rate_data, 'Arithmetic Mean', ax)
    plot_stock_data_3d(geometric_mean, interest_rate_data, 'Geometric Mean', ax)

    plt.show()

