import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from scipy.stats import gmean

def get_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def plot_stock_chart(stock_data, title, ax):
    if stock_data is not None:
        ax.plot(stock_data, label=title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Price')
        ax.legend()
        ax.grid()

if __name__ == "__main__":
    tickers = [ '2464.T', '3153.T', '2768.T']   # 日本の銘柄のティッカーシンボルのリスト
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2023, 4, 20)

    fig, ax = plt.subplots(figsize=(10, 5))

    stock_data_list = []

    for ticker in tickers:
        stock_data = get_stock_data(ticker, start_date, end_date)
        if stock_data is not None:
            plot_stock_chart(stock_data['Close'], ticker, ax)
            stock_data_list.append(stock_data['Close'])

    # Calculate the average of the stock prices
    avg_stock_data = pd.concat(stock_data_list, axis=1).mean(axis=1)
    plot_stock_chart(avg_stock_data.to_frame(), 'Average', ax)

    # Calculate the geometric mean of the stock prices
    geom_mean_stock_data = pd.concat(stock_data_list, axis=1).apply(gmean, axis=1)
    plot_stock_chart(geom_mean_stock_data.to_frame(), 'Geometric Mean', ax)

    plt.show()
