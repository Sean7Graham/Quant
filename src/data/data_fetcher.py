# data_fetcher.py

import yfinance as yf

# def fetch_data(ticker, start_date, end_date):

#     data = 0
#     # .. code to fetch the data ..

#     return data


class DataFetcher:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self, ticker):
        return yf.download(ticker, start=self.start_date, end=self.end_date)
