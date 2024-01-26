# data_fetcher.py

import yfinance as yf
from pandas_datareader import data as web
import pandas as pd
import warnings


class DataFetcher:
    def __init__(self, start_date, end_date, symbols):
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        
        pd.set_option('display.max_columns', None)
        warnings.filterwarnings('ignore')
        yf.pdr_override()


    def download_data(self, ticker):
        try:
            data = yf.download(ticker, start=self.start_date, end=self.end_date)
            return data
        except Exception as e:
            print(f"Error while downloading {ticker}: {e}")
            return None
        
    def get_symbols(self, ohlc):
        out = []
        new_symbols = []
        for symbol in self.symbols:
            try:
                df = web.get_data_yahoo(symbol, start=self.start_date, end=self.end_date)
                df = df[ohlc]
                new_symbols.append(symbol)
                out.append(df.astype('float'))
            except Exception as e:
                print(f"An error occurred while downloading {symbol}: {e}")
                continue
        if out:
            data = pd.concat(out, axis=1)
            data.columns = new_symbols
            return data.dropna(axis=1)
        else:
            return pd.DataFrame()