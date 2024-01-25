# main.py

import pandas as pd
import warnings
import math
import yfinance as yf
import 

from src.analysis.performance_metrics import cagr, mdd, sharpe
from src.data.data_fetcher import DataFetcher
from src.data.data_processing import DataProcessor
from src.strategy.traditional_pairs_trading import TraditionalPairsTrading
from strategy.kalman_pairs_trading



# pd.set_option('display.max_columns', None)
# warnings.filterwarnings('ignore')
# symbols = ['GDX','GDXJ','GLD', 'AAPL','GOOGL', 'META','AMD',
#            'NVDA','CSCO', 'ORCL', 'ATVI', 'TTWO', 'EA', 'HYG',
#            'LQD', 'JNK', 'SLV', 'USLV', 'SIVR', 'USO', 'UWT',
#            'QQQ', 'SPY', 'VOO', 'VDE', 'VTI', 'EMLP', 'VDC',
#            'FSTA', 'KXI', 'IBB', 'VHT','VNQ', 'IYR', 'MSFT',
#            'PG', 'TMF', 'UPRO', 'WFC', 'JPM', 'GS', 'CVX',
#            'XOM', 'INTC', 'COST', 'WMT', 'T', 'VZ', 'CMCSA', 'AMZN']
# yf.pdr_override()


def main():
    try:
        fetcher = DataFetcher("2017-09-01", "2023-11-26")
        close1 = fetcher.download_data("TLT")
        close2 = fetcher.download_data("IEI")
        
        processor = DataProcessor(close1, close2)
        df_pair = processor.prepare_data()
        
        results = []
        for pair in pairs

        bench = df.loc[str(Pair_Rets.index[0]):str(Pair_Rets.index[-1])].SPY.pct_change().dropna()
        Pair_Rets0 = Pair_Rets.loc[str(bench.index[0]):str(bench.index[-1])]
        ## Fix the algo to compare the portfolio to the S&P 500 returns
        # ...rest of workflow here

    except Exception as e:
        print(f"An error occured: {e}")


if __name__ == "__main__":
    main()
