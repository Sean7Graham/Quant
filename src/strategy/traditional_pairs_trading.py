# traditional_pairs_trading.py

from src.data.data_fetcher import DataFetcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.analysis.performance_metrics import sharpe, mdd


class TraditionalTradingStrategy:
    def __init__(self, symbol1, symbol2, start_date, end_date, trading_cost=0.000):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.data_fetcher = DataFetcher(start_date, end_date)
        self.df_pair = None
        self.trading_cost = trading_cost

    def run_strategy(self):
        close1 = self.data_fetcher.download_data(self.symbol1)
        close2 = self.data_fetcher.download_data(self.symbol2)
        self.df_pair = pd.DataFrame(
            {"ADJ_PRC1": close1["Close"], "ADJ_PRC2": close2["Close"]}
        )

        # ... Additional strategy logic like calculating spread, positions, returns ...

        self.calculate_spread()
        self.determine_positions()
        self.calculate_returns()
        self.plot_results()

    # ... Other methods for calculating spread, positions, returns, plotting ...
