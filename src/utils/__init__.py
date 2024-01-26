import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.analysis.performance_metrics import cagr, mdd, sharpe

class TraditionalPairsTrading:
    def __init__(self, symbol1, symbol2, start_date, end_date):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.start_date = start_date
        self.end_date = end_date
        self.df_pair = None
