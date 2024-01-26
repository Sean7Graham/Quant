import pandas as pd

from src.data.data_fetcher import DataFetcher

symbols = [
    "GDX",
    "GDXJ",
    "GLD",
    "AAPL",
    "GOOGL",
    "FB",
    "TWTR",
    "AMD",
    "NVDA",
    "CSCO",
    "ORCL",
    "ATVI",
    "TTWO",
    "EA",
    "HYG",
    "LQD",
    "JNK",
    "SLV",
    "USLV",
    "SIVR",
    "USO",
    "UWT",
    "QQQ",
    "SPY",
    "VOO",
    "VDE",
    "VTI",
    "EMLP",
    "VDC",
    "FSTA",
    "KXI",
    "IBB",
    "VHT",
    "VNQ",
    "IYR",
    "MSFT",
    "PG",
    "TMF",
    "UPRO",
    "WFC",
    "JPM",
    "GS",
    "CVX",
    "XOM",
    "INTC",
    "COST",
    "WMT",
    "T",
    "VZ",
    "CMCSA",
    "AMZN",
]

start = pd.Timestamp("2014-01-01")
end = pd.Timestamp("2020-03-05")

fetcher = DataFetcher(start, end, symbols)

prices = fetcher.get_symbols(
    symbols, data_source="yahoo", ohlc="Close", begin_date=start, end_date=end
)

combo = prices.copy()
