import pandas as pd
import numpy as np
import itertools
import datetime
import matplotlib.pyplot as plt
import math

import yfinance as yf
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates
import seaborn as sns

from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm

# from google.colab import files, output

from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
import datetime as dt
import pandas as pd
import math
import os.path
import time
import json
import requests
import pandas_market_calendars as mcal
from datetime import timedelta, datetime
from dateutil import parser
import seaborn as sns
import matplotlib as mpl
import quantstats as qs
import statsmodels.api as sm
from pykalman import KalmanFilter
from math import sqrt
import warnings
import ffn
import pyfolio as pf
from pandas_datareader import data as web


def cagr(returns, resolution):
    return np.prod(1 + returns) ** (resolution / len(returns)) - 1


def mdd(returns):
    cum_rets = (1 + returns).cumprod()
    max_cumret = cum_rets.cummax()
    drawdown = 1 - cum_rets / max_cumret
    return np.max(drawdown)


def sharpe(returns, riskfreerate=0, resolution=252):
    return (
        np.sqrt(resolution)
        * (np.mean(returns) - riskfreerate / resolution)
        / (np.std(returns))
    )


## Retrieve and format data

window = 20
entryzscore = 2
exitzscore = 0.5
trading_cost = 0.0000

close1 = yf.download("TLT", start="2017-09-01", end="2023-11-26")["Close"]
close2 = yf.download("IEI", start="2017-09-01", end="2023-11-26")["Close"]
df_pair = pd.DataFrame({"ADJ_PRC1": close1, "ADJ_PRC2": close2})

df_pair["spread"] = df_pair["ADJ_PRC2"] - df_pair["ADJ_PRC1"]
df_pair["spread_mean"] = df_pair["spread"].rolling(window).mean()
df_pair["spread_std"] = df_pair["spread"].rolling(window).std()
df_pair["RET1"] = df_pair["ADJ_PRC1"].pct_change().shift(-1)
df_pair["RET2"] = df_pair["ADJ_PRC2"].pct_change().shift(-1)
df_pair["spread_zscore"] = (df_pair["spread"] - df_pair["spread_mean"]) / df_pair[
    "spread_std"
]
df_pair.dropna(inplace=True)

## Determine holdings based on strategy

df_pair["position2"] = np.NaN
df_pair.loc[df_pair["spread_zscore"] > entryzscore, "position2"] = -0.5
df_pair.loc[df_pair["spread_zscore"] < -entryzscore, "position2"] = 0.5
df_pair.loc[np.abs(df_pair["spread_zscore"]) < exitzscore, "position2"] = 0
df_pair["position2"].fillna(method="ffill", inplace=True)
df_pair["position1"] = -df_pair["position2"]

## Calculate returns

df_rets = df_pair[["RET1", "RET2"]].rename(columns={"RET1": "one", "RET2": "two"})
df_weights = df_pair[["position1", "position2"]].rename(
    columns={"position1": "one", "position2": "two"}
)

rets = (df_rets * df_weights).sum(axis=1).fillna(0)
costs = df_weights.diff().abs().sum(axis=1).fillna(0)
costs.iloc[0] = df_weights.iloc[0].sum()
rets = rets - trading_cost * costs

df_pair["Portfolio"] = np.cumprod(1 + rets)

## Plot results

print(
    "The Sharpe ratio of our strategy is {}.".format(round(sharpe(rets, 0.025, 252), 3))
)
print("The maximum drawdown is {}.".format(round(mdd(rets), 3)))

fig, ax = plt.subplots(1, 1)

ax.plot(df_pair["ADJ_PRC1"], color="#002147")
ax.plot(df_pair["ADJ_PRC2"], color="#A3C1AD")

ax1 = ax.twinx()
ax1.plot(df_pair["Portfolio"], color="black")

## Adding legend, x-axis & y-axis label, title

## Retrieve and format data

window = 20
entryzscore = 2
exitzscore = 0.5
trading_cost = 0.0000

close1 = yf.download("TLT", start="2017-09-01", end="2023-11-26")["Close"]
close2 = yf.download("IEI", start="2017-09-01", end="2023-11-26")["Close"]
df_pair = pd.DataFrame({"ADJ_PRC1": close1, "ADJ_PRC2": close2})

df_pair["spread"] = df_pair["ADJ_PRC2"] - df_pair["ADJ_PRC1"]
df_pair["spread_mean"] = df_pair["spread"].rolling(window).mean()
df_pair["spread_std"] = df_pair["spread"].rolling(window).std()
df_pair["RET1"] = df_pair["ADJ_PRC1"].pct_change().shift(-1)
df_pair["RET2"] = df_pair["ADJ_PRC2"].pct_change().shift(-1)
df_pair["spread_zscore"] = (df_pair["spread"] - df_pair["spread_mean"]) / df_pair[
    "spread_std"
]
df_pair.dropna(inplace=True)

## Determine holdings based on strategy

df_pair["position2"] = np.NaN
df_pair.loc[df_pair["spread_zscore"] > entryzscore, "position2"] = -0.5
df_pair.loc[df_pair["spread_zscore"] < -entryzscore, "position2"] = 0.5
df_pair.loc[np.abs(df_pair["spread_zscore"]) < exitzscore, "position2"] = 0
df_pair["position2"].fillna(method="ffill", inplace=True)
df_pair["position1"] = -df_pair["position2"]

## Calculate returns

df_rets = df_pair[["RET1", "RET2"]].rename(columns={"RET1": "one", "RET2": "two"})
df_weights = df_pair[["position1", "position2"]].rename(
    columns={"position1": "one", "position2": "two"}
)

rets = (df_rets * df_weights).sum(axis=1).fillna(0)
costs = df_weights.diff().abs().sum(axis=1).fillna(0)
costs.iloc[0] = df_weights.iloc[0].sum()
rets = rets - trading_cost * costs

df_pair["Portfolio"] = np.cumprod(1 + rets)

## Plot results

print(
    "The Sharpe ratio of our strategy is {}.".format(round(sharpe(rets, 0.025, 252), 3))
)
print("The maximum drawdown is {}.".format(round(mdd(rets), 3)))

fig, ax = plt.subplots(1, 1)

ax.plot(df_pair["ADJ_PRC1"], color="#002147")
ax.plot(df_pair["ADJ_PRC2"], color="#A3C1AD")

ax1 = ax.twinx()
ax1.plot(df_pair["Portfolio"], color="black")

## Adding legend, x-axis & y-axis label, title

pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")
symbols = [
    "GDX",
    "GDXJ",
    "GLD",
    "AAPL",
    "GOOGL",
    "META",
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
yf.pdr_override()
## Eliminate outdated ticker and provide alternative datasets for the group to analyze


def get_symbols(symbols, data_source, ohlc, begin_date=None, end_date=None):
    out = []
    new_symbols = []
    for symbol in symbols:
        df = web.get_data_yahoo(symbol, start=begin_date, end=end_date)
        df = df[ohlc]
        new_symbols.append(symbol)
        out.append(df.astype("float"))
        data = pd.concat(out, axis=1)
        data.columns = new_symbols
        data = data.dropna(axis=1)
    return data.dropna(axis=1)


start = pd.Timestamp("2015-01-01")
end = pd.Timestamp("2023-11-24")


prices = get_symbols(
    symbols, data_source="yahoo", ohlc="Close", begin_date=start, end_date=end
)

start = pd.Timestamp("2015-01-01")
end = pd.Timestamp("2023-11-24")


prices = get_symbols(
    symbols, data_source="yahoo", ohlc="Close", begin_date=start, end_date=end
)

combo = prices.copy()
combo.index = pd.DatetimeIndex(combo.index)
combo.head()

combo.info()

num_stocks = len(combo.columns)
print("Number of Stocks =", num_stocks)

n_secs = len(combo.columns)
colors = cm.rainbow(np.linspace(0, 1, n_secs))
combo.div(combo.iloc[0, :]).plot(color=colors, figsize=(12, 6))  # Normalize Prices
plt.title("All Stocks Normalized Price Series")
plt.xlabel("Date")
plt.ylabel("Price (USD$)")
plt.grid(which="major", axis="both")
plt.legend(bbox_to_anchor=(1.01, 1.1), loc="upper left", ncol=1)
plt.show()

## Consider getting rid of the noisiest assets? Their pairs return rely mostly on the high vol and is not represenative of the portfolio


def KalmanFilterAverage(x):
    # Construct a Kalman filter
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=0,
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01,
    )
    # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(), index=x.index)
    return state_means


# Kalman filter regression
def KalmanFilterRegression(x, y):
    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2)  # How much random walk wiggles
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,  # y is 1-dimensional, (alpha, beta) is 2-dimensional
        initial_state_mean=[0, 0],
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=2,
        transition_covariance=trans_cov,
    )
    # Use the observations y to get running estimates and errors for the state parameters
    state_means, state_covs = kf.filter(y.values)
    return state_means


def half_life(spread):
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    spread_lag2 = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret, spread_lag2)
    res = model.fit()
    halflife = int(round(-np.log(2) / res.params[1], 0))
    if halflife <= 0:
        halflife = 1
    return halflife


def backtest(df, s1, s2):
    #############################################################
    # INPUT:
    # DataFrame of prices (df)
    # s1: the symbol of asset one
    # s2: the symbol of asset two
    # x: the price series of asset one
    # y: the price series of asset two
    # OUTPUT:
    # df1['cum rets']: cumulative returns in pandas data frame
    # sharpe: Sharpe ratio
    # CAGR: Compound Annual Growth Rate

    x = df[s1]
    y = df[s2]

    # Run regression (including Kalman Filter) to find hedge ratio and then create spread series
    df1 = pd.DataFrame({"y": y, "x": x})
    df1.index = pd.to_datetime(df1.index)
    state_means = KalmanFilterRegression(KalmanFilterAverage(x), KalmanFilterAverage(y))
    df1["hr"] = -state_means[:, 0]
    df1["spread"] = df1.y + (df1.x * df1.hr)

    # calculate half life
    halflife = half_life(df1["spread"])

    # calculate z-score with window = half life period
    meanSpread = df1.spread.rolling(window=halflife).mean()
    stdSpread = df1.spread.rolling(window=halflife).std()
    df1["zScore"] = (df1.spread - meanSpread) / stdSpread

    ##############################################################

    # trading logic
    entryZscore = 1.25
    exitZscore = -0.08

    # set up num units long
    df1["long entry"] = (df1.zScore < -entryZscore) & (
        df1.zScore.shift(1) > -entryZscore
    )
    df1["long exit"] = (df1.zScore > -exitZscore) & (df1.zScore.shift(1) < -exitZscore)
    df1["num units long"] = np.nan
    df1.loc[df1["long entry"], "num units long"] = 1
    df1.loc[df1["long exit"], "num units long"] = 0
    df1["num units long"][0] = 0
    df1["num units long"] = df1["num units long"].fillna(method="pad")

    # set up num units short
    df1["short entry"] = (df1.zScore > entryZscore) & (
        df1.zScore.shift(1) < entryZscore
    )
    df1["short exit"] = (df1.zScore < exitZscore) & (df1.zScore.shift(1) > exitZscore)
    df1.loc[df1["short entry"], "num units short"] = -1
    df1.loc[df1["short exit"], "num units short"] = 0
    df1["num units short"][0] = 0
    df1["num units short"] = df1["num units short"].fillna(method="pad")

    # set up totals: num units and returns
    df1["numUnits"] = df1["num units long"] + df1["num units short"]
    df1["spread pct ch"] = (df1["spread"] - df1["spread"].shift(1)) / (
        (df1["x"] * abs(df1["hr"])) + df1["y"]
    )
    df1["port rets"] = df1["spread pct ch"] * df1["numUnits"].shift(1)
    df1["cum rets"] = df1["port rets"].cumsum()
    df1["cum rets"] = df1["cum rets"] + 1

    ##############################################################

    try:
        sharpe = (df1["port rets"].mean() / df1["port rets"].std()) * sqrt(252)
    except ZeroDivisionError:
        sharpe = 0.0

    ##############################################################

    start_val = 1
    end_val = df1["cum rets"].iat[-1]
    start_date = df1.iloc[0].name
    end_date = df1.iloc[-1].name
    days = (end_date - start_date).days
    CAGR = (end_val / start_val) ** (252.0 / days) - 1

    df1[s1 + " " + s2 + "_cum_rets"] = df1["cum rets"]

    return df1[s1 + " " + s2 + "_cum_rets"], sharpe, CAGR


# NOTE CRITICAL LEVEL HAS BEEN SET TO 5% FOR COINTEGRATION TEST
def find_cointegrated_pairs(dataframe, critial_level=0.05):
    n = dataframe.shape[1]  # the length of dateframe
    pvalue_matrix = np.ones((n, n))  # initialize the matrix of p
    keys = dataframe.columns  # get the column names
    pairs = []  # initilize the list for cointegration
    for i in range(n):
        for j in range(i + 1, n):  # for j bigger than i
            stock1 = dataframe[keys[i]]  # obtain the price of "stock1"
            stock2 = dataframe[keys[j]]  # obtain the price of "stock2"
            result = sm.tsa.stattools.coint(stock1, stock2)  # get conintegration
            pvalue = result[1]  # get the pvalue
            pvalue_matrix[i, j] = pvalue
            if pvalue < critial_level:  # if p-value less than the critical level
                pairs.append(
                    (keys[i], keys[j], pvalue)
                )  # record the contract with that p-value
    return pvalue_matrix, pairs

    ## Examine to make sure that the criticial test is performed as desired

    # NOTE CRITICAL LEVEL HAS BEEN SET TO 5% FOR COINTEGRATION TEST


def find_cointegrated_pairs(dataframe, critial_level=0.05):
    n = dataframe.shape[1]  # the length of dateframe
    pvalue_matrix = np.ones((n, n))  # initialize the matrix of p
    keys = dataframe.columns  # get the column names
    pairs = []  # initilize the list for cointegration
    for i in range(n):
        for j in range(i + 1, n):  # for j bigger than i
            stock1 = dataframe[keys[i]]  # obtain the price of "stock1"
            stock2 = dataframe[keys[j]]  # obtain the price of "stock2"
            result = sm.tsa.stattools.coint(stock1, stock2)  # get conintegration
            pvalue = result[1]  # get the pvalue
            pvalue_matrix[i, j] = pvalue
            if pvalue < critial_level:  # if p-value less than the critical level
                pairs.append(
                    (keys[i], keys[j], pvalue)
                )  # record the contract with that p-value
    return pvalue_matrix, pairs

    ## Examine to make sure that the criticial test is performed as desired

    # Use Seaborn to plot a heatmap of our results matrix


sns.clustermap(
    pvalue_matrix_df,
    xticklabels=binance_symbols,
    yticklabels=binance_symbols,
    figsize=(12, 12),
)
plt.title("Stock P-value Matrix")
plt.tight_layout()
plt.show()

results = []
for pair in pairs:
    rets, sharpe, CAGR = backtest(df[split:], pair[0], pair[1])
    results.append(rets)
    print(
        "The pair {} and {} produced a Sharpe Ratio of {} and a CAGR of {}".format(
            pair[0], pair[1], round(sharpe, 2), round(CAGR, 4)
        )
    )
    rets0 = pd.concat(results, axis=1)

rets0.plot(figsize=(12, 6), legend=True)
plt.legend(bbox_to_anchor=(1.01, 1.1), loc="upper left", ncol=1)
plt.grid(which="major", axis="both")
plt.title("Pairs Returns")
plt.xlabel("Date")
plt.ylabel("Returns")

filename = "pairs_rets.csv"
rets0.to_csv(filename)
# concatenate together the individual equity curves into a single DataFrame
results_df = pd.concat(results, axis=1).dropna()

# equally weight each equity curve by dividing each by the number of pairs held in the DataFrame
results_df /= len(results_df.columns)

# sum up the equally weighted equity curves to get our final equity curve
final_res = results_df.sum(axis=1)

# square root of sample size for correct number of bins for returns distribution
print("Bin Count =", np.sqrt(len(final_res)))

## checking the bins actually do what they meant to do

Pair_Rets = ffn.to_returns(final_res)
Pair_Rets = pd.DataFrame(Pair_Rets)
Pair_Rets = Pair_Rets.fillna(0)
Pair_Rets.columns = ["Pairs_Returns"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.distplot(
    Pair_Rets,
    hist=True,
    kde=True,
    bins=35,
    hist_kws={"linewidth": 1, "alpha": 0.5},
    label="Pairs Returns",
    color="#4b91fa",
    ax=ax1,
)

ax1.axvline(
    x=0.000709,
    color="#ff0000",
    linewidth=1.25,
    linestyle="dashed",
    label="Returns Mean",
)

ax1.set_title("Pairs Returns Distribution")
ax1.margins(0.001)
ax1.set_xlabel("Returns (%)")
ax1.set_ylabel("Density")

stats.probplot(Pair_Rets.Pairs_Returns, plot=ax2)

plt.tight_layout()

plt.show()

##  fixing the plt to obtain the graph

Pair_Rets.Pairs_Returns.describe()

perf = final_res.calc_stats()

fig = pf.create_returns_tear_sheet(Pair_Rets.Pairs_Returns)
## change the code as to_pydatetime is outdated

bench = (
    df.loc[str(Pair_Rets.index[0]) : str(Pair_Rets.index[-1])].SPY.pct_change().dropna()
)
Pair_Rets0 = Pair_Rets.loc[str(bench.index[0]) : str(bench.index[-1])]
## Fix the algo to compare the portfolio to the S&P 500 returns
