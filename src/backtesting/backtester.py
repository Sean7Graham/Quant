# backtester.py

import pandas as pd
from pykalman import KalmanFilter

from strategy.kalman_pairs_trading import KalmanFilterStrategy


class Backtester:
    def __init__(self, strategy):
        self.strategy = strategy

    def kalman_backtest(df, s1, s2):
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

        kalman_strategy = KalmanFilterStrategy  # TODO change

        # Run regression (including Kalman Filter) to find hedge ratio and then create spread series
        df1 = pd.DataFrame({"y": y, "x": x})
        df1.index = pd.to_datetime(df1.index)
        state_means = (
            kalman_strategy.KalmanFilterAverage(x),
            kalman_strategy.KalmanFilterAverage(y),
        )
        df1["hr"] = -state_means[:, 0]
        df1["spread"] = df1.y + (df1.x * df1.hr)

        # calculate half life
        halflife = kalman_strategy.half_life(df1["spread"])

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
        df1["long exit"] = (df1.zScore > -exitZscore) & (
            df1.zScore.shift(1) < -exitZscore
        )
        df1["num units long"] = np.nan
        df1.loc[df1["long entry"], "num units long"] = 1
        df1.loc[df1["long exit"], "num units long"] = 0
        df1["num units long"][0] = 0
        df1["num units long"] = df1["num units long"].fillna(method="pad")

        # set up num units short
        df1["short entry"] = (df1.zScore > entryZscore) & (
            df1.zScore.shift(1) < entryZscore
        )
        df1["short exit"] = (df1.zScore < exitZscore) & (
            df1.zScore.shift(1) > exitZscore
        )
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
