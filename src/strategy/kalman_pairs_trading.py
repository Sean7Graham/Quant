# kalman_pairs_trading.py

from pykalman import KalmanFilter
import pandas as pd
import numpy as np
import statsmodels.api as sm


class KalmanTradingStrategy:
    def __init__(self, symbol1, symbol2, start_, trading_cost=0.000, datafetcher):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.trading_cost = trading_cost
        
        self.kf = None
        self.state_means = None
        self.state_covs = None
        self.half_life = None
        
        

    def KalmanFilterAverage(self, x):
        # Construct a Kalman filter
        self.kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=0,
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01,
        )
        # Use the observed values of the price to get a rolling mean
        state_means, _ = self.kf.filter(x.values)
        state_means = pd.Series(state_means.flatten(), index=x.index)
        return state_means

    # Kalman filter regression
    def KalmanFilterRegression(self, x, y):
        delta = 1e-3
        trans_cov = delta / (1 - delta) * np.eye(2)  # How much random walk wiggles
        obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
        self.kf = KalmanFilter(
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
        state_means, state_covs = self.kf.filter(y.values)
        return state_means
    
    def KalmanFilterRegression2(self, x, y):
        obs_mat_F = np.transpose(np.vstack([data[sym_a].values, np.ones(data.shape[0])])).reshape(-1, 1, 2)

        kf = KalmanFilter(n_dim_obs=1,                                      # y is 1-dimensional
                  n_dim_state=2,                                    #  states (alpha, beta) is 2-dimensinal
                  initial_state_mean=np.ones(2),                    #  initial value of intercept and slope theta0|0
                  initial_state_covariance=np.ones((2, 2)),         # initial cov matrix between intercept and slope P0|0
                  transition_matrices=np.eye(2),                    # G, constant
                  observation_matrices=obs_mat_F,                   # F, depends on x
                  observation_covariance=1,                         # v_t, constant
                  transition_covariance= np.eye(2))                 # w_t, constant
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
    
    def run_strategy(self):
        

    # @staticmethod
    # def KalmanFilterAverage(self, x):
    #     # Construct a Kalman filter
    #     self.kf = KalmanFilter(
    #         transition_matrices=[1],
    #         observation_matrices=[1],
    #         initial_state_mean=0,
    #         initial_state_covariance=1,
    #         observation_covariance=1,
    #         transition_covariance=0.01,
    #     )
    #     # Use the observed values of the price to get a rolling mean
    #     self.state_means, _ = self.kf.filter(x.values)
    #     self.state_means = pd.Series(self.state_means.flatten(), index=x.index)
    #

    # @staticmethod
    # def KalmanFilterRegression(self, x, y):
    #     delta = 1e-3
    #     trans_cov = delta / (1 - delta) * np.eye(2)  # How much random walk wiggles
    #     obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
    #     self.kf = KalmanFilter(
    #         n_dim_obs=1,
    #         n_dim_state=2,  # y is 1-dimensional, (alpha, beta) is 2-dimensional
    #         initial_state_mean=[0, 0],
    #         initial_state_covariance=np.ones((2, 2)),
    #         transition_matrices=np.eye(2),
    #         observation_matrices=obs_mat,
    #         observation_covariance=2,
    #         transition_covariance=trans_cov,
    #     )
    #     # Use the observations y to get running estimates and errors for the state parameters
    #     self.state_means, self.state_covs = self.kf.filter(y.values)

    # @staticmethod
    # def half_life(self, spread):
    #     spread_lag = spread.shift(1)
    #     spread_lag.iloc[0] = spread_lag.iloc[1]
    #     spread_ret = spread - spread_lag
    #     spread_ret.iloc[0] = spread_ret.iloc[1]
    #     spread_lag2 = sm.add_constant(spread_lag)
    #     model = sm.OLS(spread_ret, spread_lag2)
    #     res = model.fit()
    #     halflife = int(round(-np.log(2) / res.params[1], 0))
    #     if halflife <= 0:
    #         self.half_life = 1
