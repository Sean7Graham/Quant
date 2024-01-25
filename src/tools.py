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
