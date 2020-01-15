import numpy as np
from numpy import log, polyfit, sqrt, std, subtract
import pandas as pd
import math
import scipy.stats as stats


def calculate_covariance_matrix(historical_stock_returns, base_historical_variance):
    """
    Calculate the covariance matrix of a safe asset (money) provided stock returns
    :param historical_stock_returns: list of historical stock returns
    :return: DataFrame of the covariance matrix of stocks and money (in practice just the variance).
    """
    assets = ['stocks', 'money']
    covariances = np.cov(np.array([historical_stock_returns, np.zeros(len(historical_stock_returns))]))

    if covariances.sum().sum() == 0.:
        # If the price is stationary, revert to base historical variance
        covariances[0][0] = base_historical_variance
    return pd.DataFrame(covariances, index=assets, columns=assets)


def div0(numerator, denominator):
    """
    ignore / 0, and return 0 div0( [-1, 0, 1], 0 ) -> [0, 0, 0]
    credits to Dennis @ https://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
    :param numerator: float numerator
    :param denominator: float denominator
    :return: float answer
    """
    answer = np.true_divide(numerator, denominator)
    if not np.isfinite(answer):
        answer = 0

    return answer


def div_by_hundred(x):
    """
    Divivde input by 100
    :param x: float input
    :return: float output
    """
    return x / 100.0


def discounted_value_cash_flow(cash_flow, periods_ahead, disc_rate):
    """
    Calculate discounted values of future cash flows
    :param cash_flow: np.Array future cash flows
    :param periods_ahead: integer of amounts of periods to look ahead
    :param disc_rate: np.Array of discount rates per period
    :return: np.Array of discounted future cash flows
    """
    return cash_flow / (1 + disc_rate)**periods_ahead


def find_horizon(dcfs):
    """
    Find index at which at which the discounted_cash flows can be cut off
    :param dcfs: list of discounted cash flows
    :return: index at which the list can be cut off
    """
    for idx, cash_flow in enumerate(dcfs):
        if cash_flow < 0.01:
            return idx
    return False


def calculate_npv(dividends, discount_rates):
    """Calculates the current NPV of a stream of dividends """
    current_index = 0
    final_index = len(dividends)
    discounted_cash_flows = dividends / ((1 + discount_rates)**range(current_index, final_index))
    if find_horizon(discounted_cash_flows):
        return sum(discounted_cash_flows[:find_horizon(discounted_cash_flows)])
    else:
        return np.nan


def hurst(ts):
    """
    source: https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing
    Returns the Hurst Exponent of the time series vector ts
    """
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0


def organise_data(obs, burn_in_period=0):
    """
    Extract data in manageable format from list of orderbooks
    :param obs: object limit-orderbook
    :param burn_in_period: integer period of observations which is discarded
    :return: Pandas DataFrames of prices, returns, autocorrelation in returns, autocorr_abs_returns, volatility, volume, fundamentals
    """
    window = 20
    close_price = []
    returns = []
    autocorr_returns = []
    autocorr_abs_returns = []
    returns_volatility = []
    volume = []
    fundamentals = []
    for ob in obs:  # record
        # close price
        close_price.append(ob.tick_close_price[burn_in_period:])
        # returns
        r = pd.Series(np.array(ob.tick_close_price[burn_in_period:])).pct_change()
        returns.append(r)
        # autocorrelation returns
        ac_r = [r.autocorr(lag=lag) for lag in range(25)]
        autocorr_returns.append(ac_r)
        # autocorrelation absolute returns
        absolute_returns = pd.Series(r).abs()
        autocorr_abs_returns.append([absolute_returns.autocorr(lag=lag) for lag in range(25)])
        # volatility of returns
        roller_returns = r.rolling(window)
        returns_volatility.append(roller_returns.std(ddof=0))
        # volume
        volume.append([sum(volumes) for volumes in ob.transaction_volumes_history][burn_in_period:])
        # fundamentals
        fundamentals.append(ob.fundamental[burn_in_period:])
    mc_prices = pd.DataFrame(close_price).transpose()
    mc_returns = pd.DataFrame(returns).transpose()
    mc_autocorr_returns = pd.DataFrame(autocorr_returns).transpose()
    mc_autocorr_abs_returns = pd.DataFrame(autocorr_abs_returns).transpose()
    mc_volatility = pd.DataFrame(returns_volatility).transpose()
    mc_volume = pd.DataFrame(volume).transpose()
    mc_fundamentals = pd.DataFrame(fundamentals).transpose()

    return mc_prices, mc_returns, mc_autocorr_returns, mc_autocorr_abs_returns, mc_volatility, mc_volume, mc_fundamentals


def hypothetical_series(starting_value, returns):
    """
    input: starting_value: float starting value
    input: returns: list 
    """
    returns = list(returns)
    simulated_series = [starting_value]
    for idx in range(len(returns)):
        simulated_series.append(simulated_series[-1] * (1 + returns[idx]))
    return simulated_series


def get_specific_bootstraps_moments(full_series, bootstrap_number):
    """Get a vector with the moments of a specific bootstrap"""
    return np.array([full_series[i][bootstrap_number] for i in range(len(full_series))])


def confidence_interval(data, av):
    sample_stdev = np.std(data)
    sigma = sample_stdev/math.sqrt(len(data))
    return stats.t.interval(alpha = 0.95, df= 24, loc=av, scale=sigma)


def ornstein_uhlenbeck_evolve(init_level, previous_level, sigma, mean_reversion, seed):
    fundamental_value = [previous_level]

    error = np.random.normal(0, sigma)
    new_dr = np.exp(np.log((fundamental_value[-1]) + error + mean_reversion * (np.log(init_level) - np.log(fundamental_value[-1]))))
    if new_dr <= 0 or np.isnan(new_dr) == True:
        new_dr = fundamental_value[-1]

    return new_dr
