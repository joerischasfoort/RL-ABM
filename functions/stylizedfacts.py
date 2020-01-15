"""This file contains functions and tests to calculate the stylized facts"""
import pandas as pd
import numpy as np
from functions.helpers import div0
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts


def calculate_close(orderbook_transaction_price_history):
    closing_prices = []
    for day in orderbook_transaction_price_history:
        closing_prices.append(day[-1])
    close = pd.Series(closing_prices).pct_change()
    return close


def calculate_returns(orderbook_transaction_price_history):
    """Return the returns"""
    closing_prices = []
    for day in orderbook_transaction_price_history:
        closing_prices.append(day[-1])
    returns = pd.Series(closing_prices).pct_change()
    return returns[1:]


def zero_autocorrelation(returns, lags):
    """returns wether average autocorrelation is much different from zero"""
    autocorr_returns = [returns.autocorr(lag=lag) for lag in range(lags)]
    # if mean autocorrelation are between -0.1 and 0.1
    average_autocorrelation = np.mean(autocorr_returns[1:])
    if (average_autocorrelation < 0.1) and (average_autocorrelation > -0.1):
        return True, average_autocorrelation
    else:
        return False, np.inf


def fat_tails_kurtosis(returns):
    series_returns = pd.Series(returns)
    kurt = series_returns.kurtosis()
    if kurt > 4:
        return True, kurt
    else:
        return False, np.inf


def clustered_volatility(returns, lags):
    absolute_returns = returns.abs()
    autocorr_abs_returns = [absolute_returns.autocorr(lag=lag) for lag in range(lags)]
    average_autocorrelation = np.mean(autocorr_abs_returns[1:])
    if (average_autocorrelation < 0.1) and (average_autocorrelation > -0.1):
        return False, np.inf
    else:
        return True, average_autocorrelation


def autocorrelation_returns(returns, lags):
    """
    Calculate the average autocorrelation in a returns time series
    :param returns: time series of returns
    :param lags: the lags over which the autocorrelation is to be calculated
    :return: average autocorrelation
    """
    returns = pd.Series(returns)
    autocorr_returns = [returns.autocorr(lag=lag) for lag in range(lags)]
    average_autocorrelation = np.mean(autocorr_returns[1:])
    return average_autocorrelation


def kurtosis(returns):
    """
    Calculates the kurtosis in a time series of returns
    :param returns: time series of returns
    :return: kurtosis
    """
    series_returns = pd.Series(returns)
    return series_returns.kurtosis()


def autocorrelation_abs_returns(returns, lags):
    """
    Calculates the average autocorrelation of absolute returns in a returns time series
    :param returns: returns time series
    :param lags: lags used to calculate autocorrelations
    :return: average autocorrelation of absolute returns
    """
    returns = pd.Series(returns)
    absolute_returns = returns.abs()
    autocorr_abs_returns = [absolute_returns.autocorr(lag=lag) for lag in range(lags)]
    return np.mean(autocorr_abs_returns[1:])


def correlation_volume_volatility(volume, returns, window):
    """
    :param volume: volume time series
    :param returns: returns time series
    :param window: rolling window used to calculate return volatility
    :return: correlation between returns volatility and volume
    """
    volume = pd.Series(volume)
    returns = pd.Series(returns)
    roller_returns = returns.rolling(window)
    returns_volatility = roller_returns.std(ddof=0)
    correlation = returns_volatility.corr(volume)
    return correlation


def cointegr(fundament, price):
    """
    Calculate cointegration with fundamentals
    :param fundament:
    :param price:
    :return: ADF test statistic, ADF critical values
    """
    model = sm.OLS(fundament, price)
    res = model.fit()
    residuals = res.resid
    cadf = ts.adfuller(residuals)
    return cadf[0], cadf[4]


def true_scores(simulations, m_index):
    """
    For every moment: count the number of simulations for which the single moments (or all moments jointly) are contained in their confidence intervals.
    """
    score = 0
    for s in simulations:
        if s[m_index]:
            score += 1
    return (float(score) / len(simulations)) * 100


def get_model_moments_in_confidence(mc_rets, mc_p, mc_f, conf_int_mom):
    """
    Get moments of a particular simulation and check if they fall in the bounds
    :param mc_rets: Pandas Dataframe of simulated returns
    :param mc_p: Pandas Dataframe of simulated prices
    :param mc_f: Pandas Dataframe of simulated fundamental value
    :param conf_int_mom:
    :return: list of True and False's for all the moments which are within the confidence intervals
    """
    first_order_autocors = []
    autocors1 = []
    autocors5 = []
    mean_abs_autocor = []
    kurtoses = []
    spy_abs_auto10 = []
    spy_abs_auto25 = []
    spy_abs_auto50 = []
    spy_abs_auto100 = []
    cointegrations = []
    for col in mc_rets:
        first_order_autocors.append(autocorrelation_returns(mc_rets[col][1:], 25))
        autocors1.append(mc_rets[col][1:].autocorr(lag=1))
        autocors5.append(mc_rets[col][1:].autocorr(lag=5))
        mean_abs_autocor.append(autocorrelation_abs_returns(mc_rets[col][1:], 25))
        kurtoses.append(mc_rets[col][2:].kurtosis())
        spy_abs_auto10.append(mc_rets[col][1:].abs().autocorr(lag=10))
        spy_abs_auto25.append(mc_rets[col][1:].abs().autocorr(lag=25))
        spy_abs_auto50.append(mc_rets[col][1:].abs().autocorr(lag=50))
        spy_abs_auto100.append(mc_rets[col][1:].abs().autocorr(lag=100))
        cointegrations.append(cointegr(mc_p[col][1:], mc_f[col][1:])[0])

    moments = np.array([
        np.mean(first_order_autocors),
        np.mean(autocors1),
        np.mean(autocors5),
        np.mean(mean_abs_autocor),
        np.mean(kurtoses),
        np.mean(spy_abs_auto10),
        np.mean(spy_abs_auto25),
        np.mean(spy_abs_auto50),
        np.mean(spy_abs_auto100),
        np.mean(cointegrations)])

    mom_covered = [between_interval(i, v) for i, v in zip(conf_int_mom, moments)]
    return mom_covered


def between_interval(interval, value):
    """
    Check if all moments are within the intervals
    :param interval: tuple with lower bound, upper bound
    :param value: float value to be checked to lie within the interval
    :return:
    """
    if interval[0] <= value <= interval[1]:
        return True
    else:
        return False
