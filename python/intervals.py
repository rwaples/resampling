"""This file contains methods for calculating the confidence intervals of resampled values
"""
from scipy.stats import norm
import numpy as np


def bt_standard(bt, confidence, obs_value):
    """ calculate the standard normal confidence interval of the bootstrap values
    return the lower and upper bound
    @bt: bootstrap values
    @confidence: confidence level
    @obs_value: estimated value of the observation sample
    """
    z_score = abs(norm.ppf((1 - confidence) / 2))
    upper = obs_value + z_score * np.std(bt, ddof=1)  # change degree of freedom
    lower = obs_value - z_score * np.std(bt, ddof=1)
    assert upper > lower
    return lower, upper


def jk_delete_one(jk, confidence, obs_value):
    """calculate the standard normal confidence interval of the jackknife values (delete one methods)
    return the lower and upper bound
    @jk: jackknife values
    @confidence interval: confidence level
    @obs_value: estimated value of the observation sample
    """
    n = len(jk)
    # variance of the jk values
    var = (n - 1) * np.sum(np.square(jk - np.mean(jk))) / n
    z_score = abs(norm.ppf((1 - confidence) / 2))

    upper = obs_value + z_score * np.sqrt(var)
    lower = obs_value - z_score * np.sqrt(var)
    assert upper > lower
    return lower, upper


def jk_delete_mj(params, confidence, obs_value):
    """calculate the standard normal confidence interval of the jackknife values (delete mj methods)
    return the lower and upper bound
    @params: (jackknife values, size of each block)
    @confidence: confidence level
    @obs_value: estimated value of the observation sample
    """
    jk, sizes = params
    n, g = sum(sizes), len(jk)
    h = n / sizes

    mean = g * obs_value - np.sum((1 - sizes / n) * jk)
    var = np.sum(np.square(h * obs_value - (h - 1) * jk - mean) / (h - 1)) / g

    z_score = abs(norm.ppf((1 - confidence) / 2))
    upper = obs_value + z_score * np.sqrt(var)
    lower = obs_value - z_score * np.sqrt(var)
    assert upper > lower

    return lower, upper
