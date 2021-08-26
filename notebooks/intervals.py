from scipy.stats import norm
import numpy as np


def normal(bt, confidence, obs_value):
    '''
    bt: bootstrap values
    confidence: confidence level
    obs_value: estimate value \hat\theta
    '''
    
    qnorm = abs(norm.ppf((1 - confidence) / 2))

    upper = obs_value + qnorm * np.std(bt, ddof=1) # change degree of freedom
    lower = obs_value - qnorm * np.std(bt, ddof=1)

    return lower, upper


def quantile(bt, confidence):
    """
    bt: bootstrap values
    confidence: confidence level
    """
    
    cutoff = (1 - confidence) / 2
    lower = np.quantile(bt, cutoff)
    upper = np.quantile(bt, 1 - cutoff)

    return lower, upper


def bias_corrected_normal(bt, confidence, obs_value):
    """
    bt: bootstrap values
    confidence: confidence lvel
    obs_value: estimate value \hat\theta
    """
    
    # caculate the bias between the resampled mean and the observed sample mean
    mean = np.mean(bt)
    bias = mean - obs_value
    
    # correct the observed sample mean
    obs_value -= bias
    
    qnorm = abs(norm.ppf((1 - confidence) / 2))

    upper = obs_value + qnorm * np.std(bt, ddof=1)
    lower = obs_value - qnorm * np.std(bt, ddof=1)
    
    return lower, upper


def jk_ps(ps, confidence):
    """
    ps: pseudo values
    confidence interval: confidence level
    """
    qnorm = abs(norm.ppf((1 - confidence) / 2))

    upper = np.mean(ps) + qnorm * np.std(ps, ddof=1) / np.sqrt(len(ps)) # change degree of freedom
    lower = np.mean(ps) - qnorm * np.std(ps, ddof=1) / np.sqrt(len(ps))
    
    return lower, upper


def jk_m(jk, confidence, obs_value):
    """
    jk: jackkife values
    confidence interval: confidence level
    obs_value: estimate value
    """
    n = len(jk)
    mean = np.mean(jk)
    num = np.sum(np.square(jk - mean))
    se = np.sqrt((n - 1) * num / n)
    
    qnorm = abs(norm.ppf((1 - confidence) / 2))
    
    upper = obs_value + qnorm * se
    lower = obs_value - qnorm * se
    
    return lower, upper


def jk_mj(jk, confidence, obs_value, sizes):
    """
    jk: jackkife values
    confidence interval: confidence level
    obs_value: estimate value
    sizes: size of each block
    n_sites: number of sites in total
    """
    
    n = sum(sizes)
    g = len(jk)
    
    mean = g * obs_value - np.sum((1 - sizes / n) * jk)
    h = n / sizes
    num = np.square(h * obs_value - (h - 1) * jk - mean) / (h - 1)
    var = np.sum(num) / g
    se = np.sqrt(var)

    qnorm = abs(norm.ppf((1 - confidence) / 2))

    upper = obs_value + qnorm * se
    lower = obs_value - qnorm * se

    return lower, upper

