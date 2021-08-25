from scipy.stats import norm
import numpy as np


def normal(data, confidence, obs_value):
    '''
    @data: estimated parameters (e.g. means of 1000 resample data)
    @confidence: width of the interval
    @obs_value: value of the estimated parameter of the observed sample 
    (None for jackknife resample that uses pseudovalues)
    '''
    
    qnorm = abs(norm.ppf((1 - confidence) / 2))

    upper = obs_value + qnorm * np.std(data, ddof=1) # change degree of freedom
    lower = obs_value - qnorm * np.std(data, ddof=1)

    return lower, upper


def quantile(data, confidence):
    """
    @data: estimated parameters (e.g. means of 1000 resample data)
    @confidence: width of the interval
    """
    
    cutoff = (1 - confidence) / 2
    lower = np.quantile(data, cutoff)
    upper = np.quantile(data, 1 - cutoff)

    return lower, upper


def bias_corrected_normal(data, confidence, obs_value):
    """
    @data: estimated parameters (e.g. means of 1000 resample data)
    @confidence: width of the interval
    @obs_value: value of the estimated parameter of the observed sample
    """
    
    # caculate the bias between the resampled mean and the observed sample mean
    mean = np.mean(data)
    bias = mean - obs_value
    
    # correct the observed sample mean
    obs_value -= bias
    
    qnorm = abs(norm.ppf((1 - confidence) / 2))

    upper = obs_value + qnorm * np.std(data, ddof=1)
    lower = obs_value - qnorm * np.std(data, ddof=1)
    
    return lower, upper


def jk_pseudo(data, confidence):
    """
    confidence interval for jackknife resample that uses pseudovalues
    """
    qnorm = abs(norm.ppf((1 - confidence) / 2))

    upper = np.mean(data) + qnorm * np.std(data, ddof=1) / np.sqrt(len(data)) # change degree of freedom
    lower = np.mean(data) - qnorm * np.std(data, ddof=1) / np.sqrt(len(data))
    
    return lower, upper


def jk_m(data, confidence, obs_value):
    """
    confidence interval for jackknife resample that does not use pseudovalues
    """
    n = len(data)
    mean = np.mean(data)
    num = np.sum(np.square(data - mean))
    se = np.sqrt((n - 1) * num / n)
    
    qnorm = abs(norm.ppf((1 - confidence) / 2))
    
    upper = obs_value + qnorm * se
    lower = obs_value - qnorm * se
    
    return lower, upper


def jk_mj(data, confidence, obs_value, sizes, n_sites):
    """
    @data: estimated parameters (e.g. means of 1000 resample data)
    @confidence: width of the interval
    @est_param: value of the estimated parameter of the observed sample
    @sizes: number of sites in each block
    """
    
    n = len(data)
    
    mean = n * obs_value - np.sum((1 - sizes / n) * data)
    h = n_sites / sizes
    num = np.square(h * obs_value - (h - 1) * data - mean) / (h - 1)
    var = np.sum(num) / n
    se = np.sqrt(var)

    qnorm = abs(norm.ppf((1 - confidence) / 2))

    upper = obs_value + qnorm * se
    lower = obs_value - qnorm * se

    return lower, upper

