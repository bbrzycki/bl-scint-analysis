import numpy as np
from scipy import optimize
import scipy.stats, scipy.signal

import setigen as stg
from setigen.funcs import func_utils
from . import factors

# def autocorr(x, length=20):
#     # Returns up to length index shifts for autocorrelations
#     return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0, 1]
#                          for i in range(1, length)])

def autocorr(ts, remove_spike=False):
    """
    Calculate full autocorrelation, normalizing time series to zero mean and unit variance.
    """
    ts = (ts - np.mean(ts)) #/ np.std(ts)
    acf = np.correlate(ts, ts, 'full')[-len(ts):]
    if remove_spike:
        acf[0] = acf[1]
    acf /= acf[0] # This is essentially the variance (scaled by len(ts))
    return acf


def acf(ts, remove_spike=False):
    return autocorr(ts, remove_spike=remove_spike)
    

def get_stats(ts):
    """
    Calculate statistics based on normalized time series (to mean 1).
    """
    stats = {}
    
    stats['fchans'] = len(ts)
    stats['std'] = np.std(ts)
    stats['min'] = np.min(ts)
    stats['ks'] = scipy.stats.kstest(ts, 
                                          scipy.stats.expon.cdf)[0]

    ac = autocorr(ts)
    stats['lag1'] = ac[1]
    return stats


def acf_func(x, A, sigma, Y=0):
    return A * stg.func_utils.gaussian(x, 0, sigma) + Y * scipy.signal.unit_impulse(len(x))
    
    
def fit_acf(acf, remove_spike=False):
    if remove_spike:
        t_acf_func = lambda x, sigma: acf_func(x, 1, sigma, 0)
    else:
        t_acf_func = acf_func
    popt, a = optimize.curve_fit(t_acf_func, 
                                 np.arange(len(acf)),
                                 acf,)
#     print(a)
    if remove_spike:
        return [1, popt[0], 0]
    else:
        return popt