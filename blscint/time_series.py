import numpy as np
import scipy.stats

import setigen as stg
from setigen.funcs import func_utils
from . import factors

# def autocorr(x, length=20):
#     # Returns up to length index shifts for autocorrelations
#     return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0, 1]
#                          for i in range(1, length)])

def autocorr(ts):
    """
    Calculate full autocorrelation, normalizing time series to zero mean and unit variance.
    """
    ts = (ts - np.mean(ts)) / np.std(ts)
    acf = np.correlate(ts, ts, 'full')[-len(ts):] / len(ts)
    return acf
    

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