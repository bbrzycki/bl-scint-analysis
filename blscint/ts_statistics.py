import numpy as np
from scipy import optimize
import scipy.stats, scipy.signal
import matplotlib.pyplot as plt
import collections
import seaborn as sns
import pandas as pd

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
    
    # stats['fchans'] = len(ts)
    stats['std'] = np.std(ts)
    stats['min'] = np.min(ts)
    stats['ks'] = scipy.stats.kstest(ts, 
                                     scipy.stats.expon.cdf)[0]

    ac = autocorr(ts)
    stats['lag1'] = ac[1]
    stats['lag2'] = ac[2]
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
    
    
def ts_plots(ts, xlim=None, bins=None):
    """
    Plot time series, autocorrelation, and histogram.
    """
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 3, 1)
    plt.plot(ts)
    
    if xlim is not None:
        plt.xlim(0, xlim)
    plt.xlabel('Lag / px')
    plt.ylabel('Intensity')
    
    
    plt.subplot(1, 3, 2)
    plt.plot(autocorr(ts), label='TS AC')
    
    if xlim is not None:
        plt.xlim(0, xlim)
    plt.xlabel('Lag / px')
    plt.ylabel('Autocorrelation')
    
    plt.subplot(1, 3, 3)
    plt.hist(ts, bins=bins)
    plt.xlabel('Intensity')
    plt.ylabel('Counts')
    plt.show()
    
    print(get_stats(ts))
    
    
def ts_stat_plots(ts_arr, t_d=None, dt=None):
    """
    Plot relevant statistics over a list of time series arrays. Plot in reference to
    a given scintillation timescale and time resolution.
    """
    stats_labels = ['Standard Deviation', 
                    'Minimum',
                    'KS Statistic', 
                    'Lag-1 Autocorrelation', 
                    'Lag-2 Autocorrelation']
    fig, axs = plt.subplots(1, 5, figsize=(25, 4), sharex='col')
    
    ts_stats_dict = collections.defaultdict(list)
    for ts in ts_arr:
        ts_stats = get_stats(ts)

        for key in ts_stats:
            ts_stats_dict[key].append(ts_stats[key])
    
    for r, key in enumerate(['std', 'min', 'ks', 'lag1', 'lag2']):
        if r == 0:
            bins = np.arange(0, 2.05, 0.05)
        elif r == 1:
            bins = np.arange(-0.5, 0.55, 0.05)
        elif r == 2:
            bins = np.arange(0, 0.51, 0.01)
        else:
            bins = np.arange(-0.2, 1.02, 0.02)
            
        axs[r].hist(ts_stats_dict[key], bins=bins, histtype='step')
        
        axs[r].set_xlabel(f'{stats_labels[r]}')
        axs[r].xaxis.set_tick_params(labelbottom=True)
    axs[0].set_ylabel('Counts')
    if t_d is not None and dt is not None:
        axs[3].axvline(stg.func_utils.gaussian(1,
                                               0, 
                                               t_d / dt / factors.hwem_m), ls='--', c='k')
        axs[4].axvline(stg.func_utils.gaussian(2,
                                               0, 
                                               t_d / dt / factors.hwem_m), ls='--', c='k')
    plt.show()
    
    
def ts_ac_plot(ts_arr, t_d, dt, p=2):
    """
    Plot autocorrelations, mean += std dev at each lag over a list of time series arrays. 
    Plot in reference to scintillation timescale and time resolution, up to lag p.
    """
    ac_dict = {'lag': [], 'ac': [], 'type': []}
    
    p = min(p+1, len(ts_arr[0])) - 1
    
    for i in np.arange(0, p+1):
        ac_dict['lag'].append(i)
        ac_dict['ac'].append(stg.func_utils.gaussian(i,
                                                     0, 
                                                     t_d / dt / factors.hwem_m))
        ac_dict['type'].append('target')
        
    j = 0
    for ts in ts_arr:
        ac = autocorr(ts)
        if j==0:
            print(ac.shape)
            j=1
        for i in range(0, p+1):
            ac_dict['lag'].append(i)
            ac_dict['ac'].append(ac[i])
            ac_dict['type'].append('sim')
    data = pd.DataFrame(ac_dict)
    
#     sns.catplot(data=pd.DataFrame(ac_dict), x='lag', y='ac', kind='box',hue='type')
    ax = sns.lineplot(data=data,
                      x='lag',
                      y='ac',
                      style='type',
                      hue='type', 
                      markers=True, 
                      dashes=False, 
                      ci='sd')
    
#     ax.set_xticks(np.arange(0, p+1))
    ax.grid()
    plt.show()