"""
Generate scintillated signals matching Gaussian pulse profiles and exponential
intensity distributions using theoretical PDFs. 

Scintillation on narrowband signals references:

Cordes & Lazio 1991:
http://articles.adsabs.harvard.edu/pdf/1991ApJ...376..123C

Cordes, Lazio, & Sagan 1997:
https://iopscience.iop.org/article/10.1086/304620/pdf
"""
import sys
import numpy as np
from scipy.stats import norm
import scipy.linalg

import setigen as stg
from setigen.funcs import func_utils
from blscint import factors
from blscint import diag_stats


def find_nearest(arr, val):
    """
    Return index of closest value.

    Parameters
    ----------
    arr : np.ndarray
        Input array
    val : float
        Target value

    Returns
    -------
    idx : int
        Closest index
    """
    idx = (np.abs(np.array(arr) - val)).argmin()
    return idx


def bpdf(g1, g2, ac):
    """
    Calculate joint probability of g1, g2 separated by a temporal auto-correlation value of ac,
    according to Eq. B12 of Cordes, Lazio, & Sagan 1997.

    Parameters
    ----------
    g1 : float
        Signal gain 1
    g2 : float
        Signal gain 2
    ac : float
        Autocorrelation value in [0, 1)

    Returns
    -------
    f_2g : float
        Joint probability
    """
    # Calculate modified Bessel term, which can diverge.
    i_factor = scipy.special.i0(2*np.sqrt(g1*g2*ac)/(1-ac))
    # In divergent case, use exponential approximation to simplify terms.
    if np.any(np.isinf(i_factor)):
        #https://math.stackexchange.com/questions/376758/exponential-approximation-of-the-modified-bessel-function-of-first-kind-equatio
        return 1/(1-ac)*np.exp((-(g1+g2)+2*np.sqrt(g1*g2*ac))/(1-ac))*np.sqrt(4*np.pi*np.sqrt(g1*g2*ac)/(1-ac))
    else:
        return 1/(1-ac)*np.exp(-(g1+g2)/(1-ac))*i_factor

    
def get_ts_pdf(t_d, dt, num_samples, max_g=5, steps=1000, seed=None):
    """
    Produce time series data via bivariate pdf for the gain. With a maximum gain `max_g`
    and number of potential gain levels within [0, `max_g`].

    Parameters
    ----------
    t_d : float
        Scintillation timescale (s)
    dt : float
        Time resolution (s)
    num_samples : int
        Number of synthetic samples to produce
    max_g : float, optional
        Maximum possible gain (for computation)
    steps : int, optional
        Number of possible gain levels within [0, `max_g`] (for computation)
    seed : None, int, Generator, optional
        Random seed or seed generator

    Returns
    -------
    Y : np.ndarray
        Final synthetic scintillated time series (Y values)
    """
    rng = np.random.default_rng(seed)
    ac_arr = stg.func_utils.gaussian(np.arange(0, steps),
                                     0, 
                                     t_d / dt / factors.hwem_m)
    
    possible_g = np.linspace(0, max_g, steps, endpoint=False)
#     F_2g = np.empty(shape=(n, n))
#     for i in range(n):
#         F_2g[i, :] = bpdf(possible_g[i], possible_g, rho[1])
#     F_2g = F_2g / np.sum(F_2g, axis=1, keepdims=True)

    ts_idx = np.zeros(num_samples, dtype=int)

    init_g = max_g + 1
    while init_g > max_g:
        init_g = rng.exponential()
    ts_idx[0] = find_nearest(possible_g, init_g)
    
    update_freq = int(np.ceil(t_d / dt))
    for i in range(1, num_samples):
#         offset = i % update_freq
#         if offset == 1:
#             last_i = i - 1
#         if offset == 0:
#             offset = update_freq
#         raw_p = bpdf(possible_g[ts_idx[last_i]], possible_g, ac_arr[offset])
        
        raw_p = bpdf(possible_g[ts_idx[i-1]], possible_g, ac_arr[1])

        p = raw_p / np.sum(raw_p)
        try:
            ts_idx[i] = rng.choice(np.arange(steps), p=p)
        except:
#             print(F_2g[i])
            print(i)
            sys.exit(1)
    Y = possible_g[ts_idx]
    return Y