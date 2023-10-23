"""
Generate scintillated signals matching Gaussian pulse profiles and exponential
intensity distributions using the autogregressive to anything (ARTA) algorithm.

Cario & Nelson 1996:
https://www.sciencedirect.com/science/article/pii/016763779600017X

Scintillation on narrowband signals references:

Cordes & Lazio 1991:
http://articles.adsabs.harvard.edu/pdf/1991ApJ...376..123C

Cordes, Lazio, & Sagan 1997:
https://iopscience.iop.org/article/10.1086/304620/pdf
"""

import numpy as np
from scipy.stats import norm
import scipy.linalg

import setigen as stg
from setigen.funcs import func_utils
from blscint import factors
from blscint import diag_stats


def get_rho(t_d, dt, p, pow=5/3):
    """
    Get target autocorrelation guesses for ARTA with scintillation
    timescale t_d and time resolution dt, up to lag p. 
    Modeled as a power law with exponent 5/3 or 2. 

    Parameters
    ----------
    t_d : float
        Scintillation timescale (s)
    dt : float
        Time resolution (s)
    p : int
        Number of lags to calculate
    pow : float, optional
        Exponent for ACF fit, either 5/3 or 2 (arising from phase structure function) 

    Returns
    -------
    r : np.ndarray
        Array of autocorrelations, starting with lag 1
    """
    # Calculate sigma from width
    # r = stg.func_utils.gaussian(np.arange(1, p + 1),
    #                             0, 
    #                             t_d / dt / factors.hwem_m)
    r = diag_stats.scint_acf(np.arange(1, p + 1), t_d / dt, pow=pow)
    return r


# def psi(r):
#     """
#     Return covariance matrix for initial multivariate normal distribution.
#     """
#     # r is the array of guesses to get close to desired autocorrelations
#     p = len(r)
#     covariance = np.ones((p, p))
#     for i in range(0, p - 1):
#         for j in range(0, p - i - 1):
#             covariance[i + j + 1, j] = covariance[j, i + j + 1] = r[i]
#     return covariance


def psi(r):
    """
    Return covariance matrix for initial multivariate normal distribution.

    Parameters
    ----------
    r : np.ndarray
        Array of autocorrelation guesses, starting with lag 1

    Returns
    -------
    cov_mat : np.ndarray
        Covariance matrix
    """
    return scipy.linalg.toeplitz(np.concatenate([[1.], r[:-1]]))


def build_Z(r, T, seed=None):
    """
    Build full baseline Z array.

    Parameters
    ----------
    r : np.ndarray
        Array of autocorrelation guesses, starting with lag 1
    T : int
        Final length of array Z, should be greater than p
    seed : None, int, Generator, optional
        Random seed or seed generator

    Returns
    -------
    Z : np.ndarray
        Array of Z values, as in ARTA
    """
    rng = np.random.default_rng(seed)

    # T is final length of array Z, should be greater than p
    # r is the array of guesses to get close to desired autocorrelations
    # Returns full Z array
    p = len(r)
    assert T >= p

    Z = np.zeros(T)
    covariance = psi(r) 
    
    min_eig = np.min(np.real(np.linalg.eigvals(covariance)))
    if min_eig < 0:
        covariance -= 10*min_eig * np.eye(*covariance.shape)
    covariance += np.eye(*covariance.shape)*1e-6
    
    # Check whether covariance is nonnegative definite
#     print(np.linalg.eigvalsh(covariance))
    _ = np.linalg.cholesky(covariance)

    Z[:p] = rng.multivariate_normal(np.zeros(p), covariance)
    alpha = np.dot(r, np.linalg.inv(covariance))
#     print(np.abs(np.roots([1.]+list(-alpha))))
    try:
        assert np.all(np.abs(np.roots([1.]+list(-alpha))) <= 1.)
    except AssertionError:
        raise RuntimeError('Time series is not stationary! At least one root has magnitude larger than 1.')
        
    variance = 1 - np.dot(alpha, r)
    try:
        assert variance >= 0
    except AssertionError:
        raise RuntimeError('Variance of epsilon is negative!')

    for i in range(p, T):
        epsilon = rng.normal(0, np.sqrt(variance))
        Z[i] = np.dot(alpha, Z[i-p:i][::-1]) + epsilon
    return Z


def inv_exp_cdf(x, rate=1):
    """
    Inverse exponential distribution CDF.

    Parameters
    ----------
    x : float, np.ndarray
        Input value(s) from [0, 1)
    rate : float
        Rate parameter lambda

    Returns
    -------
    v : float, np.ndarray
        Output of inverse CDF
    """
    return -np.log(1. - x) / rate


def inv_levy_cdf(x, loc=0, scale=1):
    return scale / (norm.ppf(1 - x / 2))**2 + loc


def get_Y(Z, dist='exp'):
    """
    Get final values specific to an overall exponential distribution,
    normalized to mean of 1.

    Parameters
    ----------
    Z : np.ndarray
        Array of Z values, as in ARTA

    Returns
    -------
    Y : np.ndarray
        Final synthetic scintillated time series (Y values)
    """
    if dist == 'exp':
        Y = inv_exp_cdf(norm.cdf(Z))
    else:
        Y = inv_levy_cdf(norm.cdf(Z))
    return Y / np.mean(Y)


def get_ts_arta(t_d, dt, num_samples, p=2, pow=5/3, dist='exp'):
    """
    Produce time series data via an ARTA process. 

    Parameters
    ----------
    t_d : float
        Scintillation timescale (s)
    dt : float
        Time resolution (s)
    num_samples : int
        Number of synthetic samples to produce
    p : int, optional
        Number of lags to calculate
    pow : float, optional
        Exponent for ACF fit, either 5/3 or 2 (arising from phase structure function) 

    Returns
    -------
    Y : np.ndarray
        Final synthetic scintillated time series (Y values)
    """
    rho = get_rho(t_d, dt, p, pow)
    Z = build_Z(rho, num_samples)
    Y = get_Y(Z, dist)
    return Y

