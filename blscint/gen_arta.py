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
from . import factors
from . import ts_statistics


def get_rho(t_d, dt, p):
    """
    Get target autocorrelations with time array ts and scintillation
    timescale t_d, up to lag p. Modeled as Gaussian according to t_d. 
    """
    # Calculate sigma from width
    r = stg.func_utils.gaussian(np.arange(1, p + 1),
                                0, 
                                t_d / dt / factors.hwem_m)
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
    """
    return scipy.linalg.toeplitz(np.concatenate([[1.], r[:-1]]))


def build_Z(r, T):
    """
    Build full baseline Z array.
    """
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

    Z[:p] = np.random.multivariate_normal(np.zeros(p), covariance)
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
        epsilon = np.random.normal(0, np.sqrt(variance))
        Z[i] = np.dot(alpha, Z[i-p:i][::-1]) + epsilon
    return Z


def inv_exp_cdf(x, rate=1):
    """
    Inverse exponential distribution CDF.
    """
    return -np.log(1. - x) / rate


def get_Y(Z):
    """
    Get final values specific to an overall exponential distribution,
    normalized to mean of 1.
    """
    Y = inv_exp_cdf(norm.cdf(Z))
    return Y / np.mean(Y)


def get_ts_arta(t_d, dt, num_samples, p=2):
    """
    Produce time series data via an ARTA process. 
    """
    rho = get_rho(t_d, dt, p)
    Z = build_Z(rho, num_samples)
    Y = get_Y(Z)
    return Y

