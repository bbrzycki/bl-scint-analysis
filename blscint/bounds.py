import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
import scipy.stats
from astropy.stats import sigma_clip

import setigen as stg

from . import frame_proc


def plot_bounds(frame, l, r, use_db=False, cb=True, lw=2):
    """
    Plot frame data with overlaid bounding boxes.
    
    lw: line width
    """
    frame.plot(use_db=use_db, cb=cb)
    plt.axvline(l, ls='--', c='w', lw=lw)
    plt.axvline(r, ls='--', c='w', lw=lw)


def polyfit_bounds(spec, deg=1, snr_threshold=10):
    """
    Bounding box set by a polynomial fit to the background. Edges are set by
    where the fit intersects the data on either side of the central peak.
    
    If the signal is broad compared to the spectrum or frame, the background fit
    will highly intersect or even model out the actual signal, hiding any peaks. 
    
    Degrees to try: 1, 7.
    
    Parameters
    ----------
    spec : ndarray
        Intensity spectra
    deg : int
        Degree of polynomial fit
    snr_threshold : int, float
        Threshold for peak detection, in units of noise standard deviations
        
    Returns
    -------
    l : int
        Lower bound
    r : int
        Upper bound
    metadata : dict
        Dictionary with metadata related to peak-finding. Contains polynomial fit
        object, number of peaks, and detected peak information.
    """
    y = sigma_clip(spec)
    x = np.arange(len(spec))

    coeffs = np.polyfit(x[~y.mask], y[~y.mask], deg)
    poly = np.poly1d(coeffs)

    # Estimate noise std (with background fit subtraction)
    std = np.std(y[~y.mask] - poly(x[~y.mask]))

    # Get peaks above SNR threshold
    peaks = scipy.signal.find_peaks(spec - poly(x), prominence=snr_threshold * std)
    if len(peaks[0]) == 0:
        raise ValueError('No peaks found! Signal may be absent, too dim, or too wide')
    
    # Find highest peak
    i = np.argmax(peaks[1]['prominences'])
    peak_i = peaks[0][i]
    
    cutoffs = np.where(spec - poly(x) <= 0)[0]\
    
    i = np.digitize(peak_i, cutoffs) - 1
    l, r = cutoffs[i] + 1, cutoffs[i + 1]
    
    metadata = {
        'poly': poly, 
        'num_peaks': len(peaks[0]), 
        'peaks': peaks
    }
    return l, r, metadata


def threshold_bounds(spec, half_width=3):
    """
    Create bounds based on intensity attentuation on either side of the central
    peak. Threshold is set by ideal Gaussian profile, in units of standard deviations (sigma).
    
    Parameters
    ----------
    spec : ndarray
        Intensity spectra
    half_width : float
        Assuming a Gaussian signal profile, the half_width determines where to set the bounds,
        in units of sigma from the peak.
        
    Returns
    -------
    l : int
        Lower bound
    r : int
        Upper bound
    metadata : dict
        Dictionary with metadata. Contains noise mean and spectra maximum,
        which are used to normalize spec to the spectra maximum.
    """
    noise_spec = sigma_clip(spec, masked=True)
    norm_spec = (spec - np.mean(noise_spec)) / (np.max(spec) - np.mean(noise_spec))
    
    threshold = stg.func_utils.gaussian(half_width, 0, 1)
    cutoffs = np.where(norm_spec < threshold)[0]
    
    peak = np.argmax(norm_spec)
    i = np.digitize(peak, cutoffs) - 1
    l, r = cutoffs[i] + 1, cutoffs[i + 1]
    
    metadata = {
        'noise_mean': np.mean(noise_spec),
        'spec_max': np.max(spec)
    }
    return l, r, metadata


def gaussian_bounds(spec, half_width=3, peak_guess=None):
    """
    Create bounds based on a Gaussian fit to the central peak.
    
    Parameters
    ----------
    spec : ndarray
        Intensity spectra
    half_width : float
        Assuming a Gaussian signal profile, the half_width determines where to set the bounds,
        in units of sigma from the peak.
    peak_guess : int
        Guess for peak index. Should normally be the center of the spectrum, but can have 
        strange side effects.
        
    Returns
    -------
    l : int
        Lower bound
    r : int
        Upper bound
    metadata : dict
        Dictionary with metadata related to peak-finding. Contains Gaussian fit information.
    """
    gaussian_func = lambda x, A, x0, sigma, y: A * stg.func_utils.gaussian(x, x0, sigma) + y
    
    if peak_guess is not None:
        peak_guess = [1, peak_guess, 1, 1]
    popt, _ = optimize.curve_fit(gaussian_func, 
                                 np.arange(len(spec)),
                                 spec,
                                 p0=peak_guess)
    
    peak = int(popt[1])
    sigma = abs(popt[2])
    offset = int(sigma * half_width)
    
    metadata = {
        'popt': popt
    }
    return peak - offset, peak + offset + 1, metadata


def clipped_bounds(spec, min_empty_bins=2):
    """
    Run sigma clip on spectrum, and find central peak.
    
    This method reduces the spectrum to a mask based on sigma_clip, so it can
    be unreliable if the signal is broad compared to the size of the spectrum or frame.
    
    Parameters
    ----------
    spec : ndarray
        Intensity spectra
    min_empty_bins : int
        Minimum number of adjacent "empty", or non-clipped, bins required to 
        mark either bound with respect to the central peak.
        
    Returns
    -------
    l : int
        Lower bound
    r : int
        Upper bound
    metadata : dict
        Dictionary with metadata related to peak-finding. Contains detected
        peak information.
    """
    mask_spec = sigma_clip(spec).mask.astype(int)
    
    peaks = scipy.signal.find_peaks(mask_spec, prominence=1)
    if len(peaks[0]) == 0:
        raise ValueError('No peaks found! Signal may be absent, too dim, or too wide')
    # Get the highest peak that's *closest* to the center of the frame
    center_bin = len(spec) // 2
    prominences = peaks[1]['prominences']
    max_idx = np.where(prominences == np.max(prominences))[0]
    peak_i = peaks[0][max_idx[np.argmin(np.abs(peaks[0][max_idx] - center_bin))]]

    # Convolve with ones, to find where sequences are adjacent zeros
    convolved = np.convolve(mask_spec,
                            np.ones(min_empty_bins).astype(int),
                            'valid')
    c_mask = (convolved != 0).astype(int)
    diffs = np.diff(c_mask)
    
    # Find which range of bins the peak lies in
    l_idx = np.where(diffs > 0)[0]
    r_idx = np.where(diffs < 0)[0]
    # Max index with value under peak index, and min index with value over
    # Adjust left edge to make up for zero'd bins from the convolution,
    # since we only care about the signal
    l = l_idx[l_idx + min_empty_bins <= peak_i][-1] + min_empty_bins
    r = r_idx[r_idx >= peak_i][0] + 1
    
    metadata = {
        'num_peaks': len(peaks[0]), 
        'peaks': peaks
    }
    return l, r, metadata


def clipped_2D_bounds(frame, min_empty_bins=2, min_clipped=1, peak_prominence=4):
    """
    Run sigma clip on 2D data array, and find central peak above a certain
    number of clipped values along the time axis, per frequency bin.
    
    This will return an IndexError if the signal passes outside of the frame
    (i.e. for very wide signals).
    
    Note that this function accepts a frame instead of a 1D numpy array spectrum.
    
    Parameters
    ----------
    frame : setigen.Frame
        Frame of intensity data
    min_empty_bins : int
        Minimum number of adjacent "empty", or non-clipped, bins required to 
        mark either bound with respect to the central peak.
    min_clipped : int
        Minimum number of pixels clipped by sigma_clip in the time direction in
        order for a frequency bin to be considered part of the signal.
    peak_prominence : int
        Prominence in units of clipped pixels, for use in peak finding.
        
    Returns
    -------
    l : int
        Lower bound
    r : int
        Upper bound
    metadata : dict
        Dictionary with metadata related to peak-finding. Contains detected
        peak information.
    """
    n_frame = frame_proc.t_norm_frame(frame)
    clipped_data = sigma_clip(n_frame.data)
    mask_spec = np.sum(clipped_data.mask, axis=0)
    
    peaks = scipy.signal.find_peaks(mask_spec, prominence=peak_prominence)
#     idx = np.argmin(np.abs(peaks[0] - 128))

    # Find highest peak
#     i = np.argmax(peaks[1]['prominences'])
#     peak_i = peaks[0][i]

    # Get the highest peak that's *closest* to the center of the frame
    center_bin = frame.fchans // 2
    prominences = peaks[1]['prominences']
    max_idx = np.where(prominences == np.max(prominences))[0]
    peak_i = peaks[0][max_idx[np.argmin(np.abs(peaks[0][max_idx] - center_bin))]]
    
    # I find that np.find_peaks doesn't do a good job for bounding boxes
#     l = peaks[1]['left_bases'][idx]
#     r = peaks[1]['right_bases'][idx]

    # Change mask to thresholding by number of pixels along time axis
    mask_spec = mask_spec >= min_clipped
#     print(peak_i, mask_spec)

    # Convolve with ones, to find where sequences are adjacent zeros
    convolved = np.convolve(mask_spec,
                            np.ones(min_empty_bins).astype(int),
                            'valid')
    c_mask = (convolved != 0).astype(int)
#     print(c_mask)
    diffs = np.diff(c_mask)
#     print(diffs)
    
    # Find which range of bins the peak lies in
    l_idx = np.where(diffs > 0)[0]
    r_idx = np.where(diffs < 0)[0]
#     print(l_idx, r_idx)
    # Max index with value under peak index, and min index with value over
    # Adjust left edge to make up for zero'd bins from the convolution,
    # since we only care about the signal
    l = l_idx[l_idx + min_empty_bins <= peak_i][-1] + min_empty_bins
    r = r_idx[r_idx >= peak_i][0] + 1
    
    metadata = {
        'num_peaks': len(peaks[0]), 
        'peaks': peaks
    }
    return l, r, metadata


def boxcar_bounds(spec, window_sizes=None):
    """
    Bounding box set by boxcar filter. Simplified version of 
    matched filtering. Default window sizes: powers of 2.
    
    Parameters
    ----------
    spec : ndarray
        Intensity spectra
    window_sizes : list, array-like, optional
        List of boxcar window sizes to try. If None, automatically uses
        powers of 2 as window sizes.
        
    Returns
    -------
    l : int
        Lower bound
    r : int
        Upper bound
    metadata : dict
        Dictionary with metadata related to peak-finding. Contains peak SNR 
        corresponding to the "best" boxcar filter.
    """
    if window_sizes is None:
        p = int(np.log2(len(spec)))
        window_sizes = [2**x for x in range(p)]
        
    max_found = {'l': 0, 'r': len(spec), 'snr': 0}
    for window in window_sizes:
        corr = np.correlate(spec, np.ones(window), mode='valid') / window

        clipped_corr = sigma_clip(corr)
        m, s = np.mean(clipped_corr), np.std(clipped_corr)

        peak_snr = (np.max(corr) - m) / s
        peak_idx = np.argmax(corr)

        l = peak_idx
        r = peak_idx + window
        if peak_snr > max_found['snr']:
            max_found['l'] = l
            max_found['r'] = r
            max_found['snr'] = peak_snr
            
    metadata = {
        'peak_snr': max_found['snr']
    }
    return max_found['l'], max_found['r'], metadata