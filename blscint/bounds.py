import numpy as np
from scipy import optimize
import scipy.stats
from astropy.stats import sigma_clip
import setigen as stg

from . import factors
from .peaks import *


def clipped_bounds(frame, min_bins=2, min_clipped=1, peak_prominence=4):
    """
    Run sigma clip on 2D data array, and find central peak above a certain
    number of clipped values along the time axis, per frequency bin.
    
    This will return an IndexError if the signal passes outside of the frame
    (i.e. for very wide signals).
    """
    n_frame = t_norm_frame(frame)
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
    peak_i = peaks[0][max_idx[np.argmin(np.abs(peaks[0][max_idx] - 128))]]
    
    # I find that np.find_peaks doesn't do a good job for bounding boxes
#     l = peaks[1]['left_bases'][idx]
#     r = peaks[1]['right_bases'][idx]

    # Change mask to thresholding by number of pixels along time axis
    mask_spec = mask_spec >= min_clipped
#     print(peak_i, mask_spec)

    # Convolve with ones, to find where sequences are adjacent zeros
    convolved = np.convolve(mask_spec,
                            np.ones(min_bins).astype(int),
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
    l = l_idx[l_idx + min_bins <= peak_i][-1] + min_bins
    r = r_idx[r_idx >= peak_i][0]
    
#     cutoffs = np.where(np.abs(diffs)==1)[0]
#     print(cutoffs)
#     # Find which range of bins the peak lies in
#     i = np.digitize(peak_i, cutoffs) - 1
#     # Adjust left edge to make up for zero'd bins from the convolution,
#     # since we only care about the signal
#     l, r = cutoffs[i] + (min_bins), cutoffs[i + 1]
    
    metadata = {
        'num_peaks': len(peaks[0]), 
        'peaks': peaks
    }
    return l, r+1, metadata


def polyfit_bounds(spec, deg=7, snr_threshold=10):
    """
    Bounding box set by a polynomial fit to the background. Edges are set by
    where the fit intersects the data on either side of the central peak.
    
    spec is a numpy array representing the spectrum.
    deg is the polynomial fit degree
    """
    y = sigma_clip(spec)
    x = np.arange(len(spec))

    coeffs = np.polyfit(x[~y.mask], y[~y.mask], deg)
    poly = np.poly1d(coeffs)

    # Estimate noise std (with background fit subtraction)
    std = np.std(y[~y.mask] - poly(x[~y.mask]))

    # Get peaks above SNR threshold
    peaks = scipy.signal.find_peaks(spec - poly(x), prominence=snr_threshold * std)
#     print(peaks)
    
    # Find highest peak
    i = np.argmax(peaks[1]['prominences'])
    peak_i = peaks[0][i]
    
    cutoffs = np.where(spec - poly(x) <= 0)[0]
    
    i = np.digitize(peak_i, cutoffs) - 1
    l, r = cutoffs[i], cutoffs[i + 1]
    
    metadata = {
        'poly': poly, 
        'num_peaks': len(peaks[0]), 
        'peaks': peaks
    }
    return l, r+1, metadata


def gaussian_bounds(spec, half_width=3, peak_guess=None):
    """
    Create bounds based on a Gaussian fit to the central peak.
    
    spec is a numpy array representing the spectrum.
    half_width is how many sigma to go from center.
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
    return peak - offset, peak + offset+1, None
#     return int(peak), (-offset, offset)


def threshold_bounds(spec, half_width=3):
    """
    Create bounds based on intensity attentuation on either side of the central
    peak. Threshold is set by ideal Gaussian profile, in units of standard deviations (sigma).
    
    spec is a numpy array representing the spectrum.
    half_width is how many sigma to go from center.
    """
    noise_spec = sigma_clip(spec, masked=True)
    norm_spec = (spec - np.mean(noise_spec)) / (np.max(spec) - np.mean(noise_spec))
    
    threshold = stg.func_utils.gaussian(half_width, 0, 1)
    cutoffs = np.where(norm_spec < threshold)[0]
    
    peak = np.argmax(norm_spec)
    i = np.digitize(peak, cutoffs) - 1
    l, r = cutoffs[i], cutoffs[i + 1]
    return l, r+1, None
#     offset1, offset2 = cutoffs[i] - peak, cutoffs[i + 1] - peak
#     return peak, (offset1, offset2)