import numpy as np
from scipy import optimize
from astropy.stats import sigma_clip
import setigen as stg

from . import factors

 
# def gaussian_bounds(spec, half_width=3, peak_guess=None):
#     """
#     spec is a numpy array representing the spectrum.
#     half_width is how many sigma to go from center.
#     """
#     gaussian_func = lambda x, A, x0, sigma, y: A * stg.func_utils.gaussian(x, x0, sigma) + y
    
#     if peak_guess is not None:
#         peak_guess = [1, peak_guess, 1, 1]
#     popt, _ = optimize.curve_fit(gaussian_func, 
#                                  np.arange(len(spec)),
#                                  spec,
#                                  p0=peak_guess)
    
#     peak = popt[1]
#     sigma = abs(popt[2])
#     offset = int(sigma * half_width)
#     return int(peak), (-offset, offset)


# def threshold_bounds(spec, half_width=3):
#     """
#     spec is a numpy array representing the spectrum.
#     half_width is how many sigma to go from center.
#     """
#     noise_spec = sigma_clip(spec, masked=True)
#     norm_spec = (spec - np.mean(noise_spec)) / (np.max(spec) - np.mean(noise_spec))
    
#     threshold = stg.func_utils.gaussian(half_width, 0, 1)
#     cutoffs = np.where(norm_spec < threshold)[0]
    
#     peak = np.argmax(norm_spec)
#     i = np.digitize(peak, cutoffs) - 1
#     offset1, offset2 = cutoffs[i] - peak, cutoffs[i + 1] - peak
#     return peak, (offset1, offset2)


def t_norm_frame(frame, as_data=None):
    """
    "Normalize" frame by subtracting out noise background, along time axis.
    """
    if as_data is not None:
        # as_data is a Frame from which to get the bounds, to normalize 'frame'
        data = as_data.data
    else:
        data = frame.data
    clipped_data = sigma_clip(data, axis=1, masked=True)
    n_frame = frame.copy()
    n_frame.data = (frame.data - np.mean(clipped_data, axis=1, keepdims=True))
    return n_frame


def dedrift_frame(frame, drift_rate):
    max_offset = int(abs(drift_rate) * frame.tchans * frame.dt / frame.df)
    tr_data = np.zeros((frame.data.shape[0], frame.data.shape[1] - max_offset))

    for i in range(frame.data.shape[0]):
        offset = int(abs(drift_rate) * i * frame.dt / frame.df)
        if drift_rate >= 0:
            start_idx = 0 + offset
            end_idx = start_idx + tr_data.shape[1]
        else:
            end_idx = frame.data.shape[1] - 1 - offset
            start_idx = end_idx - tr_data.shape[1]
        tr_data[i] = frame.data[i, start_idx:end_idx]
        
    # Match frequency to truncated frame
    if frame.ascending:
        if drift_rate >= 0:
            fch1 = frame.fs[0]
        else:
            fch1 = frame.fs[max_offset]
    else:
        if drift_rate >= 0:
            fch1 = frame.fs[::-1][max_offset]
        else:
            fch1 = frame.fs[::-1][0]
        
    tr_frame = stg.Frame.from_data(frame.df, frame.dt, fch1, frame.ascending, tr_data)
    return tr_frame


def truncate_frame(frame, shift_freq=False, as_data=None, half_width=3, trunc_type='gaussian', norm_spec=False, peak_guess=None):
    """
    trunc_type='gaussian' or 'threshold'.
    shift_freq determines whether to match peaks along freq axis; otherwise don't shift at all.
    as_data uses another frame to fit.
    norm_spec is a bool for normalizing spectrum to a maximum of 1, to aid gaussian fitting.
    """
    if as_data is not None:
        # as_data is a Frame from which to get the bounds, to truncate 'frame'
        data = as_data.data
    else:
        data = frame.data

    if trunc_type == 'gaussian':
        trunc_func = lambda spec: gaussian_bounds(spec, half_width=half_width, peak_guess=peak_guess)
    else:
        trunc_func = lambda spec: threshold_bounds(spec, half_width=half_width)
    if shift_freq:
        peaks = np.zeros(data.shape[0], dtype=int)
        offs1 = np.zeros(data.shape[0], dtype=int)
        offs2 = np.zeros(data.shape[0], dtype=int)
        for i in range(data.shape[0]):
            spec = data[i]
            if norm_spec:
                spec = spec / np.max(spec)
            peaks[i], (offs1[i], offs2[i]) = trunc_func(spec)
            if trunc_type == 'threshold':
                peaks[i] += np.mean([offs1[i], offs2[i]])
#         offset = int((np.max(offs2) - np.min(offs1))/2)
        offset = int((np.mean(offs2) - np.mean(offs1))/2)

        tr_data = np.zeros((data.shape[0], offset * 2))
        for i in range(data.shape[0]):
            # Have to be a bit careful -- now this won't complain if signal doesn't fit
            l_bound = max(0, peaks[i]-offset)
            r_bound = min(frame.fchans-1, peaks[i]+offset)
            tr_data[i] = frame.data[i][l_bound:r_bound]
        peak = peaks[0]
        l_bound = max(0, peaks[0]-offset)
        r_bound = min(frame.fchans, peaks[0]+offset)
    else:
        spec = np.mean(data, axis=0)
        if norm_spec:
            spec = spec / np.max(spec)
        peak, (_, offset) = trunc_func(spec)
        
        # Have to be a bit careful -- now this won't complain if signal doesn't fit
        l_bound = max(0, peak-offset)
        r_bound = min(frame.fchans-1, peak+offset)
        tr_data = frame.data[:, l_bound:r_bound]

    # Match frequency to truncated frame
    if frame.ascending:
        fch1 = frame.fs[l_bound]
    else:
        fch1 = frame.fs[r_bound]
    
    tr_frame = stg.Frame.from_data(frame.df, frame.dt, fch1, frame.ascending, tr_data)
    sigma = offset / half_width
    return tr_frame, peak, sigma