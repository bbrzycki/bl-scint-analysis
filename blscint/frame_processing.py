import numpy as np
from scipy import optimize
from astropy.stats import sigma_clip

import blimpy as bl
import setigen as stg

from . import factors
from . import bounds


def tnorm(frame, divide_std=False, as_data=None):
    """
    Normalize frame by subtracting out noise background, along time axis.
    Additional option to divide out by the standard deviation of each 
    individual spectrum. 

    Parameters
    ----------
    frame : stg.Frame
        Raw spectrogram frame
    divide_std : bool, optional
        Normalize each spectrum by dividing by its standard deviation 
    as_data : stg.Frame, optional
        Use alternate frame to compute noise stats. If desired, use a more
        isolated region of time-frequency space for cleaner computation.

    Returns
    -------
    n_frame : stg.Frame
        Normalized frame
    """
    if as_data is not None:
        # as_data is a Frame from which to get the bounds, to normalize "frame"
        data = as_data.data
    else:
        data = frame.data
    clipped_data = sigma_clip(data, axis=1, masked=True)
    n_frame = frame.copy()
    n_frame.data = (frame.data - np.mean(clipped_data, axis=1, keepdims=True))
    if divide_std:
        n_frame.data = n_frame.data / np.std(clipped_data, axis=1, keepdims=True)
    return n_frame


def extract_ts(frame, bound='threshold', divide_std=True, as_data=None):
    """
    Extract normalized time series from dedrifted frame with centered signal, 
    as well as frequency bounds as a tuple.
    """
    spec = stg.integrate(frame)

    if bound == 'threshold':
        l, r, _ = bounds.threshold_baseline_bounds(spec)
    elif bound == 'snr':
        l, r, _ = bounds.snr_bounds(spec)
    else:
        raise ValueError("Bound should be either 'threshold' or 'snr'")
    
    n_frame = tnorm(frame, divide_std=divide_std, as_data=as_data)
    # tr_frame = n_frame.get_slice(l, r)
    tr_frame = stg.get_slice(n_frame, l, r)
    # ts = tr_frame.integrate('f')
    # ts = ts / np.mean(ts)
    ts = stg.integrate(tr_frame, axis='f', as_frame=True)
    ts.normalize()
    ts.add_metadata({"l": l, "r": r})
    return ts, (l, r)


def get_metadata(fn):
    """
    Get frame resolution from a spectrogram file without loading the 
    actual data.

    Parameters
    ----------
    fn : str
        .fil or .h5 filename
        
    Returns
    -------
    params : dict
        Dictionary with tchans, df, dt
    """
    container = bl.Waterfall(str(fn), load_data=False).container
    return {
        "tchans": container.file_shape[0],
        "df": abs(container.header["foff"]) * 1e6,
        "dt": container.header["tsamp"]
    }


def centered_frame(data_fn, center_freq, drift_rate, fchans, frame_metadata=None):
    """
    center_freq is in MHz.
    """
    if frame_metadata is None:
        frame_metadata = get_metadata(data_fn)

    tchans = frame_metadata["tchans"]
    df = frame_metadata["df"]
    dt = frame_metadata["dt"]

    adj_center_freq = center_freq + drift_rate / 1e6 * tchans / 2
    max_offset = int(abs(drift_rate) * tchans * dt / df)
    if drift_rate >= 0:
        adj_fchans = [0, max_offset]
    else:
        adj_fchans = [max_offset, 0]
    
    f_start = adj_center_freq - (fchans / 2 + adj_fchans[0]) * df / 1e6
    f_stop = adj_center_freq + (fchans / 2 + adj_fchans[1]) * df / 1e6
    frame = stg.Frame(data_fn, f_start=f_start, f_stop=f_stop)
        
    frame.add_metadata({
        'drift_rate': drift_rate,
        'center_freq': center_freq,
    })
    return frame



def _dedrift_frame(frame, drift_rate=None):
    if drift_rate is None:
        if "drift_rate" in frame.metadata:
            drift_rate = frame.metadata["drift_rate"]
        else:
            raise KeyError("Please specify a drift rate to account for")
            
    # Calculate maximum pixel offset and raise an exception if necessary
    max_offset = int(abs(drift_rate) * frame.tchans * frame.dt / frame.df)
    if max_offset >= frame.data.shape[1]:
        raise ValueError(f"The provided drift rate ({drift_rate} Hz/s) is too high for the frame dimensions")
    tr_data = np.zeros((frame.data.shape[0], frame.data.shape[1] - max_offset))

    for i in range(frame.data.shape[0]):
        offset = int(abs(drift_rate) * i * frame.dt / frame.df)
        if drift_rate >= 0:
            start_idx = 0 + offset
            end_idx = start_idx + tr_data.shape[1]
        else:
            end_idx = frame.data.shape[1] - offset
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
        
    tr_frame = stg.Frame.from_data(frame.df, 
                                   frame.dt, 
                                   fch1, 
                                   frame.ascending,
                                   tr_data,
                                   metadata=frame.metadata,
                                   waterfall=frame.check_waterfall())
#     if tr_frame.waterfall is not None and "source_name" in tr_frame.waterfall.header:
#         tr_frame.waterfall.header["source_name"] += "_dedrifted"
    return tr_frame
