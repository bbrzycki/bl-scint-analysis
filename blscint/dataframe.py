import os
import numpy as np
import pandas as pd
import setigen as stg
import blimpy as bl


def make_dataframe(dat_file):
    """
    Create Pandas dataframe from TurboSETI output.

    Parameters
    ----------
    dat_file : str
        TurboSETI .dat filename to format as a dataframe
        
    Returns
    -------
    dataframe : DataFrame
        Dataframe with extracted TurboSETI parameters
    """
    root, ext = os.path.splitext(dat_file)
    if ext != '.dat':
        dat_file = f'{root}.dat'
    columns = ['TopHitNum',
                'DriftRate',
                'SNR',
                'Uncorrected_Frequency',
                'Corrected_Frequency',
                'ChanIndx',
                'FreqStart',
                'FreqEnd',
                'SEFD', 
                'SEFD_freq',
                'CoarseChanNum',
                'FullNumHitsInRange']
    dataframe = pd.read_csv(dat_file, 
                            sep='\t',
                            names=columns, 
                            comment='#',
                            index_col=False)
    return dataframe


def get_frame_params(fn):
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
    container = bl.Waterfall(fn, load_data=False).container
    return {
        'tchans': container.file_shape[0],
        'df': abs(container.header['foff']) * 1e6,
        'dt': container.header['tsamp']
    }


def centered_frame(fn,
                   drift_rate,
                   center_freq,
                   fchans=256,
                   frame_params=None):
    """
    Here, center_freq is in MHz. 
    """
    if frame_params is not None:
        tchans = frame_params['tchans']
        df = frame_params['df']
        dt = frame_params['dt']
    else:
        container = bl.Waterfall(fn, load_data=False).container
        tchans = container.file_shape[0]
        df = abs(container.header['foff']) * 1e6
        dt = container.header['tsamp']
        
    adj_center_freq = center_freq + drift_rate/1e6 * tchans/2
    max_offset = int(abs(drift_rate) * tchans * dt / df)
    if drift_rate >= 0:
        adj_fchans = [0, max_offset]
    else:
        adj_fchans = [max_offset, 0]
    wf = bl.Waterfall(fn,
                      f_start=adj_center_freq - (fchans/2 + adj_fchans[0]) * df/1e6,
                      f_stop=adj_center_freq + (fchans/2 + adj_fchans[1]) * df/1e6)
    frame = stg.Frame(wf)
    
    frame.add_metadata({
        'drift_rate': drift_rate,
        'center_freq': center_freq,
        'fn': fn,
    })
    return frame
    
    
def turbo_centered_frame(i, 
                       dataframe,
                       fn=None,
                       fchans=256, 
                       tchans=64, 
                       df=2.7939677238464355, 
                       dt=9.305762474666658,
                       **kwargs):
    """
    Create Frame centered at a target signal from a TurboSETI-created .dat file.
    Does not remove drift -- use Frame.dedrift separately to do so.

    Parameters
    ----------
    i : int
        Signal index
    dataframe : DataFrame
        Pandas dataframe with TurboSETI parameters
    fn : str, optional
        Filename of datafile (unless filename is also in the dataframe, under 'fn')
    fchans : int, optional
        Number of frequency bins in target frame
    tchans : int, optional
        Number of time samples in data
    df : float, optional
        Frequency resolution (Hz)
    dt : float, optional
        Time resolution (Hz)
        
    Returns
    -------
    frame : stg.Frame
        Generated frame
    """
    row = dataframe.loc[i]
    drift_rate = row['DriftRate']
    center_freq = row['Uncorrected_Frequency']
    if fn is None:
        fn = row['fn']

    adj_center_freq = center_freq + drift_rate/1e6 * tchans/2
    max_offset = int(abs(drift_rate) * tchans * dt / df)
    if drift_rate >= 0:
        adj_fchans = [0, max_offset]
    else:
        adj_fchans = [max_offset, 0]
    wf = bl.Waterfall(fn,
                      f_start=adj_center_freq - (fchans/2 + adj_fchans[0]) * df/1e6,
                      f_stop=adj_center_freq + (fchans/2 + adj_fchans[1]) * df/1e6)
    frame = stg.Frame(wf)
        
    frame.add_metadata({
        'drift_rate': drift_rate,
        'center_freq': center_freq,
        'i': i,
    })
    return frame
