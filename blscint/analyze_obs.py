import os
import sys
import glob
import numpy as np
import pandas as pd
import setigen as stg
import blimpy as bl
import matplotlib.pyplot as plt
import tqdm
import collections

from astropy import units as u
from astropy.stats import sigma_clip
import scipy.stats

from turbo_seti.find_doppler.find_doppler import FindDoppler

from . import bounds
from . import dataframe
from . import frame_processing
from . import ts_statistics
from . import gen_arta

import sprofiler as sp


def as_file_list(fns, node_excludes=[], str_excludes=[]):
    """
    Expand files, using glob pattern matching, into a full list.
    In addition, user can specify strings to exclude in any filenames.
    """
    if not isinstance(fns, list):
        fns = [fns]
    fns = [fn for exp_fns in fns for fn in glob.glob(exp_fns)]
    fns.sort()
    for exclude_str in node_excludes:
        exclude_str = f"{int(exclude_str):02d}"
        fns = [fn for fn in fns if f"blc{exclude_str}" not in fn]
    for exclude_str in str_excludes:
        fns = [fn for fn in fns if exclude_st not in fn]
    return fns


def run_turboseti(obs_fns, min_drift=0.00001, max_drift=5, snr=10, out_dir='.', gpu_id=0, replace_existing=False):
    """
    Accept observation as input, return and save csv as output (via pandas).
    """
    p = sp.Profiler(logname='turboseti.log')
    turbo_dat_list = []
    for data_fn in glob.glob(obs_fns):
       
        # First, check if equivalent h5 data file exists in either source or target directory
        h5_fn_old = f"{os.path.splitext(data_fn)[0]}.h5"
        h5_fn_new = f"{out_dir}/{os.path.splitext(os.path.basename(data_fn))[0]}.h5"
        if os.path.exists(h5_fn_old):
            data_fn = h5_fn_old
        elif os.path.exists(h5_fn_new):
            print("Using H5 file in target directory")
            data_fn = h5_fn_new
        if gpu_id == 5:
            gpu_backend=False
            gpu_id=0
        else:
            gpu_backend=True
        turbo_dat_fn = f"{out_dir}/{os.path.splitext(os.path.basename(data_fn))[0]}.dat"
        if not os.path.exists(turbo_dat_fn) or replace_existing:
            p.start('turboseti')
            find_seti_event = FindDoppler(data_fn,
                                          min_drift=min_drift,
                                          max_drift=max_drift,
                                          snr=snr,
                                          out_dir=out_dir,
                                          gpu_backend=gpu_backend,
                                          gpu_id=gpu_id,
                                          precision=1)
            find_seti_event.search()
            turbo_dat_list.append(turbo_dat_fn)
            p.stop('turboseti')
    p.report()
    return turbo_dat_list


def get_bbox_frame(index, df):
    row = df.loc[index]
    param_dict = dataframe.get_frame_params(row['fn'])
    frame = dataframe.turbo_centered_frame(index, df, row['fn'], row['fchans'], **param_dict)
    frame = stg.dedrift(frame)
    return frame


def empty_ts_stats(fchans):
    ts_stats = {
        'std': None,
        'min': None,
        'ks': None,
        'anderson': None,
        'lag1': None,
        'lag2': None,
        'fchans': fchans,
        'l': None,
        'r': None,
        'acf_amp': None,
        'acf_sigma': None,
        'acf_noise': None,
    }
    return ts_stats


def run_bbox_stats(turbo_dat_fns, data_dir='.', data_ext='.fil', data_res_ext='.0005', replace_existing=False):
    """
    Accept TurboSETI .dat files as input, return and save csv as output (via pandas).
    Boundary box statistics.
    """
    p = sp.Profiler(logname='bounding_box.log', verbose=1)
    csv_list = []
    for turbo_dat_fn in as_file_list(turbo_dat_fns):
        print(f"Working on {turbo_dat_fn}")
        data_fn = f"{data_dir}/{os.path.splitext(os.path.basename(turbo_dat_fn))[0][:-5]}{data_res_ext}{data_ext}"
        csv_fn = f"{os.path.splitext(turbo_dat_fn)[0][:-5]}{data_res_ext}_bbox.csv"
        
        # Skip if csv already exists
        if not os.path.exists(csv_fn) or replace_existing:
            df = dataframe.make_dataframe(turbo_dat_fn)
            param_dict = dataframe.get_frame_params(data_fn)

            ts_stats_dict = collections.defaultdict(list)
            for index, row in tqdm.tqdm(df.iterrows()):
                found_peak = False
                fchans = 256
                while not found_peak:
                    try:
                        p.start('frame_init')
                        frame = dataframe.turbo_centered_frame(index, df, data_fn, fchans, **param_dict)
                        frame = stg.dedrift(frame)
                        p.stop('frame_init')
                                
                        spec = frame.integrate()

                        p.start('polyfit')
                        l, r, metadata = bounds.polyfit_bounds(spec, deg=1, snr_threshold=10)
                        p.stop('polyfit')

                        found_peak = True
                    except ValueError:
                        # If no fit found, or out of bounds
                        fchans *= 2
                        p.remove('polyfit')
                    except IndexError:
                        # Broadband interferer
                        l, r, metadata = None, None, None
                        ts_stats = empty_ts_stats(fchans)
                        p.remove('polyfit')
                        break

                # If IndexError... was probably not narrowband signal,
                # so just skip adding it in
                if l is not None:
                    try:
                        p.start('threshold_bounds')
                        l, r, metadata = bounds.threshold_baseline_bounds(spec)
                        # print(l,r)
                        p.stop('threshold_bounds')

                        n_frame = frame_processing.t_norm_frame(frame)
                        tr_frame = n_frame.get_slice(l, r)

                        # Get time series and normalize
                        ts = tr_frame.integrate('f')
                        ts = ts / np.mean(ts)

                        ts_stats = ts_statistics.get_stats(ts)
                        ts_stats['fchans'] = fchans
                        ts_stats['l'] = l
                        ts_stats['r'] = r

                    except IndexError:
                        p.remove('threshold_bounds')
                        ts_stats = empty_ts_stats(fchans)
                for key in ts_stats:
                    ts_stats_dict[f"{key}"].append(ts_stats[key])

            # Set statistic columns
            for key in ts_stats_dict:
                df[key] = ts_stats_dict[key]

            df['fn'] = data_fn
            df['node'] = os.path.basename(data_fn)[:5]

            df.to_csv(csv_fn, index=False)
        csv_list.append(csv_fn)
    p.report()
    return csv_list


def plot_snapshot(index, df):
    row = df.loc[index]
    
    param_dict = dataframe.get_frame_params(row['fn'])
    frame = dataframe.turbo_centered_frame(index, df, row['fn'], row['fchans'], **param_dict)
    dd_frame = stg.dedrift(frame)

    spec = dd_frame.integrate()

    l, r, metadata = bounds.threshold_baseline_bounds(spec)

    n_frame = frame_processing.t_norm_frame(dd_frame)
    tr_frame = n_frame.get_slice(l, r)

    # Get time series and normalize
    ts = tr_frame.integrate('f')
    ts = ts / np.mean(ts)

    ts_stats = ts_statistics.get_stats(ts)
    
    print(f"SNR : {row['SNR']:.3}")
    for stat in ts_stats:
        print(f"{stat:<4}: {ts_stats[stat]:.3}")
    print(f"l, r: {l}, {r}")
    
    plt.figure(figsize=(20, 3))
    plt.subplot(1, 4, 1)
    frame.bl_plot()
    plt.title(f'Index {index}')
    
    plt.subplot(1, 4, 2)
    bounds.plot_bounds(n_frame, l, r)
    plt.title(f"Drift rate: {row['DriftRate']:.3} Hz/s")
    
    plt.subplot(1, 4, 3)
    plt.plot(ts, c='k')
    plt.axhline(0, ls='--')
    plt.axhline(1, ls='-')
    plt.title('Time series')
    
    plt.subplot(1, 4, 4)
    acf = ts_statistics.autocorr(ts)
    plt.plot(acf, c='k')
    plt.axhline(0, ls='--')
    plt.title(f"ACF: ks={row['ks']:.3}")
    plt.show()
    

def plot_bounded_frame(index, df):
    row = df.loc[index]
    
    param_dict = dataframe.get_frame_params(row['fn'])
    frame = dataframe.turbo_centered_frame(index, df, row['fn'], row['fchans'], **param_dict)
    dd_frame = stg.dedrift(frame)

    spec = dd_frame.integrate()

    l, r, metadata = bounds.threshold_baseline_bounds(spec)

    tr_frame = dd_frame.get_slice(l, r)
    tr_frame.plot()
    plt.show()
    
    
def plot_random_snapshots(df, n=1):
    df_sampled = df.sample(n=n)
    for i in df_sampled.index:
        plot_snapshot(i, df_sampled)
        
        
def plot_all_snapshots(df):
    for i in df.index:
        plot_snapshot(i, df)
    


def get_bbox_df(csv_fns):
    """
    Read in dataframe with bbox statistics calculated.
    """
    df_list = [pd.read_csv(fn) for fn in as_file_list(csv_fns)]
    data_df = pd.concat(df_list, ignore_index=True)
    
    # Exclude DC bin (value depends on rawspec fftlength)
    # print('Before DC bins (may be excluded by TurboSETI):', data_df.shape)
    data_df = data_df[data_df['ChanIndx'] != 524288]
    # print('After removing:', data_df.shape)
    
    # # Exclude first compute node
    # data_df = data_df[data_df['fn'].apply(lambda x: x.split('/')[-1][3:5] != '00')]
    
    # Remove non-fit signals (which are replaced with NaN)
    data_df = data_df[data_df['ks'].notna()]
    return data_df
        
        
def plot_bbox_stats(csv_fns, plot_fn_prefix='bbox_stats'):
    """
    Make stats plots with RFI and synthetic signals.
    """
    data_df = get_bbox_df(csv_fns)
    
    # Simulate signals
    p = sp.Profiler(logname='synthetic_scintillations.log')
    n_samples = 1000

    synth_stats_dicts = {}
    sample_frame = stg.Frame.from_backend_params(
                                fchans=256,
                                obs_length=600, 
                                sample_rate=3e9, 
                                num_branches=1024,
                                fftlength=1048576,
                                int_factor=13,
                                fch1=8*u.GHz,
                                ascending=False)
    for t_d in [10, 30, 100]:
        p.start('synthesize_bbox')
        ts_stats_dict = collections.defaultdict(list)

        for _ in range(n_samples):
            ts = gen_arta.get_ts_arta(t_d, sample_frame.dt, sample_frame.tchans, p=32)
            frame = stg.Frame(**sample_frame.get_params())
            frame.add_noise_from_obs()
            signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(128), 
                                                        drift_rate=0),
                                      ts * frame.get_intensity(snr=10),
                                      stg.sinc2_f_profile(width=3*frame.df*u.Hz),
                                      stg.constant_bp_profile(level=1))
            l, r, _ = bounds.threshold_baseline_bounds(frame.integrate())

            n_frame = frame_processing.t_norm_frame(frame)
            tr_frame = n_frame.get_slice(l, r)
            tr_ts = tr_frame.integrate('f')
            tr_ts /= tr_ts.mean()

            # Just get the stats for the detected signal
            ts_stats = ts_statistics.get_stats(tr_ts)

            for key in ts_stats:
                ts_stats_dict[f"{key}"].append(ts_stats[key])

        synth_stats_dicts[t_d] = ts_stats_dict
        p.stop('synthesize_bbox')
    
    
    
    
    keys = ['std', 'min', 'ks', 'lag1']
    t_ds = [10, 30, 100]

    fig, axs = plt.subplots(1, len(keys), figsize=(20, 4), sharex='col')

    for j, key in enumerate(keys):
        key = f"{key}"
        bins=np.histogram(np.hstack([synth_stats_dicts[t_d][key] for t_d in t_ds] + [data_df[key]]), bins=40)[1]
        for i, t_d in enumerate(t_ds):
            axs[j].hist(synth_stats_dicts[t_d][key], bins=bins, histtype='step', label=f'{t_d} s')
            axs[j].set_title(f'{key.upper()}')
            axs[j].xaxis.set_tick_params(labelbottom=True)
    #         axs[j].legend()

        axs[j].hist(data_df[key], bins=bins, histtype='step', color='k', lw=2, label='Non-DC RFI')
        axs[j].set_title(f'{key.upper()}')
        axs[j].legend(loc=[1, 1, 1, 2][j])
    plt.savefig(f"{plot_fn_prefix}.pdf", bbox_inches='tight')