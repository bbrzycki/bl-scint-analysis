import os
import glob
import numpy as np
import pandas as pd
import setigen as stg
import blimpy as bl
import matplotlib.pyplot as plt

from scipy import optimize
import scipy.stats
from astropy.stats import sigma_clip

from turbo_seti.find_doppler.find_doppler import FindDoppler

from . import bounds

import sprofiler as sp


def run_turboseti(obs_fns, max_drift=5, snr=10, out_dir='.'):
    """
    Accept observation as input, return and save csv as output (via pandas).
    """
    p = sp.Profiler(logname='turboseti.log')
    dat_list = []
    for fn in glob.glob(obs_fns):
        p.start('turboseti')
        find_seti_event = FindDoppler(fn,
                                      max_drift=max_drift,
                                      snr=snr,
                                      out_dir=out_dir,
                                      gpu_backend=True,
                                      precision=1)
        find_seti_event.search()
        dat_list.append(f"{out_dir}/{os.path.basename(fn)}")
        p.stop('turboseti')
    return dat_list


def bbox_stats(dat_fns, data_dir='.', data_ext='.fil'):
    """
    Accept TurboSETI .dat files as input, return and save csv as output (via pandas).
    Boundary box statistics.
    """
    p = sp.Profiler(logname='bounding_box.log')
    csv_list = []
    for dat_fn in glob.glob(dat_fns):
        print(f"Working on {dat_fns}")
        data_fn = f"{data_dir}/{os.path.splitext(os.path.basename(dat_fn))[0]}.{data_ext}"

        df = bls.make_dataframe(dat_file)
        param_dict = bls.get_frame_params(fn)

        ts_stats_dict = collections.defaultdict(list)
        for index, row in tqdm.tqdm(df.iterrows()):
            found_peak = False
            fchans = 256
            while not found_peak:
                try:
                    frame = bls.turbo_centered_frame(index, df, data_fn, fchans, **param_dict)
                    frame = stg.dedrift(frame)
                    spec = frame.integrate()

                    p.start('polyfit')
                    l, r, metadata = bls.polyfit_bounds(spec, deg=1, snr_threshold=10)
                    p.stop('polyfit')

                    found_peak = True
                except ValueError:
                    fchans *= 2
                    p.remove('polyfit')

            p.start('threshold_bounds')
            l, r, metadata = bls.threshold_bounds(spec, half_width=3)
            p.stop('threshold_bounds')

            n_frame = bls.t_norm_frame(frame)
            tr_frame = n_frame.get_slice(l, r)

            # Get time series and normalize
            ts = tr_frame.integrate('f')
            ts = ts / np.mean(ts)

            ts_stats = bls.get_stats(ts)
            ts_stats['fchans'] = fchans
            for key in ts_stats:
                ts_stats_dict[key].append(ts_stats[key])

        # Set statistic columns
        for key in ts_stats_dict:
            df[key] = ts_stats_dict[key]

        df['fn'] = data_fn
        csv_fn = f"bbox_{os.path.splitext(dat_fn)[0]}.csv"
        df.to_csv(csv_fn)
        csv_list.append(csv_fn)
        
        
def plot_bbox_stats(csv_fns):
    df_list = [pd.read_csv(fn) for fn in glob.glob(csv_fns)]
    data_df = pd.concat(df_list, ignore_index=True)
    
    # Exclude DC bin
    data_df = data_df[data_df['ChanIndx'] != 524288]
    
    # # Exclude first compute node
    # data_df = data_df[data_df['fn'].apply(lambda x: x.split('/')[-1][3:5] != '00')]
    
    
    # Simulate signals
    p = sp.Profiler(logname='synthetic_scintillations.log')
    n_samples = 1000

    synth_stats_dicts = {}
    sample_frame = stg.Frame.from_backend_params(
                                fchans=256,
                                tchans=64, 
                                obs_length=600, 
                                sample_rate=3e9, 
                                num_branches=1024,
                                fftlength=1048576,
                                fch1=8*u.GHz,
                                ascending=False)
    for t_d in [10, 30, 100]:
        p.start('synthesize_bbox')
        ts_stats_dict = collections.defaultdict(list)

        for _ in range(n_samples):
            ts = bls.get_ts_arta(t_d, sample_frame.dt, sample_frame.tchans, p=16)
            frame = stg.Frame(fchans=sample_frame.fchans,
                              **sample_frame.get_params())
            frame.add_noise_from_obs()
            signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(128), 
                                                        drift_rate=0),
                                      ts * frame.get_intensity(snr=25),
                                      stg.sinc2_f_profile(width=30*u.Hz),
                                      stg.constant_bp_profile(level=1))
            l, r, _ = bls.threshold_bounds(frame.integrate())

            n_frame = bls.t_norm_frame(frame)
            tr_frame = n_frame.get_slice(l, r)
            tr_ts = tr_frame.integrate('f')
            tr_ts /= tr_ts.mean()

            # Just get the stats for the detected signal
            ts_stats = bls.get_stats(tr_ts)

            for key in ts_stats:
                ts_stats_dict[key].append(ts_stats[key])

        synth_stats_dicts[t_d] = ts_stats_dict
        p.stop('synthesize_bbox')
    
    
    
    
    keys = ['std', 'min', 'ks', 'lag1']
    t_ds = [10, 30, 100]

    fig, axs = plt.subplots(1, len(keys), figsize=(20, 4), sharex='col')

    for j, key in enumerate(keys):
        bins=np.histogram(np.hstack([synth_stats_dicts[t_d][key] for t_d in t_ds] + [non_dc[key]]), bins=40)[1]
        for i, t_d in enumerate(t_ds):
            axs[j].hist(synth_stats_dicts[t_d][key], bins=bins, histtype='step', label=f'{t_d} s')
            axs[j].set_title(f'{key.upper()}')
            axs[j].xaxis.set_tick_params(labelbottom=True)
    #         axs[j].legend()

        axs[j].hist(data_df[key], bins=bins, histtype='step', color='k', lw=2, label='Non-DC RFI')
        axs[j].set_title(f'{key.upper()}')
        axs[j].legend(loc=[1, 1, 1, 2][j])
    plt.savefig('bbox_stats.pdf', bbox_inches='tight')