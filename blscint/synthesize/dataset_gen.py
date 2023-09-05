from pathlib import Path
import click 
import collections
import numpy as np
import pandas as pd
import tqdm
import setigen as stg

from blscint import synthesize
from blscint import frame_processing
from blscint import diag_stats


class SignalGenerator(object):
    """
    Class to synthesize scintillated signals for use in thresholding.
    """
    def __init__(self, dt, df, tchans, seed=None, **kwargs):
        self.rng = np.random.default_rng(seed)

        self.frame_metadata = {
            'dt': dt,
            'df': df,
            'tchans': tchans
        }
        self.df = None

    def make_dataset(self, 
                     t_d, 
                     n=1000, 
                     snr=25,
                     injected=False,
                     bound='threshold',
                     gen_method='arta',
                     pow=5/3, 
                     divide_std=True,
                     file_stem=None, 
                     save_ts=False,
                     seed=None):
        """
        Create dataset of synthetic scintillated signals, and save
        statistic details to csv. 

        gen_method is either 'arta' or 'fft'.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if file_stem is not None:
            stem_path = Path(file_stem)
            csv_path = stem_path.parent / f"{stem_path.name}.diagstat.csv"
            tsdump_path = stem_path.parent / f"{stem_path.name}.tsdump.npy"
        
        stats_df = pd.DataFrame()
        ts_stats_dict = collections.defaultdict(list)
        if save_ts:
            tsdump = np.full((n, self.frame_metadata['tchans']), 
                             np.nan)
        for idx in tqdm.trange(n):
            if gen_method == 'arta':
                ts = synthesize.get_ts_arta(t_d, 
                                            self.frame_metadata['dt'],
                                            self.frame_metadata['tchans'],
                                            p=self.frame_metadata['tchans']//4,
                                            pow=pow)
            elif gen_method == 'fft':
                ts = synthesize.get_ts_fft(t_d,
                                           self.frame_metadata['dt'],
                                           self.frame_metadata['tchans'],
                                           pow=pow)
            else:
                raise ValueError("Generation method must be either 'arta' or 'fft'")
            l = r = fchans = None

            if injected:
                frame = stg.Frame(fchans=256,
                                  tchans=self.frame_metadata['tchans'],
                                  df=self.frame_metadata['df'],
                                  dt=self.frame_metadata['dt'])
                frame.add_noise_from_obs()
                signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(128), 
                                                            drift_rate=0),
                                        ts * frame.get_intensity(snr=snr),
                                        stg.sinc2_f_profile(width=2 * frame.df),
                                        stg.constant_bp_profile(level=1))

                ts, (l, r) = frame_processing.extract_ts(frame,
                                                         bound=bound,
                                                         divide_std=divide_std)
                if save_ts:
                    tsdump[idx, :] = ts
                fchans = frame.fchans
                
            ts_stats = diag_stats.get_diag_stats(ts, 
                                                 dt=self.frame_metadata['dt'])
            ts_stats.update({
                'fchans': fchans,
                't_d': t_d,
                'l': l,
                'r': r,
                'SNR': snr,
                'DriftRate': 0,
            })

            for stat in ts_stats:
                ts_stats_dict[stat].append(ts_stats[stat])

        # Set statistic columns
        for stat in ts_stats_dict:
            stats_df[stat] = ts_stats_dict[stat]

        stats_df['real'] = False

        if self.df is None:
            self.df = stats_df 
        else:
            self.df = pd.concat([self.df, stats_df], ignore_index=True)

        if file_stem is not None:
            # Save time series intensities for all signals if option enabled
            if save_ts:
                stats_df['tsdump_fn'] = tsdump_path.resolve()
                np.save(tsdump_path, tsdump)

            stats_df.to_csv(csv_path, index=False)


@click.command(name='synthesize',
               short_help='Make datasets of sythetic scintillated signals',
               no_args_is_help=True,)
# @click.argument('filename')
@click.option('-d', '--save-dir', 
              help='Directory to save output files')
@click.option('-t', '--tscint', multiple=True, type=float,
              help='Scintillation timescales to synthesize')
@click.option('-n', '--sample-number', type=int, default=1000, show_default=True,
              help='Number of samples per scintillation timescale')
@click.option('--dt', type=float,
              help='Time resolution of observations')
@click.option('--tchans', type=int,
              help='Number of time channels in each observation')
@click.option('--df', type=float, 
              help='Frequency resolution of observations')
@click.option('--rfi-csv', 
              help='Diagstat csv for RFI observations')
@click.option('--data-csv', 
              help='Diagstat csv for data observations')
@click.option('-i', '--injected', is_flag=True,
              help='Whether to inject synthetic signals and extract noisy intensity time series')
@click.option('--store-idp', is_flag=True,
              help='Store intermediate data products')
def synthesize_dataset(save_dir, tscint, sample_number, dt, tchans, df,
                       rfi_csv, data_csv, injected, store_idp):
    """
    Make datasets of sythetic scintillated signals for use in thresholding
    """
    generator = SignalGenerator(dt=dt, df=df, tchans=tchans)
    for t_d in tscint:
        generator.make_dataset(t_d=t_d,
                               n=sample_number,
                               injected=injected,
                               file_stem=Path(save_dir) / f"synthetic_{t_d}s")