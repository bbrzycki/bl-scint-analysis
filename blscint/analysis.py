import os
from pathlib import Path
import click 
import collections
import shutil
import psutil
import subprocess
import tqdm

import numpy as np
import setigen as stg

from . import diag_stats
from . import frame_processing
from . import bounds
from . import hit_parser


def as_file_list(fns, node_excludes=[], str_excludes=[]):
    """
    Expand files, using glob pattern matching, into a full list.
    In addition, user can specify strings to exclude in any filenames.
    
    Parameters
    ----------
    fns : list
        List of files or patterns (i.e. for use with glob)
    node_excludes : list, optional
        List which nodes should be excluded from analysis, particularly
        for overlapped spectrum
    str_excludes : list, optional
        List of strings that shouldn't appear in filenames
        
    Returns
    -------
    fns : list
        Returned list of all suitable filenames
    """
    if not isinstance(fns, list):
        fns = [fns]
    fns = [fn for exp_fns in fns for fn in glob.glob(exp_fns)]
    fns.sort()
    for exclude_str in node_excludes:
        exclude_str = f"{int(exclude_str):02d}"
        fns = [fn for fn in fns if f"blc{exclude_str}" not in fn]
    for exclude_str in str_excludes:
        fns = [fn for fn in fns if exclude_str not in fn]
    return fns


@click.command(short_help='Run deDoppler analysis on observations')
@click.argument('filename', nargs=-1)
@click.option('-d', '--hits-dir', 
              help='Target directory for hits files, if different from data directory')
@click.option('-s', '--snr', default=25,
              help='SNR detection threshold')
@click.option('-M', '--max-drift', default=10,
              help='Maximum drift rate threshold')
# @click.option('-m', '--min-drift', default=0.00001,
#               help='Minimum drift rate threshold')
@click.option('-g', '--gpu/--no-gpu', default=False,
              help='Option to use GPU for computation')
def dedoppler(filename, hits_dir, snr, max_drift, gpu):
    """
    Run deDoppler analysis on observation files. First tries
    seticore, then turboSETI. If GPU is not enabled, then use
    turboSETI.
    """
    seticore_path = shutil.which('seticore')
    turboseti_path = shutil.which('turboSETI')

    for data_fn in filename:
        if gpu:
            if seticore_path is not None:
                print('Running seticore')
                p = psutil.Popen([seticore_path, 
                                  data_fn,
                                  '--output', hits_dir,
                                  '--snr', str(snr),
                                  '--max_drift', str(max_drift)],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            elif turboseti_path is not None:
                print('Running turboSETI with GPU')
                p = psutil.Popen([turboseti_path, 
                                  data_fn,
                                  '--out_dir', hits_dir,
                                  '--snr', str(snr),
                                  '--max_drift', str(max_drift),
                                  '--gpu', 'y'],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            else:
                raise FileNotFoundError('No GPU deDoppler code found in PATH')
        else:
            if turboseti_path is not None:
                print('Running turboSETI')
                p = psutil.Popen([turboseti_path, 
                                  data_fn,
                                  '--out_dir', hits_dir,
                                  '--snr', str(snr),
                                  '--max_drift', str(max_drift)],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            else:
                raise FileNotFoundError('No deDoppler code found in PATH')


@click.command(short_help='Compute diagnostic statistics for each detected signal',
               no_args_is_help=True,)
@click.argument('filename', nargs=-1)
@click.option('-d', '--hits-dir', 
              help='Directory with .dat hits files, if different from data directory')
@click.option('-b', '--bound-type', default='threshold',
              type=click.Choice(['threshold', 'snr'], 
                                case_sensitive=False),
              help='How to frequency bound signals (threshold, snr)')
@click.option('-r', '--replace/--no-replace', default=False,
              help='Replace existing diagnostic statistic csv files')
@click.option('--save-ts/--no-save-ts', default=False,
              help='Save time series intensities to file')
@click.option('--threshold-fn', 
              help='Filename of thresholding file')
def diagstat(filename, 
             hits_dir, 
             bound_type, 
             replace, 
             save_ts,
             threshold_fn, 
             init_fchans=256,
             divide_std=True):
    """
    Compute diagnostic statistics for each detected signal
    """
    csv_path_list = []
    # filename is read in as an array
    for data_fn in filename:
        data_path = Path(data_fn)
        click.echo(f"Working on {data_path}")
        hits_path = Path(hits_dir) / f"{data_path.stem}.dat"
        csv_path = Path(hits_dir) / f"{data_path.stem}_diagstat.csv"
        tsdump_path = Path(hits_dir) / f"{data_path.stem}_tsdump.npy"

        # Skip if csv already exists
        if not csv_path.exists or replace:
            hp = hit_parser.HitParser(hits_path)

            if save_ts:
                tsdump = np.full((len(hp.df), hp.frame_metadata["tchans"]), 
                                 np.nan)

            ts_stats_dict = collections.defaultdict(list)
            for idx, row in tqdm.tqdm(hp.df.iterrows()):
                found_peak = False
                fchans = init_fchans 
                while not found_peak:
                    try:
                        frame = hp.centered_frame(idx, 
                                                  fchans=fchans, 
                                                  data_fn=data_path)
                        # Dedrift using hit metadata
                        frame = stg.dedrift(frame) 
                        spec = frame.integrate()
                        l, r, _ = bounds.polyfit_bounds(spec, 
                                                        deg=1, 
                                                        snr_threshold=10)
                        found_peak = True
                    except ValueError:
                        # If no fit found, or out of bounds
                        fchans *= 2
                    except IndexError:
                        # Broadband interferer
                        l, r, metadata = None, None, None
                        ts_stats = diag_stats.empty_diag_stats(fchans)
                        break

                # If IndexError, it most likely was not a narrowband signal,
                # so skip it
                if l is not None:
                    try:
                        if bound_type == 'snr':
                            l, r, _ = bounds.snr_bounds(spec, snr=5)
                        else:
                            l, r, _ = bounds.threshold_baseline_bounds(spec)

                        n_frame = frame_processing.normalize_frame(frame, 
                                                                   divide_std=divide_std)
                        tr_frame = n_frame.get_slice(l, r)

                        # Get time series and normalize
                        ts = tr_frame.integrate('f')
                        ts = ts / np.mean(ts)
                        if save_ts:
                            tsdump[idx, :] = ts

                        ts_stats = diag_stats.get_diag_stats(ts)
                        ts_stats['fchans'] = fchans
                        ts_stats['l'] = l
                        ts_stats['r'] = r
                    except IndexError:
                        ts_stats = diag_stats.empty_diag_stats(fchans)

                for stat in ts_stats:
                    ts_stats_dict[stat].append(ts_stats[stat])

            # Set statistic columns
            for stat in ts_stats_dict:
                hp.df[stat] = ts_stats_dict[stat]

            # Save absolute path to dataframe
            hp.df['data_fn'] = data_path.resolve()
            hp.df['node'] = data_path.name[:5]
            hp.df.to_csv(csv_path, index=False)

            # Save time series intensities for all signals if option enabled
            if save_ts:
                hp.df['tsdump_fn'] = tsdump_path.resolve()
                np.save(tsdump_path, tsdump)
        csv_path_list.append(csv_path)
    return csv_path_list





