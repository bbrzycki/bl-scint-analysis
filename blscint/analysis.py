import os
import sys
import glob
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
        List of files or patterns, as strings or pathlib.Paths
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
    
    paths = []
    for pattern in fns:
        if str(pattern)[0] == "/":
            paths.extend(Path("/").glob(str(pattern)[1:]))
        else:
            paths.extend(Path().glob(str(pattern)))
    
    paths.sort()
    for exclude_str in node_excludes:
        exclude_str = f"{int(exclude_str):02d}"
        paths = [fn for fn in paths if f"blc{exclude_str}" not in fn.name]
    for exclude_str in str_excludes:
        paths = [fn for fn in paths if exclude_str not in fn.name]
    return paths


@click.command(name='dedoppler',
               short_help='Run deDoppler analysis on observations')
@click.argument('filename', nargs=-1)
@click.option('-d', '--hits-dir', default=None,
              help='Target directory for hits files, if different from data directory')
@click.option('-s', '--snr', default=25, show_default=True,
              help='SNR detection threshold')
@click.option('-M', '--max-drift', default=10, show_default=True,
              help='Maximum drift rate threshold')
# @click.option('-m', '--min-drift', default=0.00001, show_default=True,
#               help='Minimum drift rate threshold')
@click.option('-g', '--gpu/--no-gpu', default=False, 
              help='Use GPU for computation')
@click.option('-t', '--turbo', is_flag=True, default=False, 
              help='Force use of turboSETI')
def dedoppler(filename, hits_dir=None, snr=25, max_drift=10, gpu=False, turbo=False):
    """
    Run deDoppler analysis on observation files. First tries
    seticore, then turboSETI. If GPU is not enabled, then use
    turboSETI.
    """
    seticore_path = shutil.which('seticore')
    turboseti_path = shutil.which('turboSETI')

    seticore_used = False

    for data_fn in filename:
        data_path = Path(data_fn)
        if hits_dir is None:
            hits_dir_path = data_path.parent
        else:
            hits_dir_path = Path(hits_dir)
        if gpu:
            if seticore_path is not None and not turbo:
                print('Running seticore')
                seticore_used = True
                command = [
                    str(seticore_path), 
                    str(data_path),
                    '--output', str(hits_dir_path) + '/',
                    '--snr', str(snr),
                    '--max_drift', str(max_drift),
                ]
            elif turboseti_path is not None:
                print('Running turboSETI with GPU')
                command = [
                    str(turboseti_path), 
                    str(data_path),
                    '--out_dir', str(hits_dir_path) + '/',
                    '--snr', str(snr),
                    '--max_drift', str(max_drift),
                    '--gpu', 'y',
                ]
            else:
                raise FileNotFoundError('No GPU deDoppler code found in PATH')
        else:
            if turboseti_path is not None:
                print('Running turboSETI')
                command = [
                    str(turboseti_path), 
                    str(data_path),
                    '--out_dir', str(hits_dir_path) + '/',
                    '--snr', str(snr),
                    '--max_drift', str(max_drift),
                ]
            else:
                raise FileNotFoundError('No deDoppler code found in PATH')
            
        p = psutil.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
        print(f"Subprocess PID: {p.pid} ({data_fn})\n")

        for line in p.stdout:
            sys.stdout.write(line)

        if seticore_used and 'unable' in line:
            raise RuntimeError(line)


@click.command(name='diagstat',
               short_help='Compute diagnostic statistics for each detected signal',
               no_args_is_help=True,)
@click.argument('filename', nargs=-1)
@click.option('-d', '--hits-dir', default=None, 
              help='Directory with .dat hits files, if different from data directory')
@click.option('-b', '--bound', default='threshold', show_default=True,
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
             hits_dir=None, 
             bound='threshold', 
             replace=False, 
             save_ts=False,
             threshold_fn=None, 
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

        if hits_dir is None:
            hits_dir_path = data_path.parent
        else:
            hits_dir_path = Path(hits_dir)
        hits_path = hits_dir_path / f"{data_path.stem}.dat"
        csv_path = hits_dir_path / f"{data_path.stem}.diagstat.csv"
        tsdump_path = hits_dir_path / f"{data_path.stem}.tsdump.npy"

        # Skip if csv already exists
        if not csv_path.exists() or replace:
            hp = hit_parser.HitParser(hits_path)

            if save_ts:
                tsdump = np.full((len(hp.df), hp.frame_metadata["tchans"]), 
                                 np.nan)

            ts_stats_dict = collections.defaultdict(list)
            for idx, row in tqdm.tqdm(hp.df.iterrows(), total=hp.df.shape[0]):
                found_peak = False
                fchans = init_fchans 
                while not found_peak:
                    try:
                        frame = hp.centered_frame(idx, 
                                                  fchans=fchans, 
                                                  data_fn=data_path)
                        # Dedrift using hit metadata
                        frame = stg.dedrift(frame) 
                        spec = stg.integrate(frame)
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
                        ts, (l, r) = frame_processing.extract_ts(frame,
                                                                 bound=bound,
                                                                 divide_std=divide_std)
                            
                        ts = ts.array()
                        if save_ts:
                            tsdump[idx, :] = ts

                        ts_stats = diag_stats.get_diag_stats(ts, dt=frame.dt)
                        ts_stats.update({
                            'fchans': fchans,
                            'l': l,
                            'r': r,
                        })
                    except IndexError:
                        ts_stats = diag_stats.empty_diag_stats(fchans)

                for stat in ts_stats:
                    ts_stats_dict[stat].append(ts_stats[stat])

            # Set statistic columns
            for stat in ts_stats_dict:
                hp.df[stat] = ts_stats_dict[stat]

            # Save absolute path to dataframe
            hp.df['data_fn'] = str(data_path.resolve())
            hp.df['node'] = data_path.name[:5]

            # Save time series intensities for all signals if option enabled
            if save_ts:
                hp.df['tsdump_fn'] = str(tsdump_path.resolve())
                np.save(tsdump_path, tsdump)

            hp.df.to_csv(csv_path, index=False)
        csv_path_list.append(csv_path)
    return csv_path_list





