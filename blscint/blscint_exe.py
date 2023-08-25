#!/usr/bin/env python3
import os
import click

from ._version import __version__
from . import _analyze_obs
from . import analysis


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, '-V', '--version')
@click.pass_context
def cli(ctx):
    """
    Scintillation analysis suite for narrowband SETI
    """
    pass 


# @click.command(short_help='Compute diagnostic statistics for each detected signal',
#                no_args_is_help=True,)
# @click.argument('filename')
# @click.option('-hd', '--hits-dir', 
#               help='Directory with .dat hits files, if different from data directory')
# @click.option('-b', '--bound', default='threshold',
#               type=click.Choice(['threshold', 'snr'], 
#                                 case_sensitive=False),
#               help='How to frequency bound signals (threshold, snr)')
# @click.option('-t', '--threshold-fn', 
#               help='Filename of thresholding file')
# def diagstat(filename, hits_dir, bound, threshold_fn):
#     """
#     Compute diagnostic statistics for each detected signal
#     """
#     _analyze_obs.run_bbox_stats(turbo_dat_fns=filename,
#                                data_dir=filename)


@click.command(short_help='Perform Monte Carlo simulations of expected scattering parameters',
               no_args_is_help=True,)
@click.argument('filename')
@click.option('-n', '--sample-number', 
              help='Number of samples per pointing')
@click.option('-p', '--pointing', multiple=True,
              help='Telescope pointings in Galactic coordinates')
@click.option('-d', '--distance', multiple=True,
              help='Distances between which to sample')
@click.option('-f', '--frequency', multiple=True,
              help='Frequencies between which to sample')
@click.option('-v', '--velocity', multiple=True,
              help='Transverse velocities between which to sample')
@click.option('-w', '--distance-weighting',
              type=click.Choice(['carrollostlie', 'mcmillan', 'uniform'], 
                                case_sensitive=False),
              help='How to weight distance sampling (carrollostlie, uniform, mcmillan)')
@click.option('--weight-by-flux', is_flag=True,
              help='Add weighting by flux (distance square-law)')
@click.option('-r', '--scint-regime',
              type=click.Choice(['kolmogorov', 'squarelaw'], 
                                case_sensitive=False),
              help='Scintillation regime to use: kolmogorov (5/3) or squarelaw (2)')
@click.option('--store-idp', is_flag=True,
              help='Store intermediate data products')
def montecarlo(filename, sample_number, pointing, distance, frequency, velocity, 
               distance_weighting, weight_by_flux, scint_regime, store_idp):
    """
    Perform Monte Carlo simulations of expected scattering parameters
    """
    pass


@click.command(short_help='Make datasets of sythetic scintillated signals',
               no_args_is_help=True,)
@click.argument('filename')
@click.option('-t', '--tscint', multiple=True,
              help='Scintillation timescales to synthesize')
@click.option('-n', '--sample-number', 
              help='Number of samples per scintillation timescale')
@click.option('--dt', 
              help='Time resolution of observations')
@click.option('--tchans', 
              help='Number of time channels in each observation')
@click.option('--df', 
              help='Frequency resolution of observations')
@click.option('--rfi-csv', 
              help='Diagstat csv for RFI observations')
@click.option('--data-csv', 
              help='Diagstat csv for data observations')
@click.option('--store-idp', is_flag=True,
              help='Store intermediate data products')
def synthesize(filename, tscint, sample_number, dt, tchans, df,
               rfi_csv, data_csv, store_idp):
    """
    Make datasets of sythetic scintillated signals for use in thresholding
    """
    pass


cli.add_command(analysis.dedoppler)
cli.add_command(analysis.diagstat)
cli.add_command(montecarlo)
cli.add_command(synthesize)

if __name__ == '__main__':
    cli()