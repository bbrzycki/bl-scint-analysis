#!/usr/bin/env python3
import os
import click

from ._version import __version__
from . import analysis
from . import simulations
from .observations import bl_obs


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, '-V', '--version')
@click.pass_context
def cli(ctx):
    """
    Scintillation analysis suite for narrowband SETI
    """
    pass 


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


cli.add_command(analysis.dedoppler)
cli.add_command(analysis.diagstat)
cli.add_command(montecarlo)
cli.add_command(simulations.synthesize_dataset)
# cli.add_command(bl_obs.observability)

if __name__ == '__main__':
    cli()