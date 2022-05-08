import numpy as np
import tqdm
import matplotlib.pyplot as plt
from astropy import units as u
import astropy.coordinates as coord
coord.galactocentric_frame_defaults.set('v4.0') 

from astropy.stats import sigma_clip
from scipy.stats import median_absolute_deviation
from galpy.potential.mwpotentials import McMillan17

from . import ne2001


# Stellar density model from McMillan 2017, 'The mass distribution and gravitational potential of 
# the Milky Way'. All units in M_sun, kpc. 
def density_bulge(R, z):
    q = 0.5
    r0 = 0.075
    alpha = 1.8
    rcut = 2.1
    rprime = np.sqrt(R**2 + (z/q)**2)
    rho0b = 97.3e9
    return rho0b / (1 + rprime/r0)**alpha * np.exp(-(rprime/rcut)**2)

def density_thin(R, z):
    S0 = 886.7e6
    zd = 0.300
    Rd = 2.53
    return S0/(2*zd) * np.exp(-np.abs(z)/zd - np.abs(R)/Rd)

def density_thick(R, z):
    S0 = 156.7e6
    zd = 0.900
    Rd = 3.38
    return S0/(2*zd) * np.exp(-np.abs(z)/zd - np.abs(R)/Rd)

def density_tot(R, z):
    return density_bulge(R, z) + density_thin(R, z) + density_thick(R, z)


def mc_sample(l, 
              b, 
              d=(1e-3, 20), 
              f=(4, 8), 
              v=(5, 100), 
              n=1000, 
              d_sampling_type='density',
              galaxy_rotation=False,
              d_steps=1000,
              scint_regime='moderate'):
    """
    Monte Carlo sampling of scintillation timescales. d, f, v can be single values or a tuple range.
    
    Uses uniform distributions for frequency and transverse velocity. n is number of samples.
    
    With parameter `d_sampling_type`, specify how distance is sampled: stellar 'density' 
    or 'uniform'.
    
    With parameter `galaxy_rotation`, specify if transverse velocity adds appropriate galactic
    rotation term.
    """
    try:
        if d_sampling_type == 'density':
            # Sample by density
            d = np.linspace(d[0], d[1], d_steps) * u.kpc
            cs = coord.SkyCoord(l=l*u.deg, b=b*u.deg,
                                distance=d,
                               frame='galactic')
            gcs = cs.transform_to(coord.Galactocentric(galcen_distance=8.21*u.kpc)) 
            gcs.representation_type = 'cylindrical'
            densities = density_tot(gcs.rho.to(u.kpc).value, gcs.z.to(u.kpc).value)
            
            norm_densities = densities / np.sum(densities)
            d = np.random.choice(d, p=norm_densities, size=n)
        else:
            d = np.random.uniform(d[0], d[1], n)
    except TypeError:
        d = np.repeat(d, n)
        
    try:
        f = np.random.uniform(f[0], f[1], n)
    except TypeError:
        f = np.repeat(f, n)
        
    try:
        v = np.random.uniform(v[0], v[1], n)
    except TypeError:
        v = np.repeat(v, n)
    if galaxy_rotation:
        for i in range(len(d)):
            cs = coord.SkyCoord(l=l*u.deg, b=b*u.deg,
                                distance=d[i],
                               frame='galactic')
            gcs = cs.transform_to(coord.Galactocentric(galcen_distance=8.21*u.kpc)) 
            gcs.representation_type = 'cylindrical'
            R = gcs.rho.to(u.kpc).value
            v_circ = np.sum([p.vcirc(R*u.kpc)**2 for p in McMillan17])**0.5
    
    t_ds = np.empty(n)
    for i in tqdm.tqdm(range(n)):
        t_ds[i] = ne2001.get_t_d(l, b, d[i], f[i], v[i], regime=scint_regime).value
    return t_ds


def coverage(t_ds, start=None, stop=None):
    """
    Get fractional coverage of scintillation timescales for a specified range.
    """
    if start is None:
        start = np.min(t_ds)
    if stop is None:
        stop = np.max(t_ds)
    return np.sum((t_ds >= start) & (t_ds <= stop)) / len(t_ds)


def central(t_ds, p):
    """
    Get central fraction p of array.
    """
    t_ds = np.sort(t_ds)
    cut = int(len(t_ds) * (1 - p) / 2)
    return t_ds[cut:-cut]


def visualize_mc(t_ds):
    print(f'without sigmaclip: Med,std,mad: {np.median(t_ds):.1f}+-{np.std(t_ds):.1f}: {median_absolute_deviation(t_ds):.1f}')
    print(f'coverage: {coverage(t_ds, start=np.median(t_ds)-median_absolute_deviation(t_ds), stop=np.median(t_ds)+median_absolute_deviation(t_ds))}')
    t_ds = sigma_clip(t_ds, masked=False)
    
    plt.hist(t_ds, bins=25)
    plt.xlabel('Scintillation timescale (s)')
    plt.ylabel('Counts')
    plt.title(f'Med,std,mad: {np.median(t_ds):.1f}+-{np.std(t_ds):.1f}: {median_absolute_deviation(t_ds):.1f}')
#     plt.legend()
    
    plt.tight_layout()