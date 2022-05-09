import numpy as np
import tqdm
import matplotlib.pyplot as plt
from astropy import units as u
import astropy.coordinates as coord
coord.galactocentric_frame_defaults.set('v4.0') 

import scipy.stats

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
              # galaxy_rotation=False,
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
    # if galaxy_rotation:
    #     for i in range(len(d)):
    #         cs = coord.SkyCoord(l=l*u.deg, b=b*u.deg,
    #                             distance=d[i],
    #                            frame='galactic')
    #         gcs = cs.transform_to(coord.Galactocentric(galcen_distance=8.21*u.kpc)) 
    #         gcs.representation_type = 'cylindrical'
    #         R = gcs.rho.to(u.kpc).value
    #         v_circ = np.sum([p.vcirc(R*u.kpc)**2 for p in McMillan17])**0.5
    
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
    

class NESampler(object):
    """
    Class for sampling scintillation estimates from NE2001 electron density model.
    """
    def __init__(self, l, b, 
                 d=(1e-3, 20),
                 n=1000, 
                 d_sampling_type='density',
                 d_steps=1000):
        self.l, self.b = l, b
        self.d = d
        self.n = n
        
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
            
        self.base_t_ds = np.empty(n)
        self.base_nu_ds = np.empty(n)
        for i in tqdm.tqdm(range(n)):
            self.base_t_ds[i] = ne2001.query_ne2001(l, b, d[i], field='SCINTIME').value
            self.base_nu_ds[i] = ne2001.query_ne2001(l, b, d[i], field='SBW').to(u.Hz).value
        self.base_data = SampledNEData(t_d=self.base_t_ds, nu_d=self.base_nu_ds)
            
    def tau_d(self, nu_d, C1=1.16):
        """
        Temporal broadening.
        C1 is 1.53 for a medium with a square-law structure function. NE2001 paper.
        """
        return C1 / (2 * np.pi * nu_d)
    
    def nu_sb(self, t_d):
        """
        Spectral broadening.
        """
        C2 = 2.02
        return C2 / (2 * np.pi * t_d)
    
    def sample(self, f=(4, 8), v=(5, 100), scint_regime='moderate'):
        """
        Sample frequencies, transverse velocities, and scale base model values appropriately.
        """
        try:
            f = np.random.uniform(f[0], f[1], self.n)
        except TypeError:
            f = np.repeat(f, self.n)

        try:
            v = np.random.uniform(v[0], v[1], self.n)
        except TypeError:
            v = np.repeat(v, self.n)
            
        # Scintillation timescale scaling    
        if scint_regime == 'very_strong':
            # According to Cordes & Lazio 1991, there should be an inner scale l1 scaling here as well.
            t_ds = self.base_data.t_d * (f / 1)**1 * (np.abs(v) / 100)**(-1)
            nu_ds = self.base_data.nu_d * (f / 1)**4
        else:
            t_ds = self.base_data.t_d * (f / 1)**1.2 * (np.abs(v) / 100)**(-1)
            nu_ds = self.base_data.nu_d * (f / 1)**4.4
        
        # return {
        #     't_d': t_ds,
        #     'tau_d': self.tau_d(nu_ds),
        #     'nu_d': nu_ds,
        #     'nu_sb': self.nu_sb(t_ds),
        # }
        return SampledNEData(t_d=t_ds, nu_d=nu_ds)
    
    
class SampledNEData(object):
    def __init__(self, **kwargs):
        self.t_d = kwargs['t_d']
        self.nu_d = kwargs['nu_d']
        
        self.labels = {
            't_d': 'Scintillation Timescale',
            'nu_d': 'Scintillation Bandwidth',
            'tau_d': 'Temporal Broadening',
            'nu_sb': 'Spectral Broadening',
        }
        self.units = {
            't_d': 's',
            'nu_d': 'Hz',
            'tau_d': 's',
            'nu_sb': 'Hz',
        }
            
    @property
    def tau_d(self):
        return 1.16 / (2 * np.pi * self.nu_d)
                
    @property
    def nu_sb(self):
        return 2.02 / (2 * np.pi * self.t_d)
    
    @property
    def data_dict(self):
        return {
            't_d': self.t_d,
            'nu_d': self.nu_d,
            'tau_d': self.tau_d,
            'nu_sb': self.nu_sb,
        }
        
    def __add__(self, other):
        t_d = np.concatenate(self.t_d, other.t_d)
        nu_d = np.concatenate(self.nu_d, other.nu_d)
        return SampledNEParams(t_d=t_d, nu_d=nu_d)
    
    def report(self):
        for quantity in self.data_dict:
            print(self.labels[quantity])
            data = self.data_dict[quantity]
            print(f"Median: {np.median(data):.3}")
            print(f"MAD: {median_absolute_deviation(data):.3}")
            print(f"Std: {np.std(data):.3}")
            print(f'Coverage: {coverage(data, start=np.median(data)-median_absolute_deviation(data), stop=np.median(data)+median_absolute_deviation(data)):.3}')
            
            print(f"IQ: {np.quantile(data, 0.25):.3} to {np.quantile(data, 0.75):.3}")
            print(f"IQR: {scipy.stats.iqr(data):.3}")
            print("-"*10)
            
    def plot(self):
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        for quantity in self.data_dict:
            if quantity == 't_d':
                plt.sca(axs[0, 0])
            if quantity == 'nu_sb':
                plt.sca(axs[0, 1])
            if quantity == 'nu_d':
                plt.sca(axs[1, 0])
            if quantity == 'tau_d':
                plt.sca(axs[1, 1])
            data = self.data_dict[quantity]
            clipped_data = sigma_clip(data, maxiters=10, masked=False)
            _, bins = np.histogram(clipped_data, bins=25)
            plt.hist(data, bins=bins)
            plt.xlabel(f'{self.labels[quantity]} ({self.units[quantity]})')
            plt.ylabel('Counts')
            
            plt.axvline(np.median(data), ls='-', c='k')
            plt.axvline(np.quantile(data, 0.25), ls='--', c='k')
            plt.axvline(np.quantile(data, 0.75), ls='--', c='k')
            plt.axvline(np.median(data)-median_absolute_deviation(data), ls=':', c='b')
            plt.axvline(np.median(data)+median_absolute_deviation(data), ls=':', c='b')
            plt.title(f'{np.median(data):.3} \u00B1 {median_absolute_deviation(data):.3} {self.units[quantity]} (std: {np.std(sigma_clip(sigma_clip(data, masked=False), masked=False)):.3})')
        plt.tight_layout()