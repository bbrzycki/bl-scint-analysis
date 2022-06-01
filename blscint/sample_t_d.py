import numpy as np
import tqdm
import matplotlib.pyplot as plt
from astropy import units as u
import astropy.coordinates as coord
coord.galactocentric_frame_defaults.set('v4.0') 

try:
    import cPickle as pickle
except:
    import pickle
    
import scipy.stats

from astropy.stats import sigma_clip
from scipy.stats import median_absolute_deviation
from galpy.potential.mwpotentials import McMillan17

from . import ne2001


# Stellar density model from McMillan 2017, 'The mass distribution and gravitational potential of 
# the Milky Way'. All units in M_sun, kpc. 
def mcmillan_rho_bulge(R, z):
    q = 0.5
    r0 = 0.075
    alpha = 1.8
    rcut = 2.1
    rprime = np.sqrt(R**2 + (z/q)**2)
    rho0b = 97.3e9
    return rho0b / (1 + rprime/r0)**alpha * np.exp(-(rprime/rcut)**2)

def mcmillan_rho_thin(R, z):
    S0 = 886.7e6
    zd = 0.300
    Rd = 2.53
    return S0/(2*zd) * np.exp(-np.abs(z)/zd - np.abs(R)/Rd)

def mcmillan_rho_thick(R, z):
    S0 = 156.7e6
    zd = 0.900
    Rd = 3.38
    return S0/(2*zd) * np.exp(-np.abs(z)/zd - np.abs(R)/Rd)

def mcmillan_rho_tot(R, z):
    return mcmillan_rho_bulge(R, z) + mcmillan_rho_thin(R, z) + mcmillan_rho_thick(R, z)


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
    
    
def transition_freqs(l, b, d=(1e-3, 20), d_steps=1000):
    """
    Get transition from strong to weak scattering.
    """
    freqs = np.empty(d_steps)
    d = np.linspace(d[0], d[1], d_steps)
    for i in tqdm.tqdm(range(d_steps)):
        freqs[i] = ne2001.query_ne2001(l, b, d[i], field='NU_T').to(u.GHz).value
    return freqs


def min_d_ss(l, b, d=(1e-3, 20), f=(4, 8), delta_d=0.01):
    """
    Get min distince for strong scattering.
    """
    f_max = np.max(f)
    d = np.arange(d[0], d[1] + delta_d / 2, delta_d)
    freqs = np.empty(d.size)
    for i in range(d.size):
        if ne2001.query_ne2001(l, b, d[i], field='NU_T').to(u.GHz).value > f_max:
            return d[i] * u.kpc
    return None
    

class NESampler(object):
    """
    Class for sampling scintillation estimates from NE2001 electron density model.
    """
    def __init__(self, l, b, 
                 d=(0.01, 20),
                 delta_d=0.01,
                 galcen_distance=8.21):
        self.l, self.b = l, b
        self.delta_d = delta_d
        
        try:
            # Sample by density
            self.d = np.arange(d[0], d[1] + delta_d / 2, delta_d)
            cs = coord.SkyCoord(l=l*u.deg, b=b*u.deg,
                                distance=self.d*u.kpc,
                                frame='galactic')
            gcs = cs.transform_to(coord.Galactocentric(galcen_distance=galcen_distance*u.kpc)) 
            gcs.representation_type = 'cylindrical'
            self.d_rel_prob = mcmillan_rho_tot(gcs.rho.to(u.kpc).value, gcs.z.to(u.kpc).value)
            
            raw_t_ds = np.empty(self.d.size)
            raw_nu_ds = np.empty(self.d.size)
            for i in tqdm.tqdm(range(self.d.size)):
                raw_t_ds[i] = ne2001.query_ne2001(l, b, self.d[i], field='SCINTIME').value
                raw_nu_ds[i] = ne2001.query_ne2001(l, b, self.d[i], field='SBW').to(u.Hz).value
        except (TypeError, IndexError):
            self.d = d
            raw_t_ds = ne2001.query_ne2001(l, b, d, field='SCINTIME').value
            raw_nu_ds = ne2001.query_ne2001(l, b, d, field='SBW').to(u.Hz).value
            print(f"Transition Frequency is {ne2001.query_ne2001(l, b, d=d, field='NU_T')}")

        self.raw_data = NEData(t_d=raw_t_ds, nu_d=raw_nu_ds)
            
    def save_pickle(self, filename):
        """
        Save entire sampler (including NE2001 calculations) as a pickled file (.pickle).
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_pickle(cls, filename):
        """
        Load sampler object from a pickled file (.pickle), 
        created with NESampler.save_pickle.
        """
        with open(filename, 'rb') as f:
            sampler = pickle.load(f)
        return sampler
            
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
    
    def sample(self, n=1000, f=(4, 8), v=(5, 100), d=None, 
               d_sampling_type='density', scint_regime='moderate', verbose=True):
        """
        Sample frequencies, transverse velocities, and scale base model values appropriately.
        """
        min_d = min_d_ss(self.l, self.b, d=(1e-3, 20), f=f, delta_d=self.delta_d)
        if min_d is None:
            raise RuntimeError('Strong regime is never achieved along the line of sight!')
        else:
            min_d = min_d.to(u.kpc).value
            if verbose:
                print(f"Min distance for strong scattering is {min_d:.3} kpc")
        
        if isinstance(self.d, (int, float)):
            sampled_t_ds = np.repeat(self.raw_data.t_d, n)
            sampled_nu_ds = np.repeat(self.raw_data.nu_d, n)
            sampled_ds = np.repeat(self.d, n)
        else:
            d_idx = np.arange(self.d.size)
            # Enforce strong scattering regime, and optional distance cut
            if d is None:
                d_idx = d_idx[(self.d >= min_d)]
            else:
                d_idx = d_idx[(self.d >= np.max(d[0], min_d)) & (self.d <= d[1])]
                
            if d_sampling_type == 'density':
                # d = self.d[d_idx]
                # print(self.d_rel_prob.shape, d_idx.shape)
                d_prob = self.d_rel_prob[d_idx] / np.sum(self.d_rel_prob[d_idx])
                sampled_idx = np.random.choice(d_idx, size=n, p=d_prob)
            else:
                sampled_idx = np.random.choice(d_idx, size=n)
            sampled_t_ds = self.raw_data.t_d[sampled_idx]
            sampled_nu_ds = self.raw_data.nu_d[sampled_idx]
            sampled_ds = self.d[sampled_idx]
        
        try:
            f = np.random.uniform(f[0], f[1], n)
        except (TypeError, IndexError):
            f = np.repeat(f, n)

        try:
            v = np.random.uniform(v[0], v[1], n)
        except (TypeError, IndexError):
            v = np.repeat(v, n)
            
        # Scintillation timescale scaling    
        if scint_regime == 'very_strong':
            # According to Cordes & Lazio 1991, there should be an inner scale l1 scaling here as well.
            t_ds = sampled_t_ds * (f / 1)**1 * (np.abs(v) / 100)**(-1)
            nu_ds = sampled_nu_ds * (f / 1)**4
        else:
            t_ds = sampled_t_ds * (f / 1)**1.2 * (np.abs(v) / 100)**(-1)
            nu_ds = sampled_nu_ds * (f / 1)**4.4
        
        # return {
        #     't_d': t_ds,
        #     'tau_d': self.tau_d(nu_ds),
        #     'nu_d': nu_ds,
        #     'nu_sb': self.nu_sb(t_ds),
        # }
        return NEData(t_d=t_ds, nu_d=nu_ds)
    
    
class NEData(object):
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
    
    def mask(self, mask):
        return SampledNEParams(t_d=self.t_d[mask], nu_d=self.nu_d[mask])
    
    def report(self):
        for quantity in self.data_dict:
            print(f"{self.labels[quantity]}:")
            data = self.data_dict[quantity]
            clipped_data = sigma_clip(data, maxiters=10, masked=False)
            vals, bins = np.histogram(clipped_data, bins=25)
            mode_idx = np.argmax(vals)
            mode = (bins[mode_idx] + bins[mode_idx + 1]) / 2
            
            print(f"Median: {np.median(data):.3} {self.units[quantity]}")
            print(f"MAD: {median_absolute_deviation(data):.3} {self.units[quantity]}")
            print(f'Coverage: {coverage(data, start=np.median(data)-median_absolute_deviation(data), stop=np.median(data)+median_absolute_deviation(data)):.3}')
            print(f"Std: {np.std(data):.3} {self.units[quantity]}")
            print(f"Mode: {mode:.3} {self.units[quantity]}")
            
            print(f"IQ: {np.quantile(data, 0.25):.3} {self.units[quantity]} -- {np.quantile(data, 0.75):.3} {self.units[quantity]}")
            print("-"*10)
            
    def diagnostic_plot(self):
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
            vals, bins = np.histogram(clipped_data, bins=25)
            mode_idx = np.argmax(vals)
            mode = (bins[mode_idx] + bins[mode_idx + 1]) / 2
            plt.hist(data, bins=bins)
            plt.xlabel(f'{self.labels[quantity]} ({self.units[quantity]})')
            plt.ylabel('Counts')
            
            plt.axvline(np.median(data), ls='-', c='k')
            plt.axvline(np.quantile(data, 0.25), ls='--', c='k')
            plt.axvline(np.quantile(data, 0.75), ls='--', c='k')
            plt.axvline(np.median(data)-median_absolute_deviation(data), ls=':', c='b')
            plt.axvline(np.median(data)+median_absolute_deviation(data), ls=':', c='b')
            plt.axvline(mode, ls='-', c='g')
            plt.title(f'{np.median(data):.3}({np.quantile(data, 0.25):.3}, {np.quantile(data, 0.75):.3}) {self.units[quantity]} (mode: {mode:.3} {self.units[quantity]})')
        plt.tight_layout()
            
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
            vals, bins = np.histogram(clipped_data, bins=25)
            mode_idx = np.argmax(vals)
            mode = (bins[mode_idx] + bins[mode_idx + 1]) / 2
            
            plt.hist(data, bins=bins, histtype='step', color='k')
            plt.xlabel(f'{self.labels[quantity]} ({self.units[quantity]})')
            plt.ylabel('Counts')
            
            plt.axvline(np.median(data), ls='--', c='k')
            plt.axvline(np.quantile(data, 0.25), ls=':', c='k')
            plt.axvline(np.quantile(data, 0.75), ls=':', c='k')
            # plt.title(f'{np.median(data):.3}({np.quantile(data, 0.25):.3}, {np.quantile(data, 0.75):.3}) {self.units[quantity]} (mode: {mode:.3} {self.units[quantity]})')
        plt.tight_layout()