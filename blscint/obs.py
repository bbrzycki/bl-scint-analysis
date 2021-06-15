import os
import subprocess

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy import constants as const


def to_galactic(ra, dec=None):
    """
    Convert RA/Dec to galactic coordinates (l, b).
    
    Parameters
    ----------
    ra : str, float, or astropy.Quantity
        Right ascension as a string or float in degrees, or a full string 
        that includes both RA and Dec
    dec : str, float, or astropy.Quantity, optional
        Declination as a string or float in degrees
        
    Returns
    -------
    l, b : float
        Galactic coordinates
    """
    if dec is None:
        assert isinstance(ra, str)
        c = SkyCoord(ra, unit=(u.hourangle, u.deg))
    else:
        if isinstance(ra, str) and isinstance(dec, str):
            c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        elif type(ra) in [int, float] and type(dec) in [int, float]:
            c = SkyCoord(ra, dec, unit=(u.deg, u.deg))
        else:
            c = SkyCoord(ra, dec)
    gal = c.galactic
    return gal.l.value, gal.b.value

def query_ne2001(l, b, d, field=None):
    """
    Query NE2001 model for various parameters, as described in 
    Cordes & Lazio 2002.
    
    """
    current_path = os.path.abspath(os.path.dirname(__file__))
    exec_path = os.path.join(current_path, 'NE2001/bin.NE2001/run_NE2001.pl')
    
    cwd = os.getcwd()
    os.chdir(os.path.join(current_path, 'NE2001/bin.NE2001/'))
    
    if field is None:
        field = 'ALL'
    output = subprocess.run(['./run_NE2001.pl',
                             str(l),
                             str(b), 
                             str(d), 
                             '-1', 
                             field],
                            stdout=subprocess.PIPE).stdout.decode('utf-8')
    os.chdir(cwd)
    
    if field == 'ALL':
        print(output)
        return
             
    # Get unit
    unit = (output.split()[3].replace('pc-', 'pc.')
                             .replace('^{', '(')
                             .replace('}', ')'))
    unit = u.Unit(unit)
    val = float(output.split()[2])
    return val * unit

def get_standard_tscint(l, b, d):
    """
    Use NE2001 to estimate scintillation time at 1 GHz and 1 km/s transverse velocity.
    
    Parameters
    ----------
    l : float
        Galactic longitude
    b : float
        Galactic latitude
    d : float
        Distance in kpc
        
    Returns
    -------
    t_d : float
        Scintillation timescale in s
    """
    return query_ne2001(l, b, d, field='SCINTIME')

def scale_tscint(t_d, f=1, v=100, regime='moderate'):
    """
    Scale scintillation time by frequency and effective transverse velocity of 
    the diffraction pattern with respect to the observer. Changes exponential 
    scaling based on scattering regime, which is 'moderate' by default, or 
    'very_strong' (as in Cordes & Lazio 1991, Section 4.3).
    
    Parameters
    ----------
    t_d : float
        Scintillation time (s) at 1 GHz and 100 km/s
    f : float
        Frequency in GHz
    v : float
        Transverse velocity in km/s
    regime : str
        String determining frequency scaling, can be 'moderate' or 'very_strong'
        
    Returns
    -------
    t_d : float
        Scintillation timescale in s
    """
    if regime == 'very_strong':
        f_exp = 1
    else:
        f_exp = 1.2
    return t_d * (f / 1)**(f_exp) * (v / 100)**(-1)

def get_tscint(l, b, d, f=1, v=100, regime='moderate'):
    """
    Use NE2001 to estimate scintillation time at a specified frequency and 
    effective transverse velocity of the diffraction pattern with respect to
    the observer. Changes exponential scaling based on scattering regime, which
    is 'moderate' by default, or 'very_strong' (as in Cordes & Lazio 1991, Section
    4.3).
    
    Parameters
    ----------
    l : float
        Galactic longitude
    b : float
        Galactic latitude
    d : float
        Distance in kpc
    f : float
        Frequency in GHz
    v : float
        Transverse velocity in km/s
    regime : str
        String determining frequency scaling, can be 'moderate' or 'very_strong'
        
    Returns
    -------
    t_d : float
        Scintillation timescale in s
    """
    t_st = get_standard_tscint(l, b, d)
    return scale_tscint(t_st, f, v, regime)
    

def get_fresnel(f, D, normalize=True):
    """
    Get Fresnel scale. If normalize=True, use definition with 1/2pi in the sqrt.
    
    Parameters
    ----------
    f : float
        Frequency in GHz
    D : float
        Distance in kpc
    normalize : bool
        Whether to scale by sqrt(1/2pi)
    """
    
    wl = const.c / (f * u.GHz)
    l_f = np.sqrt(wl * (D * u.kpc)).to(u.cm)
    if normalize:
        l_f = np.sqrt(l_f / (2 * np.pi))
    return l_f