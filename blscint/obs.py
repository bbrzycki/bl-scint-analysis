import os
import subprocess

from astropy import units as u
from astropy.coordinates import SkyCoord


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


def get_tscint(l, b, d):
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
    current_path = os.path.abspath(os.path.dirname(__file__))
    exec_path = os.path.join(current_path, 'NE2001/bin.NE2001/run_NE2001.pl')
    print(exec_path)
    
    cwd = os.getcwd()
    os.chdir(os.path.join(current_path, 'NE2001/bin.NE2001/'))
    
    output = subprocess.run(['./run_NE2001.pl',
                             str(l),
                             str(b), 
                             str(d), 
                             '-1', 
                             'SCINTIME'],
                            stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(output)
    os.chdir(cwd)
             
    t_d = float(output.split()[2])
    return t_d