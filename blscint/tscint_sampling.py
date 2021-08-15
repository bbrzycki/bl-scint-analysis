import numpy as np
import tqdm

from . import ne2001

def mc_sample(l, b, d=(1, 20), f=(4, 8), v=(5, 50), n=100, regime='moderate'):
    """
    Monte Carlo sampling of scintillation timescales. d, f, v can be single values or a tuple range.
    
    Uses uniform distributions. n is number of samples.
    """
    try:
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
    
    t_ds = np.empty(n)
    for i in tqdm.tqdm(range(n)):
        t_ds[i] = ne2001.get_tscint(l, b, d[i], f[i], v[i], regime=regime).value
    return t_ds