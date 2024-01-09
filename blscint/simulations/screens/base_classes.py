from abc import ABC, abstractmethod
import numpy as np 
import scipy.special
from astropy import units as u
from astropy.constants import a0, alpha
import setigen as stg
from tqdm import tqdm


def get_wavelength(f):
    return stg.cast_value(f, u.Hz).to(u.cm, equivalencies=u.spectral())


def get_k(f):
    return 2 * np.pi / get_wavelength(f)


def get_rF(distance, f):
    return (distance / (2 * np.pi / get_wavelength(f)))**0.5


class BaseRadioSource(ABC):
    @abstractmethod
    def emit(self):
        return
    

class BasePhaseSpectrum(ABC):
    @abstractmethod
    def Phi(self, q, f):
        return


class BaseScreen(ABC):
    @abstractmethod
    def phases(self, f):
        return
    
    @abstractmethod
    def propagate_phase_screen(self, E, f):
        return
    
    @abstractmethod
    def propagate_free_space(self, E, z, f):
        return
    

class BaseScatteringModel(ABC):
    @abstractmethod
    def propagate_free_space(self, E, z, f):
        return
    
    @abstractmethod
    def observer_electric_field(self, f):
        return
    
    @abstractmethod
    def observer_dynamic_spectrum(self, fmin, df, fchans):
        return
    

