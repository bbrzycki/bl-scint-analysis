"""
Multiplicative factors to take a standard deviation (sigma) value
to various measures such as full width at half maximum (FWHM).
"""
import numpy as np

# Half max
hwhm_f = hwhm_m = np.sqrt(2 * np.log(2))
fwhm_f = fwhm_m = 2 * hwhm_m

# 1/e max
hwem_f = hwem_m = np.sqrt(2)
fwem_f = fwem_m = 2 * hwem_m
