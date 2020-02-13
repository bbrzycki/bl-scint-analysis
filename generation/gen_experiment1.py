import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import time
import csv

import blimpy as bl

import sys, os, glob
sys.path.insert(0, "../../setigen")
import setigen as stg

start_time = time.time()



obs_fn = '/datax/scratch/bbrzycki/old/data/blc00_guppi_58331_12383_DIAG_SGR_B2_0014.gpuspec.0000.fil'

output_dir = '/datax/scratch/bbrzycki/scint_data/experiment1/data/'

labels_fn = output_dir + 'labels.csv'




fchans = 256

# Iterator for getting fil data from observations
fil_iter = stg.split_fil_generator(obs_fn, fchans)




# Generation loop
fieldnames = [
    'filename', 
    'snr',
    'tscint',
    'start_index',
    'end_index',
    'width'
]
with open(labels_fn, 'a') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate over every possible 
    for i, fil in enumerate(fil_iter):
        
        frame = stg.Frame(fil=fil)
        
        # Only inject signals in 50% of images
        if np.random.random() < 0.5:

            ##### SIGNAL PARAMETERS #####

            # Choose scintillation timescale
            tscint = np.random.normal(30, 10)
            while tscint < 20:
                tscint = np.random.normal(30, 10)

            # Choose FWHM of signal in freq space
            width = np.random.uniform(5, 30) * u.Hz

            # Choose brightness
            snr = np.random.uniform(10, 100)


            # Signal injection
            start_index = np.random.randint(frame.fchans)
            end_index = np.random.uniform(0, frame.fchans) 
            drift_rate = (end_index - start_index) * frame.df / (frame.dt * frame.tchans)

            p = 2
            rho = stg.get_rho(frame.ts, tscint, p)

            Z = stg.build_Z(rho, frame.tchans)
            Y = stg.get_Y(Z)

            signal = frame.add_signal(stg.constant_path(f_start=frame.fs[start_index], drift_rate=drift_rate),
                                      Y[:frame.tchans] * frame.compute_intensity(snr),
                                      stg.gaussian_f_profile(width=width),
                                      stg.constant_bp_profile(level=1))
        else:
            # If no signal, set everything to 0
            snr = 0
            tscint = 0
            start_index = 0
            end_index = 0
            width = 0
        
        
        # Save results out
        data_fn = output_dir + '{:06d}.npy'.format(i)
        frame.save_data(data_fn)
        
        logged_data = {
            'filename': data_fn,
            'snr': snr,
            'tscint': tscint,
            'start_index': start_index,
            'end_index': end_index,
            'width': width,
        }
        writer.writerow(logged_data)
        print(logged_data)
        
print('Generating {:d} images took {:.2f} seconds'.format(i, time.time() - start_time))