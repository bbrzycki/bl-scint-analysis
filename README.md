# blscint
#### Analysis code for identifying ISM scintillation in detected narrowband radio signals

This library provides methods for probing the presence of strong ISM scintillation in detected narrowband radio signals as a method of analyzing technosignatures. There are a few main sections: scintillation timescale estimation using NE2001 ([Cordes & Lazio 2002](https://arxiv.org/abs/astro-ph/0207156)), scintillated intensity time series synthesis, and statistical analysis of intensity time series (for characterizing local RFI and filtering technosignature candidates). All of these are described in detail in [Brzycki et al. 2023](https://iopscience.iop.org/article/10.3847/1538-4357/acdee0/meta) and Brzcyki et al. 2024 (submitted). 

### Scintillation timescale estimation

The `blscint.ne2001` modules use the original Fortran implementation of the NE2001 model to estimate quantities relevant in detecting ISM scintillation in narrowband technosignatures. The most basic function is `query_ne2001`, which simply runs the model to retrieve quantities at 1 GHz and transverse velocity 100 km/s. For scintillation searches, we care most about the scintillation timescale, which can be estimated as a function of detected frequency and transverse velocity using `get_t_d`. 

To help figure out which scintillation timescales are likely given a set of observational parameters, we can use Monte Carlo random sampling. We decide on a galaxy model for the number density of stars (Carroll & Ostlie, McMillan, or otherwise), and use that to simulate the distribution of plausible emitting civilizations. We provide a simple integration-based estimation of the number of stars in a given sky direction and beamwidth with the `count_stars` function. The `NESampler` class is used to do Monte Carlo sampling for a given sky direction (l, b) over each free parameter (frequency, transverse velocity, distance). 

### Synthetic scintillated intensity time series and likelihood estimation

To provide a point of reference, we provide functions that synthesize scintillated intensity time series. This can be done in multiple ways. For direct time series simulations, we can produce the expected theoretical statistics with an FFT-based approach, with `get_ts_fft`. We can also use an autoreggressive process described in [Cario & Nelson 1996](https://www.sciencedirect.com/science/article/pii/016763779600017X) with `get_ts_arta`. These have their pros and cons; the FFT approach is generally faster but has constraints in the size of the time series, but the ARTA approach can generate arbitrarily sized time series more reliably. 

The `SignalGenerator` class allows you to quickly produce a dataset of synthetic signals, and in particular produces a `.csv` file with summary statistics. This is useful for comparing summary statistics with those for signals detected within observations. 

We can estimation the likelihood that a detected signal is scintillated using `KDERanker`, which inherits from `BaseSyntheticDistRanker`. These classes use synthetic scintillated datasets (with specific scintillation timescales) produced with the above methods to calculate scintillation likelihoods. The base class lets you produce plots as in Brzycki et al. 2024, with methods such as `plot_ranking_vs_frequency` and `plot_diagstat_corner_plot`. These employ the ranking method `KDERanker.rank(hit)`.

### Timeseries analysis

Most of `blscint` is dedicated to analysis of detected signals for scintillated properties. We largely use [Setigen](https://github.com/bbrzycki/setigen) Frames and Cadences to facilitate analysis and data operations.

In `blscint.analysis`, we have command-line utilities wrapped with `Click` to run both the deDoppler analysis (to detect and report signal frequency and drift rates) and the diagnostic statistic analysis (to extract signal intensities and compute statistics relevant to ISM scintillation).

In `blscint.bounds`, we provide a set of many methods for localization a detected signal in frequency. This is important to tightly constrain narrowband signals so that we can extract as much signal power and as little noise power as possible. However, there isn't a singular way to do this, so depending on your goals or conventions, we provide a host of methods (fractional thresholds, SNR-based thresholds, etc.). 

In `blscint.frame_processing`, we have functions for the actual extraction of a normalized time series from data, given the location of a detection narrowband signal. This includes background noise normalization in `tnorm` (not quite removal, but taking it to a 0 mean 1 std baseline). We also provide methods for creating Setigen frames and cadences centered around each signal to simplify analysis. The function `extract_ts` combines the above functions and returns a normalized intensity time series for a narrowband signal of interest. 

In `blscint.diag_stats`, we compute the diagnostic statistics given a normalized intensity time series. This includes fitting a theoretical autocorrelation function to determine the best-fit scintillation timescale. Please note that since radio observations (especially those with high spectral resolution) typically do not have a high number of time bins, this best-fit is quite noisy and shouldn't be taken at face value as the correct timescale. Nevertheless, this timescale fit is still a diagnostic statistic that tells us something about the likelihood that a given signal is scintillated.

The class `SignalManager` allows you to combine the contents of diagstat csvs from both real signals (observations) and synthetic signal datasets, so that you can plot histograms of various quantities or statistics and compare between observations and theory. 

### Analysis of narrowband signals in radio observations

We provide a command-line utility `blscint` to run analysis routines from the command-line. For a dedicated search, this utility is called with various options to run deDoppler detection and diagnostic statistic extraction. 

For examples of usage, we include scripts used in Brzycki et al. 2024 (submitted to AJ) in `analysis_scripts/`. These include custom scripts for distributed computing over many compute nodes containing observational data. Specifically, `node_manager.py` is run from the command line on any node on the system (even a login node) using command line arguments, which specify the analysis routines to run and the set of compute nodes that should execute that analysis (on the data located on those compute nodes). Each compute node will accordingly execute `node_worker.py` with a given analysis step (subset of `dedoppler`, `diagstat`, and/or `scp`). The manager script also lets you kill all analysis-related processes on the compute nodes for any reason (e.g. node GPUs need to be used instead for reducing radio observations).

In `blscint.remote`, we provide methods to easily retrieve (and subsequently plot) observational cadences around detected signals even though the data is located on other machines. In particular, `dsc_cadence` uses a `DSCadence` object so that you can use indices to look up specific signals. 
