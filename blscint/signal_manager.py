import click
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import setigen as stg 

from . import frame_processing


class SignalManager(object):
    """
    Class to manage real and synthetic signal datasets.
    """
    def __init__(self):
        self.df = None

    def add_real(self, csv_file, label=None):
        real_df = pd.read_csv(csv_file)
        real_df['real'] = True
        if label is not None:
            real_df['label'] = label

        # Exclude DC bin (value depends on rawspec fftlength)
        # print('Before DC bins (may be excluded by TurboSETI):', data_df.shape)
        real_df = real_df[real_df['ChanIndx'] != 524288]
        # print('After removing:', data_df.shape)
        
        # # Exclude first compute node
        # data_df = data_df[data_df['fn'].apply(lambda x: x.split('/')[-1][3:5] != '00')]
        
        # Remove non-fit signals (which are replaced with NaN)
        real_df = real_df[real_df['ks'].notna()]

        if self.df is None:
            self.df = real_df
        else:
            self.df = pd.concat([self.df, real_df], ignore_index=True)

    def add_synthetic(self, csv_file, label=None):
        synth_df = pd.read_csv(csv_file)
        synth_df['real'] = False
        if label is not None:
            synth_df['label'] = label
        if self.df is None:
            self.df = synth_df
        else:
            self.df = pd.concat([self.df, synth_df], ignore_index=True)

    def plot_histograms(self, 
                        filters=[], 
                        statistics=['std', 'min', 'ks', 'fit_t_d'], 
                        titles=['Standard Deviation', 
                                'Minimum', 
                                'Kolmogorov-Smirnoff Statistic', 
                                'Scintillation Timescale Fit (s)'],
                        statistic_bounds=None,
                        # legend_loc=[1, 2, 1, 1],
                        **kwargs):
        """
        Plot histograms of signal statistics.
        """
        fig, axs = plt.subplots(1, 
                                len(statistics), 
                                figsize=(5 * len(statistics), 4), 
                                sharey='col')

        for j, stat in enumerate(statistics):
            if j == 0:
                axs[j].set_ylabel('Counts')
            
            all_vals = np.hstack([filter['df'][stat] for filter in filters if stat in filter['df']])
            all_vals = all_vals[~np.isnan(all_vals)]
            # all_vals = all_vals[np.isfinite(all_vals)]
            if statistic_bounds is not None:
                if statistic_bounds[j] is not None:
                    all_vals = all_vals[(all_vals > statistic_bounds[j][0])
                                        & (all_vals < statistic_bounds[j][1])]
                    # axs[j].set_xlim(statistic_bounds[j])

            bins=np.histogram(all_vals, bins=kwargs.get('bins', 40))[1]

            for i, filter in enumerate(filters):
                try:
                    axs[j].hist(filter['df'][stat], 
                                bins=bins, 
                                histtype='step', 
                                label=filter['label'],
                                color=filter.get('color'),
                                linewidth=filter.get('linewidth'),
                                linestyle=filter.get('linestyle'))
                except ValueError:
                    pass
                
            axs[j].xaxis.set_tick_params(labelbottom=True)
            axs[j].set_xlabel(titles[j])
            
            if 'legend_loc' in kwargs:
                axs[j].legend(loc=kwargs['legend_loc'][j])
            else:
                axs[j].legend()

    def plot_diagstat_histograms(self, t_ds=[], rfi_labels=[], data_labels=[], **kwargs):
        """
        Convenience method for plotting most common diagnostic statistic
        histograms. 
        """
        filters = []

        # Add synthetic signals
        for t_d in t_ds:
            filters.append({
                'label': f'{t_d} s',
                'df': self.df[self.df['t_d'] == t_d]
            })

        # Add rfi and data signals
        if len(rfi_labels) > 0:
            regex = '|'.join(rfi_labels)
            filters.append({
                'label': 'RFI',
                'df': self.df[self.df['data_fn'].str.contains(regex, na=False)],
                'color': 'k'
            })

        if len(data_labels) > 0:
            regex = '|'.join(data_labels)
            filters.append({
                'label': 'Data',
                'df': self.df[self.df['data_fn'].str.contains(regex, na=False)],
                'color': 'k',
                'linewidth': 2
            })

        self.plot_histograms(filters=filters, **kwargs)
    
    def centered_frame(self, idx, fchans=None, frame_metadata=None):
        """
        Create setigen frame centered around a given signal hit, 
        specified by its index in the dataframe. 
        """
        signal_info = self.df.loc[idx]
        drift_rate = signal_info["DriftRate"]
        center_freq = signal_info["Uncorrected_Frequency"]
        data_fn = signal_info["data_fn"]
        if fchans is None:
            fchans = signal_info["fchans"]
        if frame_metadata is None:
            # Make the assumption that all file metadata is consistent
            frame_metadata = frame_processing.get_metadata(data_fn)

        tchans = frame_metadata["tchans"]
        df = frame_metadata["df"]
        dt = frame_metadata["dt"]

        adj_center_freq = center_freq + drift_rate / 1e6 * tchans / 2
        max_offset = int(abs(drift_rate) * tchans * dt / df)
        if drift_rate >= 0:
            adj_fchans = [0, max_offset]
        else:
            adj_fchans = [max_offset, 0]
        
        f_start = adj_center_freq - (fchans / 2 + adj_fchans[0]) * df / 1e6
        f_stop = adj_center_freq + (fchans / 2 + adj_fchans[1]) * df / 1e6
        frame = stg.Frame(data_fn, f_start=f_start, f_stop=f_stop)
            
        frame.add_metadata({
            'drift_rate': drift_rate,
            'center_freq': center_freq,
            'idx': idx,
        })
        return frame

    def estimate_thresholds(self, *args, **kwargs):
        pass

    def overplot_signal(self, ts, *args, **kwargs):
        pass

    def save(self, filename):
        """
        Save DataSynthesizer as a pickled file (.pickle).
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """
        Load DataSynthesizer from a pickled file (.pickle).
        """
        with open(filename, 'rb') as f:
            frame = pickle.load(f)
        return frame