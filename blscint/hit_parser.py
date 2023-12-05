from pathlib import Path
import pandas as pd 
import setigen as stg

from . import frame_processing


class HitParser(object):
    """
    Class to load hits from TurboSETI output and interact with data.
    """
    def __init__(self, *args):
        dfs = []

        dat_columns = [
            "TopHitNum",
            "DriftRate",
            "SNR",
            "Uncorrected_Frequency",
            "Corrected_Frequency",
            "ChanIndx",
            "FreqStart",
            "FreqEnd",
            "SEFD", 
            "SEFD_freq",
            "CoarseChanNum",
            "FullNumHitsInRange"
        ]
        for hits_fn in args:
            hits_path = Path(hits_fn)
            if hits_path.suffix == ".dat":
                df = pd.read_csv(hits_path, 
                                 sep='\t',
                                 names=dat_columns, 
                                 comment='#',
                                 index_col=False)
            else:
                df = pd.read_csv(hits_path,
                                 index_col=False)
            dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True)

        # Populates once you attempt to access observational data
        self.frame_metadata = None

    def centered_frame(self, idx, fchans=256, data_fn=None):
        """
        Create setigen frame centered around a given signal hit, 
        specified by its index in the dataframe. 
        """
        signal_info = self.df.loc[idx]
        drift_rate = signal_info["DriftRate"]
        center_freq = signal_info["Uncorrected_Frequency"]
        if data_fn is None:
            data_fn = signal_info["data_fn"]
        if self.frame_metadata is None:
            # Make the assumption that all file metadata is consistent
            self.frame_metadata = frame_processing.get_metadata(data_fn)

        tchans = self.frame_metadata["tchans"]
        df = self.frame_metadata["df"]
        dt = self.frame_metadata["dt"]

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

            