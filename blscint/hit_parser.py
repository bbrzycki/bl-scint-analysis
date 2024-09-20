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

        frame = frame_processing.centered_frame(data_fn,
                                                center_freq,
                                                drift_rate,
                                                fchans,
                                                frame_metadata=self.frame_metadata)
        frame.add_metadata(dict(idx=idx))
        return frame

            