from pathlib import Path
import pandas as pd

from blscint import analysis


class DSFile():
    def __init__(self, filename):
        self.path = Path(filename)

        try:
            stem_parts = self.path.name.split(".")[0].split("_")

            self.node = stem_parts[0]
            self.timestamp = "_".join(stem_parts[2:4])
            self.target = "_".join(stem_parts[4:-1])
            self.scan = stem_parts[-1]
        
            d, s = self.timestamp.split("_")
            self.unix = (int(d) - 40587) * 86400 + int(s)
            self.mjd = int(d) + int(s) / 86400
        except IndexError:
            self.node = None
            self.timestamp = None
            self.target = None
            self.scan = None
            self.unix = None
            self.mjd = None


class DSPointing():
    def __init__(self, dsfiles=[], excluded_nodes=[], order_label=None, session_idx=None):
        self.hits_df = pd.concat([pd.read_csv(dsf.path) for dsf in dsfiles], ignore_index=True)

        self.excluded_nodes = excluded_nodes 
        for i, node in enumerate(self.excluded_nodes):
            if "blc" not in node:
                self.excluded_nodes[i] = f"blc{int(node):02d}"
            
        self.hits_df = self.hits_df[~self.hits_df["node"].isin(self.excluded_nodes)]
        self.nodes = sorted(set(self.hits_df["node"]))

        racks = sorted(set([node[3] for node in self.nodes]))
        all_potential_nodes = [f"blc{r}{n}" for r in racks for n in range(8)]
        self.missing_nodes = sorted(set(all_potential_nodes) - set(self.nodes + self.excluded_nodes))
        
        self.order_label = order_label 
        self.session_idx = session_idx 

        self.unix = dsfiles[0].unix 
        self.mjd = dsfiles[0].mjd
        self.data_fn_template = dsfiles[0].path

    def get_data_fn(self, node):
        path = self.data_fn_template
        return path.parent / '_'.join([node] + path.name.split('_')[1:])


class DSCadence():
    def __init__(self, dspointings=[]):
        """
        dspointings is a list of DSPointing objects, intended to manage 
        cadences of diagstat files.
        """
        self.pointings = dspointings 
        self.order = "".join(p.order_label for p in self.pointings)
        self.session_idx = self.pointings[0].session_idx

    def find_event(self):
        pass 