from pathlib import Path

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
        except IndexError:
            self.node = None
            self.timestamp = None
            self.target = None
            self.scan = None


class DSPointing():
    def __init__(self, dsfiles=[], excluded_nodes=[], order_label=None, session_idx=None):
        self.hits_df = pd.concat([pd.read_csv(dsf.path) for dsf in dsfiles], ignore_index=True)

        self.excluded_nodes = excluded_nodes 
        self.hits_df = self.hits_df[self.hits_df["node"].isin(self.excluded_nodes)]
        self.nodes = sorted(set(self.hits_df["node"]))

        racks = sorted(set([node[3] for node in self.nodes]))
        all_potential_nodes = [f"blc{r}{n}" for r in racks for n in range(8)]
        self.missing_nodes = sorted(set(all_potential_nodes) - set(self.nodes + self.excluded_nodes))
        
        self.order_label = order_label 
        self.session_idx = session_idx 


class DSCadence():
    def __init__(self, dspointings=[]):
        """
        dspointings is a list of DSPointing objects, intended to manage 
        cadences of diagstat files.
        """
        self.pointings = dspointings 
        self.order = "".join(p.order_label for p in self.pointings)
        self.session_idx = self.pointings[0].session_idx