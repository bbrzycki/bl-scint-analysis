import collections
from pathlib import Path
import pandas as pd
import numpy as np
import setigen as stg

from blscint import analysis
from blscint.frame_processing import centered_cadence


class DSFile():
    def __init__(self, filename):
        self.path = Path(filename)

        try:
            stem_parts = self.path.name.split('.')[0].split('_')

            self.node = stem_parts[0]
            self.timestamp = '_'.join(stem_parts[2:4])
            self.target = '_'.join(stem_parts[4:-1])
            self.scan = stem_parts[-1]
        
            d, s = self.timestamp.split('_')
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
        temp_hits = [pd.read_csv(dsf.path) for dsf in dsfiles]
        self.hits = pd.concat([hits for hits in temp_hits if len(hits) > 0], ignore_index=True)

        # Exclude overlapping nodes, as specified
        self.excluded_nodes = excluded_nodes 
        for i, node in enumerate(self.excluded_nodes):
            if 'blc' not in node:
                self.excluded_nodes[i] = f'blc{int(node):02d}'
            
        self.hits = self.hits[~self.hits['node'].isin(self.excluded_nodes)]

        # Exclude DC bin peaks that slipped through deDoppler
        # mask_dc = np.abs(self.hits['ChanIndx'] - 524288) <= 1
        # self.hits_dc = self.hits[mask_dc]
        # self.hits = self.hits[~mask_dc]

        self.nodes = sorted(set(self.hits['node']))

        racks = sorted(set([node[3] for node in self.nodes]))
        all_potential_nodes = [f'blc{r}{n}' for r in racks for n in range(8)]
        self.missing_nodes = sorted(set(all_potential_nodes) - set(self.nodes + self.excluded_nodes))
        
        self.order_label = order_label 
        self.session_idx = session_idx 

        self.unix = dsfiles[0].unix 
        self.mjd = dsfiles[0].mjd
        self.target = dsfiles[0].target 
        self.scan = dsfiles[0].scan
        
        self.ds_fn_template = dsfiles[0].path
        self.data_fn_template = Path(self.hits.iloc[0]['data_fn'])
    
    def get_ds_fn(self, node):
        path = self.ds_fn_template
        return path.parent / '_'.join([node] + path.name.split('_')[1:])

    def get_data_fn(self, node):
        path = self.data_fn_template
        return path.parent / '_'.join([node] + path.name.split('_')[1:])

    def __str__(self):
        text = f'{self.__class__.__name__} \'{self.target}\''
    
        if self.order_label is not None:
            text += f' [{self.order_label}]'
        return f'<{text}>'

    def centered_frame(self):
        pass


class DSCadence(collections.abc.MutableSequence):
    def __init__(self, dspointings=[]):
        '''
        dspointings is a list of DSPointing objects, intended to manage 
        cadences of diagstat files.
        '''
        self.pointings = dspointings 
        self.session_idx = self.pointings[0].session_idx

    @property 
    def unique_targets(self):
        # targets = [p.target for p in self.pointings]
        # return list(set(targets))

        # Preserve order
        targets = []
        for p in self.pointings:
            if p.target not in targets:
                targets.append(p.target)
        return targets

    @property 
    def order(self):
        return ''.join(p.order_label for p in self.pointings)

    def __len__(self): 
        return len(self.pointings)

    def __getitem__(self, i): 
        if isinstance(i, slice):
            return self.__class__(self.pointings[i])
        elif isinstance(i, (list, np.ndarray, tuple)):
            return self.__class__(np.array(self.pointings)[i])
        else:
            return self.pointings[i]

    def __delitem__(self, i): 
        del self.pointings[i]

    def __setitem__(self, i, v):
        self.pointings[i] = v

    def insert(self, i, v):
        self.pointings.insert(i, v)

    def __str__(self):
        text = f'{self.__class__.__name__} {self.unique_targets} [{self.order}]'
        return f'<{text}>'

    def on_off_split(self, on_label=None):
        """
        Return pair of lists of ON and OFF pointings, based on the ON label.
        """
        if on_label is None:
            on_label = self.pointings[0].order_label
        on_pointings = [p for p in self.pointings 
                        if p.order_label == on_label]
        off_pointings = [p for p in self.pointings 
                         if p.order_label != on_label]
        return self.__class__(on_pointings), self.__class__(off_pointings)