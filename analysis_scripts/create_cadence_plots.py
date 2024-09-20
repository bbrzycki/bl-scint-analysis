import matplotlib.pyplot as plt

import socket
import collections
from pathlib import Path
import tqdm 
from fabric import Connection
from ast import literal_eval

import pandas as pd
import numpy as np
import setigen as stg
import blscint as bls
from blscint.observations.organization import DSFile, DSPointing, DSCadence
from blscint.remote import dsc_cadence 

from context import DATA_DIR, DB_PATH, DIAGSTAT_DIR, EVENT_PLOT_DIR, EVENT_PLOT_DIR_BY_NODE

MODEL_FRAME = stg.Frame.from_backend_params(fchans=256, 
                                            obs_length=600,
                                            int_factor=7)


def plot_event(event):
    first_hit = event.iloc[0]
    cadence = bls.centered_cadence(data_fns=literal_eval(first_hit['cadence_data_fns']),
                                   pointing_idx=first_hit['pointing_idx'],
                                   center_freq=first_hit['Uncorrected_Frequency'],
                                   drift_rate=first_hit['DriftRate'],
                                   fchans=first_hit['fchans'],
                                   frame_metadata=MODEL_FRAME.get_params(),
                                   order='ABAB')
    
    fig = plt.figure(figsize=(8, 6))
    cadence.plot(slew_times=True)
    ax = plt.gcf().add_axes([1.05, 0, .5, 1])
    ax.set_axis_off()

    print(event)
    print(event[['TopHitNum', 'DriftRate', 'SNR', 'Uncorrected_Frequency', 
                    'ChanIndx', 'std', 'min', 'ks', 'anderson']])
    print(str(event[['TopHitNum', 'DriftRate', 'SNR', 'Uncorrected_Frequency', 
                    'ChanIndx', 'std', 'min', 'ks', 'anderson']]))
    
    plt.text(0.0, 
             0.8, 
             str(event[['TopHitNum', 'DriftRate', 'SNR', 'Uncorrected_Frequency', 
                    'ChanIndx']]), 
             horizontalalignment='left',
             verticalalignment='center', 
             transform=plt.gca().transAxes)
    plt.text(0.0, 
             0.65, 
             str(event[['std', 'min', 'ks', 'anderson']]), 
             horizontalalignment='left',
             verticalalignment='center', 
             transform=plt.gca().transAxes)
    plt.text(0.0, 
             0.5, 
             str(event[['lag1', 'lag2', 'fit_t_d', 
                    'fit_A', 'fit_W', 'fchans', ]]), 
             horizontalalignment='left',
             verticalalignment='center', 
             transform=plt.gca().transAxes)
    plt.text(0.0, 
             0.35, 
             str(event[['l', 'r', 'node',
                    'found_with', 'event_idx', 'target_event_idx']]), 
             horizontalalignment='left',
             verticalalignment='center', 
             transform=plt.gca().transAxes)
    plt.text(0.0, 
             0.2, 
             str(event[['data_fn', 'pointing_idx']]), 
             horizontalalignment='left',
             verticalalignment='center', 
             transform=plt.gca().transAxes)
    
    plt.savefig(EVENT_PLOT_DIR_BY_NODE / f"event_{first_hit['target_event_idx']}.pdf", bbox_inches='tight')


def plot_all_events():
    EVENT_PLOT_DIR_BY_NODE.mkdir(parents=True, exist_ok=True)
    for target_event_idx, event in tqdm.tqdm(pd.read_csv(EVENT_PLOT_DIR / f'events_{socket.gethostname()}.csv').groupby('target_event_idx')):
        plot_event(event)


def partition_cadence_by_nodes(cadence_list):
    events_by_node = {} 
    for dscadence in tqdm.tqdm(cadence_list):
        for node, events in dscadence.events.groupby('node'):
            # print(node)
            data_fns = [str(p.get_data_fn(node)) for p in dscadence.pointings]
            # print(data_fns)

            events['cadence_data_fns'] = [data_fns] * len(events)
            events['pointing_idx'] = events['data_fn'].apply(lambda fn: data_fns.index(fn))
            # print(events.shape)
            if node in events_by_node:
                events_by_node[node] = pd.concat([events_by_node[node], events])
            else:
                events_by_node[node] = events  
    return events_by_node