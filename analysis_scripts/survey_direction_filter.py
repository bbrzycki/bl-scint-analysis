import collections
try:
    import cPickle as pickle
except:
    import pickle
from pathlib import Path
import tqdm 

import pandas as pd
import numpy as np
import setigen as stg
import blscint as bls
from blscint.observations.organization import DSFile, DSPointing, DSCadence
from blscint.remote import dsc_cadence 

from context import DATA_DIR, DB_PATH, DIAGSTAT_DIR


nested_target_list = [
    [
        [
            ["DIAG_SCINT_GP_NGP"],
            ["DIAG_SCINT_GP_L5_B2", "DIAG_SCINT_GP_L5_B1"],
            ["DIAG_SCINT_GP_L5_B0", "DIAG_SCINT_GP_L5_B-1"],
            ["DIAG_SCINT_GP_L5_B-2", "DIAG_SCINT_GP_L4_B-2"],
            ["DIAG_SCINT_GP_L4_B-1", "DIAG_SCINT_GP_L4_B0"],
            ["DIAG_SCINT_GP_L4_B1", "DIAG_SCINT_GP_L4_B2"],
            ["DIAG_SCINT_GP_L3_B2", "DIAG_SCINT_GP_L3_B1"],
            ["DIAG_SCINT_GP_L3_B0", "DIAG_SCINT_GP_L3_B-1"]
        ],
        ["07", "40", "47", "50", "57", "60"]
    ],
    [
        [
            ["DIAG_SCINT_GP_NGP"],
            ["DIAG_SCINT_GP_L3_B-2", "DIAG_SCINT_GP_L2_B-2"],
            ["DIAG_SCINT_GP_L2_B-1", "DIAG_SCINT_GP_L2_B0"],
            ["DIAG_SCINT_GP_L2_B1", "DIAG_SCINT_GP_L2_B2"],
            ["DIAG_SCINT_GP_L1_B2", "DIAG_SCINT_GP_L1_B1"],
            ["DIAG_SCINT_GP_L1_B0", "DIAG_SCINT_GP_L1_B-1"],
            ["DIAG_SCINT_GP_L1_B-2", "DIAG_SCINT_GP_L0_B-2"],
            ["DIAG_SCINT_GP_L0_B-1", "DIAG_SCINT_GP_L0_B1"],
            ["DIAG_SCINT_GP_L0_B2", "DIAG_SCINT_GP_L-1_B2"],
            ["DIAG_SCINT_GP_L-1_B1", "DIAG_SCINT_GP_L-1_B0"]
        ],
        ["47", "50", "57", "60", "67", "70"]
    ],
    [
        [
            ["DIAG_SCINT_GP_NGP"],
            ["DIAG_SCINT_GP_L-1_B-1", "DIAG_SCINT_GP_L-1_B-2"],
            ["DIAG_SCINT_GP_L-2_B-2", "DIAG_SCINT_GP_L-2_B-1"],
            ["DIAG_SCINT_GP_L-2_B0", "DIAG_SCINT_GP_L-2_B1"],
            ["DIAG_SCINT_GP_L-2_B2", "DIAG_SCINT_GP_L-3_B2"],
            ["DIAG_SCINT_GP_L-3_B1", "DIAG_SCINT_GP_L-3_B0"],
            ["DIAG_SCINT_GP_L-3_B-1", "DIAG_SCINT_GP_L-3_B-2"],
            ["DIAG_SCINT_GP_L-4_B-2", "DIAG_SCINT_GP_L-4_B-1"],
            ["DIAG_SCINT_GP_L-4_B0", "DIAG_SCINT_GP_L-4_B1"]
        ],
        ["47", "50", "57", "60", "67", "70"]
    ],
    [
        [
            ["DIAG_SCINT_GP_NGP"],
            ["DIAG_SCINT_GP_L-4_B2", "DIAG_SCINT_GP_L-5_B2"],
            ["DIAG_SCINT_GP_L-5_B1", "DIAG_SCINT_GP_L-5_B0"],
            ["DIAG_SCINT_GP_L-5_B-1", "DIAG_SCINT_GP_L-5_B-2"],
            ["DIAG_SCINT_GC_A00", "DIAG_SCINT_GC_C01"],
            ["DIAG_SCINT_GC_C01", "DIAG_SCINT_GC_C07"],
            ["DIAG_SCINT_GC_B01", "DIAG_SCINT_GC_B04"],
            ["DIAG_SCINT_GC_B02", "DIAG_SCINT_GC_B05"],
            ["DIAG_SCINT_GC_B03", "DIAG_SCINT_GC_B06"],
            ["DIAG_SCINT_GC_C02", "DIAG_SCINT_GC_C04"],
            ["DIAG_SCINT_GC_C03", "DIAG_SCINT_GC_C05"]
        ],
        ["47", "50", "57", "60", "67", "70"]
    ],
    [
        [
            ["DIAG_SCINT_GC_NGP"],
            ["DIAG_SCINT_GC_C08", "DIAG_SCINT_GC_C06"],
            ["DIAG_SCINT_GC_C11", "DIAG_SCINT_GC_C09"],
            ["DIAG_SCINT_GC_C10", "DIAG_SCINT_GC_C12"]
        ],
        ["47", "50", "57", "60", "67", "70"]
    ]
]


def construct_cadence_list(nested_target_list):
    NGP_pointings = []
    cadence_list = []
    for session_idx, (target_list, excluded_nodes) in enumerate(nested_target_list):
        session_min_timestamp = 0

        for target_keywords in target_list:
            dspointings = [None] * (2 * len(target_keywords))
            ordering_list = []

            for target_idx, target in enumerate(target_keywords):
                # print(target)
                paths = bls.as_file_list(DIAGSTAT_DIR / f"*{target}*.diagstat.csv",
                                        excluded_nodes=excluded_nodes)
                timestamps = ["_".join([dsf.timestamp]) for fn in paths for dsf in [DSFile(fn)]]
                # print(sorted(set(timestamps)))

                if 'NGP' in target:
                    try:
                        timestamp = sorted(set(timestamps))[session_idx]
                    except IndexError:
                        timestamp = sorted(set(timestamps))[0]
                    session_min_timestamp = int(timestamp[:5])

                    dsfiles = [DSFile(path) 
                            for path in bls.as_file_list(DIAGSTAT_DIR / f"*{timestamp}_{target}*.diagstat.csv")]
                    NGP_pointings.append(DSPointing(dsfiles, 
                                                    excluded_nodes=excluded_nodes, 
                                                    session_idx=session_idx))
                    continue
                else:
                    valid_timestamps = [ts 
                                        for ts in timestamps 
                                        if int(ts[:5]) >= session_min_timestamp]
                    valid_timestamps = sorted(set(valid_timestamps))
                    
                    for iteration_idx, timestamp in enumerate(valid_timestamps):
                        dsfiles = [DSFile(path) 
                                for path in bls.as_file_list(DIAGSTAT_DIR / f"*{timestamp}_{target}*.diagstat.csv")]
                        # print(iteration_idx, target_keywords, target_idx, iteration_idx * len(target_keywords) + target_idx)
                        dspointings[iteration_idx * len(target_keywords) + target_idx] = DSPointing(dsfiles, 
                                                                                                    excluded_nodes=excluded_nodes,
                                                                                                    order_label=chr(65 + target_idx),
                                                                                                    session_idx=session_idx)
            if len(target_keywords) > 1:
                cadence_list.append(DSCadence(dspointings))
        print(f'Completed session {session_idx} ({session_idx + 1} of {len(nested_target_list)})')
    return cadence_list, NGP_pointings


def search_for_nearby_hits(cadence, max_drift_rate=10):
    for p1_idx, pointing1 in enumerate(cadence):
        for p2_idx, pointing2 in enumerate(cadence):
            # print(p1_idx, p2_idx, p1_idx == p2_idx)
            if p1_idx != p2_idx:
                Delta_t = pointing2.unix - pointing1.unix

                num_near_col = []
                idx_near_col = []
                num_near_max_col = []
                idx_near_max_col = []
                for hit_idx, row in pointing1.hits.iterrows():
                    center = row["Uncorrected_Frequency"] * 1e6
                    new_center = center + row["DriftRate"] * Delta_t

                    half_range = np.abs(2 * row["DriftRate"] * Delta_t) 
                    lower_bound = new_center - half_range
                    upper_bound = new_center + half_range

                    mask = (
                        (pointing2.hits["Uncorrected_Frequency"] * 1e6 > lower_bound)
                        & (pointing2.hits["Uncorrected_Frequency"] * 1e6 < upper_bound)
                    )
                    num_near_col.append(len(pointing2.hits[mask]))
                    idx_near_col.append(pointing2.hits[mask].index)

                    # Now do for max drift rate for RFI check
                    half_range = np.abs(2 * max_drift_rate * Delta_t) 
                    lower_bound = new_center - half_range
                    upper_bound = new_center + half_range

                    mask = (
                        (pointing2.hits["Uncorrected_Frequency"] * 1e6 > lower_bound)
                        & (pointing2.hits["Uncorrected_Frequency"] * 1e6 < upper_bound)
                    )
                    num_near_max_col.append(len(pointing2.hits[mask]))
                    idx_near_max_col.append(pointing2.hits[mask].index)

                pointing1.hits[f"num_hits_{pointing2.target}_{pointing2.scan}"] = num_near_col
                pointing1.hits[f"idx_hits_{pointing2.target}_{pointing2.scan}"] = idx_near_col
                pointing1.hits[f"num_hits_rfi_{pointing2.target}_{pointing2.scan}"] = num_near_max_col
                pointing1.hits[f"idx_hits_rfi_{pointing2.target}_{pointing2.scan}"] = idx_near_max_col
    return cadence


def direction_filter(cadence, on_labels='A', verbose=False):
    """
    Implement direction on sky filter and collect events in Pandas dataframe.

    
    """
    events = []

    if not isinstance(on_labels, list):
        on_labels = [on_labels]

    for on_label in on_labels:
        on_pointings, off_pointings = cadence.on_off_split(on_label=on_label)
        num_ons = len(on_pointings)
        
        event_idx = 0
        unique_first_hit_idx = []

        for p_ref_idx in range(num_ons):
            for hit_idx, hit in on_pointings[p_ref_idx].hits.iterrows():
                pass_checks = True
                event_hits_idx = [None] * num_ons

                # Check for signals in OFF pointings
                for p_off in off_pointings:
                    if hit[f'num_hits_rfi_{p_off.target}_{p_off.scan}'] > 0:
                        pass_checks = False 

                # Check for signals in ON pointings and track hit indices
                for p_idx in range(num_ons):
                    if p_idx != p_ref_idx:
                        p_on = on_pointings[p_idx]
                        if hit[f'num_hits_{p_on.target}_{p_on.scan}'] == 0:
                            pass_checks = False 
                        else:
                            if verbose and hit[f'num_hits_{p_on.target}_{p_on.scan}'] > 1:
                                print(f'More than 1 hit at {p_on.target}_{p_on.scan} for hit {hit_idx}!')
                            event_hits_idx[p_idx] = hit[f'idx_hits_{p_on.target}_{p_on.scan}'][0]
                    else:
                        event_hits_idx[p_idx] = hit_idx

                # If passed ON-OFF checks and not found already, save event
                if pass_checks and event_hits_idx[0] not in unique_first_hit_idx:
                    event = pd.concat([on_pointings[i].hits.loc[[event_hits_idx[i]]] 
                                       for i in range(num_ons)], 
                                      ignore_index=False)
                    event['found_with'] = f'{on_label}{p_ref_idx}'
                    event['event_idx'] = event_idx
                    event['target_event_idx'] = f'{on_pointings[p_ref_idx].target}_{event_idx:04d}' 
                    # print(f'{on_pointings[p_ref_idx].target}_{event_idx:04d}')

                    unique_first_hit_idx.append(event_hits_idx[0])
                    events.append(event)
    
                    event_idx += 1
                    
    if len(events) > 0:
        events = pd.concat(events, ignore_index=False)
    else:
        columns = list(cadence[0].hits.columns)
        columns += ['found_with', 'event_idx', 'target_event_idx']
        events = pd.DataFrame(columns=columns)
    return events


def filter_cadence_list(cadence_list):
    total_events = 0
    total_hits = 0
    
    for c in tqdm.tqdm(cadence_list):
        print(f'Working on {c}')
        c = search_for_nearby_hits(c, max_drift_rate=10)
    
        c.events = direction_filter(c, on_labels=['A', 'B'])
    
        num_A = len(c.events[c.events['found_with'].str.contains('A')].groupby('target_event_idx').first())
        num_B = len(c.events[c.events['found_with'].str.contains('B')].groupby('target_event_idx').first())
        total_events += num_A + num_B 
        
        hits_A = len(c[0].hits) + len(c[2].hits)
        hits_B = len(c[1].hits) + len(c[3].hits)
        total_hits += hits_A + hits_B
        print(f'Events - A: {num_A}, B: {num_B}')
        print(f'Hits   - A: {hits_A}, B: {hits_B}')
    print(f'Total events: {total_events}')
    print(f'Total hits: {total_hits}')
    with open('processed_cadences.pickle', 'wb') as f:
        pickle.dump(cadence_list, f)
    return cadence_list





