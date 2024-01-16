import tempfile
import contextlib
import sqlite3
from fabric import Connection
import pandas as pd
import setigen as stg
from pathlib import Path

from . import signal_manager
from . import frame_processing


CONDA_ACTIVATE_PATH = Path("/home/bryanb/miniconda3/bin/activate")


def fetch_pipeline_progress(node_db_path):
    """
    Read remote pipeline sqlite file from a path of the 
    form {host}:{sqlite_path}.
    """
    node, sqlite_path = str(node_db_path).split(":")
    handle, temp_path = tempfile.mkstemp()
    with Connection(node) as c:
        c.get(sqlite_path, local=temp_path)
    with contextlib.closing(sqlite3.connect(temp_path)) as con:
        df = pd.read_sql("SELECT * FROM jobs", con)
    return df


def centered_frame(node_csv_path, idx, fchans=256):
    node, csv_path = str(node_csv_path).split(":")
    temp_path = Path(tempfile.gettempdir()) / "bls_frame.pickle"

    python_code = (
        f"import tempfile;"
        f"save_loc = tempfile.gettempdir() + \"/bls_remote_frame.pickle\";"
        f"import blscint as bls;"
        f"m = bls.SignalManager();"
        f"m.add_real(\"{csv_path}\");"
        f"fr = m.centered_frame({idx}, fchans={fchans});"
        f"fr.save_pickle(save_loc);"
        f"print(save_loc);"
    )
    with Connection(node) as c:
        result = c.run(f"source {CONDA_ACTIVATE_PATH}; conda activate bl; python -c '{python_code}'")
        remote_temp_path = result.stdout.strip()

        c.get(remote_temp_path, local=str(temp_path))
    return stg.Frame.load_pickle(temp_path)