from pathlib import Path
import socket

DATA_DIR = Path("/datax2/users/bryanb")
DB_PATH = DATA_DIR / "pipeline.sqlite3"
DIAGSTAT_DIR = DATA_DIR / "scintillation_diagstats"

EVENT_PLOT_DIR = Path("scintillation_plots")
EVENT_PLOT_DIR_BY_NODE = EVENT_PLOT_DIR / socket.gethostname()

CONDA_ACTIVATE_PATH = Path("/home/bryanb/miniconda3/bin/activate")
PIPELINE_PATH = Path("/home/bryanb/blscint/analysis_scripts/")
WORKER_PATH = PIPELINE_PATH / "node_worker.py"