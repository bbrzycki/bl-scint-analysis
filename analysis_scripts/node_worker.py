import sqlite3 
import shlex
import time 
import socket
import jort 
import glob
import psutil 
import click
from pathlib import Path
import contextlib
from datetime import datetime
from collections import Counter

from context import DATA_DIR, DB_PATH


def get_target_files(dir):
    return list(DATA_DIR.glob("*.0006.h5"))
    # return list(glob.glob(f"{dir}/*.0006.h5"))


def parse_fn(fn):
    """
    Get observation info from filename, fn is Path object.
    """
    name = fn.name
    obs_els = name.split("_")

    node = obs_els[0]
    short_name = "_".join(obs_els[4:-1])
    scan = obs_els[-1].split(".")[0]

    return (name, short_name, node, scan)


def setup_status_db():
    with contextlib.closing(sqlite3.connect(DB_PATH)) as con:
        cur = con.cursor()
        cur.execute((
            "CREATE TABLE IF NOT EXISTS jobs ("
            "    id INTEGER PRIMARY KEY,"
            "    name TEXT,"
            "    short_name TEXT,"
            "    node TEXT,"
            "    scan TEXT,"
            "    dedoppler_status TEXT,"
            "    dedoppler_start TEXT,"
            "    dedoppler_runtime TEXT,"
            "    dedoppler_gpu TEXT,"
            "    diagstat_status TEXT,"
            "    diagstat_start TEXT,"
            "    diagstat_runtime TEXT,"
            "    error_message TEXT"
            ")"
        ))
        con.commit()

        # Populate full list of existing files, to keep track of data.
        full_file_list = sorted(get_target_files(DATA_DIR))
        for fn in full_file_list:
            # First check that it doesn't exist
            res = cur.execute(
                "SELECT * FROM jobs WHERE name = ?",
                (fn.name,)
            ).fetchone() 
            if res is None:
                cur.execute(
                    "INSERT INTO jobs(name, short_name, node, scan) VALUES (?,?,?,?)",
                    parse_fn(fn)
                )
        con.commit()


def completed(data_fn, step):
    if step == "scp":
        data_path = Path(data_fn)
        csv_path = data_path.parent / f"{data_path.stem}.diagstat.csv"
        return not csv_path.is_file()
    with contextlib.closing(sqlite3.connect(DB_PATH)) as con:
        cur = con.cursor()
        res = cur.execute(
            f"SELECT {step}_status FROM jobs WHERE name = ?",
            (data_fn.name,)
        )
        status = res.fetchone()[0]
        return status == "success"
    

def progress_summary(step):
    with contextlib.closing(sqlite3.connect(DB_PATH)) as con:
        cur = con.cursor()
        res = cur.execute(
            f"SELECT {step}_status FROM jobs"
        )
        res = res.fetchall()
        status = [s[0] for s in res]
        status_dict = dict(Counter(status))
        if None in status_dict:
            status_dict["todo"] = status_dict.pop(None)
        else:
            status_dict["todo"] = 0
        if "running" in status_dict:
            status_dict["todo"] += 1
        for status in ["success", "error"]:
            if status not in status_dict:
                status_dict[status] = 0
        return f'{status_dict["todo"]},{status_dict["success"]},{status_dict["error"]}'


def execute_analysis_step(data_fn, step):
    data_path = Path(data_fn)
    csv_path = data_path.parent / f"{data_path.stem}.diagstat.csv"
    if step == "dedoppler":
        # command = (
        #     f"blscint dedoppler -s 10 {data_fn}"
        # )
        command = (
            f"blscint dedoppler -s 10 -g {data_fn}"
        )
    elif step == "diagstat":
        command = (
            f"blscint diagstat -r {data_fn}"
        )
    elif step == "scp":
        time.sleep(1)
        command = (
            f"scp {csv_path} blc00:/datax2/users/bryanb/scintillation_diagstats/{csv_path.name}"
        )
    else:
        raise ValueError(f"{step} isn't a valid option (choose from dedoppler, diagstat, or scp)")

    if step in ["dedoppler", "diagstat"]:
        # Mark as running
        with contextlib.closing(sqlite3.connect(DB_PATH)) as con:
            cur = con.cursor()
            cur.execute((
                f"UPDATE jobs SET {step}_status = ?, {step}_start = ? "
                f"WHERE name = ?"
            ),
                ("running", datetime.utcnow().isoformat(), data_fn.name)
            )
            if step == "dedoppler":
                cur.execute((
                    "UPDATE jobs SET dedoppler_gpu = ? "
                    "WHERE name = ?"
                ),
                    ("True", data_fn.name)
                )
            con.commit()

    # Run and collect payload
    payload = jort.track_new(command, to_db=True, session_name=step)

    if step in ["dedoppler", "diagstat"]:
        # Mark as finished
        with contextlib.closing(sqlite3.connect(DB_PATH)) as con:
            cur = con.cursor()
            cur.execute((
                f"UPDATE jobs SET {step}_status = ?, {step}_runtime = ? "
                f"WHERE name = ?"
            ),
                (payload["status"], payload["runtime"], data_fn.name)
            )
            if payload["status"] == "error":
                cur.execute((
                    "UPDATE jobs SET error_message = ? "
                    "WHERE name = ?"
                ),
                    (payload["error_message"], data_fn.name)
                )
            con.commit()


@click.command()
@click.argument('step')
@click.option('-p', '--preference', multiple=True, 
              help='observation keyword for preferred / high priority analysis')
@click.option('-e', '--end-after-pref', is_flag=True,
              help='end after preferred analysis')
@click.option('-f', '--force', is_flag=True,
              help='force run on all files')
def search(step, preference, end_after_pref, force):
    if force:
        with contextlib.closing(sqlite3.connect(DB_PATH)) as con:
            cur = con.cursor()
            cur.execute(f"UPDATE jobs SET {step}_status = NULL, {step}_runtime = NULL")
            con.commit()

    full_file_list = get_target_files(DATA_DIR)
    callbacks = []

    tr = jort.Tracker()
    tr.start(f"{step} - {socket.gethostname()} - high priority")
    # Do preferred files first
    pref_file_list = []
    for keyword in preference:
        pref_file_list = [fn for fn in full_file_list if keyword in str(fn)]
        for data_fn in pref_file_list:
            if not completed(data_fn, step):
                execute_analysis_step(data_fn, step)
    tr.stop(callbacks=callbacks)

    if not end_after_pref:
        tr.start(f"{step} - {socket.gethostname()} - normal priority")

        # Iterate over all data files that haven't been successfully completed
        remaining_fns = sorted(set(full_file_list) - set(pref_file_list))
        for data_fn in remaining_fns:
            if not completed(data_fn, step):
                execute_analysis_step(data_fn, step)
        tr.stop(callbacks=callbacks)


if __name__ == "__main__":
    setup_status_db()
    search()