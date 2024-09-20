import sqlite3 
import shlex
import jort 
import glob
import subprocess
import psutil 
import click
import paramiko 
import time
from datetime import datetime
import numpy as np
from braceexpand import braceexpand
from pathlib import Path

from context import (
    DATA_DIR, DB_PATH, DIAGSTAT_DIR, EVENT_PLOT_DIR, EVENT_PLOT_DIR_BY_NODE,
    CONDA_ACTIVATE_PATH, PIPELINE_PATH, WORKER_PATH
)


@click.command()
@click.argument('steps', nargs=-1)
@click.option('-p', '--preference', multiple=True, 
              help='observation keyword for preferred analysis')
@click.option('-n', '--nodes', multiple=True,
              help='which nodes to run on')
@click.option('-f', '--force', is_flag=True,
              help='force run on all files')
@click.option('--frequency', default=60,
              help='update frequency, in seconds')
@click.option('--reset-db', is_flag=True,
              help='reset database for any reason')
def delegate(steps, preference, nodes, force, frequency, reset_db):
    """
    steps: dedoppler diagstat scp 

    Note that nodes must be specified with quotes to avoid bash expansion.
    """
    nodes = [y for x in nodes for y in list(braceexpand(x))]
    if len(nodes) == 0:
        nodes = [f"blc{x}{y}" for x in range(8) for y in range(8)]
    nodes = np.array(nodes)
    print(f"Target nodes: {' '.join(nodes)}")

    preference_options = ' '.join([f'-p {n}' for n in preference])
    for step in steps:
        # Only reason we don't have to pass in jort config is that blc nodes share the config file as of now
        if step == "kill":
            full_command = "ps ux | egrep -v 'ssh|screen|bash|PID|vscode' | awk '{print $2}' | xargs -t kill"
        elif step == "plot":
            python_code = (
                f"from create_cadence_plots import plot_all_events;"
                f"plot_all_events();"
            )
            full_command = (
                f"source {CONDA_ACTIVATE_PATH};"
                f"conda activate bl;"
                f"jort init;"
                f"cd {PIPELINE_PATH};"
                f"python -c '{python_code}'"
            )
        else:
            # full_command = (
            #     f"source {CONDA_ACTIVATE_PATH};"
            #     f"conda activate bl;"
            #     f"jort init;"
            #     f"python {WORKER_PATH} {step} {preference_options} -e"
            # )
            full_command = (
                f"source {CONDA_ACTIVATE_PATH};"
                f"conda activate bl;"
                f"jort init;"
                f"python {WORKER_PATH} {step} {preference_options}"
            )
            if force:
                full_command += " --force"
        if reset_db:
            full_command = f"rm {DB_PATH};" + full_command

        nodes_stdout = {}
        problem_nodes = []
        for node in nodes:
            try:
                s = paramiko.SSHClient()
                s.load_system_host_keys()
                s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                print(f"Executing on node {node}")
                s.connect(node)
                stdin, stdout, stderr = s.exec_command(full_command)
                nodes_stdout[node] = stdout
            except Exception as e:
                print(f"Problem with node {node}")
                problem_nodes.append(node)
        time.sleep(1)

        # Reset which nodes are available
        nodes = np.array(list(nodes_stdout.keys()))
        
        nodes_finished = np.array([nodes_stdout[node].channel.exit_status_ready() for node in nodes])
        while False in nodes_finished:
            print(f"Running ({datetime.now().isoformat()} @ GBT): {' '.join(nodes[np.invert(nodes_finished)])}")
            if step in ["dedoppler", "diagstat"]:
                progress_update(nodes, step)
            time.sleep(frequency)
            nodes_finished = np.array([nodes_stdout[node].channel.exit_status_ready() for node in nodes])
        print(f'Completed `{step}`')

        if len(problem_nodes) > 0:
            print(f"Nodes {problem_nodes} had issues!")
        else:
            print("No nodes with issues!")


def progress_update(nodes, step):
    """
    Print formatted display of progress
    """
    progress_dict = {}
    for node in nodes:
        s = None
        try:
            s = paramiko.SSHClient()
            s.load_system_host_keys()
            s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            s.connect(node)
            command = (
                f"cd {PIPELINE_PATH};"
                f"source {CONDA_ACTIVATE_PATH};"
                f"conda activate bl;"
                f"python -c 'from node_worker import progress_summary; print(progress_summary(\"{step}\"))'"
            )
            stdin, stdout, stderr = s.exec_command(command)
            stdout = stdout.readlines()
            progress_dict[node] = stdout[0].strip()
        except Exception as e:
            pass
        finally:
            if s:
                s.close()

    x_print = [0, 4, 5, 6, 7]
    print()
    print(f"{step} progress: TODO SUCCESS ERROR")
    row = "| "
    for x in x_print:
        row += f"{'Node':<5} {'T':>4}{'S':>4}{'E':>4} | "
    print("+" + "-" * (len(row) - 3) + "+")
    print(row)
    print("|" + "=" * (len(row) - 3) + "|")
    for y in range(8):
        row = "| "
        for x in x_print:
            node = f"blc{x}{y}"
            try:
                counts = progress_dict[node].split(",")
            except KeyError:
                counts = "--,--,--".split(",")
            row += f"{node}:{counts[0]:>4}{counts[1]:>4}{counts[2]:>4} | "
        print(row)
    print("+" + "-" * (len(row) - 3) + "+")
    print()


if __name__ == "__main__":
    delegate()