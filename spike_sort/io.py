"""Loading raw EMG signal from disk."""
import os
import sys

import numpy as np
import pandas as pd


def lookup_sample_rate(filename, header_path="Header.txt", default=24000):
    if not os.path.isfile(header_path):
        return default
    with open(header_path) as f:
        next(f, None)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[3] == os.path.basename(filename):
                return float(parts[2])
    return default


def load_signal(filename, fs, start, duration):
    start_sample = int(round(start * fs))
    n_rows = int(round(duration * fs)) if duration is not None else None
    try:
        data = pd.read_csv(filename, header=None, skiprows=start_sample,
                            nrows=n_rows, dtype=np.float64).to_numpy().ravel()
    except pd.errors.EmptyDataError:
        data = np.empty(0)
    if data.size == 0:
        sys.exit(f"--start {start}s skips past the end of {filename} "
                  f"({start_sample} samples requested at {fs} Hz) — lower --start "
                  f"(or START_S) for this file.")
    return data
