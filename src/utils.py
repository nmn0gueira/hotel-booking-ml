import os
import pandas as pd


def load_subsample_indices(indices_path):
    indices = []
    with open(indices_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                indices.append(int(line))
    return indices


def log_experiment(results, filepath="experiments.csv"):
    df = pd.DataFrame([results])
    if os.path.exists(filepath):
        df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df.to_csv(filepath, index=False)
