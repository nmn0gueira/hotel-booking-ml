import os

import numpy as np
import pandas as pd


def load_subsample_indices(indices_path):
    indices = []
    with open(indices_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                indices.append(int(line))
    return indices


def representation_id(feature_set: str, scaler: str) -> str:
    """Stable short ID for a (feature_set, scaler) pair."""
    fs_code = {"full": "F", "no_value_block": "NV", "no_context": "NC", "complexity_only": "CO"}
    sc_code = {"standard": "std", "robust": "rob", "minmax": "mm", "mean": "mr"}
    return f"{fs_code.get(feature_set, feature_set)}-{sc_code.get(scaler, scaler)}"


def log_experiment(results: dict, filepath: str = "experiments.csv") -> None:
    df = pd.DataFrame([results])
    if os.path.exists(filepath):
        df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df.to_csv(filepath, index=False)


def save_labels(
    labels: np.ndarray,
    model: str,
    k: int,
    seed,
    feature_set: str,
    scaler: str,
    labels_dir: str = "labels",
) -> None:
    os.makedirs(labels_dir, exist_ok=True)
    seed_str = "none" if seed is None else str(seed)
    fname = f"{model}_k{k}_s{seed_str}_{feature_set}_{scaler}.npy"
    np.save(os.path.join(labels_dir, fname), labels)


def load_labels(
    model: str,
    k: int,
    seed,
    feature_set: str,
    scaler: str,
    labels_dir: str = "labels",
) -> np.ndarray:
    seed_str = "none" if seed is None else str(seed)
    fname = f"{model}_k{k}_s{seed_str}_{feature_set}_{scaler}.npy"
    return np.load(os.path.join(labels_dir, fname))
