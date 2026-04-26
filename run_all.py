import os

import pandas as pd

from src.preprocessing import preprocess_data
from src.clustering import fit_predict
from src.evaluation import evaluate_clustering
from src.utils import load_subsample_indices, log_experiment

DATA_PATH = "data/hotel_bookings_course_release_v1.csv"
INDICES_PATH = "data/subsample_indices_v1_n30000_seed12345.txt"
OUTPUT_PATH = "experiments.csv"

K_VALUES = [3, 4, 5, 6]
SEEDS = [0, 1, 2, 3, 4]
#ALGORITHMS = ["kmeans", "ikmeans", "gmm"]
ALGORITHMS = ["kmeans", "gmm"]
FEATURE_SETS = ["full", "no_value_block"]
SCALERS = ["standard", "robust"]


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    indices = load_subsample_indices(INDICES_PATH)
    df = df.iloc[indices].reset_index(drop=True)
    print(f"Subsample shape: {df.shape}")

    # Load completed runs to allow resuming after a crash
    completed = set()
    if os.path.exists(OUTPUT_PATH):
        prior = pd.read_csv(OUTPUT_PATH)
        for _, row in prior.iterrows():
            completed.add((row["model"], int(row["k"]), row["seed"], row["feature_set"], row["scaler"]))
        print(f"Resuming: {len(completed)} runs already completed.")

    total = len(FEATURE_SETS) * len(SCALERS) * (
        len(SEEDS) * 2 * len(K_VALUES) + 1 * len(K_VALUES)
    )
    done = len(completed)

    for feature_set in FEATURE_SETS:
        for scaler in SCALERS:
            print(f"\n--- feature_set={feature_set}  scaler={scaler} ---")
            X, _ = preprocess_data(df, feature_set=feature_set, scaler=scaler)

            for algorithm in ALGORITHMS:
                seeds = [None] if algorithm == "ikmeans" else SEEDS
                for k in K_VALUES:
                    for seed in seeds:
                        key = (algorithm, k, seed, feature_set, scaler)
                        if key in completed:
                            continue
                        labels, _ = fit_predict(X, k, seed, algorithm)
                        metrics = evaluate_clustering(X, labels)
                        log_experiment(
                            {
                                "model": algorithm,
                                "k": k,
                                "seed": seed,
                                "feature_set": feature_set,
                                "scaler": scaler,
                                **metrics,
                            },
                            filepath=OUTPUT_PATH,
                        )
                        done += 1
                        print(
                            f"  [{done}/{total}] {algorithm} k={k} seed={seed}"
                            f"  sil={metrics['silhouette']:.4f}"
                            f"  db={metrics['davies_bouldin']:.4f}"
                        )

    print(f"\nDone. Results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
