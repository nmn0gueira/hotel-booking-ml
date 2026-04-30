import os

import pandas as pd

from src.preprocessing import preprocess_data
from src.clustering import fit_predict
from src.evaluation import evaluate_clustering
from src.utils import load_subsample_indices, log_experiment

DATA_PATH = "data/hotel_bookings_course_release_v1.csv"
INDICES_PATH = "data/subsample_indices_v1_n30000_seed12345.txt"
OUTPUT_PATH = "experiments.csv"

K_VALUES = [2, 3, 4, 5, 6]
SEEDS = [0, 1, 2, 3]
ALGORITHMS = ["kmeans", "ikmeans", "gmm"]
# full: main representation (booking-creation segmentation, with context + adr and other stuff)
# no_value_block: drops adr, deposit_type, previous_*, is_repeated_guest
# no_context: drops hotel_binary, arrival_month_sin/cos
# complexity_only: drops VALUE_BLOCK + PROFILE_BLOCK; pure friction/workload features
FEATURE_SETS = ["full", "no_value_block", "no_context", "complexity_only"]
SCALERS = ["standard", "robust"]


def _total_runs() -> int:
    per_representation = (2 * len(SEEDS) + 1) * len(K_VALUES)
    return len(FEATURE_SETS) * len(SCALERS) * per_representation


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    indices = load_subsample_indices(INDICES_PATH)
    df = df.iloc[indices].reset_index(drop=True)
    print(f"Subsample shape: {df.shape}")

    completed: set = set()
    if os.path.exists(OUTPUT_PATH):
        prior = pd.read_csv(OUTPUT_PATH)
        for _, row in prior.iterrows():
            completed.add(
                (row["model"], int(row["k"]), row["seed"], row["feature_set"], row["scaler"])
            )
        print(f"Resuming: {len(completed)} runs already completed.")

    total = _total_runs()
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
                        labels, runtime = fit_predict(X, k, seed, algorithm)
                        metrics = evaluate_clustering(X, labels)
                        log_experiment(
                            {
                                "model": algorithm,
                                "k": k,
                                "seed": seed,
                                "feature_set": feature_set,
                                "scaler": scaler,
                                **metrics,
                                "runtime" : runtime
                            },
                            filepath=OUTPUT_PATH,
                        )
                        done += 1
                        print(
                            f"  [{done}/{total}] {algorithm} k={k} seed={seed}"
                            f"  sil={metrics['silhouette']:.4f}"
                            f"  db={metrics['davies_bouldin']:.4f}"
                            f"  ch={metrics['calinski_harabasz']:.4f}"
                            f"  runtime={runtime:.4f}"
                        )

    print(f"\nDone. {done} runs written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
