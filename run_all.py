import argparse
import os
import uuid
from datetime import date as _date

import pandas as pd

from src.preprocessing import preprocess_data
from src.clustering import fit_predict
from src.evaluation import evaluate_clustering
from src.utils import load_subsample_indices, log_experiment, save_labels, representation_id

DATA_PATH = "data/hotel_bookings_course_release_v1.csv"
INDICES_PATH = "data/subsample_indices_v1_n30000_seed12345.txt"
OUTPUT_PATH = "experiments.csv"

K_VALUES = [2, 3, 4, 5, 6, 7, 8]
SEEDS = list(range(10))
ALGORITHMS = ["kmeans", "ikmeans", "gmm"]
# full: main representation (booking-creation segmentation, with context + adr and other stuff)
# no_value_block: drops adr, deposit_type, previous_*, is_repeated_guest
# no_context: drops hotel_binary, arrival_month_sin/cos
# complexity_only: drops VALUE_BLOCK + PROFILE_BLOCK; pure friction/workload features
FEATURE_SETS = ["full", "no_value_block", "no_context", "complexity_only"]
SCALERS = ["standard", "robust"]


def _total_runs() -> int:
    kmeans_gmm_runs = 2 * len(SEEDS) * len(K_VALUES)   # kmeans + gmm, 10 seeds each
    ikmeans_runs = len(K_VALUES)                        # deterministic, 1 run per K
    per_pair = kmeans_gmm_runs + ikmeans_runs
    return len(FEATURE_SETS) * len(SCALERS) * per_pair


def main(fast: bool = False):
    # ── Fast-check mode overrides ─────────────────────────────────────────────
    # Reduces the grid to a small but representative subset so the full
    # pipeline (preprocessing → clustering → evaluation → logging → labels/)
    # can be verified in ~2 minutes without running all 1176 combinations.
    global K_VALUES, SEEDS, FEATURE_SETS, SCALERS
    if fast:
        K_VALUES     = [3, 4]
        SEEDS        = [0, 1]
        FEATURE_SETS = ["no_value_block"]
        SCALERS      = ["robust"]
        print("FAST-CHECK MODE: K=[3,4], seeds=[0,1], "
              "feature_sets=[no_value_block], scalers=[robust]")

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    indices = load_subsample_indices(INDICES_PATH)
    df = df.iloc[indices].reset_index(drop=True)
    print(f"Subsample shape: {df.shape}")

    completed: set = set()
    if os.path.exists(OUTPUT_PATH):
        prior = pd.read_csv(OUTPUT_PATH)
        for _, row in prior.iterrows():
            seed_val = None if pd.isna(row["seed"]) else row["seed"]
            completed.add(
                (row["model"], int(row["k"]), seed_val, row["feature_set"], row["scaler"])
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
                        try:
                            labels, runtime, n_iter = fit_predict(X, k, seed, algorithm)
                        except ValueError as exc:
                            print(f"  SKIP {algorithm} k={k} seed={seed}: {exc}")
                            done += 1
                            continue
                        metrics = evaluate_clustering(X, labels)
                        run_id = str(uuid.uuid4())[:8]
                        rep_id = representation_id(feature_set, scaler)
                        log_experiment(
                            {
                                "run_id": run_id,
                                "date": _date.today().isoformat(),
                                "model": algorithm,
                                "k": k,
                                "seed": seed,
                                "feature_set": feature_set,
                                "scaler": scaler,
                                "representation_id": rep_id,
                                **metrics,
                                "runtime": runtime,
                                "diagnostics": n_iter,   # convergence iterations (model.n_iter_)
                                "sample_rule": os.path.basename(INDICES_PATH),
                                "notes": "",
                            },
                            filepath=OUTPUT_PATH,
                        )
                        save_labels(labels, algorithm, k, seed, feature_set, scaler)
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
    parser = argparse.ArgumentParser(
        description="Run full clustering experiment grid."
    )
    parser.add_argument(
        "--fast", action="store_true",
        help=(
            "Fast-check mode: verifies end-to-end pipeline integrity without "
            "running the full grid. Uses 2 seeds, K in [3, 4], 1 feature set "
            "(no_value_block), and 1 scaler (robust). Completes in ~2 minutes."
        ),
    )
    args = parser.parse_args()
    main(fast=args.fast)