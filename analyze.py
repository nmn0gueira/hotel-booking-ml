"""
Post-run analysis: reads experiments.csv + labels/, generates all figures and tables.
Run AFTER run_all.py completes.

Usage:
    python analyze.py
    python analyze.py --main-k 4   # override the selected K for profiles
"""
import argparse
import os
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture

from src.anomaly import kmeans_anomaly_scores, top_anomalies
from src.evaluation import compute_ari
from src.preprocessing import preprocess_data
from src.profiling import compute_cluster_profiles
from src.utils import load_labels, load_subsample_indices

DATA_PATH = "data/hotel_bookings_course_release_v1.csv"
INDICES_PATH = "data/subsample_indices_v1_n30000_seed12345.txt"
EXPERIMENTS_PATH = "experiments.csv"
LABELS_DIR = "labels"
FIGURES_DIR = "figures"
TABLES_DIR = "tables"

MAIN_FS = "no_value_block"
MAIN_SCALER = "robust"

SEEDS = list(range(10))
K_VALUES = [2, 3, 4, 5, 6, 7, 8]
FEATURE_SETS = ["full", "no_value_block", "no_context", "complexity_only"]
SCALERS = ["standard", "robust"]


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    indices = load_subsample_indices(INDICES_PATH)
    return df.iloc[indices].reset_index(drop=True)


def generate_stability_metrics(exp: pd.DataFrame) -> None:
    """Mean ± std of internal indices across seeds per (model, k, feature_set, scaler)."""
    stochastic = exp[exp["model"].isin(["kmeans", "gmm"])]
    rows = []
    for name, group in stochastic.groupby(["model", "k", "feature_set", "scaler"]):
        model, k, fs, sc = name
        row = {"model": model, "k": k, "feature_set": fs, "scaler": sc,
               "n_seeds": len(group)}
        for col in ["silhouette", "davies_bouldin", "calinski_harabasz"]:
            row[f"{col}_mean"] = round(group[col].mean(), 4)
            row[f"{col}_std"] = round(group[col].std(), 4)
        rows.append(row)
    out = f"{TABLES_DIR}/stability_seed_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  {out}")


def generate_seed_ari() -> None:
    """Pairwise ARI across seeds for kmeans and gmm per (k, feature_set, scaler)."""
    rows = []
    for fs in FEATURE_SETS:
        for sc in SCALERS:
            for algo in ["kmeans", "gmm"]:
                for k in K_VALUES:
                    label_list = []
                    for seed in SEEDS:
                        try:
                            lbl = load_labels(algo, k, seed, fs, sc, LABELS_DIR)
                            label_list.append(lbl)
                        except FileNotFoundError:
                            pass
                    if len(label_list) < 2:
                        continue
                    aris = [compute_ari(a, b)
                            for a, b in combinations(label_list, 2)]
                    aris = [v for v in aris if not np.isnan(v)]
                    rows.append({
                        "model": algo, "k": k, "feature_set": fs, "scaler": sc,
                        "ari_mean": round(float(np.mean(aris)), 4) if aris else float("nan"),
                        "ari_std": round(float(np.std(aris)), 4) if aris else float("nan"),
                        "n_pairs": len(aris),
                    })
    out = f"{TABLES_DIR}/stability_seed_ari.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  {out}")


def generate_controlled_comparison_ari() -> None:
    """ARI between main representation (R0: no_value_block+robust, seed=0) and others."""
    rows = []
    for k in K_VALUES:
        try:
            labels_r0 = load_labels("kmeans", k, 0, MAIN_FS, MAIN_SCALER, LABELS_DIR)
        except FileNotFoundError:
            continue
        for fs in FEATURE_SETS:
            for sc in SCALERS:
                if fs == MAIN_FS and sc == MAIN_SCALER:
                    continue
                try:
                    labels_alt = load_labels("kmeans", k, 0, fs, sc, LABELS_DIR)
                except FileNotFoundError:
                    continue
                ari = compute_ari(labels_r0, labels_alt)
                rows.append({"k": k, "vs_feature_set": fs, "vs_scaler": sc,
                             "ari": round(ari, 4)})
    out = f"{TABLES_DIR}/controlled_comparison_ari.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  {out}")


def generate_profiles(raw_df: pd.DataFrame, main_k: int) -> None:
    """Cluster profiles for main representation at selected K."""
    X, feature_names = preprocess_data(raw_df, feature_set=MAIN_FS, scaler=MAIN_SCALER)
    labels = load_labels("kmeans", main_k, 0, MAIN_FS, MAIN_SCALER, LABELS_DIR)
    profile = compute_cluster_profiles(X, labels, feature_names, raw_df=raw_df)
    out = f"{TABLES_DIR}/profile_k{main_k}_{MAIN_FS}_{MAIN_SCALER}.csv"
    profile.to_csv(out)
    print(f"  {out}")


def generate_gmm_aic_bic(raw_df: pd.DataFrame) -> None:
    """AIC/BIC curve for GMM on main representation (family-specific diagnostic)."""
    X, _ = preprocess_data(raw_df, feature_set=MAIN_FS, scaler=MAIN_SCALER)
    aic_vals, bic_vals = [], []
    for k in K_VALUES:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
        gmm.fit(X)
        aic_vals.append(gmm.aic(X))
        bic_vals.append(gmm.bic(X))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(K_VALUES, aic_vals, marker="o", label="AIC")
    ax.plot(K_VALUES, bic_vals, marker="s", label="BIC")
    ax.set_xlabel("K")
    ax.set_ylabel("Score (lower is better)")
    ax.set_title(f"GMM model selection — {MAIN_FS} + {MAIN_SCALER}")
    ax.legend()
    fig.tight_layout()
    out = f"{FIGURES_DIR}/gmm_aic_bic.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out}")


def generate_anomaly_analysis(raw_df: pd.DataFrame, main_k: int) -> None:
    """
    Extension E1: top-20 anomalies by normalised centroid distance (k-means).
    Sensitivity figure: anomaly rank correlation under standard vs robust scaling.
    Stability: compare across 3 seeds.
    """
    X_rob, _ = preprocess_data(raw_df, feature_set=MAIN_FS, scaler=MAIN_SCALER)
    labels_rob = load_labels("kmeans", main_k, 0, MAIN_FS, MAIN_SCALER, LABELS_DIR)
    centroids_rob = np.array([
        X_rob[labels_rob == c].mean(axis=0)
        for c in sorted(np.unique(labels_rob))
    ])
    scores_rob = kmeans_anomaly_scores(X_rob, labels_rob, centroids_rob)

    top20_idx = top_anomalies(scores_rob, n=20)
    raw_cols = [c for c in ["lead_time", "adults", "children", "babies",
                             "total_of_special_requests", "required_car_parking_spaces",
                             "adr", "deposit_type", "market_segment",
                             "customer_type", "is_canceled"] if c in raw_df.columns]
    raw_slice = raw_df.iloc[:len(labels_rob)].reset_index(drop=True)
    top_df = raw_slice.loc[top20_idx, raw_cols].copy()
    top_df.insert(0, "anomaly_score", scores_rob[top20_idx].round(3))
    top_df.insert(1, "cluster", labels_rob[top20_idx])
    out_table = f"{TABLES_DIR}/anomaly_top20.csv"
    top_df.to_csv(out_table)
    print(f"  {out_table}")

    # Sensitivity: robust vs standard scaling anomaly rank correlation
    X_std, _ = preprocess_data(raw_df, feature_set=MAIN_FS, scaler="standard")
    labels_std = load_labels("kmeans", main_k, 0, MAIN_FS, "standard", LABELS_DIR)
    centroids_std = np.array([
        X_std[labels_std == c].mean(axis=0)
        for c in sorted(np.unique(labels_std))
    ])
    scores_std = kmeans_anomaly_scores(X_std, labels_std, centroids_std)

    n = min(len(scores_rob), len(scores_std))
    corr, _ = spearmanr(scores_rob[:n], scores_std[:n])

    rank_rob = np.argsort(np.argsort(-scores_rob[:n]))
    rank_std = np.argsort(np.argsort(-scores_std[:n]))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(rank_rob[:500], rank_std[:500], alpha=0.3, s=5)
    ax.set_xlabel("Anomaly rank — RobustScaler (R0)")
    ax.set_ylabel("Anomaly rank — StandardScaler")
    ax.set_title("E1: Anomaly rank sensitivity (top 500 shown)")
    ax.text(0.05, 0.95, f"Spearman r = {corr:.3f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    out_fig = f"{FIGURES_DIR}/anomaly_sensitivity.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"  {out_fig}")

    # Stability: compare top-20 sets across seeds 0, 1, 2
    seed_top20 = {}
    for seed in [0, 1, 2]:
        try:
            lbl = load_labels("kmeans", main_k, seed, MAIN_FS, MAIN_SCALER, LABELS_DIR)
            cen = np.array([X_rob[lbl == c].mean(axis=0) for c in sorted(np.unique(lbl))])
            sc = kmeans_anomaly_scores(X_rob, lbl, cen)
            seed_top20[seed] = set(top_anomalies(sc, n=20).tolist())
        except FileNotFoundError:
            pass
    if len(seed_top20) >= 2:
        seeds = sorted(seed_top20)
        overlap_rows = []
        for sa, sb in combinations(seeds, 2):
            inter = len(seed_top20[sa] & seed_top20[sb])
            overlap_rows.append({"seed_a": sa, "seed_b": sb, "top20_overlap": inter})
        out_stab = f"{TABLES_DIR}/anomaly_stability.csv"
        pd.DataFrame(overlap_rows).to_csv(out_stab, index=False)
        print(f"  {out_stab}")


def main(main_k: int = 4) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    print("Loading data...")
    raw_df = _load_data()
    exp = pd.read_csv(EXPERIMENTS_PATH)

    print("Generating stability tables...")
    generate_stability_metrics(exp)
    generate_seed_ari()
    generate_controlled_comparison_ari()

    print(f"Generating cluster profiles (K={main_k})...")
    generate_profiles(raw_df, main_k)

    print("Generating GMM AIC/BIC figure...")
    generate_gmm_aic_bic(raw_df)

    print("Generating anomaly analysis (E1)...")
    generate_anomaly_analysis(raw_df, main_k)

    print(f"\nDone. Check {FIGURES_DIR}/ and {TABLES_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-k", type=int, default=4,
                        help="K value to use for cluster profiles and anomaly analysis")
    args = parser.parse_args()
    main(main_k=args.main_k)
