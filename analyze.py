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

from src.anomaly import kmeans_anomaly_scores, gmm_anomaly_scores, top_anomalies
from src.clustering import run_ikmeans
from src.evaluation import compute_ari, evaluate_clustering
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



#    Stability across seeds: mean ± std of internal indices, pairwise ARI across seeds.
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



#   Pairwise ARI across seeds for kmeans and gmm per (k, feature_set, scaler).
def generate_seed_ari() -> None:

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



#  Controlled comparison: ARI between main representation (R0: no_value_block+robust, seed=0) and others.
def generate_controlled_comparison_ari() -> None:

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



#    Cluster profiles for main representation at selected K.
def generate_profiles(raw_df: pd.DataFrame, main_k: int) -> None:

    X, feature_names = preprocess_data(raw_df, feature_set=MAIN_FS, scaler=MAIN_SCALER)
    labels = load_labels("kmeans", main_k, 0, MAIN_FS, MAIN_SCALER, LABELS_DIR)
    profile = compute_cluster_profiles(X, labels, feature_names, raw_df=raw_df)
    out = f"{TABLES_DIR}/profile_k{main_k}_{MAIN_FS}_{MAIN_SCALER}.csv"
    profile.to_csv(out)
    print(f"  {out}")



# AIC/BIC curve for GMM on main representation (family-specific diagnostic).
def generate_gmm_aic_bic(raw_df: pd.DataFrame) -> None:
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



#  Rule-based classification: data-quality anomaly vs rare-but-plausible booking.
def _classify_anomaly_type(row: pd.Series) -> str:
   
    # No guests at all — impossible booking
    adults   = row.get("adults",   1)
    children = row.get("children", 0)
    babies   = row.get("babies",   0)
    if pd.notna(adults) and pd.notna(children) and pd.notna(babies):
        if int(adults) == 0 and int(children) == 0 and int(babies) == 0:
            return "data_quality"

    # Negative ADR — impossible price
    adr = row.get("adr", np.nan)
    if pd.notna(adr) and float(adr) < 0:
        return "data_quality"

    # Extreme lead_time (>700 days ~2 years) — likely data entry error
    lead_time = row.get("lead_time", np.nan)
    if pd.notna(lead_time) and float(lead_time) > 700:
        return "data_quality"

    # Unrealistically large party (>10 adults) — likely entry error
    if pd.notna(adults) and int(adults) > 10:
        return "data_quality"

    return "rare_plausible"


#    Build a justification string for why this booking is unusual.
def _build_reason(row: pd.Series, cluster_means: dict) -> str:
   
    NUMERIC_COLS = ["lead_time", "adults", "children", "babies", "adr",
                    "total_of_special_requests", "required_car_parking_spaces"]

    cluster_id = int(row["cluster"]) if "cluster" in row.index else -1
    means = cluster_means.get(cluster_id, {})

    # Flag impossible values first
    flags = []
    adr = row.get("adr", np.nan)
    if pd.notna(adr) and float(adr) < 0:
        flags.append(f"adr={float(adr):.1f} (impossible: negative price)")

    adults   = row.get("adults",   np.nan)
    children = row.get("children", np.nan)
    babies   = row.get("babies",   np.nan)
    if (pd.notna(adults) and pd.notna(children) and pd.notna(babies)
            and int(adults) == 0 and int(children) == 0 and int(babies) == 0):
        flags.append("adults=children=babies=0 (impossible: no guests)")

    lead_time = row.get("lead_time", np.nan)
    if pd.notna(lead_time) and float(lead_time) > 700:
        flags.append(f"lead_time={int(lead_time)} (>700 days: likely entry error)")

    # Find top deviant numeric fields (by absolute deviation from cluster mean)
    deviations = []
    for col in NUMERIC_COLS:
        val = row.get(col, np.nan)
        mean_val = means.get(col, np.nan)
        if pd.isna(val) or pd.isna(mean_val):
            continue
        dev = abs(float(val) - float(mean_val))
        deviations.append((dev, col, float(val), float(mean_val)))

    deviations.sort(reverse=True)
    top_devs = []
    for _, col, val, mean_val in deviations[:3]:
        # Skip if already flagged above
        if any(col in f for f in flags):
            continue
        top_devs.append(f"{col}={val:.0f} (cluster mean={mean_val:.1f})")

    parts = flags + top_devs
    return "; ".join(parts) if parts else "no dominant deviation identified"



#    Extension E1: top-20 anomalies
def generate_anomaly_analysis(raw_df: pd.DataFrame, main_k: int) -> None:
    
    X_rob, _ = preprocess_data(raw_df, feature_set=MAIN_FS, scaler=MAIN_SCALER)
    labels_rob = load_labels("kmeans", main_k, 0, MAIN_FS, MAIN_SCALER, LABELS_DIR)
    centroids_rob = np.array([
        X_rob[labels_rob == c].mean(axis=0)
        for c in sorted(np.unique(labels_rob))
    ])
    scores_kmeans = kmeans_anomaly_scores(X_rob, labels_rob, centroids_rob)

    # GMM anomaly scores (sensitivity: k-means vs GMM)
    scores_gmm = gmm_anomaly_scores(X_rob, k=main_k, seed=0)

    top20_kmeans = set(top_anomalies(scores_kmeans, n=20).tolist())
    top20_gmm    = set(top_anomalies(scores_gmm,    n=20).tolist())
    overlap_kg   = len(top20_kmeans & top20_gmm)
    print(f"  k-means vs GMM top-20 overlap: {overlap_kg}/20 points in common")

    # Build cluster means dict for reason generation
    NUMERIC_COLS = ["lead_time", "adults", "children", "babies", "adr",
                    "total_of_special_requests", "required_car_parking_spaces"]
    raw_slice = raw_df.iloc[:len(labels_rob)].reset_index(drop=True)
    raw_slice["_cluster"] = labels_rob
    cluster_means = {}
    for cid in sorted(np.unique(labels_rob)):
        subset = raw_slice[raw_slice["_cluster"] == cid]
        cluster_means[int(cid)] = {
            col: float(subset[col].mean())
            for col in NUMERIC_COLS if col in subset.columns
        }
    raw_slice = raw_slice.drop(columns=["_cluster"])

    # Build top-20 table (k-means scores, primary ranking)
    top20_idx = np.array(sorted(top20_kmeans))
    raw_cols  = [c for c in ["lead_time", "adults", "children", "babies",
                              "total_of_special_requests",
                              "required_car_parking_spaces",
                              "adr", "deposit_type", "market_segment",
                              "customer_type", "is_canceled"]
                 if c in raw_df.columns]

    top_df = raw_slice.loc[top20_idx, raw_cols].copy()
    top_df.insert(0, "anomaly_score_kmeans", scores_kmeans[top20_idx].round(3))
    top_df.insert(1, "anomaly_score_gmm",    scores_gmm[top20_idx].round(3))
    top_df.insert(2, "cluster",              labels_rob[top20_idx])

    # anomaly_type column
    top_df["anomaly_type"] = top_df.apply(_classify_anomaly_type, axis=1)

    # reason column — needs cluster column already present
    top_df["reason"] = top_df.apply(
        lambda row: _build_reason(row, cluster_means), axis=1
    )

    out_table = f"{TABLES_DIR}/anomaly_top20.csv"
    top_df.to_csv(out_table)
    print(f"  {out_table}")

    # k-means vs GMM overlap summary table
    only_kmeans = top20_kmeans - top20_gmm
    only_gmm    = top20_gmm    - top20_kmeans
    both        = top20_kmeans & top20_gmm

    kg_rows = []
    for idx in sorted(top20_kmeans | top20_gmm):
        in_km  = idx in top20_kmeans
        in_gmm = idx in top20_gmm
        kg_rows.append({
            "original_index": idx,
            "in_kmeans_top20": in_km,
            "in_gmm_top20":    in_gmm,
            "in_both":         in_km and in_gmm,
            "score_kmeans":    round(float(scores_kmeans[idx]), 3),
            "score_gmm":       round(float(scores_gmm[idx]),    3),
        })
    out_kg = f"{TABLES_DIR}/anomaly_kmeans_vs_gmm.csv"
    pd.DataFrame(kg_rows).to_csv(out_kg, index=False)
    print(f"  {out_kg}")

    #Figure: k-means vs GMM score scatter (all points) 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(scores_kmeans, scores_gmm, alpha=0.15, s=4, color="steelblue",
               label="all points")
    # Highlight the 3 groups
    both_arr = np.array(sorted(both))
    ok_arr   = np.array(sorted(only_kmeans))
    og_arr   = np.array(sorted(only_gmm))
    if len(both_arr):
        ax.scatter(scores_kmeans[both_arr], scores_gmm[both_arr],
                   color="red", s=60, zorder=5, label=f"both top-20 ({len(both_arr)})")
    if len(ok_arr):
        ax.scatter(scores_kmeans[ok_arr], scores_gmm[ok_arr],
                   color="orange", s=60, zorder=4,
                   label=f"k-means only ({len(ok_arr)})", marker="^")
    if len(og_arr):
        ax.scatter(scores_kmeans[og_arr], scores_gmm[og_arr],
                   color="green", s=60, zorder=4,
                   label=f"GMM only ({len(og_arr)})", marker="s")
    ax.set_xlabel("k-means anomaly score (norm. centroid dist.)")
    ax.set_ylabel("GMM anomaly score (neg. log-likelihood)")
    ax.set_title(
        f"E1: k-means vs GMM anomaly scores (k={main_k})\n"
        f"top-20 overlap = {overlap_kg}/20"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_fig_kg = f"{FIGURES_DIR}/anomaly_kmeans_vs_gmm.png"
    fig.savefig(out_fig_kg, dpi=150)
    plt.close(fig)
    print(f"  {out_fig_kg}")

    # Figure: anomaly_type breakdown bar chart
    type_counts = top_df["anomaly_type"].value_counts()
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    bars = ax2.bar(type_counts.index, type_counts.values,
                   color=["tomato" if t == "data_quality" else "steelblue"
                          for t in type_counts.index],
                   edgecolor="white")
    for bar, val in zip(bars, type_counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Count (out of 20)")
    ax2.set_title(f"E1: Anomaly type breakdown (k={main_k})")
    ax2.set_ylim(0, 22)
    fig2.tight_layout()
    out_fig_type = f"{FIGURES_DIR}/anomaly_type_breakdown.png"
    fig2.savefig(out_fig_type, dpi=150)
    plt.close(fig2)
    print(f"  {out_fig_type}")

    # Sensitivity: robust vs standard scaling rank correlation
    X_std, _ = preprocess_data(raw_df, feature_set=MAIN_FS, scaler="standard")
    labels_std = load_labels("kmeans", main_k, 0, MAIN_FS, "standard", LABELS_DIR)
    centroids_std = np.array([
        X_std[labels_std == c].mean(axis=0)
        for c in sorted(np.unique(labels_std))
    ])
    scores_std = kmeans_anomaly_scores(X_std, labels_std, centroids_std)

    n = min(len(scores_kmeans), len(scores_std))
    corr, _ = spearmanr(scores_kmeans[:n], scores_std[:n])

    rank_rob = np.argsort(np.argsort(-scores_kmeans[:n]))
    rank_std = np.argsort(np.argsort(-scores_std[:n]))

    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.scatter(rank_rob[:500], rank_std[:500], alpha=0.3, s=5)
    ax3.set_xlabel("Anomaly rank — RobustScaler (R0)")
    ax3.set_ylabel("Anomaly rank — StandardScaler")
    ax3.set_title("E1: Anomaly rank sensitivity (top 500 shown)")
    ax3.text(0.05, 0.95, f"Spearman r = {corr:.3f}",
             transform=ax3.transAxes, va="top")
    fig3.tight_layout()
    out_fig_sens = f"{FIGURES_DIR}/anomaly_sensitivity.png"
    fig3.savefig(out_fig_sens, dpi=150)
    plt.close(fig3)
    print(f"  {out_fig_sens}")

    # Stability: top-20 overlap across seeds 0/1/2
    seed_top20 = {}
    for seed in [0, 1, 2]:
        try:
            lbl = load_labels("kmeans", main_k, seed, MAIN_FS, MAIN_SCALER, LABELS_DIR)
            cen = np.array([X_rob[lbl == c].mean(axis=0)
                            for c in sorted(np.unique(lbl))])
            sc  = kmeans_anomaly_scores(X_rob, lbl, cen)
            seed_top20[seed] = set(top_anomalies(sc, n=20).tolist())
        except FileNotFoundError:
            pass
    if len(seed_top20) >= 2:
        seeds_list = sorted(seed_top20)
        overlap_rows = []
        for sa, sb in combinations(seeds_list, 2):
            inter = len(seed_top20[sa] & seed_top20[sb])
            overlap_rows.append({"seed_a": sa, "seed_b": sb,
                                  "top20_overlap": inter})
        out_stab = f"{TABLES_DIR}/anomaly_stability.csv"
        pd.DataFrame(overlap_rows).to_csv(out_stab, index=False)
        print(f"  {out_stab}")


def generate_ikmeans_bootstrap_stability(raw_df: pd.DataFrame, main_k: int, n_boot: int = 10, frac: float = 0.8, seed: int = 0) -> None:

    X, _ = preprocess_data(raw_df, feature_set=MAIN_FS, scaler=MAIN_SCALER)
    n_total = len(X)
    subsample_size = int(n_total * frac)

    rng = np.random.default_rng(seed)

    run_indices = []   # original row indices for each boot
    run_labels  = []   # cluster labels aligned to run_indices
    metric_rows = []

    print(f"  iK-means bootstrap: k={main_k}, n_boot={n_boot}, "
          f"frac={frac}, subsample_size={subsample_size}")

    for b in range(n_boot):
        idx = rng.choice(n_total, size=subsample_size, replace=False)
        idx_sorted = np.sort(idx)
        X_boot = X[idx_sorted]

        try:
            labels_boot, runtime, _ = run_ikmeans(X_boot, main_k)
        except ValueError as exc:
            print(f"    boot {b}: SKIP — {exc}")
            continue

        metrics = evaluate_clustering(X_boot, labels_boot)
        run_indices.append(idx_sorted)
        run_labels.append(labels_boot)

        metric_rows.append({
            "boot_run": b,
            "n_points": subsample_size,
            "runtime": round(runtime, 4),
            **{k: round(v, 4) for k, v in metrics.items()},
        })
        print(f"    boot {b}: sil={metrics['silhouette']:.4f}  "
              f"db={metrics['davies_bouldin']:.4f}  "
              f"ch={metrics['calinski_harabasz']:.4f}  "
              f"runtime={runtime:.2f}s")

    # Save per-run metrics table
    metrics_df = pd.DataFrame(metric_rows)
    out_metrics = f"{TABLES_DIR}/ikmeans_bootstrap_stability.csv"
    metrics_df.to_csv(out_metrics, index=False)
    print(f"  {out_metrics}")

    # Pairwise ARI on shared indices
    ari_rows = []
    n_runs = len(run_indices)
    for i, j in combinations(range(n_runs), 2):
        shared = np.intersect1d(run_indices[i], run_indices[j])
        if len(shared) < 2:
            continue
        # Map shared global indices back to positions in each boot subsample
        pos_i = np.searchsorted(run_indices[i], shared)
        pos_j = np.searchsorted(run_indices[j], shared)
        lbl_i = run_labels[i][pos_i]
        lbl_j = run_labels[j][pos_j]
        ari = compute_ari(lbl_i, lbl_j)
        ari_rows.append({
            "run_a": i, "run_b": j,
            "n_shared": len(shared),
            "ari": round(float(ari), 4) if not np.isnan(ari) else float("nan"),
        })

    ari_df = pd.DataFrame(ari_rows)
    out_ari = f"{TABLES_DIR}/ikmeans_bootstrap_ari.csv"
    ari_df.to_csv(out_ari, index=False)
    print(f"  {out_ari}")

    if not ari_rows:
        print("  WARNING: no valid ARI pairs computed (too few successful boot runs).")
        return

    valid_aris = [r["ari"] for r in ari_rows if not np.isnan(r["ari"])]
    print(f"  ARI across {len(valid_aris)} pairs: "
          f"mean={np.mean(valid_aris):.4f}  std={np.std(valid_aris):.4f}  "
          f"min={np.min(valid_aris):.4f}  max={np.max(valid_aris):.4f}")

    # Figure: box-plots of internal indices across bootstrap runs
    if len(metrics_df) < 2:
        print("  WARNING: fewer than 2 successful runs — skipping figure.")
        return

    metrics_to_plot = [
        ("silhouette",        "Silhouette (higher is better)",        "steelblue"),
        ("davies_bouldin",    "Davies-Bouldin (lower is better)",     "tomato"),
        ("calinski_harabasz", "Calinski-Harabasz (higher is better)", "seagreen"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, ylabel, color) in zip(axes, metrics_to_plot):
        vals = metrics_df[col].dropna().tolist()
        bp = ax.boxplot(vals, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.6)
        ax.scatter([1] * len(vals), vals, color=color, alpha=0.8,
                   zorder=3, s=40, edgecolors="black", linewidths=0.5)
        mean_v = float(np.mean(vals))
        std_v  = float(np.std(vals))
        ax.set_title(
            f"iK-means bootstrap (k={main_k})\n"
            f"{ylabel}\nmean={mean_v:.4f} +/- std={std_v:.4f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_ylabel(ylabel, fontsize=9)

    fig.suptitle(
        f"iK-means Bootstrap Stability  |  k={main_k}  |  "
        f"{n_boot} runs x {int(frac*100)}% subsample  "
        f"(seed={seed}, fs={MAIN_FS}, scaler={MAIN_SCALER})",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out_fig = f"{FIGURES_DIR}/ikmeans_bootstrap_metrics.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_fig}")

    #Figure: ARI distribution across pairs
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(valid_aris, bins=min(10, len(valid_aris)),
             color="steelblue", edgecolor="white", linewidth=0.6)
    mean_ari = float(np.mean(valid_aris))
    std_ari  = float(np.std(valid_aris))
    ax2.axvline(mean_ari, color="red", linestyle="--", linewidth=1.5,
                label=f"mean={mean_ari:.3f}")
    ax2.axvline(mean_ari - std_ari, color="orange", linestyle=":", linewidth=1.2,
                label=f"mean+/-std=[{mean_ari-std_ari:.3f}, {mean_ari+std_ari:.3f}]")
    ax2.axvline(mean_ari + std_ari, color="orange", linestyle=":", linewidth=1.2)
    ax2.set_xlabel("Pairwise ARI (on shared indices)")
    ax2.set_ylabel("Count")
    ax2.set_title(
        f"iK-means Bootstrap ARI Distribution\n"
        f"k={main_k}, {n_boot} runs x {int(frac*100)}% subsample "
        f"({len(valid_aris)} pairs)"
    )
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    out_fig2 = f"{FIGURES_DIR}/ikmeans_bootstrap_ari.png"
    fig2.savefig(out_fig2, dpi=150)
    plt.close(fig2)
    print(f"  {out_fig2}")


def main(main_k: int = 4) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    raw_df = _load_data()
    exp = pd.read_csv(EXPERIMENTS_PATH)

    generate_stability_metrics(exp)
    generate_seed_ari()
    generate_controlled_comparison_ari()

    print(f"Generating cluster profiles (K={main_k})...")
    generate_profiles(raw_df, main_k)

    generate_gmm_aic_bic(raw_df)

    generate_anomaly_analysis(raw_df, main_k)

    generate_ikmeans_bootstrap_stability(raw_df, main_k)

    print(f"\nDone. Check {FIGURES_DIR}/ and {TABLES_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-k", type=int, default=4,
                        help="K value to use for cluster profiles and anomaly analysis")
    args = parser.parse_args()
    main(main_k=args.main_k)