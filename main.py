# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python UL
#     language: python
#     name: hotel-booking
# ---

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.max_columns", 100)

# %%
df = pd.read_csv("data/hotel_bookings_course_release_v1.csv")
df.head()

# %%
n, d_total = df.shape
n, d_total

# %% [markdown]
# ## Quick EDA

# %%
## Change the columns and parameters used

# Two univariate views (choose any two features)
f1, f2 = X_raw.columns[0], X_raw.columns[1]

plt.figure()
plt.hist(X_raw[f1].values, bins=25)
plt.title(f"Histogram — {f1}")
plt.xlabel(f1); plt.ylabel("count")
plt.show()

plt.figure()
plt.hist(X_raw[f2].values, bins=25)
plt.title(f"Histogram — {f2}")
plt.xlabel(f2); plt.ylabel("count")
plt.show()

# One 2‑D view (choose any two features)
xcol, ycol = X_raw.columns[0], X_raw.columns[9]
plt.figure()
plt.scatter(X_raw[xcol], X_raw[ycol], s=25)
plt.xlabel(xcol); plt.ylabel(ycol)
plt.title(f"Scatter — {xcol} vs {ycol}")
plt.show()

print("Ranges (min/max):")
display(pd.DataFrame({"min": X_raw.min(), "max": X_raw.max()}))


# %% [markdown]
# ## Feature Engineering

# %%
# na
# Clamp reservation dates to seasons or months for example

# %% [markdown]
# ## Data quality snapshot
#
# Before modeling, produce a **data quality snapshot**:
# - Missing values per column
# - #unique values per column (to detect ID-like / high-cardinality fields)
# - Duplicate rows (exact duplicates)
# - Very large/small values (sanity checks)

# %%
def data_quality_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    snap = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=False),                 # NaN counts as a distinct value in n_unique (dropna=False).
        "pct_unique": (df.nunique(dropna=False) / n * 100).round(2),
    }).sort_values(["pct_missing", "pct_unique"], ascending=False)
    return snap

snap = data_quality_snapshot(df)
snap

# %%
# Check exact duplicate rows (rare here, but important in general)
n_dups = df.duplicated().sum()
n_dups


# %%
# Check for outliers somewhere. Range of high and low values with median

# %% [markdown]
# ## Sampling 
#
# Compare statistics for the full dataset and a sample of n size (e.g., n=50).

# %%
def summarize_numeric(df_num: pd.DataFrame, cols):
    return pd.DataFrame({
        "mean": df_num[cols].mean(),
        "std": df_num[cols].std(ddof=0),
        "min": df_num[cols].min(),
        "max": df_num[cols].max(),
    })

cols_demo = ["mean radius", "mean texture", "mean area", "mean concavity"]

rng = np.random.default_rng(123)
idx50 = rng.choice(X_raw.index, size=50, replace=False)
idx10 = rng.choice(X_raw.index, size=10, replace=False)

summ_full = summarize_numeric(X_raw, cols_demo)
summ_50 = summarize_numeric(X_raw.loc[idx50], cols_demo)
summ_10 = summarize_numeric(X_raw.loc[idx10], cols_demo)

summ_full.join(summ_50, lsuffix="_full", rsuffix="_n50").join(summ_10.add_suffix("_n10"))

# %%
# Visual check: distribution of a variable in full vs sample
var = "mean radius"

plt.figure()
plt.hist(X_raw[var], bins=30, alpha=0.6, label="full")
plt.hist(X_raw.loc[idx50, var], bins=15, alpha=0.6, label="n=50")
plt.hist(X_raw.loc[idx10, var], bins=10, alpha=0.6, label="n=10")
plt.xlabel(var)
plt.ylabel("count")
plt.title("Sampling: full vs subsamples")
plt.legend()
plt.show()

# %% [markdown]
# Compute proximity matrices on small tables so you can interpret the numbers.

# %%
rng = np.random.default_rng(42)
idx_small = rng.choice(df_mixed.index, size=8, replace=False)

cols_small_numeric = ["mean radius", "mean texture", "mean area"]
cols_small_mixed = cols_small_numeric + ["radius_q", "clinic", "high_concavity"]

df_small = df_mixed.loc[idx_small, cols_small_mixed].copy()
df_small


# %%
# Efficient version
def pairwise_l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    a2 = np.sum(X * X, axis=1)[:, None]
    D2 = a2 + a2.T - 2.0 * (X @ X.T)
    D2[D2 < 0] = 0.0  # numerical guard
    return np.sqrt(D2)

def pairwise_l1(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    # Broadcasting: (n,1,p) - (1,n,p) -> (n,n,p)
    return np.sum(np.abs(X[:, None, :] - X[None, :, :]), axis=2)

def pairwise_cosine_distance(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, eps)
    S = Xn @ Xn.T  # cosine similarity
    S = np.clip(S, -1.0, 1.0)
    return 1.0 - S

def knn_indices(D: np.ndarray, k: int) -> np.ndarray:
    """Return indices of k nearest neighbors for each point (excluding self)."""
    n = D.shape[0]
    idx = np.argsort(D, axis=1)
    # drop self (distance 0 at idx[:,0]) then take k
    return idx[:, 1:k+1]

###############################################################################
# quick smoke test
X = X_raw.values
D_l2 = pairwise_l2(X)
D_l1 = pairwise_l1(X)
D_cos = pairwise_cosine_distance(X)

print(D_l2.shape, D_l1.shape, D_cos.shape, "all finite:", np.isfinite(D_l2).all())

## Inefficient version
absX_num = df_small[cols_small_numeric].to_numpy(dtype=float)

def minkowski_distance(x: np.ndarray, y: np.ndarray, p: float) -> float:
    if p == np.inf:
        return np.max(np.abs(x - y))
    return np.sum(np.abs(x - y) ** p) ** (1.0 / p)

def pairwise_distance_matrix(X: np.ndarray, metric: str = "l2", p: float = 2.0) -> np.ndarray:
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            if metric == "minkowski":
                dij = minkowski_distance(X[i], X[j], p=p)
            elif metric == "l1":
                dij = minkowski_distance(X[i], X[j], p=1.0) # manhattan
            elif metric == "l2":
                dij = minkowski_distance(X[i], X[j], p=2.0) # euclidian
            elif metric == "linf":
                dij = minkowski_distance(X[i], X[j], p=np.inf) # chebyshev
            else:
                raise ValueError("Unknown metric")
            D[i, j] = D[j, i] = dij
    return D

D_l1 = pairwise_distance_matrix(X_num, metric="l1")
D_l2 = pairwise_distance_matrix(X_num, metric="l2")
D_linf = pairwise_distance_matrix(X_num, metric="linf")

pd.DataFrame(D_l2, index=df_small.index, columns=df_small.index)

# %% [markdown]
# ## Data Preprocessing 
# - Scaling: why Euclidean distance can be meaningless without it
# - Handling outliers (robust scaling or trimming + z-score, for example)
# - Handling missing data (“Drop rows” often changes geometry the most because it removes objects entirely, so neighborhoods are defined on a different set of points. Between mean vs median imputation, differences usually appear when the variable is skewed/outlier-prone (median is more robust).
# - Discretizaion: ordinal variables and ordinal distances (ordinal encoding?)
# - Nominal attributes and one-hot encoding. Avoid one-hot encoding many categorical variables as this can create a very high-dimensional matrix with mostly zeros (extreme sparsity).

# %% [markdown]
# ### Scaling Pipelines

# %%
# TODO: Missing MeanRange Scaler (needs manual implementation)

scalers = {
    "raw": None,
    "standard": StandardScaler(),
    "robust": RobustScaler(),
    "minmax": MinMaxScaler(),
}

def apply_scaler(Xdf: pd.DataFrame, scaler):
    if scaler is None:
        return Xdf.values.copy()
    return scaler.fit_transform(Xdf.values)

X_scaled = {name: apply_scaler(X_raw, sc) for name, sc in scalers.items()}

# choose k for neighbor sets
KNN_K = 10

def mean_jaccard(idxA: np.ndarray, idxB: np.ndarray) -> float:
    js = []
    for a, b in zip(idxA, idxB):
        A, B = set(a), set(b)
        inter = len(A & B)
        union = len(A | B)
        js.append(inter / union if union else 1.0)
    return float(np.mean(js))

DISTANCES = {
    "L2": pairwise_l2,
    "L1": pairwise_l1,
    "cosine": pairwise_cosine_distance,
}

reps = list(X_scaled.keys())
metrics = list(DISTANCES.keys())

# Pre-compute kNN index sets for every (representation, distance)
knn_cache = {}
for metric, D_fn in DISTANCES.items():
    for rep in reps:
        D = D_fn(X_scaled[rep])
        knn_cache[(rep, metric)] = knn_indices(D, KNN_K)

# ---------------------------------------------------------------------
# (A) Representation sensitivity (for each distance): 
#     stability = average overlap vs the OTHER representations
# ---------------------------------------------------------------------
rows = []
for metric in metrics:
    for rep in reps:
        overlaps = []
        for rep2 in reps:
            if rep2 == rep:
                continue
            overlaps.append(mean_jaccard(knn_cache[(rep, metric)], knn_cache[(rep2, metric)]))
        rows.append({
            "distance": metric,
            "rep": rep,
            "knn_k": KNN_K,
            "mean_jaccard_vs_other_reps": float(np.mean(overlaps)),
        })

df_knn_rep_stability = pd.DataFrame(rows)

print("Representation sensitivity (per distance): higher = more similar neighborhoods across representations")
display(df_knn_rep_stability.sort_values(["distance", "mean_jaccard_vs_other_reps"], ascending=[True, False]))

# One-number summary per representation (average across distances)
df_knn_rep_overall = (
    df_knn_rep_stability
    .groupby("rep", as_index=False)["mean_jaccard_vs_other_reps"]
    .mean()
    .rename(columns={"mean_jaccard_vs_other_reps": "mean_jaccard_avg_across_distances"})
)

print("Overall neighborhood stability per representation (averaged across distances):")
display(df_knn_rep_overall.sort_values("mean_jaccard_avg_across_distances", ascending=False))

# Optional: show a full overlap matrix for one distance (L2)
def overlap_matrix_for_metric(metric: str) -> pd.DataFrame:
    M = np.zeros((len(reps), len(reps)))
    for i, r1 in enumerate(reps):
        for j, r2 in enumerate(reps):
            if r1 == r2:
                M[i, j] = 1.0
            else:
                M[i, j] = mean_jaccard(knn_cache[(r1, metric)], knn_cache[(r2, metric)])
    return pd.DataFrame(M, index=reps, columns=reps)

print("Pairwise representation overlap matrix (example: L2)")
display(overlap_matrix_for_metric("L2"))

# ---------------------------------------------------------------------
# (B) Distance sensitivity at FIXED representation (one example):
#     compare how changing L2/L1/cosine changes neighborhoods
# ---------------------------------------------------------------------
REP_FOR_DISTANCE_COMPARE = "standard"  # change if you want (e.g., try "raw")
rows = []
for metric in metrics:
    overlaps = []
    for metric2 in metrics:
        if metric2 == metric:
            continue
        overlaps.append(
            mean_jaccard(
                knn_cache[(REP_FOR_DISTANCE_COMPARE, metric)],
                knn_cache[(REP_FOR_DISTANCE_COMPARE, metric2)],
            )
        )
    rows.append({
        "rep": REP_FOR_DISTANCE_COMPARE,
        "distance": metric,
        "knn_k": KNN_K,
        "mean_jaccard_vs_other_distances": float(np.mean(overlaps)),
    })

df_knn_dist_stability = pd.DataFrame(rows)

print(f"Distance sensitivity at fixed representation = {REP_FOR_DISTANCE_COMPARE}:")
display(df_knn_dist_stability.sort_values("mean_jaccard_vs_other_distances", ascending=False))

def overlap_matrix_distances(rep: str) -> pd.DataFrame:
    M = np.zeros((len(metrics), len(metrics)))
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            if m1 == m2:
                M[i, j] = 1.0
            else:
                M[i, j] = mean_jaccard(knn_cache[(rep, m1)], knn_cache[(rep, m2)])
    return pd.DataFrame(M, index=metrics, columns=metrics)

display(overlap_matrix_distances(REP_FOR_DISTANCE_COMPARE))

