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
# # Lab Tutorial — T2 Data Foundations
#
# This tutorial is **about T2 concepts** (data objects/attributes, measurement scales, data quality, preprocessing choices, and proximity measures).
#
#
# **Dataset:** *Wisconsin Diagnostic Breast Cancer (WDBC)* — originally from the **UCI Machine Learning Repository** and available locally via `scikit-learn`.
# It is a **record (table) dataset** with:
# - objects = tumor samples (patients)
# - attributes = numeric measurements computed from digitized images
# - an **outcome/label** column (`target`) that we treat as *post‑hoc profiling only* (do not use it to define similarity).
#
# We use **small subsamples** (e.g., *n = 6–10 objects*) so you can propely analyse what distances, scaling, missingness, and encoding do.
#
# ---
#
# ## Learning outcomes
#
# 1. Identify **objects** and **attributes**, and assign **measurement scales** (nominal/ordinal/interval/ratio).
# 2. Produce a **data quality snapshot** (missingness, duplicates, suspicious ranges, high-cardinality / ID-like columns).
# 3. Explain how **representation defines geometry**: encoding + scaling + missing-value handling changes proximity.
# 4. Compute and compare **proximity measures** (Euclidean/Manhattan/Chebyshev, cosine, correlation; SMC vs Jaccard; mixed-type distance).
# 5. Diagnose **high-dimensional effects** (distance concentration, sparsity).
# 6. Use **sampling** to make analysis computationally and cognitively manageable.
#
# > **Answer** the questions bellow in Markdown cells.  
#
#

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.datasets import load_breast_cancer

np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.max_columns", 100)

# %% [markdown]
# ## 1) Load the dataset (record data)
#
# We use `scikit-learn` to load WDBC locally (no download needed).

# %%
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()  # includes features + target
df.head()

# %%
n, d_total = df.shape
n, d_total

# %% [markdown]
# Q1 (Objects & attributes).**  
# a) What is the **data object** here?  
# b) What is an **attribute**? Give 2 examples from the dataset.  
# c) What are `n` and `d` for this dataset? What is included in `d_total`?
#
# *Write your answer below.*
#
# ### Solution
#
# a) **Object**: one tumor (breast mass) observation / patient sample (one row).
#
# b) **Attribute**: one measured property/feature describing the object (one column), e.g. `mean radius`, `mean texture`, `mean area`.
#
# c) Here:
# - **n = 569** objects (rows).
# - **d = 30** measured features.
# - `d_total = 31` because the loaded frame also includes `target` (a label/outcome column).

# %% [markdown]
# ### Separate “inputs” from “outcome” (leakage awareness)
#
# The column `target` is a label (malignant/benign). In unsupervised settings, treat it like an *outcome/post-event* field:
# - do not use it to define similarity/distances,
# - you may use it later to profile/interpret patterns.
#
# Create:
# - `X_raw`: attributes used to define proximity (features only)
# - `y`: outcome/label for post-hoc inspection

# %%
X_raw = df.drop(columns=["target"]).copy()
y = df["target"].copy()

X_raw.shape, y.shape

# %% [markdown]
# **Q2 **  
# Why is it a mistake to include `target` when computing distances between objects (even if you never train a classifier)?
#
# ### Solution
#
# Including `target` in distances is **information leakage**:
# - It bakes the “potencial answer” (benign vs malignant) into the geometry.
# - Objects with the same label become artificially close even if their measured attributes differ.
# - Any later “pattern discovery” becomes circular because similarity was defined using the outcome.
#
# The `target` variable should be used only after proximity is defined (post‑hoc profiling/interpretation).

# %% [markdown]
# ## 2) Attribute types & measurement scales (nominal / ordinal / interval / ratio)
#
# All original features in WDBC are numeric, but **“numeric-looking” does not automatically mean ratio/interval**.
# For this lab, assume these measurements behave like **ratio-scale** variables (i.e. differences and ratios are meaningful).
#
# We create **derived attributes**:
# - **Ordinal**: discretize a continuous variable into ordered bins (quartiles).
# - **Nominal**: create categories with no ordering (synthetic “clinic site”).
# - **Binary**: create a presence/absence indicator (asymmetric example).
#
# These derived attributes are for didactic purposes, only.

# %%
# Create a small derived mixed-type version of the dataset
df_mixed = X_raw.copy()

# Ordinal attribute from a continuous measurement (quartiles)
ord_levels = ["Q1_small","Q2","Q3","Q4_large"]
df_mixed["radius_q"] = pd.qcut(df_mixed["mean radius"], q=4, labels=ord_levels)
df_mixed["radius_q"] = df_mixed["radius_q"].astype(pd.CategoricalDtype(categories=ord_levels, ordered=True))

# Nominal attribute (synthetic "clinic") 
df_mixed["clinic"] = pd.Series(np.mod(np.arange(len(df_mixed)), 5), index=df_mixed.index).map(
    {0:"A", 1:"B", 2:"C", 3:"D", 4:"E"}
)

# Asymmetric binary: "very high concavity present?"
# 1 = presence is informative, 0 = absence less informative
threshold = df_mixed["mean concavity"].quantile(0.90)
df_mixed["high_concavity"] = (df_mixed["mean concavity"] >= threshold).astype(int)

df_mixed[["mean radius","radius_q","clinic","mean concavity","high_concavity"]].head()


# %% [markdown]
# **Q3 (Measurement scales).**  
# For each of the following attributes, state a reasonable **type** (nominal/ordinal/interval/ratio) and justify briefly:
#
# - `mean radius`
# - `radius_q`
# - `clinic`
# - `high_concavity`
#
# ### Solution
#
# - `mean radius`: **ratio** (physical measurement with meaningful differences and a meaningful zero).
# - `radius_q`: **ordinal** (ordered bins; Q4 > Q3 > Q2 > Q1, but gaps are not guaranteed equal).
# - `clinic`: **nominal** (A–E are labels; no inherent order).
# - `high_concavity`: **binary**; typically treated as **asymmetric binary** if 1 = “rare presence” is informative and 0 = “absence” is less informative.

# %% [markdown]
# ## 3) Data quality snapshot (missingness, duplicates, high-cardinality)
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

snap = data_quality_snapshot(df_mixed[["radius_q","clinic","high_concavity"] + list(X_raw.columns[:6])])
snap

# %%
# Check exact duplicate rows (rare here, but important in general)
n_dups = df_mixed.duplicated().sum()
n_dups

# %% [markdown]
#  **Q4 (Quality).**  
# a) In the snapshot table above, which columns have the highest **% unique** values?  
# b) Why can an *ID-like* high-cardinality column be dangerous if you one-hot encode it?
#
# ### Solution
#
# a) The highest **% unique** columns are typically the **continuous numeric** ones (e.g., `mean area`, `mean texture`, …) because many rows have distinct real values.
#
# b) Problems related to high‑cardinality **ID-like** columns:
# - one‑hot encoding turns it into *many* sparse columns (almost all zeros),
# - each object becomes “unique” due to its ID,
# - distances/neighborhoods become dominated by ID matches instead of meaningful attributes.

# %% [markdown]
# ## 4) Working with small subsamples (distances are tangible)
#
# Take a **small subsample** and a **small set of attributes**.
#
# Compute proximity matrices on these small tables so you can interpret the numbers.

# %%
rng = np.random.default_rng(42)
idx_small = rng.choice(df_mixed.index, size=8, replace=False)

cols_small_numeric = ["mean radius", "mean texture", "mean area"]
cols_small_mixed = cols_small_numeric + ["radius_q", "clinic", "high_concavity"]

df_small = df_mixed.loc[idx_small, cols_small_mixed].copy()
df_small

# %% [markdown]
# **Q5 (Interpretation).**  
# Pick two objects (two rows) from `df_small`.  
# a) Which attributes make them look “similar”?  
# b) Which attributes make them look “different”?  
# Answer in plain language (no formulas yet).
#
# ### Solution (example)
#
# Example pair that looks **similar**: objects `396` and `486`.
# - numeric values are close-ish (`mean radius`, `mean texture`, `mean area`)
# - both have `radius_q = Q3`, `clinic = B`, and `high_concavity = 0`
#
# Example pair that looks **different**: objects `245` and `369`.
# - `mean area` and `mean radius` differ a lot
# - `radius_q` differs (Q1_small vs Q4_large)
# - `high_concavity` differs (0 vs 1)

# %% [markdown]
# ## 5) Numeric proximity: Minkowski distances (L1, L2, L∞)
#
# For numeric features, a common family is **Minkowski distance** with parameter `p`:
# - `p=1` → Manhattan (L1)
# - `p=2` → Euclidean (L2)
# - `p→∞` → Chebyshev (L∞)
#
# We will compute pairwise distance matrices on the small sample using only the numeric columns.

# %%
X_num = df_small[cols_small_numeric].to_numpy(dtype=float)

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
                dij = minkowski_distance(X[i], X[j], p=1.0)
            elif metric == "l2":
                dij = minkowski_distance(X[i], X[j], p=2.0)
            elif metric == "linf":
                dij = minkowski_distance(X[i], X[j], p=np.inf)
            else:
                raise ValueError("Unknown metric")
            D[i, j] = D[j, i] = dij
    return D

D_l1 = pairwise_distance_matrix(X_num, metric="l1")
D_l2 = pairwise_distance_matrix(X_num, metric="l2")
D_linf = pairwise_distance_matrix(X_num, metric="linf")

pd.DataFrame(D_l2, index=df_small.index, columns=df_small.index)


# %% [markdown]
# **Q6 (By hand).**  
# Choose two rows `i` and `j` from the distance matrix above.  
# Compute the **Euclidean distance** by hand using the three numeric attributes and confirm it matches the matrix value.
#
# Show your calculation (at least intermediate steps).
#
# ### Solution (one worked example)
#
# Take objects **48** and **396** using the three numeric attributes.
#
# - For 48: `(12.05, 14.63, 449.3)`
# - For 396: `(13.51, 18.89, 558.1)`
#
# Differences:
# - Δradius = 13.51 − 12.05 = **1.46**
# - Δtexture = 18.89 − 14.63 = **4.26**
# - Δarea = 558.1 − 449.3 = **108.8**
#
# Euclidean distance:
#
# \[
# d = \sqrt{1.46^2 + 4.26^2 + 108.8^2}
# \]
#
# Compute:
# - 1.46² = 2.1316
# - 4.26² = 18.1476
# - 108.8² = 11837.44
#
# Sum = 11857.7192 → √sum ≈ **108.893** (matches the matrix).

# %% [markdown]
# **Q7 (Effect of p).**  
# Compare L1 vs L2 vs L∞ on the same pair `(i,j)` used in Q6.
# - Which one is largest? smallest?
# - What is the *interpretation* of L∞ in plain language?
#
# ### Solution
#
# For the same pair (48, 396):
#
# - **L1** = |1.46| + |4.26| + |108.8| = **114.52**
# - **L2** ≈ **108.893**
# - **L∞** = max(|1.46|, |4.26|, |108.8|) = **108.8**
#
# So: **L1 is largest**, **L∞ is smallest** here.
#
# **Interpretation of L∞**: “How far apart are the two objects on their **single most different** attribute?” (here: `mean area`).

# %% [markdown]
# ## 6) Scaling: why Euclidean distance can be meaningless without it
#
# Euclidean distance assumes variables are comparable in scale.
# Here we intentionally included `mean area`, which has much larger magnitude than `mean radius` and `mean texture`.
#
# We compare:
# - raw distances
# - z-score standardization
# - robust scaling (median + IQR)
#
# Check how **nearest neighbors** change.

# %%
def zscore_scale(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma

def robust_scale_iqr(X: np.ndarray) -> np.ndarray:
    med = np.median(X, axis=0)
    q1 = np.quantile(X, 0.25, axis=0)
    q3 = np.quantile(X, 0.75, axis=0)
    iqr = np.where((q3 - q1) == 0, 1.0, (q3 - q1))
    return (X - med) / iqr

def k_nearest_neighbors(D: np.ndarray, k: int = 3):
    # For each i, return indices of k nearest neighbors (excluding itself)
    nn = []
    for i in range(D.shape[0]):
        order = np.argsort(D[i])
        order = order[order != i]
        nn.append(order[:k])
    return nn

X_raw_num = X_num
X_z = zscore_scale(X_raw_num)
X_r = robust_scale_iqr(X_raw_num)

D_raw = pairwise_distance_matrix(X_raw_num, metric="l2")
D_z = pairwise_distance_matrix(X_z, metric="l2")
D_r = pairwise_distance_matrix(X_r, metric="l2")

nn_raw = k_nearest_neighbors(D_raw, k=2)
nn_z = k_nearest_neighbors(D_z, k=2)
nn_r = k_nearest_neighbors(D_r, k=2)

def nn_table(df_small, nn_list, title):
    rows = []
    idx = list(df_small.index)
    for i, neigh in enumerate(nn_list):
        rows.append({
            "object": idx[i],
            "nn1": idx[neigh[0]],
            "nn2": idx[neigh[1]],
        })
    out = pd.DataFrame(rows)
    out.index = out["object"]
    out = out.drop(columns=["object"])
    print(title)
    display(out)

nn_table(df_small, nn_raw, "Nearest neighbors with RAW Euclidean")
nn_table(df_small, nn_z, "Nearest neighbors with Z-score Euclidean")
nn_table(df_small, nn_r, "Nearest neighbors with ROBUST(IQR) Euclidean")

# %% [markdown]
# Q8 (Scaling changes neighborhoods).**  
# Pick one object (row). Compare its nearest neighbors under:
# - raw Euclidean
# - z-score Euclidean
# - robust(IQR) Euclidean
#
# a) Did the neighbors change?  
# b) Explain **why** (which variable dominated, which got “fixed”).
#
# ### Solution (one worked example)
#
# Pick object **48**.
#
# Nearest neighbors:
# - Raw Euclidean: **50**, **247**
# - Z‑score Euclidean: **247**, **486**
# - Robust(IQR) Euclidean: **247**, **396**
#
# a) Yes — neighbors change.
#
# b) Explanation: with raw Euclidean, `mean area` dominates because it has a much larger scale.  
# After scaling, all features contribute comparably, so neighborhood structure shifts to reflect multi-feature similarity rather than just the largest-range variable.

# %% [markdown]
# ## 7) Outliers: how a single extreme point can distort scaling
#
# We create an artificial outlier by multiplying `mean area` for one object and observe:
# - how summary statistics change,
# - how scaling changes,
# - how distances/neighborhoods change.

# %%
df_out = df_small.copy()
outlier_row = df_out.index[0]
df_out.loc[outlier_row, "mean area"] *= 5  # artificial outlier

X_out = df_out[cols_small_numeric].to_numpy(float)

# Compare raw vs z-score distances after introducing outlier
D_raw_out = pairwise_distance_matrix(X_out, metric="l2")
D_z_out = pairwise_distance_matrix(zscore_scale(X_out), metric="l2")

pd.DataFrame({
    "before_outlier_mean_area": df_small["mean area"],
    "after_outlier_mean_area": df_out["mean area"],
})

# %% [markdown]
# **Q9 (Outlier impact).**  
# a) What happens to the z-score scaling parameters (mean/std) when you add the outlier?  
# b) Which of the three scalers (none / z-score / robust) would you prefer in the presence of outliers, and why?
#
# ### Solution
#
# a) Adding the outlier mainly affects the **mean** and **standard deviation** of the outlier feature (`mean area`):
# - the mean increases,
# - the standard deviation increases a lot,
# so z-scores for *all* points shift.
#
# b) Prefer **robust scaling** (median + IQR, or median + MAD) when outliers are present because it is less sensitive to extreme values.  
# (Alternative strategies: trimming/winsorization + z-score.)

# %% [markdown]
# ## 8) Missing values: “data imputation alters neighbourhood structure”
#
# The original dataset has no missing values; inject missingness to study its effect.
#
# We will compare:
# - listwise deletion (drop rows with missing)
# - mean imputation
# - median imputation
#
# Then we measure how much the **distance matrix changes**.

# %%
df_miss = df_small.copy()

# Inject missingness in a controlled way
df_miss.loc[df_miss.index[:2], "mean texture"] = np.nan
df_miss.loc[df_miss.index[3], "mean area"] = np.nan

df_miss


# %%
def impute_mean(df_num: pd.DataFrame) -> pd.DataFrame:
    return df_num.fillna(df_num.mean())

def impute_median(df_num: pd.DataFrame) -> pd.DataFrame:
    return df_num.fillna(df_num.median())

num_miss = df_miss[cols_small_numeric]

# Strategy 1: drop rows with missing
num_drop = num_miss.dropna(axis=0)

# Strategy 2: mean imputation
num_mean = impute_mean(num_miss)

# Strategy 3: median imputation
num_med = impute_median(num_miss)

num_drop.shape, num_mean.shape, num_med.shape


# %%
def distance_matrix_for_df(df_num: pd.DataFrame) -> pd.DataFrame:
    X = df_num.to_numpy(float)
    D = pairwise_distance_matrix(X, metric="l2")
    return pd.DataFrame(D, index=df_num.index, columns=df_num.index)

D_mean = distance_matrix_for_df(num_mean)
D_med = distance_matrix_for_df(num_med)

# Compare distance matrices (mean vs median imputation) on the same index set
delta = (D_mean - D_med).abs()
delta

# %% [markdown]
#  **Q10 (Missingness strategy).**  
# a) Which strategy (drop / mean / median) changes the geometry the most? Why?  
# b) If missingness is *structural* (“not applicable”), why can mean-imputation be misleading?
#
# ### Solution
#
# a) “Drop rows” often changes geometry the most because it removes objects entirely, so neighborhoods are defined on a different set of points.
# Between mean vs median imputation, differences usually appear when the variable is skewed/outlier-prone (median is more robust).
#
# b) If missingness is structural (“not applicable”), mean-imputation can be misleading because it inserts a fake typical value and can create artificial similarity (two “not applicable” cases become “average” instead of “missing for a reason”).

# %% [markdown]
# ## 9) Discretization → ordinal variables and ordinal distance
#
# Ordinal attributes have order but differences are not necessarily equal.
# A common approach:
# 1) map ordered levels to ranks `1..M`
# 2) normalize ranks to `[0,1]`
# 3) treat as interval-scale for distance computations
#
# Implement this for `radius_q`.

# %%
# Extract ordinal column for the small sample
ord_series = df_small["radius_q"].astype("category")

# Define an explicit order (already Q1_small < Q2 < Q3 < Q4_large)
ord_levels = ["Q1_small","Q2","Q3","Q4_large"]
ord_series = ord_series.cat.set_categories(ord_levels, ordered=True)

# Map to ranks 1..M then normalize to [0,1]
ranks = ord_series.cat.codes + 1  # Q1_small -> 1, ..., Q4_large -> 4
M = len(ord_levels)
z = (ranks - 1) / (M - 1)

pd.DataFrame({"radius_q": ord_series, "rank": ranks, "z_in_[0,1]": z}, index=df_small.index)

# %% [markdown]
#  **Q11 (Ordinal distance).**  
# Pick two objects with different `radius_q` values.
# a) Compute their normalized ordinal distance `|z_i - z_j|`.  
# b) Explain why this distance is *not the same as* subtracting the original `mean radius`.
#
# ### Solution (example)
#
# Pick objects **245** (Q1_small) and **369** (Q4_large).
#
# Ranks: Q1_small=1, Q4_large=4.  
# Normalize to \[0,1\]: z = (rank−1)/(4−1).
#
# - z_245 = (1−1)/3 = **0**
# - z_369 = (4−1)/3 = **1**
#
# Ordinal distance: |0 − 1| = **1**.
#
# b) This is not the same as subtracting `mean radius` because ordinal bins preserve **order** but not the exact magnitudes or unequal spacing in the original continuous values.

# %% [markdown]
# ## 10) Binary similarity: SMC vs Jaccard (symmetric vs asymmetric)
#
# We use the derived binary variable `high_concavity`:
#
# - If we interpret **1 and 0 as equally informative** → *symmetric binary* → use **SMC**.
# - If we interpret **1 = presence is informative, 0 = absence is uninformative** → *asymmetric binary* → use **Jaccard**.
#
# We compute both on a tiny binary matrix.

# %%
B = df_small[["high_concavity"]].to_numpy(int)  # shape (n,1)

def contingency_counts(x: np.ndarray, y: np.ndarray):
    # x,y are 1D binary vectors (0/1)
    f11 = np.sum((x == 1) & (y == 1))
    f10 = np.sum((x == 1) & (y == 0))
    f01 = np.sum((x == 0) & (y == 1))
    f00 = np.sum((x == 0) & (y == 0))
    return f11, f10, f01, f00

def smc_similarity(x: np.ndarray, y: np.ndarray) -> float:
    f11, f10, f01, f00 = contingency_counts(x, y)
    d = f11 + f10 + f01 + f00
    return (f11 + f00) / d if d else 0.0

def jaccard_similarity(x: np.ndarray, y: np.ndarray) -> float:
    f11, f10, f01, f00 = contingency_counts(x, y)
    denom = f11 + f10 + f01
    return f11 / denom if denom else 0.0

# Show SMC and Jaccard similarities for the first 4 objects
idx = list(df_small.index)
for a in range(4):
    for b in range(a+1, 4):
        x = B[a]
        y_ = B[b]
        print(f"{idx[a]} vs {idx[b]}: SMC={smc_similarity(x,y_):.3f}, Jaccard={jaccard_similarity(x,y_):.3f}, values={x[0]} vs {y_[0]}")

# %% [markdown]
#  **Q12 (Binary interpretation).**  
# a) In what situation is `high_concavity=0` *uninformative*?  
# b) In that situation, why does Jaccard make more sense than SMC?
#
# ### Solution
#
# a) `high_concavity=0` can be “uninformative” if it simply means “not extremely high” — it groups together many heterogeneous cases.
#
# b) In that setting, Jaccard makes more sense because it **ignores 0–0 matches** (shared absence).  
# You don’t want two objects to be considered “similar” just because they both *lack* a rare property.

# %% [markdown]
# ## 11) Nominal attributes and one-hot encoding → sparsity
#
# Nominal attributes have no ordering. Common options:
# - use simple matching on categories directly
# - or one-hot encode, then use a binary similarity/distance (SMC/Jaccard)
#
# Do one-hot encode the attribute `clinic` and inspect sparsity.

# %%
clinic_oh = pd.get_dummies(df_small["clinic"], prefix="clinic", dtype=int)
clinic_oh

# %%
sparsity = 1.0 - clinic_oh.to_numpy().mean()  # fraction of zeros in one-hot
sparsity


# %% [markdown]
#  **Q13 (Sparsity).**  
# a) What is the sparsity of the one-hot representation? Interpret it.  
# b) Why can sparsity become extreme when you one-hot encode many categorical variables?
#
# ### Solution
#
# a) With 5 clinic categories, one-hot has exactly 1 one per row → average value per cell is 1/5 = 0.2.  
# So sparsity = 1 − 0.2 = **0.8** → **80% of entries are zeros**.
#
# b) If one-hot encode many categorical variables (especially with many levels), this creates a very high-dimensional matrix with mostly zeros (extreme sparsity).

# %% [markdown]
# ## 12) Cosine similarity vs Euclidean distance (especially in sparse/high-dimensional spaces)
#
# Cosine similarity compares **angle/shape** (direction) rather than magnitude.
# Euclidean distance compares **absolute magnitude differences**.
#
# Compare cosine similarity on one-hot encoded data vs Euclidean distance.

# %%
def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    num = float(np.dot(x, y))
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return num / den if den else 0.0

OH = clinic_oh.to_numpy(int)

# Cosine similarities between first 4 objects (one-hot)
for a in range(4):
    for b in range(a+1, 4):
        print(f"{idx[a]} vs {idx[b]}: cosine={cosine_similarity(OH[a], OH[b]):.3f}, eucl={minkowski_distance(OH[a], OH[b], p=2):.3f}")

# %% [markdown]
# **Q14 (Cosine vs Euclidean on one-hot).**  
# For one-hot vectors, when do one gets cosine similarity 1? 0?  
# Explain in words.
#
# ### Solution
# - Cosine similarity is **1** when two objects are in the **same category** (identical one-hot vectors).
# - Cosine similarity is **0** when they are in **different categories** (orthogonal vectors → dot product 0).
#
# Cosine becomes a clean “same vs different” check for single one-hot attributes.

# %% [markdown]
# ## 13) Mixed-type dissimilarity
#
# With mixed data you must explicitly design dissimilarities and then aggregate them.
#
# Implement a simple mixed distance on the small sample using:
# - numeric: range-normalized absolute difference
# - ordinal: normalized rank difference
# - nominal: 0 if same category else 1
# - asymmetric binary: Jaccard-style contribution (ignore 0-0 matches)
#
# Compute:
# - overall mixed distance matrix
# - per-feature contributions for one pair

# %%
# Identify feature groups in df_small
num_cols = cols_small_numeric
ord_cols = ["radius_q"]
nom_cols = ["clinic"]
asym_bin_cols = ["high_concavity"]

def gower_like_distance(df: pd.DataFrame,
                        num_cols, ord_cols, nom_cols, asym_bin_cols) -> pd.DataFrame:
    n = len(df)
    idx = df.index

    # Precompute ranges for numeric
    num = df[num_cols].astype(float)
    ranges = (num.max(axis=0) - num.min(axis=0)).replace(0, 1.0)

    # Ordinal: map to z in [0,1] (as earlier)
    ord_df = df[ord_cols].copy()
    z_ord = {}
    for c in ord_cols:
        cat = ord_df[c].astype("category")
        # assume ordered categories already
        codes = cat.cat.codes
        M = len(cat.cat.categories)
        z = (codes) / (M - 1) if M > 1 else codes * 0.0
        z_ord[c] = z.astype(float)
    z_ord = pd.DataFrame(z_ord, index=idx)

    D = np.zeros((n, n), float)

    for i in range(n):
        for j in range(i+1, n):
            contribs = []
            weights = []

            # numeric
            for c in num_cols:
                xi, xj = num.loc[idx[i], c], num.loc[idx[j], c]
                if pd.isna(xi) or pd.isna(xj):
                    continue
                contribs.append(abs(xi - xj) / ranges[c])
                weights.append(1.0)

            # ordinal
            for c in ord_cols:
                xi, xj = z_ord.loc[idx[i], c], z_ord.loc[idx[j], c]
                if pd.isna(xi) or pd.isna(xj):
                    continue
                contribs.append(abs(xi - xj))
                weights.append(1.0)

            # nominal
            for c in nom_cols:
                xi, xj = df.loc[idx[i], c], df.loc[idx[j], c]
                if pd.isna(xi) or pd.isna(xj):
                    continue
                contribs.append(0.0 if xi == xj else 1.0)
                weights.append(1.0)

            # asymmetric binary (Jaccard contribution per feature)
            for c in asym_bin_cols:
                xi, xj = int(df.loc[idx[i], c]), int(df.loc[idx[j], c])
                # ignore 0-0 matches
                if xi == 0 and xj == 0:
                    continue
                contribs.append(0.0 if (xi == 1 and xj == 1) else 1.0)
                weights.append(1.0)

            if len(weights) == 0:
                dij = 0.0
            else:
                dij = float(np.sum(np.array(contribs) * np.array(weights)) / np.sum(weights))

            D[i, j] = D[j, i] = dij

    return pd.DataFrame(D, index=idx, columns=idx)

D_mixed = gower_like_distance(df_small, num_cols, ord_cols, nom_cols, asym_bin_cols)
D_mixed

# %% [markdown]
# **Q15 (Mixed distance understanding).**  
# Pick one pair of objects `(i,j)`:
# a) Which features contributed to making them distant?  
# b) Which features contributed 0 distance?  
# c) Do you think treating `high_concavity` as asymmetric is reasonable here? Why or why not?
#
# ### Solution (example)
#
# a) Features that increase distance:
# - large normalized differences in numeric features (`mean area` often contributes a lot),
# - different ordinal ranks (e.g., Q1 vs Q4 adds 1),
# - different nominal category (clinic mismatch adds 1),
# - asymmetric binary mismatch (0 vs 1 contributes 1, while 0–0 contributes nothing).
#
# b) Features contributing 0:
# - identical categories for `clinic`,
# - same `radius_q`,
# - numeric features that are equal/very close after range normalization,
# - (for asymmetric binary) a shared 1 gives 0; shared 0 is ignored.
#
# c) Treating `high_concavity` as asymmetric is reasonable if “high concavity” is rare and meaningful as a *presence* indicator. If 0 is also meaningfully informative, then SMC-style treatment would be more appropriate.

# %% [markdown]
# ## 14) Correlation vs cosine vs Euclidean: invariances
#
# We reproduce a classic comparison:
# - Cosine is invariant to **scaling**
# - Correlation is invariant to **scaling and translation**
# - Euclidean is invariant to neither
#
# Then you will interpret which one makes sense for different notions of “similarity”.

# %%
x = np.array([1, 2, 4, 3, 0, 0, 0], dtype=float)
y = np.array([1, 2, 3, 4, 0, 0, 0], dtype=float)

y_scaled = y * 2
y_translated = y + 5

def corr(x, y):
    x0 = x - x.mean()
    y0 = y - y.mean()
    den = np.linalg.norm(x0) * np.linalg.norm(y0)
    return float(np.dot(x0, y0) / den) if den else 0.0

def eucl(x, y):
    return float(np.linalg.norm(x - y))

rows = []
for name, yy in [("y", y), ("y_scaled", y_scaled), ("y_translated", y_translated)]:
    rows.append({
        "vector": name,
        "cosine(x,·)": cosine_similarity(x, yy),
        "corr(x,·)": corr(x, yy),
        "eucl(x,·)": eucl(x, yy),
    })

pd.DataFrame(rows)


# %% [markdown]
# **Q16 (Choosing proximity).**  
# a) Give an example where you care about **shape** (pattern) more than magnitude → which measure fits?  
# b) Give an example where **absolute magnitude** matters → which measure fits?  
# c) Explain why “zero correlation does not imply independence” matters for similarity judgments.
#
# ### Solution
#
# a) **Shape/pattern** similarity example: two temperature time series that rise/fall together (regardless of absolute level) → **correlation** (or sometimes cosine on centered series).
#
# b) **Absolute magnitude** example: two locations’ temperatures today in °C where the actual values matter → **Euclidean distance**.
#
# c) “Zero correlation ≠ independence” matters because correlation measures only **linear** association.  
# Two variables can be strongly related nonlinearly (e.g., y = x²) yet have correlation near 0, so correlation-based similarity can miss real dependence.

# %% [markdown]
# ## 15) Dimensionality & distance concentration 
#
# T2 mentions: in high dimensions, distances tend to become less contrastive.
#
# We will simulate random points in `[0,1]^d` for various `d` and observe:
# - distribution of pairwise distances
# - contrast ratio `max(distance) / min(distance)` (or similar)
#
# This is not about clustering — it’s about how geometry changes with `d`.

# %%
def distance_concentration_demo(d_list=(2, 10, 50, 200), n_points=300, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for d in d_list:
        X = rng.random((n_points, d))
        # compute pairwise distances efficiently
        # (n^2) is ok for n=300
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        # take upper triangle excluding diagonal
        tri = D[np.triu_indices(n_points, k=1)]
        results.append({
            "d": d,
            "mean": float(np.mean(tri)),
            "std": float(np.std(tri)),
            "min": float(np.min(tri)),
            "max": float(np.max(tri)),
            "max/min": float(np.max(tri) / np.min(tri)),
            "p95-p05": float(np.quantile(tri, 0.95) - np.quantile(tri, 0.05)),
        })
    return pd.DataFrame(results)

conc = distance_concentration_demo()
conc

# %%
# Add dimensionless contrast measures (quantitative “distance contrast”)
conc = conc.copy()
conc["cv"] = conc["std"] / conc["mean"]
conc["rel_iqr"] = conc["p95-p05"] / conc["mean"]
conc

# Plot: CV (relative spread) vs dimension
plt.figure()
plt.plot(conc["d"], conc["cv"], marker="o")
plt.xscale("log")
plt.xlabel("dimension d (log scale)")
plt.ylabel("CV = std/mean of pairwise L2 distances")
plt.title("Relative distance contrast decreases with dimension")
plt.show()


# %% [markdown]
# **Q17 (High-dimensional — quantify “distance contrast”).**  
#
# You computed the distribution of pairwise Euclidean (L2) distances between random points in \([0,1]^d\), for several values of `d`.
#
# a) **Quantify contrast:** Define a *dimensionless* measure of distance contrast. Use:  
# - **Coefficient of variation:** \(\mathrm{CV} = \dfrac{\mathrm{std}(D)}{\mathrm{mean}(D)}\)  
# and (optionally)  
# - **Relative interquantile range:** \(\dfrac{p95 - p05}{\mathrm{mean}(D)}\).
#
# Compute these quantities for each `d` (add them as columns in `conc`).
#
# b) **Describe the trend:** As `d` increases, what happens to CV (and relative IQR)? State the trend *using the numbers* (not only the plot).
#
# c) **Interpretation (2–3 lines):** Explain why decreasing distance contrast makes **nearest-neighbor reasoning** fragile in high dimension.  
# Hint: if many points are almost equally far, small perturbations (scaling, noise, missingness handling) can reorder neighbors.
#
# d) *(Optional)* Identify the smallest `d` in `d_list` for which \(\mathrm{CV} < 0.05\). What would that imply for “nearest vs. farthest” separation?
#
# ### Solution
#
# a) The key point is to use a *dimensionless* spread measure; the coefficient of variation is standard:
# \[
# \mathrm{CV} = \frac{\mathrm{std}(D)}{\mathrm{mean}(D)}.
# \]
# The code cell above computes `cv = std/mean` and `rel_iqr = (p95-p05)/mean`.
#
# b) Both `cv` and `rel_iqr` **decrease as `d` increases** (relative spread shrinks). This is a numerical signature of *distance concentration*: the distribution of pairwise distances becomes tighter around its mean.
#
# c) When relative spread is small, many points are at almost the same distance from a reference point. Therefore, the identity of the “nearest” points becomes **unstable**: small perturbations (rescaling, noise, missing-value handling) can change the neighbor ordering substantially.
#
# d) Once `cv` is below a few percent (e.g., < 0.05), the separation between “near” and “far” becomes weak in relative terms—nearest-neighbor intuition becomes unreliable without additional structure/assumptions.
#

# %% [markdown]
# ## 16) Sampling: make analysis feasible and keep it representative
#
# Compare summary statistics for:
# - the full dataset
# - a random sample (n=50)
# - a smaller sample (n=10)
#
# Check how close the samples are to the full distribution.

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
# ## 17) Representative sample  
# a) Compare `n=10` vs `n=50`: which is more representative and why?  
# b) Name one situation where a “representative sample” is hard to obtain.
#
# ### Solution
#
# a) `n=50` is usually more representative than `n=10` because sampling variability is lower; the empirical distribution of the sample is closer to the population distribution.
#
# b) Representative sampling is hard when:
# - rare but important subgroups exist (class imbalance, rare events),
# - the data are not i.i.d. (time dependence, drift),
# - sampling is biased (only certain sources/users appear).

# %% [markdown]
# ## 18) Mini-report checklist (what you should be able to explain)
#
# Use this as a checklist when writing up your answers:
#
# 1) **Objects & attributes:** what are they here? (Q1)  
# 2) **Leakage awareness:** what must be excluded from similarity? (Q2)  
# 3) **Measurement scales:** why does type matter for distance? (Q3)  
# 4) **Data quality snapshot:** missingness, duplicates, ID-like fields (Q4)  
# 5) **Distances:** how do L1/L2/L∞ differ? (Q6–Q7)  
# 6) **Scaling & outliers:** how do neighborhoods change? (Q8–Q9)  
# 7) **Missingness strategy:** how does imputation change geometry? (Q10)  
# 8) **Binary & categorical proximity:** SMC vs Jaccard, one-hot sparsity, cosine (Q12–Q14)  
# 9) **Mixed distance:** how do you design it? (Q15)  
# 10) **High-dimensional effects:** distance concentration (Q17)  
# 11) **Sampling:** representativeness (Q18)
#

# %%
