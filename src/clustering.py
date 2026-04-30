from dataclasses import dataclass

import time
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class APCluster:
    indices: list[int]
    centroid_raw: FloatArray
    centroid_std: FloatArray
    size: int
    scatter_pct: float


def compute_feature_statistics(
    X: FloatArray,
    use_unit_ranges: bool = False,
) -> tuple[FloatArray, FloatArray, float]:
    """Returns (grand_mean, scales, total_scatter). Scales use range; zeros replaced by 1."""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    if use_unit_ranges:
        scales = np.ones(X.shape[1], dtype=np.float64)
    else:
        scales = X.max(axis=0) - X.min(axis=0)
        scales[scales == 0.0] = 1.0
    Y = (X - mean) / scales
    total_scatter = float(np.sum(Y ** 2))
    return mean, scales, total_scatter


def normalized_squared_distances(
    X: FloatArray,
    indices: list[int],
    scales: FloatArray,
    reference: FloatArray,
) -> FloatArray:
    """δr(x_i, reference) for each i in indices."""
    rows = X[indices].astype(np.float64)
    diff = (rows - reference) / scales
    return np.sum(diff ** 2, axis=1)


def cluster_centroid(X: FloatArray, indices: list[int]) -> FloatArray:
    return X[indices].astype(np.float64).mean(axis=0)


def separate_cluster(
    X: FloatArray,
    indices: list[int],
    scales: FloatArray,
    a: FloatArray,
    b: FloatArray,
) -> list[int]:
    """Return indices where δr(x_i, a) < δr(x_i, b). Strict inequality. Ties stay with b."""
    rows = X[indices].astype(np.float64)
    dist_a = np.sum(((rows - a) / scales) ** 2, axis=1)
    dist_b = np.sum(((rows - b) / scales) ** 2, axis=1)
    mask = dist_a < dist_b
    return sorted(np.array(indices)[mask].tolist()) # "Cluster indices must always refer to row numbers of the original matrix X. For deterministic behaviour, return them in increasing order."


def extract_anomalous_cluster(
    X: FloatArray,
    indices: list[int],
    scales: FloatArray,
    mean: FloatArray,
    initial_centroid: FloatArray,
    seed_index: int,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> tuple[list[int], FloatArray]:
    """Alternating assign/update loop for one anomalous cluster."""
    c = np.asarray(initial_centroid, dtype=np.float64).copy()
    S_prev: list[int] = []

    for _ in range(max_iter):
        S = separate_cluster(X, indices, scales, c, mean)
        if not S:
            S = [seed_index]
        c_new = cluster_centroid(X, S)
        if set(S) == set(S_prev):
            return S, c_new
        if np.linalg.norm(c_new - c) <= tol:
            return S, c_new
        c = c_new
        S_prev = S

    return S, c


def ikmeans_initialize(
    X: FloatArray,
    min_cluster_size: int,
    tol: float = 1e-12,
    max_iter: int = 10_000,
    use_unit_ranges: bool = False,
) -> tuple[list[APCluster], FloatArray]:
    """
    Mirkin iK-means initialisation.

    Extracts anomalous clusters one by one until all points are assigned,
    then returns the standardised centroids of clusters with size >= min_cluster_size
    as seeds for downstream k-means.

    Raises ValueError if no cluster satisfies the size threshold.
    """
    X = np.asarray(X, dtype=np.float64)
    mean, scales, D = compute_feature_statistics(X, use_unit_ranges)

    remains: list[int] = list(range(len(X)))
    ap_clusters: list[APCluster] = []

    while remains:
        dists = normalized_squared_distances(X, remains, scales, mean)
        seed_pos = int(np.argmax(dists))
        seed_index = remains[seed_pos]
        seed = X[seed_index].copy()

        S, c = extract_anomalous_cluster(
            X, remains, scales, mean, seed, seed_index, tol, max_iter
        )

        z = (c - mean) / scales
        scatter_pct = float(100.0 * len(S) * float(np.sum(z ** 2)) / D) if D > 0.0 else 0.0

        ap_clusters.append(APCluster(
            indices=sorted(S),
            centroid_raw=c.copy(),
            centroid_std=z.copy(),
            size=len(S),
            scatter_pct=scatter_pct,
        ))

        S_set = set(S)
        remains = [i for i in remains if i not in S_set]

    retained = [cl for cl in ap_clusters if cl.size >= min_cluster_size]
    if not retained:
        raise ValueError(
            f"No anomalous cluster has size >= {min_cluster_size}. "
            f"Cluster sizes found: {[cl.size for cl in ap_clusters]}"
        )

    init_centroids = np.stack([cl.centroid_std for cl in retained])
    return ap_clusters, init_centroids

#Single k-means run (n_init=1)
#def fit_kmeans_once(X, K, init_method="k-means++", seed=42):
#    
#    model = KMeans(
#        n_clusters=K,
#        init=init_method,
#        n_init=1,         
#        random_state=seed,
#        max_iter=300,
#        algorithm="lloyd",
#    )
#    labels = model.fit_predict(X)
#    centers = model.cluster_centers_
#    return model, labels, centers

#Single K-Means run. Returns labels, model, and wall-clock runtime (s).
#def run_kmeans(X, K, seed=0, init_method="k-means++"): 
#    t0 = time.perf_counter()
#    model, labels, centers = fit_kmeans_once(X, K, init_method=init_method, seed=seed)
#    runtime = time.perf_counter() - t0
#    return labels, model, runtime

def run_kmeans(X: FloatArray, k: int, seed: int = 0):
    t0 = time.perf_counter()
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(X)
    runtime = time.perf_counter() - t0
    return labels, runtime

def run_ikmeans(X: FloatArray, k: int):
    """
    iK-means: use AP extraction centroids (raw space) to initialise k-means.
    Takes the first k clusters in extraction order.
    """
    t0 = time.perf_counter()
    ap_clusters, _ = ikmeans_initialize(X, min_cluster_size=1)
    if len(ap_clusters) < k:
        raise ValueError(
            f"iK-means extracted only {len(ap_clusters)} clusters, need {k}. "
            "Reduce k or lower min_cluster_size."
        )
    init_raw = np.stack([cl.centroid_raw for cl in ap_clusters[:k]]).astype(np.float64)
    model = KMeans(n_clusters=k, init=init_raw, n_init=1, random_state=0)
    labels = model.fit_predict(X)
    runtime = time.perf_counter() - t0
    return labels, runtime


def run_gmm(X: FloatArray, k: int, seed: int = 0):
    t0 = time.perf_counter()
    model = GaussianMixture(
        n_components=k, covariance_type="full", random_state=seed
    )
    model.fit(X)
    labels = model.predict(X)
    runtime = time.perf_counter() - t0
    return labels, runtime


def fit_predict(X: FloatArray, k: int, seed: int, algorithm: str):
    if algorithm == "kmeans":
        return run_kmeans(X, k, seed)
    elif algorithm == "ikmeans":
        return run_ikmeans(X, k)
    elif algorithm == "gmm":
        return run_gmm(X, k, seed)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")
