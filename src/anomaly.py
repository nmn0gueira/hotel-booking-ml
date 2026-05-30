import numpy as np
from sklearn.mixture import GaussianMixture


def kmeans_anomaly_scores(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Normalised centroid distance for each point.
    score = ||x - c_k|| / (mean per-feature std of cluster k + epsilon)
    High score = far from centroid relative to cluster spread.
    """
    scores = np.zeros(len(X), dtype=np.float64)
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        cluster_pts = X[mask]
        c = centroids[cluster_id]
        cluster_std = cluster_pts.std(axis=0).mean() + 1e-8
        scores[mask] = np.linalg.norm(cluster_pts - c, axis=1) / cluster_std
    return scores


def gmm_anomaly_scores(
    X: np.ndarray,
    k: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Negative log-likelihood under a fitted GMM. Higher = more anomalous.
    Fits a fresh GMM with the same k as the main clustering.
    """
    model = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
    model.fit(X)
    return -model.score_samples(X)


def top_anomalies(scores: np.ndarray, n: int = 20) -> np.ndarray:
    """Return indices of top-n anomalies (highest score)."""
    return np.argsort(scores)[::-1][:n]
