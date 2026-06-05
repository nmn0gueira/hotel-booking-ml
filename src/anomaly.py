import numpy as np
from sklearn.mixture import GaussianMixture

# Anomaly scoring functions for clustering-based methods.
def kmeans_anomaly_scores(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray,) -> np.ndarray:
   
    scores = np.zeros(len(X), dtype=np.float64)
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        cluster_pts = X[mask]
        c = centroids[cluster_id]
        cluster_std = cluster_pts.std(axis=0).mean() + 1e-8
        scores[mask] = np.linalg.norm(cluster_pts - c, axis=1) / cluster_std
    return scores

#  Fits a GMM with the same k as the main clustering.
def gmm_anomaly_scores(X: np.ndarray,k: int,seed: int = 0, ) -> np.ndarray:
   
    model = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
    model.fit(X)
    return -model.score_samples(X)

# For a given anomaly score vector, return the indices of the top-n anomalies.
def top_anomalies(scores: np.ndarray, n: int = 20) -> np.ndarray:
    return np.argsort(scores)[::-1][:n]
