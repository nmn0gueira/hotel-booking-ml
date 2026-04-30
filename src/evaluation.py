from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def evaluate_clustering(X, labels):
    """Return three internal clustering indices computed in the representation space of X.

    silhouette: higher is better [-1, 1].
    davies_bouldin: lower is better [0, inf).
    calinski_harabasz: higher is better (0, inf).
    Returns NaN for all indices when fewer than 2 clusters are present.
    """
    if len(set(labels)) < 2:
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }
    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
    }