from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)

def compute_ari(labels_a, labels_b) -> float:
    """ARI between two label vectors. Returns NaN if either has <2 clusters."""
    if len(set(labels_a)) < 2 or len(set(labels_b)) < 2:
        return float("nan")
    return float(adjusted_rand_score(labels_a, labels_b))


def evaluate_clustering(X, labels):
   
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