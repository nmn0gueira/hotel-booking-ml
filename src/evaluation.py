from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

metric_info = {
    "silhouette": {
        "label": "Silhouette",
        "func": silhouette_score,
        "direction": "max",
    },
    "calinski_harabasz": {
        "label": "Calinski–Harabasz",
        "func": calinski_harabasz_score,
        "direction": "max",
    },
    "davies_bouldin": {
        "label": "Davies–Bouldin",
        "func": davies_bouldin_score,
        "direction": "min",
    },
}

def evaluate_clustering(X, labels, runtime=None):

    result = {}
 
    if len(set(labels)) < 2:
        for name in metric_info:
            result[name] = float("nan")
    else:
        for name, info in metric_info.items():
            result[name] = info["func"](X, labels)
 
    result["runtime_s"] = runtime  # None if not provided
 
    return result