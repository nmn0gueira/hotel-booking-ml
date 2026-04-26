import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def run_kmeans(X, k, seed=0):
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(X)
    return labels, model


# TODO
def run_ikmeans(X, k):
    pass


def run_gmm(X, k, seed=0):
    model = GaussianMixture(
        n_components=k, covariance_type="full", random_state=seed
    )
    model.fit(X)
    labels = model.predict(X)
    return labels, model


def fit_predict(X, k, seed, algorithm):
    if algorithm == "kmeans":
        return run_kmeans(X, k, seed)
    elif algorithm == "ikmeans":
        return run_ikmeans(X, k)
    elif algorithm == "gmm":
        return run_gmm(X, k, seed)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")
