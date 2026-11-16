"""Initial medoids."""
import numpy as np


def initial_medoids(X, k, init, assign, rng, sample_per_cluster=100):
    """Rows of X to start from; returns (indices, distance evaluations used).

    array of k rows : used as given.
    "random"      : k points uniformly at random (what e-means does).
    "k-medoids++" : k-means++-style seeding. Each new medoid is drawn with
        probability proportional to its distance to the nearest medoid chosen
        so far, which spreads the seeds over the data. Run on a random
        subsample of `sample_per_cluster * k` points, so it costs about
        sample_per_cluster * k^2 evaluations, typically well below one epoch.
    """
    n = X.shape[0]
    if not isinstance(init, str):
        return np.asarray(init, np.int64), 0
    if init == "random":
        return rng.choice(n, k, replace=False), 0
    if init != "k-medoids++":
        raise ValueError(f"unknown init {init!r}")

    sample = rng.choice(n, min(n, sample_per_cluster * k), replace=False)
    medoids = np.empty(k, np.int64)
    medoids[0] = rng.choice(sample)
    nearest = assign(X, sample, X[medoids[:1]])[1]
    for j in range(1, k):
        medoids[j] = rng.choice(sample, p=nearest / nearest.sum())
        nearest = np.minimum(nearest, assign(X, sample, X[medoids[j:j + 1]])[1])
    return medoids, sample.size * k
