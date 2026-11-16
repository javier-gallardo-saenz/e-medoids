"""Reference k-medoids methods, built on the same distance kernels as e-medoids."""
import numpy as np

from .init import initial_medoids
from .kernels import build_kernels, group
from .metrics import resolve


class AlternatingKMedoids:
    """Classic "k-means-style" k-medoids: assign all points, recompute every
    medoid exactly, repeat until nothing changes.

    No distance matrix is stored, but each medoid update costs sum_j n_j^2
    distance evaluations, which is what makes it slow on large clusters.
    """

    def __init__(self, n_clusters, metric="euclidean", max_iter=100,
                 init="k-medoids++", random_state=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state

    def fit(self, X):
        X = np.ascontiguousarray(X)
        n, k = X.shape[0], self.n_clusters
        K = build_kernels(resolve(self.metric))
        rng = np.random.default_rng(self.random_state)
        everything = np.arange(n, dtype=np.int64)

        medoids, evals = initial_medoids(X, k, self.init, K.assign, rng)
        history = []
        for n_iter in range(1, self.max_iter + 1):
            labels, dmin, _ = K.assign(X, everything, X[medoids])
            history.append(dmin.mean())
            order, starts = group(labels, k)
            previous = medoids.copy()
            K.exact_medoids(X, order, starts, medoids)
            evals += n * k + int((np.diff(starts) ** 2).sum())
            if np.array_equal(medoids, previous):
                break
        else:  # stopped by max_iter: labels must match the last medoids
            labels, dmin, _ = K.assign(X, everything, X[medoids])
            evals += n * k

        self.n_iter_ = n_iter

        self.medoid_indices_ = medoids
        self.cluster_centers_ = X[medoids]
        self.labels_ = labels
        self.inertia_ = float(dmin.sum())
        self.n_distance_evals_ = int(evals)
        self.history_ = np.array(history)
        return self


class FasterPAM:
    """FasterPAM swap-based k-medoids (Schubert & Rousseeuw) via the `kmedoids`
    package. Usually the best local optimum, but needs the full n x n matrix:
    n(n-1)/2 distance evaluations and 4 n^2 bytes of memory.
    """

    def __init__(self, n_clusters, metric="euclidean", random_state=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.random_state = random_state

    def fit(self, X):
        import kmedoids  # optional dependency

        X = np.ascontiguousarray(X)
        n = X.shape[0]
        D = build_kernels(resolve(self.metric)).pairwise(X)
        res = kmedoids.fasterpam(D, self.n_clusters, random_state=self.random_state)

        self.medoid_indices_ = np.asarray(res.medoids, np.int64)
        self.cluster_centers_ = X[self.medoid_indices_]
        self.labels_ = np.asarray(res.labels, np.int64)
        self.inertia_ = float(res.loss)
        self.n_distance_evals_ = n * (n - 1) // 2
        return self


class OneBatchPAM:
    """OneBatchPAM (de Mathelin et al., AAAI 2025) via the authors' `onebatch`
    package: FasterPAM-style swaps over all n candidates, with the loss estimated
    on one random batch of m points. Needs the (n, m) distance matrix: n m
    evaluations and 4 n m bytes; m defaults to 100 ln(n k).

    The matrix is computed with our own kernels, so the distance code is the
    same as for the other methods. The swap loop is the authors' (single thread).
    `seconds_` splits the time into "distances", "swaps" and "assign" (the full
    assignment needed for labels, which the package itself does not produce).
    """

    def __init__(self, n_clusters, metric="euclidean", batch_size="auto", weighting=True,
                 random_state=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.batch_size = batch_size
        self.weighting = weighting  # the package's default importance weighting
        self.random_state = random_state

    def fit(self, X):
        import time

        import onebatch  # optional dependency

        X = np.ascontiguousarray(X)
        n, k = X.shape[0], self.n_clusters
        K = build_kernels(resolve(self.metric))
        seconds = {}

        def distance(A, B, n_jobs=None):
            start = time.perf_counter()
            D = K.cross(A, np.ascontiguousarray(B))
            seconds["distances"] = time.perf_counter() - start
            return D

        start = time.perf_counter()
        model = onebatch.OneBatchPAM(n_medoids=k, distance=distance, batch_size=self.batch_size,
                                     weighting=self.weighting,
                                     random_state=self.random_state).fit(X)
        seconds["swaps"] = time.perf_counter() - start - seconds["distances"]

        start = time.perf_counter()
        medoids = np.asarray(model.medoid_indices_, np.int64)
        labels, dmin, _ = K.assign(X, np.arange(n, dtype=np.int64), X[medoids])
        seconds["assign"] = time.perf_counter() - start

        self.medoid_indices_ = medoids
        self.cluster_centers_ = X[medoids]
        self.labels_ = labels
        self.inertia_ = float(dmin.sum())
        self.batch_size_ = len(model.labels_)
        self.n_distance_evals_ = n * self.batch_size_ + n * k
        self.n_iter_ = model.n_iter_
        self.seconds_ = seconds
        return self
