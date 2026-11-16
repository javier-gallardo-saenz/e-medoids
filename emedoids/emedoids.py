"""e-medoids: mini-batch k-medoids with exponentially averaged candidate costs.

e-means keeps, per cluster, an exponential moving average (EMA) of the sum and
count of the points assigned to it; the centroid is their ratio. A medoid has no
such closed form: it is the *data point* minimising the summed distance to its
cluster. e-medoids therefore keeps, per cluster, a small pool of candidate
points and an EMA of each candidate's summed distance to the batch points
assigned to the cluster. The medoid is the candidate with the lowest average.

The same distances also yield, at no extra cost, EMA estimates of what each
medoid is worth and what each candidate would save as an additional medoid.
These drive PAM-style swaps that move medoids between clusters.

Per mini-batch of B points this costs B * (k + pool_size - 1) distance
evaluations and never touches an n x n matrix, whereas an exact medoid update
costs sum_j n_j^2.
"""
import numpy as np

from .init import initial_medoids
from .kernels import build_kernels, group
from .metrics import resolve
from .schedule import cosine_step, default_batch_size


class EMedoids:
    """Mini-batch k-medoids with EMA candidate statistics.

    Parameters
    ----------
    n_clusters : number of medoids k.
    metric : name in `emedoids.metrics.METRICS` or a jitted ``dist(x, y)``.
    epochs : passes over the data (T).
    pool_size : candidates kept per cluster (m), including the medoid.
    n_new : candidates replaced per batch (default: half the pool).
    batch_size : mini-batch size (default: about ten points per cluster).
    alpha0 : initial EMA step, cosine-annealed to alpha0 / 10. With 1, the
        first batches overwrite the statistics and averaging sets in gradually.
    init : "random", "k-medoids++" (spread-out seeding; rarely needed, since
        swaps repair a poor start), or an array of k row indices.
    swap : every `swap_every` batches, move medoids that are cheap to remove to
        candidates that would save more as extra medoids (see `_swap`).
    finalize : end with one exact pass choosing, for every cluster, its best
        pooled candidate on the full data. Never increases the loss.

    Attributes
    ----------
    medoid_indices_, cluster_centers_, labels_ : the clustering.
    inertia_ : sum of distances of all points to their medoid.
    n_distance_evals_ : distance evaluations used (hardware-independent cost).
    n_swaps_ : medoids relocated by swaps.
    history_ : per-batch mean distance to the medoid (a noisy training curve).
    """

    def __init__(self, n_clusters, metric="euclidean", epochs=5, pool_size=16,
                 n_new=None, batch_size=None, alpha0=1.0, init="random",
                 swap=True, swap_every=2, finalize=True, random_state=None):
        self.n_clusters = n_clusters
        self.metric = metric
        self.epochs = epochs
        self.pool_size = pool_size
        self.n_new = n_new
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.init = init
        self.swap = swap
        self.swap_every = swap_every
        self.finalize = finalize
        self.random_state = random_state

    def fit(self, X):
        X = np.ascontiguousarray(X)
        n, k, m, T = X.shape[0], self.n_clusters, self.pool_size, self.epochs
        K = build_kernels(resolve(self.metric))
        rng = np.random.default_rng(self.random_state)

        B = self.batch_size or default_batch_size(n, k)
        n_batches = max(1, n // B)
        total = T * n_batches
        n_new = self.n_new or m // 2

        # State (see kernels.ema_step); all statistics start at zero.
        pool = np.full((k, m), -1, np.int64)
        pool[:, 0], evals = initial_medoids(X, k, self.init, K.assign, rng)
        S, G, A = np.zeros((k, m)), np.zeros((k, m)), np.zeros((k, m))
        R = np.zeros(k)
        med_slot = np.zeros(k, np.int64)
        rows = np.arange(k)

        history, n_swaps, t = [], 0, 0
        for _ in range(T):
            for idx in np.array_split(rng.permutation(n), n_batches):
                alpha = cosine_step(t, total, self.alpha0)
                labels, d1, d2 = K.assign(X, idx, X[pool[rows, med_slot]])
                order, starts = group(labels, k)
                evals += idx.size * k + K.ema_step(
                    X, idx, d1, d2, order, starts, pool, S, G, A, R, med_slot, alpha, n_new)
                history.append(d1.mean())
                t += 1
                if self.swap and t % self.swap_every == 0:
                    n_swaps += _swap(pool, S, G, A, R, med_slot)

        medoids = pool[rows, med_slot]
        everything = np.arange(n, dtype=np.int64)
        if self.finalize:
            # Exact alternate step restricted to the pool: assign all points,
            # then give each cluster its best candidate on its true members.
            labels = K.assign(X, everything, X[medoids])[0]
            order, starts = group(labels, k)
            cost = K.pool_costs(X, everything, order, starts, pool)
            empty = np.isinf(cost).all(axis=1)
            medoids = pool[rows, np.where(empty, med_slot, cost.argmin(axis=1))]
            evals += n * k + int(np.diff(starts) @ (pool >= 0).sum(axis=1))

        labels, d1, _ = K.assign(X, everything, X[medoids])
        evals += n * k

        # Rare: two clusters can end on the same point (pools may overlap). The
        # redundant ones move to the points currently worst served.
        repeated = np.setdiff1d(rows, np.unique(medoids, return_index=True)[1])
        if repeated.size:
            medoids[repeated] = np.argsort(d1)[-repeated.size:]
            labels, d1, _ = K.assign(X, everything, X[medoids])
            evals += n * k

        self.medoid_indices_ = medoids
        self.cluster_centers_ = X[medoids]
        self.labels_ = labels
        self.inertia_ = float(d1.sum())
        self.n_distance_evals_ = int(evals)
        self.n_swaps_ = n_swaps
        self.history_ = np.array(history)
        self.batch_size_ = B
        return self

    def predict(self, X):
        """Index of the nearest medoid for new points."""
        X = np.ascontiguousarray(X)
        K = build_kernels(resolve(self.metric))
        return K.assign(X, np.arange(X.shape[0], dtype=np.int64), self.cluster_centers_)[0]


def _swap(pool, S, G, A, R, med_slot):
    """Relocate medoids, PAM-style, using the EMA statistics. Returns the count.

    Removing medoid j costs about R[j]; making candidate c of another cluster i
    an extra medoid saves at least G[i, c] / A[i, c] (it only counts points of
    i). Whenever the saving exceeds the cost, cluster j restarts from c. This
    lets medoids cross between clusters, which assign/update steps alone cannot
    do, and it subsumes e-means' dead-cluster splitting (a dead cluster has
    R = 0). Each cluster takes part in at most one swap per call, since a swap
    invalidates its statistics.
    """
    k = R.size
    gain = np.where((pool >= 0) & (A > 0), G / np.maximum(A, 1e-300), 0.0)
    best = gain.argmax(axis=1)
    saving = gain[np.arange(k), best]
    cost = R.copy()
    n_swaps = 0
    for _ in range(k // 2):
        j = cost.argmin()
        cost[j] = np.inf  # j is used up, as a target or as a non-candidate
        saved, saving[j] = saving[j], -np.inf
        i = saving.argmax()
        if saving[i] <= R[j]:
            saving[j] = saved
            break
        pool[j], S[j], G[j], A[j], med_slot[j] = -1, 0.0, 0.0, 0.0, 0
        pool[j, 0], pool[i, best[i]] = pool[i, best[i]], -1
        R[j] = saving[i]  # what the new medoid is believed to be worth
        G[i], A[i] = 0.0, 0.0  # measured against the old partition of i
        cost[i], saving[i] = np.inf, -np.inf
        n_swaps += 1
    return n_swaps
