"""Compiled inner loops, specialised once per distance function.

`build_kernels(dist)` returns the kernels shared by e-medoids and the baselines,
so every method is timed on exactly the same distance code.

Conventions: `X` is the (n, d) data, `idx` holds the rows of `X` in the current
batch, and (`order`, `starts`) is the CSR grouping of batch positions by cluster
returned by `group`: cluster j owns positions order[starts[j]:starts[j + 1]].
"""
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from numba import njit, prange


@njit
def group(labels, k):
    """Group positions by cluster label (stable counting sort)."""
    starts = np.zeros(k + 1, np.int64)
    for l in labels:
        starts[l + 1] += 1
    for j in range(k):
        starts[j + 1] += starts[j]
    order = np.empty(labels.shape[0], np.int64)
    fill = starts[:-1].copy()
    for i in range(labels.shape[0]):
        order[fill[labels[i]]] = i
        fill[labels[i]] += 1
    return order, starts


@lru_cache(maxsize=None)
def build_kernels(dist):
    @njit(parallel=True)
    def assign(X, idx, centers):
        """Nearest center of every point X[idx[i]].

        Returns its index, its distance, and the distance to the second nearest.
        """
        n = idx.shape[0]
        labels = np.empty(n, np.int64)
        d1 = np.empty(n, np.float64)
        d2 = np.empty(n, np.float64)
        for i in prange(n):
            x = X[idx[i]]
            best, second, arg = np.inf, np.inf, 0
            for j in range(centers.shape[0]):
                d = dist(x, centers[j])
                if d < best:
                    best, second, arg = d, best, j
                elif d < second:
                    second = d
            labels[i] = arg
            d1[i] = best
            d2[i] = second
        return labels, d1, d2

    @njit(parallel=True)
    def ema_step(X, idx, d1, d2, order, starts, pool, S, G, A, R, age, med_slot, alpha,
                 n_new, probation):
        """One e-medoids update of every cluster from one mini-batch (in place).

        pool[j]  : rows of X that are candidate medoids of cluster j (-1 = empty)
        med_slot : pool slot of the current medoid
        With x running over the batch points assigned to cluster j, the EMAs are
        S[j, c]  : sum of dist(c, x), the cost of candidate c as medoid of j
        G[j, c]  : sum of max(0, d1(x) - dist(c, x)), the loss saved if c became
                   an *additional* medoid; A[j, c] is its EMA weight (G / A is
                   unbiased however young the candidate is)
        R[j]     : sum of d2(x) - d1(x), the loss added if medoid j were removed
        age[j, c]: number of batches candidate c has been scored on

        Up to n_new points enter each pool per batch. A candidate is on
        probation until it has been scored on `probation` batches: before that
        it can neither be evicted nor become the medoid.

        Returns the number of distance evaluations performed.
        """
        k, m = pool.shape
        keep = 1.0 - alpha
        evals = 0
        for j in prange(k):
            lo, hi = starts[j], starts[j + 1]
            nj = hi - lo
            ms = med_slot[j]

            # The most promising extra medoid is kept in the pool regardless of S.
            guard, guard_gain = -1, 0.0
            for c in range(m):
                if A[j, c] > 0 and G[j, c] / A[j, c] > guard_gain:
                    guard, guard_gain = c, G[j, c] / A[j, c]

            # 1. Newcomers: the first batch members of the cluster (a uniform
            #    sample, since batches are shuffled) replace the worst-scoring
            #    candidates that are off probation. They inherit the medoid's cost, i.e. they are
            #    presumed as good as the medoid until the data says otherwise.
            if nj >= 2:
                for t in range(min(n_new, nj)):
                    cand = idx[order[lo + t]]
                    worst, worst_s, pooled = -1, -np.inf, False
                    for c in range(m):
                        if pool[j, c] == cand:
                            pooled = True
                            break
                        if c == ms or c == guard:
                            continue
                        if pool[j, c] < 0:
                            s = np.inf  # empty slots first
                        elif age[j, c] < probation:
                            continue
                        else:
                            s = S[j, c]
                        if s > worst_s:
                            worst, worst_s = c, s
                    if pooled or worst < 0:
                        continue
                    pool[j, worst] = cand
                    S[j, worst] = S[j, ms]
                    G[j, worst] = A[j, worst] = 0.0
                    age[j, worst] = 0

            # 2. Batch statistics, then the e-means EMA update. All candidates of
            #    a cluster are scored on the same points, so their differences
            #    are paired comparisons.
            removal = 0.0
            for t in range(lo, hi):
                removal += d2[order[t]] - d1[order[t]]
            R[j] = keep * R[j] + alpha * removal
            for c in range(m):
                p = pool[j, c]
                if p < 0:
                    continue
                cost = 0.0
                if c == ms:  # already known from the assignment step
                    for t in range(lo, hi):
                        cost += d1[order[t]]
                else:
                    gain = 0.0
                    for t in range(lo, hi):
                        i = order[t]
                        d = dist(X[p], X[idx[i]])
                        cost += d
                        if d < d1[i] and idx[i] != p:
                            gain += d1[i] - d
                    evals += nj
                    if age[j, c] == 0:  # a newcomer is in the batch: drop its d(x, x) = 0
                        cost *= nj / (nj - 1)
                    G[j, c] = keep * G[j, c] + alpha * gain
                    A[j, c] = keep * A[j, c] + alpha
                S[j, c] = keep * S[j, c] + alpha * cost
                age[j, c] += 1

            # 3. The medoid is the candidate off probation with the lowest averaged cost.
            best, new = np.inf, ms
            for c in range(m):
                if pool[j, c] < 0 or S[j, c] >= best:
                    continue
                if c == ms or age[j, c] >= probation:
                    best, new = S[j, c], c
            if new != ms:  # gains were measured against the old medoid
                G[j, new] = A[j, new] = G[j, ms] = A[j, ms] = 0.0
            med_slot[j] = new
        return evals

    @njit(parallel=True)
    def pool_costs(X, idx, order, starts, pool):
        """Exact cost of every pooled candidate over its cluster's members.

        Empty slots and clusters without members are left at +inf.
        """
        k, m = pool.shape
        out = np.full((k, m), np.inf)
        for j in prange(k):
            if starts[j] == starts[j + 1]:
                continue
            for c in range(m):
                p = pool[j, c]
                if p < 0:
                    continue
                s = 0.0
                for t in range(starts[j], starts[j + 1]):
                    s += dist(X[p], X[idx[order[t]]])
                out[j, c] = s
        return out

    @njit(parallel=True)
    def exact_medoids(X, order, starts, medoids):
        """Exact medoid of every cluster, in place (sum_j n_j^2 evaluations)."""
        for j in range(medoids.shape[0]):
            lo, nj = starts[j], starts[j + 1] - starts[j]
            if nj == 0:
                continue
            cost = np.empty(nj)
            for a in prange(nj):
                s = 0.0
                for b in range(nj):
                    s += dist(X[order[lo + a]], X[order[lo + b]])
                cost[a] = s
            medoids[j] = order[lo + np.argmin(cost)]

    @njit(parallel=True)
    def pairwise(X):
        """Full (n, n) float32 distance matrix, for methods that need one."""
        n = X.shape[0]
        D = np.zeros((n, n), np.float32)
        for i in prange(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = dist(X[i], X[j])
        return D

    @njit(parallel=True)
    def cross(X, Y):
        """(len(X), len(Y)) float32 distances between two sets of points."""
        D = np.empty((X.shape[0], Y.shape[0]), np.float32)
        for i in prange(X.shape[0]):
            for j in range(Y.shape[0]):
                D[i, j] = dist(X[i], Y[j])
        return D

    return SimpleNamespace(
        cross=cross,
        assign=assign,
        ema_step=ema_step,
        pool_costs=pool_costs,
        exact_medoids=exact_medoids,
        pairwise=pairwise,
    )
