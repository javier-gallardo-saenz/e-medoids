"""Shared helpers to time clustering methods and print a comparison table."""
import time

import numpy as np

from emedoids import AlternatingKMedoids, EMedoids, FasterPAM
from emedoids.kernels import build_kernels
from emedoids.metrics import resolve


def warm_up(metric, X):
    """Trigger numba compilation on a tiny problem so timings exclude it."""
    small = X[:256]
    for model in (EMedoids(4, metric, epochs=1, batch_size=64),
                  AlternatingKMedoids(4, metric, max_iter=2)):
        model.fit(small)
    build_kernels(resolve(metric)).pairwise(small)


def default_methods(k, metric, n, seed=0, pam_limit=20_000):
    """e-medoids and its baselines; FasterPAM only where its n x n matrix fits."""
    methods = {
        "e-medoids (T=2)": EMedoids(k, metric, epochs=2, random_state=seed),
        "e-medoids (T=5)": EMedoids(k, metric, epochs=5, random_state=seed),
        "alternating": AlternatingKMedoids(k, metric, random_state=seed),  # k-medoids++ seeded
    }
    if n <= pam_limit:
        try:
            import kmedoids  # noqa: F401
            methods["FasterPAM"] = FasterPAM(k, metric, random_state=seed)
        except ImportError:
            print("(install `kmedoids` to add the FasterPAM reference)")
    return methods


def compare(methods, X):
    """Fit every method on X; return rows of (name, seconds, evals, mean distance)."""
    rows = []
    for name, model in methods.items():
        start = time.perf_counter()
        model.fit(X)
        rows.append((name, time.perf_counter() - start, model.n_distance_evals_,
                     model.inertia_ / len(X)))
    return rows


def print_table(rows, title):
    """Loss is the mean distance to the assigned medoid; gap is relative to the best."""
    best = min(r[3] for r in rows)
    print(f"\n{title}")
    print(f"{'method':<18}{'time [s]':>10}{'dist. evals':>14}{'loss':>10}{'gap':>9}")
    for name, seconds, evals, loss in rows:
        print(f"{name:<18}{seconds:>10.2f}{evals:>14.2e}{loss:>10.4f}"
              f"{100 * (loss / best - 1):>8.2f}%")
