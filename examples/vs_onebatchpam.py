"""e-medoids against OneBatchPAM on large data.

    python examples/vs_onebatchpam.py zinc --k 100
    python examples/vs_onebatchpam.py chembl --k 100          # 500k ChEMBL compounds
    python examples/vs_onebatchpam.py blobs --n 500000 --k 100
    python examples/vs_onebatchpam.py zinc --k 100 --threads 1

Both methods use the same compiled distance code. OneBatchPAM stores an (n, m)
float32 matrix; if its default m = 100 ln(n k) does not fit in free memory, m is
reduced and the table says so.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numba
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blobs import make_blobs  # noqa: E402
from datasets import chembl_smiles, load_fingerprints  # noqa: E402

from emedoids import EMedoids, OneBatchPAM  # noqa: E402
from emedoids.kernels import build_kernels  # noqa: E402
from emedoids.metrics import resolve  # noqa: E402


def onebatch_size(n, k, memory_fraction=0.6):
    """OneBatchPAM's default batch size, reduced if the matrix would not fit."""
    default = int(100 * math.log(n * k))
    fits = int(memory_fraction * psutil.virtual_memory().available / (4 * n))
    return min(default, fits), default


def oracle_loss(X, centers, metric):
    """Loss when each true blob center is represented by its nearest data point."""
    K = build_kernels(resolve(metric))
    everything = np.arange(len(X), dtype=np.int64)
    nearest = K.assign(centers, np.arange(len(centers), dtype=np.int64), X)[0]
    return K.assign(X, everything, X[nearest])[1].mean()


def run(make, X, seeds):
    """Fit one method per seed; return per-seed (seconds, loss) and the last model."""
    seconds, losses = [], []
    for seed in seeds:
        start = time.perf_counter()
        model = make(seed).fit(X)
        seconds.append(time.perf_counter() - start)
        losses.append(model.inertia_ / len(X))
    return np.array(seconds), np.array(losses), model


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data", choices=["zinc", "chembl", "blobs"])
    p.add_argument("--n", type=int, help="number of points (default: all of ZINC, else 500k)")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--dim", type=int, default=16, help="blob dimension")
    p.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 5])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--threads", type=int, default=numba.get_num_threads())
    p.add_argument("--smiles", help="SMILES file to use instead of ZINC-250k")
    p.add_argument("--batch", type=int, help="force OneBatchPAM's batch size m")
    args = p.parse_args()
    numba.set_num_threads(args.threads)
    seeds = range(args.seeds)

    reference = None
    if args.data == "zinc":
        X, metric = load_fingerprints(args.smiles, args.n), "tanimoto"
    elif args.data == "chembl":
        X, metric = load_fingerprints(chembl_smiles(args.n or 500_000)), "tanimoto"
    else:
        X, _, centers = make_blobs(args.n or 500_000, args.k, args.dim)
        metric = "manhattan"
        reference = oracle_loss(X, centers, metric)
    n, k = len(X), args.k

    # compile everything on a small problem first
    EMedoids(4, metric, epochs=1, batch_size=64).fit(X[:512])
    OneBatchPAM(4, metric, batch_size=64).fit(X[:512])

    m, m_default = onebatch_size(n, k)
    m = args.batch or m
    note = "" if m == m_default else f" (its default is {m_default})"
    print(f"\n{args.data}: n = {n}, k = {k}, {metric} distance, {args.threads} thread(s), "
          f"{args.seeds} seeds\nOneBatchPAM batch m = {m}{note}, matrix {4 * n * m / 2**30:.2f} GB")
    if reference is not None:
        print(f"reference loss (data points nearest the true centers): {reference:.4f}")

    rows = []
    for T in args.epochs:
        sec, loss, model = run(lambda s: EMedoids(k, metric, epochs=T, random_state=s), X, seeds)
        rows.append((f"e-medoids (T={T})", sec, loss, model.n_distance_evals_,
                     f"h={model.probation_}, {model.n_updates_} updates"))
    sec, loss, model = run(lambda s: OneBatchPAM(k, metric, batch_size=m, random_state=s), X, seeds)
    parts = model.seconds_
    rows.append(("OneBatchPAM", sec, loss, model.n_distance_evals_,
                 f"distances {parts['distances']:.1f}s + swaps {parts['swaps']:.1f}s "
                 f"+ assign {parts['assign']:.1f}s, {model.n_iter_} sweeps"))

    best = min(loss.mean() for _, _, loss, _, _ in rows)
    print(f"{'method':<18}{'time [s]':>9}{'dist. evals':>13}{'loss':>10}{'+-':>8}{'gap':>8}  details")
    for name, sec, loss, evals, details in rows:
        print(f"{name:<18}{np.median(sec):>9.2f}{evals:>13.2e}{loss.mean():>10.4f}"
              f"{loss.std():>8.4f}{100 * (loss.mean() / best - 1):>7.2f}%  {details}")


if __name__ == "__main__":
    main()
