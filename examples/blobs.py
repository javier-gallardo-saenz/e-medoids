"""Sanity check on synthetic data with known structure (no downloads needed).

    python examples/blobs.py

Gaussian blobs with a non-Euclidean (Manhattan) distance. With clear clusters,
every method should land close to FasterPAM's loss; the interest is in the cost.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import compare, default_methods, print_table, warm_up  # noqa: E402


def make_blobs(n, k, dim, spread=1.0, seed=0):
    """n points around k random centers; cluster sizes are deliberately unequal."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=6.0, size=(k, dim))
    weights = rng.dirichlet(np.full(k, 2.0))
    labels = rng.choice(k, n, p=weights)
    return centers[labels] + rng.normal(scale=spread, size=(n, dim)), labels


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    X, _ = make_blobs(args.n, args.k, args.dim, seed=args.seed)
    warm_up("manhattan", X)
    rows = compare(default_methods(args.k, "manhattan", args.n, args.seed), X)
    print_table(rows, f"{args.n} points in {args.dim}-d, k = {args.k}, Manhattan distance")


if __name__ == "__main__":
    main()
