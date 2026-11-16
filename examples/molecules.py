"""Cluster molecules by Tanimoto distance between Morgan (ECFP4) fingerprints.

    python examples/molecules.py                          # ZINC-250k subsets
    python examples/molecules.py --sizes 10000 --k 50
    python examples/molecules.py --smiles mine.smi --k 500

The medoids are real molecules, so they double as cluster representatives
(e.g. a diverse subset to screen or label).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import compare, default_methods, print_table, warm_up  # noqa: E402
from datasets import load_fingerprints  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smiles", help="SMILES file, one molecule per line (default: ZINC-250k)")
    p.add_argument("--sizes", type=int, nargs="+", default=[10_000, 50_000, 250_000])
    p.add_argument("--k", type=int, default=100, help="number of clusters")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    X = load_fingerprints(args.smiles)
    warm_up("tanimoto", X)
    for n in sorted({min(n, len(X)) for n in args.sizes}):
        Xn = load_fingerprints(args.smiles, n, args.seed)
        rows = compare(default_methods(args.k, "tanimoto", n, args.seed), Xn)
        print_table(rows, f"{n} molecules, k = {args.k}, Tanimoto distance")


if __name__ == "__main__":
    main()
