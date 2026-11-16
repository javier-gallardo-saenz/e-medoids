"""Dissimilarities as numba-jitted scalar functions ``dist(x, y) -> float``.

Every algorithm in this package only ever calls ``dist`` on pairs of rows, so a
custom metric is just another jitted function with the same signature.
"""
import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.extending import intrinsic


@intrinsic
def _popcount(typingctx, x):
    """Number of set bits in a uint64 (compiles to the CPU's POPCNT)."""

    def codegen(context, builder, signature, args):
        fn = builder.module.declare_intrinsic("llvm.ctpop", [ir.IntType(64)])
        return builder.call(fn, args)

    return types.uint64(types.uint64), codegen


@njit(fastmath=True)
def tanimoto(a, b):
    """Tanimoto (Jaccard) distance between bit vectors packed with `pack_bits`."""
    inter = np.uint64(0)
    union = np.uint64(0)
    for w in range(a.shape[0]):
        inter += _popcount(a[w] & b[w])
        union += _popcount(a[w] | b[w])
    return 1.0 - inter / union if union > 0 else 0.0


@njit(fastmath=True)
def euclidean(a, b):
    s = 0.0
    for i in range(a.shape[0]):
        d = a[i] - b[i]
        s += d * d
    return np.sqrt(s)


@njit(fastmath=True)
def manhattan(a, b):
    s = 0.0
    for i in range(a.shape[0]):
        s += abs(a[i] - b[i])
    return s


@njit(fastmath=True)
def cosine(a, b):
    ab = aa = bb = 0.0
    for i in range(a.shape[0]):
        ab += a[i] * b[i]
        aa += a[i] * a[i]
        bb += b[i] * b[i]
    return 1.0 - ab / np.sqrt(aa * bb) if aa > 0 and bb > 0 else 1.0


METRICS = {
    "tanimoto": tanimoto,
    "euclidean": euclidean,
    "manhattan": manhattan,
    "cosine": cosine,
}


def resolve(metric):
    """Return the jitted distance for a metric name (jitted callables pass through)."""
    if callable(metric):
        return metric
    try:
        return METRICS[metric]
    except KeyError:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(METRICS)}") from None


def pack_bits(bits):
    """Pack an (n, n_bits) 0/1 array into (n, ceil(n_bits / 64)) uint64 words."""
    packed = np.packbits(np.asarray(bits, dtype=bool), axis=1)
    packed = np.pad(packed, ((0, 0), (0, -packed.shape[1] % 8)))
    return np.ascontiguousarray(packed).view(np.uint64)
