"""Batch size and EMA step schedule.

e-means sets both with scaling laws fitted for k-means on Euclidean embeddings.
Those did not transfer in our tests (see README): for medoids the best batch
held roughly ten points per cluster whatever n, and the largest initial step
was best. Only the cosine decay is kept from the paper.
"""
import math


def default_batch_size(n, k, per_cluster=10):
    """About `per_cluster` points per cluster, rounded to a power of two."""
    return int(min(n, 2 ** round(math.log2(per_cluster * k))))


def cosine_step(t, total, alpha0, floor=0.1):
    """Step for update t of `total`: cosine decay from alpha0 to floor * alpha0."""
    return alpha0 * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t / total)))
