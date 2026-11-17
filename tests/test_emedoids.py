import numpy as np
import pytest

from emedoids import AlternatingKMedoids, EMedoids, pack_bits
from emedoids.metrics import manhattan, tanimoto


def blobs(n=3000, k=10, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=6.0, size=(k, dim))
    return centers[rng.integers(k, size=n)] + rng.normal(size=(n, dim))


def test_tanimoto_matches_definition():
    bits = np.random.default_rng(0).random((20, 200)) < 0.2
    X = pack_bits(bits)
    for i, j in [(0, 1), (3, 7), (5, 5)]:
        expected = 1 - (bits[i] & bits[j]).sum() / (bits[i] | bits[j]).sum()
        assert tanimoto(X[i], X[j]) == pytest.approx(expected)


def test_output_is_a_valid_clustering():
    X = blobs()
    model = EMedoids(10, "manhattan", random_state=0).fit(X)
    assert len(np.unique(model.medoid_indices_)) == 10
    # every point sits with its nearest medoid, and the loss is what it claims
    D = np.abs(X[:, None] - X[model.medoid_indices_][None]).sum(-1)
    assert np.allclose(D[np.arange(len(X)), model.labels_], D.min(axis=1))
    assert model.inertia_ == pytest.approx(D.min(axis=1).sum())
    assert np.array_equal(model.predict(X), model.labels_)


def test_seed_makes_runs_reproducible():
    X = blobs()
    a = EMedoids(10, "manhattan", random_state=1).fit(X)
    b = EMedoids(10, "manhattan", random_state=1).fit(X)
    assert np.array_equal(a.medoid_indices_, b.medoid_indices_)


def test_finalization_never_hurts():
    X = blobs()
    raw = EMedoids(10, "manhattan", finalize=False, random_state=2).fit(X)
    final = EMedoids(10, "manhattan", finalize=True, random_state=2).fit(X)
    assert final.inertia_ <= raw.inertia_ + 1e-9


def test_matches_exact_alternating_on_easy_data():
    X = blobs()
    ours = min(EMedoids(10, "manhattan", random_state=s).fit(X).inertia_ for s in range(3))
    ref = min(AlternatingKMedoids(10, "manhattan", random_state=s).fit(X).inertia_
              for s in range(3))
    assert ours <= 1.01 * ref


def test_swaps_repair_a_bad_start():
    # all medoids start inside one blob; only swaps can spread them out
    X = blobs()
    crowded = np.argsort(np.abs(X - X[0]).sum(axis=1))[:10]

    def loss(swap):
        return EMedoids(10, "manhattan", init=crowded, swap=swap, random_state=0).fit(X).inertia_

    assert loss(swap=True) < 0.8 * loss(swap=False)


def test_custom_metric():
    X = blobs(n=500, k=4)
    model = EMedoids(4, manhattan, epochs=2, random_state=0).fit(X)
    assert model.labels_.shape == (500,)


def test_probation_default_follows_the_number_of_updates():
    X = blobs()
    short = EMedoids(10, "manhattan", epochs=1, random_state=0).fit(X)
    long = EMedoids(10, "manhattan", epochs=5, random_state=0).fit(X)
    assert (short.n_updates_, short.probation_) == (23, 1)
    longer = EMedoids(10, "manhattan", epochs=10, random_state=0).fit(X)
    assert (long.n_updates_, long.probation_) == (115, 2)
    assert (longer.n_updates_, longer.probation_) == (230, 3)
    assert EMedoids(10, "manhattan", probation=3, random_state=0).fit(X).probation_ == 3
