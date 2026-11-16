# e-medoids

A k-medoids counterpart of [e-means](https://arxiv.org/abs/2611.09501) (Bessa et al., 2026):
mini-batch updates with exponentially averaged statistics, for arbitrary dissimilarities
and without ever building an n x n distance matrix.

## Idea

e-means keeps, for every cluster, an exponential moving average (EMA) of the *sum* and
*count* of the batch points assigned to it; the centroid is their ratio. A medoid has no
such closed form: it is the data point minimising the summed distance to its cluster.
The statistic worth averaging is therefore the **cost of candidate medoids**.

Each cluster `j` keeps a small pool of `m` candidate points. For a mini-batch:

1. **Assign** every batch point to its nearest medoid (`B k` distances), also recording
   the distance `d2` to the second nearest.
2. **Refresh the pool**: a few batch members of the cluster replace its worst-scoring
   candidates. A newcomer inherits the medoid's cost, so it starts level with the
   incumbent and has to beat it on data.
3. **EMA update**, as in e-means, with `x` running over the batch points assigned to `j`:

   | statistic | batch term | meaning |
   |---|---|---|
   | `S[j, c]` | `sum d(c, x)` | cost of candidate `c` as medoid of `j` |
   | `G[j, c]` | `sum max(0, d1(x) - d(c, x))` | loss saved if `c` became an *extra* medoid |
   | `R[j]` | `sum d2(x) - d1(x)` | loss added if medoid `j` were removed |

   `G` and `R` reuse distances already computed for `S`, so they are free.
4. **Medoid** `j` is the candidate with the smallest `S[j, c]`.
5. **Swap** (every other batch): while some candidate's saving `G` exceeds the cheapest
   medoid's removal cost `R`, that medoid's cluster restarts from the candidate. This is
   PAM's swap, estimated from EMAs, and it generalises e-means' dead-cluster splitting.

All candidates of a cluster are scored on the same points, so their scores differ by
EMAs of *paired* differences, which are far less noisy than independent estimates. The
step follows e-means' cosine decay, and a final exact pass picks each cluster's best
pooled candidate on the full data, which cannot increase the loss.

| | distance evaluations |
|---|---|
| e-medoids, `T` epochs | `T n (k + m - 1)`, plus `n (2k + m)` to finalise and label |
| alternating k-medoids, per iteration | `n k + sum_j n_j^2` |
| PAM / FasterPAM | `n^2 / 2`, and `4 n^2` bytes of memory |

## Usage

```python
from emedoids import EMedoids, pack_bits

X = pack_bits(fingerprints)                    # (n, n_bits) 0/1 array -> uint64 words
model = EMedoids(n_clusters=100, metric="tanimoto", random_state=0).fit(X)
model.medoid_indices_, model.labels_, model.inertia_
```

Built-in metrics: `tanimoto`, `euclidean`, `manhattan`, `cosine`. Any numba-jitted
`dist(x, y) -> float` works as `metric`.

## Results

Loss is the mean distance to the assigned medoid; gap is relative to the best method in
the table. One seed per row, 16 threads, compilation excluded. All methods share the same
distance kernels. Reproduce with `python examples/molecules.py` and `python examples/blobs.py`.

**Molecules**: ZINC-250k, 2048-bit Morgan fingerprints (radius 2), Tanimoto distance.

| n | k | method | time [s] | dist. evals | loss | gap |
|---|---|---|---|---|---|---|
| 20k | 100 | e-medoids (T=5) | 0.07 | 1.6e7 | 0.7176 | 1.2% |
| | | alternating | 0.09 | 1.8e7 | 0.7466 | 5.3% |
| | | FasterPAM | 20.1 | 2.0e8 | 0.7090 | 0% |
| 250k | 100 | e-medoids (T=2) | 0.44 | 1.1e8 | 0.7160 | 0% |
| | | e-medoids (T=5) | 0.90 | 2.0e8 | 0.7160 | 0% |
| | | alternating | 7.3 | 1.8e9 | 0.7478 | 4.4% |
| 20k | 1000 | e-medoids (T=5) | 0.30 | 1.4e8 | 0.6183 | 3.1% |
| | | alternating | 0.85 | 1.0e8 | 0.6350 | 5.9% |
| | | FasterPAM | 29.8 | 2.0e8 | 0.5996 | 0% |
| 250k | 1000 | e-medoids (T=5) | 4.9 | 1.8e9 | 0.6324 | 0% |
| | | alternating | 8.0 | 1.5e9 | 0.6675 | 5.6% |

**Blobs**: 20k points, 16-d, 50 unequal Gaussian clusters, Manhattan distance.

| method | time [s] | dist. evals | loss | gap |
|---|---|---|---|---|
| e-medoids (T=5) | 0.03 | 8.8e6 | 14.322 | 0.04% |
| alternating (k-medoids++ seeded) | 0.07 | 4.8e7 | 18.981 | 32.6% |
| FasterPAM | 8.6 | 2.0e8 | 14.316 | 0% |

Reading these:

- Where FasterPAM was run (up to 20k points) it gives the lowest loss. e-medoids is
  within about 1% of it at k = 100 in 100-300 times less time, and 3% behind at
  k = 1000 with only 20 points per cluster, its weakest regime.
- FasterPAM's matrix takes `4 n^2` bytes (10 GB at 50k points, 250 GB at 250k), so it
  was not run on the larger sets. There e-medoids is both faster and 4-6% lower in
  loss than alternating k-medoids.
- Alternating k-medoids stalls after 2-4 iterations on fingerprints, barely improving
  on its starting medoids.
- The time advantage over alternating shrinks when k is large, since both are then
  dominated by the `n k` assignment.

### What carried over from e-means, and what did not

Sweeps over 3 seeds on 15k-100k molecules and on the blobs:

- **EMA + cosine decay: yes.** Without averaging (step fixed at 1) swaps became erratic
  on the blobs (gap 4% +- 5% instead of 0.4%); on molecules the effect was within noise.
- **Scaling laws for batch size and step: no.** The best batch held about 10 points per
  cluster regardless of n, and an initial step of 1 was best throughout, so those are
  the defaults.
- **Dead-cluster splitting: superseded** by swaps, which cut the gap to FasterPAM from
  20% to 0.4% on blobs and from 2.6% to 1.6% on 15k molecules.
- **Random initialisation: yes.** With swaps, k-medoids++ seeding gave no further gain.

These conclusions rest on two datasets; the defaults are not tuned beyond them.

## Other problems we could look at:

Each needs only a jitted `dist(x, y)`; the cost of one distance is what makes avoiding
`n^2` and `sum n_j^2` matter.

- **Clustered chemistry**: ChEMBL or a combinatorial library has tighter series than
  ZINC's random drug-like molecules, so it tests whether swaps recover real structure
  (`examples/molecules.py --smiles file.smi`).
- **MD conformations** with RMSD after alignment: expensive distance, strongly
  unequal cluster sizes (metastable states).
- **Atomic environments** (e.g. SOAP descriptors with a kernel distance): medoids as
  representative configurations for training-set selection.
- **Sequences** under edit distance, or **time series** under DTW: no vector
  embedding exists, so k-means and e-means do not apply at all.
- **Very large k** (k ~ n / 20), as in diversity picking: the regime where e-medoids
  was weakest above.

## Layout

| file | content |
|---|---|
| [emedoids/emedoids.py](emedoids/emedoids.py) | the `EMedoids` estimator and the swap step |
| [emedoids/kernels.py](emedoids/kernels.py) | compiled loops, shared with the baselines |
| [emedoids/metrics.py](emedoids/metrics.py) | distances and fingerprint bit-packing |
| [emedoids/schedule.py](emedoids/schedule.py) | default batch size, cosine decay |
| [emedoids/init.py](emedoids/init.py) | random and k-medoids++ seeding |
| [emedoids/baselines.py](emedoids/baselines.py) | alternating k-medoids, FasterPAM wrapper |
| [examples/](examples/) | the benchmarks above |

## Setup

```
pip install -e .[examples,dev]
pytest
```
