# e-medoids

A k-medoids counterpart of [e-means](https://arxiv.org/abs/2611.09501) (Bessa et al., 2026):
mini-batch updates with exponentially averaged statistics, for arbitrary dissimilarities
and without ever building an n x n distance matrix.

## Method

**Problem.** Given points $x_1, \dots, x_n$ and a dissimilarity $d$, choose $k$ of the
points as medoids $\mu_1, \dots, \mu_k$ to minimise

$$L = \frac{1}{n} \sum_{i=1}^{n} \min_{j} d(x_i, \mu_j).$$

**From e-means to medoids.** For a mini-batch, e-means computes each cluster's batch sum
$s_j$ and count $n_j$, and replaces Lloyd's full sums by exponential moving averages (EMAs):

$$S_j \leftarrow (1-\alpha) S_j + \alpha s_j, \qquad N_j \leftarrow (1-\alpha) N_j + \alpha n_j, \qquad \text{centroid}_j = S_j / N_j.$$

A medoid has no such closed form: it is the data point with the smallest summed distance
to its cluster. So e-medoids keeps, for each cluster $j$, a small pool $P_j$ of candidate
points (the medoid among them; $m = 16$ by default, plus the probation slots described
below), and averages the **cost of each candidate**.

### One update

Update $t$ draws a batch $\mathcal{B}$ of $B$ points and does the following.

**1. Assign.** For every $x \in \mathcal{B}$, let $d_1(x) \le d_2(x)$ be its distances to the
nearest and second-nearest medoid, and assign $x$ to the nearest. Write $\mathcal{B}_j$ for
the batch points assigned to cluster $j$, and $n_j$ for their number.

**2. Refresh the pool.** Up to $m/2$ points of $\mathcal{B}_j$ enter $P_j$, replacing the
candidates with the largest $S_{jc}$ among those off probation (see Probation). Two
candidates are never replaced: the medoid, and the one with the largest estimated saving
$\hat{g}_{jc}$ (defined under Swaps), which is the cluster's best offer for a swap. A
newcomer $c$ starts with

$$S_{jc} = S_{j\mu_j}, \qquad G_{jc} = A_{jc} = 0,$$

i.e. it is presumed exactly as good as the current medoid.

**3. Batch statistics.** For every candidate $c \in P_j$:

$$s_{jc} = \sum_{x \in \mathcal{B}_j} d(c, x) \qquad \text{cost of } c \text{ as medoid of } j$$

$$g_{jc} = \sum_{x \in \mathcal{B}_j,\ x \ne c} \max\big(0,\ d_1(x) - d(c, x)\big) \qquad \text{loss saved if } c \text{ became an extra medoid}$$

and for every cluster:

$$r_j = \sum_{x \in \mathcal{B}_j} \big(d_2(x) - d_1(x)\big) \qquad \text{loss added if medoid } \mu_j \text{ were removed}$$

$g$ and $r$ reuse the distances computed for $s$ and for the assignment, so they are free.
A newcomer is itself in the batch and contributes $d(c, c) = 0$ to its own sum, so its
cost is rescaled to $s_{jc} \cdot n_j / (n_j - 1)$.

**4. EMA update.** With step $\alpha_t$, and all statistics initially zero:

$$S_{jc} \leftarrow (1-\alpha_t) S_{jc} + \alpha_t s_{jc}$$

$$G_{jc} \leftarrow (1-\alpha_t) G_{jc} + \alpha_t g_{jc}, \qquad A_{jc} \leftarrow (1-\alpha_t) A_{jc} + \alpha_t$$

$$R_j \leftarrow (1-\alpha_t) R_j + \alpha_t r_j$$

Unrolled, each statistic is a weighted sum of its past batch values, e.g.

$$S_{jc}^{(t)} = \sum_{\tau \le t} w_{\tau t}\ s_{jc}^{(\tau)}, \qquad w_{\tau t} = \alpha_\tau \prod_{u=\tau+1}^{t} (1-\alpha_u),$$

so a batch seen $q$ updates ago is down-weighted by roughly $(1-\alpha)^q$. This matters
because clusters keep changing: old batches were scored against old assignments.

**5. New medoid.**

$$\mu_j \leftarrow \arg\min_{c \in P_j} S_{jc}$$

taken over the candidates that are off probation. If the medoid changes, $G$ and $A$ of the old and new medoid are reset to zero, since
savings are measured relative to the medoid.

### Why newcomers inherit the medoid's cost

Because of the initialisation in step 2, a candidate $c$ that entered at update $t_0$
differs from the medoid $\mu$ it was compared with by

$$S_{jc}^{(t)} - S_{j\mu}^{(t)} = \sum_{\tau = t_0}^{t} w_{\tau t} \big(s_{jc}^{(\tau)} - s_{j\mu}^{(\tau)}\big),$$

an EMA of *paired* differences: both are scored on the same points $\mathcal{B}_j$, so the
noise from which points happened to be in the batch largely cancels. Only the ordering of
the $S_{jc}$ within a cluster is ever used, so no count $N_j$ is needed.

### Step size

As in e-means, the step is cosine-annealed over the $M = T \lfloor n/B \rfloor$ updates:

$$\alpha_t = \alpha_0 \left[\epsilon + (1-\epsilon)\ \tfrac{1}{2} \left(1 + \cos\frac{\pi t}{M}\right)\right], \qquad \alpha_0 = 1,\ \epsilon = 0.1.$$

Early updates ($\alpha \approx 1$) follow the current batch; late updates average over
roughly the last $1/\alpha = 10$ batches.

### Swaps

Steps 1-5 only ever replace a medoid by a point of its own cluster. Every second update,
medoids are also moved *between* clusters, as in PAM. The estimated saving of a candidate is

$$\hat{g}_{ic} = G_{ic} / A_{ic},$$

where $A_{ic}$ is the total EMA weight $c$ has received since it entered. Dividing by it
turns $G_{ic}$ into a weighted average of the batch savings $g_{ic}$, so candidates of
different ages are comparable. Let $j$ be the cluster with the smallest $R_j$ and
$(i, c)$, $i \ne j$, the candidate off probation with the largest $\hat{g}_{ic}$. If

$$\hat{g}_{ic} > R_j,$$

cluster $j$ restarts from $c$: $\mu_j \leftarrow c$, $P_j \leftarrow \lbrace c \rbrace$,
$S_j = G_j = A_j = 0$, $R_j \leftarrow \hat{g}_{ic}$. In the donor cluster, $c$ leaves $P_i$
and $G_i = A_i = 0$, because its savings were measured against the old partition. This
repeats with the remaining clusters, each taking part in at most one swap per round.

The test is conservative: $R_j$ assumes every point of $j$ falls back to its
second-nearest medoid (some may instead be picked up by $c$), and $\hat{g}_{ic}$ counts
only savings inside cluster $i$. A cluster that attracts no points has $R_j \to 0$ and is
relocated first, which is e-means' dead-cluster splitting as a special case.

### Probation

A newcomer's first score rests on a single batch, about ten points. With $m/2$ newcomers
per cluster and batch, one of them often beats the medoid, or shows a large saving
$\hat{g}$, by luck alone. Acting on that replaces or relocates a medoid for no real gain.

So a candidate is on **probation** until it has been scored on $h$ batches. Let $a_{jc}$
be the number of batches candidate $c$ has been scored on. While $a_{jc} < h$, it

- cannot be evicted from the pool (step 2),
- cannot become the medoid (step 5),
- cannot be used in a swap.

Its statistics are updated as usual in the meantime, so when it is finally judged the
evidence covers $h$ batches instead of one. Medoids set by the initialisation or by a
swap are exempt.

Candidates on probation sit in extra slots, so that waiting does not reduce the number
of newcomers. The pool holds $m/2$ established candidates plus $h$ cohorts of $m/2$
newcomers:

$$m_h = \frac{m}{2} + h \cdot \frac{m}{2}, \qquad m_1 = m = 16, \quad m_2 = 24.$$

$h = 1$ is the algorithm without probation. The default depends on the number of updates
$M$, which is known before the first batch:

$$h = 2 \ \text{ if } M \ge 60, \qquad h = 1 \ \text{ otherwise}.$$

Waiting costs updates, and a short run cannot afford it. Measured change in loss from
$h = 1$ to $h = 2$ on ZINC fingerprints (5-8 seeds, paired, $\pm$ standard error):

| n | k | epochs | updates $M$ | change in loss |
|---|---|---|---|---|
| 20k | 1000 | 5 | 10 | +0.15% ± 0.03 |
| 20k | 300 | 5 | 20 | +0.04% ± 0.05 |
| 20k | 100 | 2 | 38 | +0.10% ± 0.02 |
| 100k | 1000 | 5 | 60 | −0.12% ± 0.03 |
| 20k | 100 | 5 | 95 | −0.16% ± 0.03 |
| 100k | 300 | 5 | 120 | −0.10% ± 0.03 |
| 100k | 100 | 2 | 194 | −0.12% ± 0.07 |
| 100k | 100 | 5 | 488 | −0.25% ± 0.05 |
| 100k | 20 | 5 | 1950 | −0.23% ± 0.04 |

The gain comes from the second and third rule. At 100k molecules and $k = 100$, blocking
eviction alone changed nothing (+0.04% ± 0.08), also blocking promotion gave −0.13% ± 0.07,
and also blocking swaps gave −0.25% ± 0.05. A plain pool of the same 24 slots without
probation gave +0.06% ± 0.10. The extra slots cost 5-20% more distance evaluations.

Two things are open: whether 60 is the right threshold, and when $h = 3$ pays off. A
second dataset (ChEMBL, 3-5 seeds) agrees on the second and disagrees on the first:

| updates $M$ | ZINC, $h = 2$ | ZINC, $h = 3$ | ChEMBL, $h = 2$ | ChEMBL, $h = 3$ |
|---|---|---|---|---|
| 10-12 | +0.15% | +0.54% | −0.08% | +0.16% |
| 20-24 | +0.04% | +0.35% | −0.30% | −0.23% |
| 36-38 | +0.10% | +0.03% | −0.21% | −0.22% |
| 60-61 | −0.12% | −0.08% | −0.48% | −0.57% |
| 120-122 | −0.10% | −0.09% | −0.63% | −0.79% |
| 194-305 | −0.12% | −0.21% | −0.70% | −0.92% |
| 488 | −0.25% | −0.32% | −0.51% | −0.56% |
| 1950-2440 | −0.23% | −0.28% | −0.55% | −0.68% |

Standard errors are 0.03-0.10%. Rows pair runs with a similar number of updates; $n$ and
$k$ differ between the two datasets.

- **Threshold.** On ChEMBL $h = 2$ already helps from about 20 updates, where on ZINC it
  is neutral or slightly harmful. 60 is safe on both but leaves gains unused on ChEMBL.
- **$h = 3$.** From about 200 updates on, $h = 3$ was better than $h = 2$ in all six
  comparisons across both datasets, by 0.05-0.2%. Below that it is not reliably better.
- Probation is worth more on ChEMBL (0.5-0.9%) than on ZINC (0.1-0.3%).

### Finalisation

After the last update, all $n$ points are assigned once, giving exact clusters $C_j$, and

$$\mu_j \leftarrow \arg\min_{c \in P_j} \sum_{x \in C_j} d(c, x).$$

Since $\mu_j \in P_j$, this cannot increase $L$. A last assignment gives the labels.

### Cost

| | distance evaluations |
|---|---|
| e-medoids, $T$ epochs | $T n (k + m_h - 1)$, plus $n (2k + m_h)$ to finalise and label |
| alternating k-medoids, per iteration | $n k + \sum_j n_j^2$ |
| PAM / FasterPAM | $n^2 / 2$, and $4 n^2$ bytes of memory |

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
distance kernels. Times come from a single run and varied by up to 3x between runs on the
same machine; distance evaluations do not. Reproduce with `python examples/molecules.py` and `python examples/blobs.py`.

**Molecules**: ZINC-250k, 2048-bit Morgan fingerprints (radius 2), Tanimoto distance.

| n | k | method | time [s] | dist. evals | loss | gap |
|---|---|---|---|---|---|---|
| 20k | 100 | e-medoids (T=5) | 0.07 | 1.7e7 | 0.7169 | 1.1% |
| | | alternating | 0.08 | 1.8e7 | 0.7466 | 5.3% |
| | | FasterPAM | 21.2 | 2.0e8 | 0.7090 | 0% |
| 250k | 100 | e-medoids (T=2) | 0.48 | 1.2e8 | 0.7147 | 0.4% |
| | | e-medoids (T=5) | 0.96 | 2.1e8 | 0.7117 | 0% |
| | | alternating | 8.1 | 1.8e9 | 0.7478 | 5.1% |
| 20k | 1000 | e-medoids (T=5) | 0.47 | 1.4e8 | 0.6183 | 3.1% |
| | | alternating | 1.83 | 1.0e8 | 0.6350 | 5.9% |
| | | FasterPAM | 36.7 | 2.0e8 | 0.5996 | 0% |
| 250k | 1000 | e-medoids (T=5) | 3.8 | 1.8e9 | 0.6301 | 0% |
| | | alternating | 7.5 | 1.5e9 | 0.6675 | 5.9% |

The 20k, k = 1000 run has only 10 updates, so it runs without probation.

**Blobs**: 20k points, 16-d, 50 unequal Gaussian clusters, Manhattan distance.

| method | time [s] | dist. evals | loss | gap |
|---|---|---|---|---|
| e-medoids (T=5) | 0.13 | 9.7e6 | 14.316 | 0.00% |
| alternating (k-medoids++ seeded) | 0.21 | 4.8e7 | 18.981 | 32.6% |
| FasterPAM | 14.4 | 2.0e8 | 14.316 | 0% |

Reading these:

- Where FasterPAM was run (up to 20k points) it gives the lowest loss. e-medoids is
  within about 1% of it at k = 100 in 100-300 times less time, and 3% behind at
  k = 1000 with only 20 points per cluster, its weakest regime.
- FasterPAM's matrix takes `4 n^2` bytes (10 GB at 50k points, 250 GB at 250k), so it
  was not run on the larger sets. There e-medoids is both faster and about 5% lower
  in loss than alternating k-medoids.
- Alternating k-medoids stalls after 2-4 iterations on fingerprints, barely improving
  on its starting medoids.
- The time advantage over alternating shrinks when k is large, since both are then
  dominated by the `n k` assignment.

### Against OneBatchPAM

[OneBatchPAM](https://arxiv.org/abs/2501.19285) (de Mathelin et al., AAAI 2025) is the
closest published method: FasterPAM-style swaps over all $n$ candidates, with the loss
estimated on one random batch of $m$ points. It computes an $n \times m$ distance matrix
once ($n m$ evaluations, $4 n m$ bytes) and then sweeps over it.

Setup: the authors' package (`onebatch` 0.1.0) with its defaults, $m = 100 \ln(n k)$
and importance weighting on. Its distance matrix is computed with our kernels, so both
methods run the same distance code. Times include assigning all points to their medoid.
Reproduce with `python examples/vs_onebatchpam.py zinc --k 100` (or `chembl`, or
`blobs --n 500000`).

ChEMBL is a random sample of 500k compounds from ChEMBL 37, with the same fingerprints
and distance as ZINC. It has more close neighbours: in sets of equal size, 36% of ChEMBL
compounds have a neighbour within Tanimoto distance 0.3, against 14% for ZINC.

**16 threads**, 3 seeds, median time and mean loss. OneBatchPAM's swap loop is
single-threaded; only its distance matrix uses the threads.

| data | k | method | time [s] | dist. evals | loss |
|---|---|---|---|---|---|
| ZINC 250k | 100 | e-medoids (T=1) | 0.30 | 0.9e8 | 0.7168 |
| | | e-medoids (T=5) | 0.95 | 2.1e8 | 0.7120 |
| | | OneBatchPAM, m = 1703 | 3.27 | 4.5e8 | 0.7256 |
| ZINC 250k | 1000 | e-medoids (T=1) | 2.5 | 7.6e8 | 0.6395 |
| | | e-medoids (T=5) | 7.6 | 1.8e9 | 0.6299 |
| | | OneBatchPAM, m = 1933 | 5.9 | 7.3e8 | 0.6600 |
| ChEMBL 500k | 100 | e-medoids (T=1) | 0.83 | 1.7e8 | 0.7302 |
| | | e-medoids (T=5) | 2.23 | 4.2e8 | 0.7274 |
| | | OneBatchPAM, m = 1772 | 6.29 | 9.4e8 | 0.7427 |
| ChEMBL 500k | 1000 | e-medoids (T=1) | 3.4 | 1.5e9 | 0.6483 |
| | | e-medoids (T=5) | 8.3 | 3.6e9 | 0.6415 |
| | | OneBatchPAM, m = 2003 | 12.4 | 1.5e9 | 0.6841 |
| blobs 500k | 100 | e-medoids (T=1) | 0.33 | 1.7e8 | 13.975 |
| | | e-medoids (T=5) | 1.13 | 4.2e8 | 13.858 |
| | | OneBatchPAM, m = 1772 | 5.94 | 9.4e8 | 14.531 |
| blobs 1M | 100 | e-medoids (T=1) | 0.81 | 3.5e8 | 13.850 |
| | | e-medoids (T=5) | 2.44 | 8.4e8 | 13.743 |
| | | OneBatchPAM, m = 641 (memory-capped) | 5.1 | 7.4e8 | 16.224 |

The blobs are 16-d with unequal sizes, Manhattan distance. Using the data point nearest
each true center as medoid gives 13.932 (500k) and 13.792 (1M). At 1M points the default
$m = 1842$ needs a 6.9 GB matrix, which did not fit on the 16 GB test machine, so that
row uses the largest batch that did and is not a like-for-like comparison.

**1 thread**, 2 seeds, which removes the threading difference:

| data | k | e-medoids (T=1) | e-medoids (T=5) | OneBatchPAM |
|---|---|---|---|---|
| ZINC 250k | 100 | 1.2 s, 0.7172 | 3.2 s, 0.7119 | 7.0 s, 0.7255 |
| ZINC 250k | 1000 | 7.5 s, 0.6395 | 17.4 s, 0.6300 | 12.1 s, 0.6597 |
| ChEMBL 500k | 100 | 3.9 s, 0.7303 | 8.7 s, 0.7272 | 19.0 s, 0.7425 |
| ChEMBL 500k | 1000 | 14.2 s, 0.6480 | 35.6 s, 0.6415 | 24.2 s, 0.6842 |
| blobs 500k | 100 | 1.2 s, 13.979 | 3.5 s, 13.851 | 8.8 s, 14.549 |

**Is this only because OneBatchPAM's default batch is small?** No. A larger batch does
improve its loss, but at a cost. In a separate test on 100k ZINC molecules ($k = 100$,
16 threads), a batch four times the default with weighting off brought OneBatchPAM from
0.7246 to 0.7133, level with e-medoids' 0.7137 on the same data. That run took 10.4 s
and a 2.6 GB matrix, against under 1 s for e-medoids, and the same batch at 250k points
would need 6.8 GB. On 100k blobs it closed most of the gap (14.83 to 14.18, with
e-medoids at 14.11).

Reading these:

- With default settings e-medoids reaches a lower loss in less time on every problem,
  with one thread or sixteen. The exception in time is $k = 1000$ at five epochs on one
  thread, where e-medoids is slower but 4.5% (ZINC) and 6.2% (ChEMBL) lower in loss;
  one epoch is already both faster and lower.
- The gap in loss is larger on ChEMBL than on ZINC (2.1% against 1.9% at $k = 100$,
  6.6% against 4.8% at $k = 1000$), the dataset with more real structure.
- Memory is the practical limit: its matrix grows with $n$, while e-medoids keeps
  $k \cdot m_h$ candidates whatever $n$.
- The $\log n$ in OneBatchPAM's $O(n \log n)$ grows slowly ($m$ rises 8% from 250k to 1M
  points), so the gap is mostly a constant factor: per point it evaluates $m \approx 1700$
  distances, against $k + m_h - 1 = 123$ per epoch for e-medoids at $k = 100$. At
  $k = 1000$ the two budgets are similar.
- $k = 1000$ is outside the range OneBatchPAM's paper tested ($k \le 100$). Its default
  batch then holds about two points per cluster.

### What carried over from e-means, and what did not

Sweeps over 3 seeds on 15k-100k molecules and on the blobs, run before probation was
added:

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

- **More clustered chemistry**: ChEMBL is covered above. A single target's compounds or
  a combinatorial library would have tighter series still, with clusters closer in size
  to the series themselves (`examples/molecules.py --smiles file.smi`).
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
| [emedoids/baselines.py](emedoids/baselines.py) | alternating k-medoids, FasterPAM and OneBatchPAM wrappers |
| [examples/](examples/) | the benchmarks above |

## Setup

```
pip install -e .[examples,dev]
pytest
```
