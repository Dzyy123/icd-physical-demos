"""
acq_design.py — REAL information-theoretic / optimal-design acquisition scores.

Motivation (adversarial-review E1): the old code aliased `eig` to `coverage`
(same branch, same incident-edge count). That is not an EIG. Here we implement
two genuinely different, principled baselines that plug into the SAME closed-form
discovery engine (acq_closed.discover / do-criterion); only target SELECTION
changes.

Belief state. `cons` + `undirected` is a CPDAG/interventional essential graph.
Samples must be consistent DAG extensions of that object, not arbitrary acyclic
orientations (which can introduce forbidden v-structures). We therefore use a
randomized Dor--Tarsi sink-elimination procedure.

Two scores over this belief:

  EIG (true expected information gain). Under the (idealized) do-criterion,
  intervening on node k reveals the orientation of every skeleton edge incident
  to k. The information gained is the mutual information between "the orientation
  pattern of edges incident to k" and the full graph, which (since the graph
  determines the pattern) equals the Shannon ENTROPY of that pattern's
  distribution under the belief. score_EIG(k) = H(pattern_k).  This differs from
  coverage: it discounts incident edges whose orientation is already forced by
  acyclicity (zero entropy) and accounts for correlations among edges.

  HB (Hauser-Buhlmann / optimal experiment design). The classic objective is the
  EXPECTED NUMBER OF EDGES ORIENTED by the intervention, counting Meek
  propagation. score_HB(k) = E_belief[ #oriented skeleton edges after fixing the
  edges incident to k + Meek closure ].  This differs from coverage by the Meek
  propagation term (coverage = the incident count only, no propagation, no
  acyclicity).
"""
from collections import Counter
import numpy as np


def random_topo_order(cons, K, prng):
    """Random linear extension of the partial DAG `cons` (cons[a,b]=1 => a->b)
    via randomized Kahn's algorithm."""
    indeg = (cons > 0.5).sum(0).astype(int)          # in-degree of each node
    order, remaining = [], set(range(K))
    avail = [v for v in range(K) if indeg[v] == 0]
    while avail:
        v = avail.pop(int(prng.integers(len(avail))))
        order.append(v); remaining.discard(v)
        for w in range(K):
            if cons[v, w] > 0.5:
                indeg[w] -= 1
                if indeg[w] == 0:
                    avail.append(w)
    if remaining:                                     # cons had a cycle: shouldn't happen
        order.extend(sorted(remaining))
    return order


def sample_orientations(cons, undirected, K, n_samples, prng):
    """List of samples; each sample = dict edge(i,j)->+1 (i->j) / -1 (j->i),
    a consistent DAG extension of the current essential graph."""
    skel = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            if cons[i, j] > 0.5:
                skel[i, j] = skel[j, i] = 1.0
    for i, j in undirected:
        skel[i, j] = skel[j, i] = 1.0

    def is_undirected(d, i, j):
        return (
            skel[i, j] > 0.5
            and d[i, j] < 0.5
            and d[j, i] < 0.5
        )

    def draw_extension():
        d = (np.asarray(cons) > 0.5).astype(float).copy()
        remaining = set(range(K))
        while remaining:
            admissible = []
            for x in remaining:
                if any(d[x, y] > 0.5 for y in remaining - {x}):
                    continue
                parents = [
                    y for y in remaining - {x} if d[y, x] > 0.5
                ]
                und = [
                    y for y in remaining - {x}
                    if is_undirected(d, x, y)
                ]
                if not all(
                    skel[u, v] > 0.5
                    for idx, u in enumerate(und)
                    for v in und[idx + 1:]
                ):
                    continue
                if not all(
                    skel[u, p] > 0.5
                    for u in und for p in parents
                ):
                    continue
                admissible.append((x, und))
            if not admissible:
                return None
            x, und = admissible[int(prng.integers(len(admissible)))]
            for u in und:
                d[u, x] = 1.0
            remaining.remove(x)
        return d

    samples = []
    for _ in range(n_samples):
        d = draw_extension()
        if d is None:
            raise ValueError(
                "current PDAG has no consistent DAG extension; EIG is undefined")
        samples.append({
            (i, j): (1 if d[i, j] > 0.5 else -1)
            for (i, j) in undirected
        })
    return samples


def _incident(undirected, K):
    inc = {k: [] for k in range(K)}
    for (i, j) in undirected:
        inc[i].append((i, j)); inc[j].append((i, j))
    return inc


def eig_scores(cons, undirected, K, avail, prng, n_samples=160):
    """Expected information gain per candidate = entropy of the incident-edge
    orientation pattern under the belief. Tie-broken by incident-edge count."""
    scores = np.full(K, -1.0)
    if not undirected:
        return scores
    samples = sample_orientations(cons, undirected, K, n_samples, prng)
    inc = _incident(undirected, K)
    for k in avail:
        if not inc[k]:
            continue
        cnt = Counter(tuple(s[e] for e in inc[k]) for s in samples)
        tot = sum(cnt.values())
        ent = -sum((c / tot) * np.log(c / tot) for c in cnt.values())
        scores[k] = ent + 1e-6 * len(inc[k])          # tiny coverage tie-break
    return scores


def hb_scores(cons, undirected, K, avail, prng, meek_fn, skel, n_samples=48):
    """Hauser-Buhlmann optimal-design score = expected number of skeleton edges
    oriented after intervening on k (incident edges + Meek propagation)."""
    scores = np.full(K, -1.0)
    if not undirected:
        return scores
    samples = sample_orientations(cons, undirected, K, n_samples, prng)
    inc = _incident(undirected, K)
    tri = [(a, b) for a in range(K) for b in range(a + 1, K) if skel[a, b] > 0]
    for k in avail:
        if not inc[k]:
            continue
        tot = 0.0
        for s in samples:
            d = cons.copy()
            for (i, j) in inc[k]:
                if s[(i, j)] == 1:
                    d[i, j] = 1
                else:
                    d[j, i] = 1
            meek_fn(d, skel, K)
            tot += sum(1 for (a, b) in tri if d[a, b] or d[b, a])
        scores[k] = tot / len(samples)
    return scores
