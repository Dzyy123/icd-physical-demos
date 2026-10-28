"""
min_static_v2.py — STATIC engine for active causal discovery, SCORE-GAP version.

Key fix over v0/v1: a continuous learner over-commits (orients reversible edges
confidently, often wrong) and gives no honest uncertainty. The honest signal is
the SCORE GAP between an edge's two forced orientations:
  gap{i,j} = | score(force i->j) - score(force j->i) |   (score = total log-MSE)
Validated behaviour (dbg_scoregap.py):
  - reversible / MEC-ambiguous edge:  gap ~ 0      (score-equivalent)
  - v-structure / determined edge:    gap >> 0
  - after do(k): edges incident to k get large gap (resolved, correct direction)

Engine per round:
  1. For every skeleton edge fit both forced orientations -> (preferred dir, gap).
  2. Consensus orientation = lower-score (preferred) direction.   -> SHD
  3. Node uncertainty = #incident edges with gap < thr (still ambiguous).
  4. ACTIVE intervenes on the highest-uncertainty node; RANDOM picks uniformly;
     PASSIVE only adds observational data.
Metric: orientation-SHD vs #interventions (given-skeleton isolates orientation).

Run:
  python min_static_v2.py --K 6 --given_skeleton --seeds 12 --rounds 6 --cap 1
"""
import argparse, json, time
import numpy as np
import torch
import torch.nn as nn


# ----------------------------- SCM -----------------------------
def random_dag(K, p_edge, rng):
    perm = rng.permutation(K)
    A = np.zeros((K, K))
    for a in range(K):
        for b in range(a + 1, K):
            if rng.random() < p_edge:
                A[perm[a], perm[b]] = 1.0
    return A


def random_weights(A, rng, wlow=0.8, whigh=1.5):
    W = np.zeros_like(A)
    for (i, j) in np.argwhere(A > 0):
        W[i, j] = rng.choice([-1.0, 1.0]) * rng.uniform(wlow, whigh)
    return W


def topo_order(A):
    K = A.shape[0]
    indeg = A.sum(0).copy(); A2 = A.copy()
    avail = [k for k in range(K) if indeg[k] == 0]; order = []
    while avail:
        n = avail.pop(0); order.append(n)
        for j in range(K):
            if A2[n, j] > 0:
                A2[n, j] = 0; indeg[j] -= 1
                if indeg[j] == 0: avail.append(j)
    return order


def sample_scm(W, n, rng, noise, do_node=None):
    K = W.shape[0]
    if np.isscalar(noise): noise = np.full(K, float(noise))
    A = (np.abs(W) > 0).astype(float); order = topo_order(A)
    Z = np.zeros((n, K))
    for node in order:
        if do_node is not None and node == do_node:
            Z[:, node] = rng.normal(0.0, 1.0, size=n)
        else:
            parents = np.where(A[:, node] > 0)[0]
            mean = np.zeros(n)
            for p in parents:
                mean = mean + W[p, node] * Z[:, p]
            Z[:, node] = mean + rng.normal(0.0, noise[node], size=n)
    return Z


# -------------------- masked linear learner --------------------
class MaskedW(nn.Module):
    def __init__(self, K, dirmask):
        super().__init__()
        self.W = nn.Parameter(0.01 * torch.randn(K, K))
        self.register_buffer("m", torch.tensor(dirmask, dtype=torch.float32))

    def We(self):
        return self.W * self.m

    def data_score(self, Z, M):
        pred = Z @ self.We()
        resid = (Z - pred) ** 2
        keep = 1.0 - M
        mse = (resid * keep).sum(0) / (keep.sum(0) + 1e-8)
        return torch.log(mse + 1e-8).sum()

    def acyc(self):
        WW = self.We() ** 2
        return torch.trace(torch.matrix_exp(WW)) - WW.shape[0]


def fit_score(Z, M, K, dirmask, restart, outer, inner, lr=0.05, l1=0.02):
    """Fit masked linear SCM, return best (lowest) DATA score over restarts."""
    best = np.inf
    for _ in range(restart):
        model = MaskedW(K, dirmask)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        rho, alpha, hp = 1.0, 0.0, np.inf
        for o in range(outer):
            for t in range(inner):
                opt.zero_grad()
                h = model.acyc()
                loss = model.data_score(Z, M) + l1 * model.We().abs().sum() + alpha * h + 0.5 * rho * h * h
                loss.backward(); opt.step()
            with torch.no_grad():
                hn = float(model.acyc())
            alpha += rho * hn
            if hn > 0.25 * hp: rho *= 2.0
            hp = hn
            if hn < 1e-6: break
        with torch.no_grad():
            best = min(best, float(model.data_score(Z, M)))
    return best


def score_gap_orient(Z_np, M_np, K, skel, restart, outer, inner):
    """For each skeleton edge: fit both forced orientations, orient by lower
    score, record gap. Returns consensus directed adj + per-node ambiguity."""
    Z = torch.tensor(Z_np, dtype=torch.float32)
    M = torch.tensor(M_np, dtype=torch.float32)
    base = (1 - np.eye(K)) * skel
    cons = np.zeros((K, K))
    gaps = {}
    for i in range(K):
        for j in range(i + 1, K):
            if skel[i, j] == 0:
                continue
            m_ij = base.copy(); m_ij[j, i] = 0.0
            m_ji = base.copy(); m_ji[i, j] = 0.0
            s_ij = fit_score(Z, M, K, m_ij, restart, outer, inner)
            s_ji = fit_score(Z, M, K, m_ji, restart, outer, inner)
            if s_ij <= s_ji:
                cons[i, j] = 1
            else:
                cons[j, i] = 1
            gaps[(i, j)] = abs(s_ij - s_ji)
    return cons, gaps


def _ols_logmse(Z, j, parents, keep):
    """log mean residual^2 of OLS  z_j ~ z_parents  on rows `keep` (j not intervened)."""
    y = Z[keep, j]
    if y.shape[0] < 3:
        return 0.0
    if len(parents) == 0:
        resid = y - y.mean()
    else:
        X = Z[keep][:, parents]
        X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        resid = y - X1 @ beta
    return float(np.log((resid ** 2).mean() + 1e-8))


def score_gap_orient_closed(Z_np, M_np, K, skel):
    """Closed-form (OLS) score-equivalent orientation — exact for the
    linear-Gaussian latent layer, hundreds x faster than the gradient-descent
    fit. For each skeleton edge, compare its two forced orientations by the two
    incident nodes' log residual variance; each node regresses on all skeleton
    neighbours, and the edge's direction decides whether i has j as a parent or
    vice-versa (other nodes' parent sets are unchanged, so they cancel)."""
    Z = np.asarray(Z_np, dtype=float); M = np.asarray(M_np)
    nbr = {j: [k for k in range(K) if skel[j, k] > 0 or skel[k, j] > 0] for j in range(K)}
    cons = np.zeros((K, K)); gaps = {}
    for i in range(K):
        for j in range(i + 1, K):
            if skel[i, j] == 0:
                continue
            keep_i = M[:, i] == 0; keep_j = M[:, j] == 0
            ni, nj = nbr[i], nbr[j]
            ni_noj = [x for x in ni if x != j]
            nj_noi = [x for x in nj if x != i]
            s_ij = _ols_logmse(Z, j, nj, keep_j) + _ols_logmse(Z, i, ni_noj, keep_i)
            s_ji = _ols_logmse(Z, i, ni, keep_i) + _ols_logmse(Z, j, nj_noi, keep_j)
            if s_ij <= s_ji:
                cons[i, j] = 1
            else:
                cons[j, i] = 1
            gaps[(i, j)] = abs(s_ij - s_ji)
    return cons, gaps


def shd(true_A, pred_A):
    K = true_A.shape[0]
    t = (true_A > 0.5).astype(int); p = (pred_A > 0.5).astype(int)
    return sum(1 for i in range(K) for j in range(i + 1, K)
               if (t[i, j], t[j, i]) != (p[i, j], p[j, i]))


# ----------------------------- experiment -----------------------------
def make_scm(seed, K, p_edge):
    rng = np.random.default_rng(seed)
    A_true = random_dag(K, p_edge, rng)
    W = random_weights(A_true, rng)
    noise = rng.uniform(0.4, 1.2, size=K)
    return A_true, W, noise, rng


def cpdag_reversible_count(A_true):
    """#undirected edges in the CPDAG = #reversible edges that OBSERVATION alone
    cannot orient (only intervention can). Standard: orient v-structures, then
    Meek rules R1/R2 to a fixpoint; remaining skeleton edges are reversible."""
    K = A_true.shape[0]
    skel = ((A_true + A_true.T) > 0)
    directed = np.zeros((K, K), dtype=int)   # directed[i,j]=1 means i->j fixed
    # v-structures i->k<-j with i,j non-adjacent
    for k in range(K):
        for i in range(K):
            for j in range(i + 1, K):
                if skel[i, k] and skel[j, k] and not skel[i, j] \
                        and A_true[i, k] > 0 and A_true[j, k] > 0:
                    directed[i, k] = 1; directed[j, k] = 1
    # Meek R1/R2 to fixpoint
    changed = True
    while changed:
        changed = False
        for i in range(K):
            for j in range(K):
                if i == j or not skel[i, j] or directed[i, j] or directed[j, i]:
                    continue
                orient = False
                # R1: k->i, i-j, k not adjacent j  => i->j
                for k in range(K):
                    if k != i and k != j and directed[k, i] and not skel[k, j]:
                        orient = True; break
                # R2: i->k->j  => i->j
                if not orient:
                    for k in range(K):
                        if k != i and k != j and directed[i, k] and directed[k, j]:
                            orient = True; break
                if orient:
                    directed[i, j] = 1; changed = True
    rev = sum(1 for i in range(K) for j in range(i + 1, K)
              if skel[i, j] and not directed[i, j] and not directed[j, i])
    return rev


def make_scm_reversible(seed, K, p_edge, min_rev=2, max_tries=300):
    """Rejection-sample a DAG with >= min_rev reversible edges, so passive
    observation provably cannot fully orient it and active intervention has
    something to do. Avoids the 'observation already solves it' degenerate seeds."""
    base = np.random.default_rng(seed)
    A_last = None
    for _ in range(max_tries):
        s2 = int(base.integers(1 << 31))
        rng = np.random.default_rng(s2)
        A = random_dag(K, p_edge, rng)
        A_last = (A, rng)
        if cpdag_reversible_count(A) >= min_rev:
            W = random_weights(A, rng)
            noise = rng.uniform(0.4, 1.2, size=K)
            return A, W, noise, rng
    A, rng = A_last
    return A, random_weights(A, rng), rng.uniform(0.4, 1.2, size=K), rng


def node_ambiguity(gaps, K, thr):
    score = np.zeros(K)
    for (i, j), g in gaps.items():
        if g < thr:
            score[i] += 1.0; score[j] += 1.0
    return score


def run_one_seed(seed, args):
    K = args.K
    A_true, W, noise, rng = make_scm(seed, K, args.p_edge)
    if not args.given_skeleton:
        raise SystemExit("v2 currently supports --given_skeleton only")
    skel = ((A_true + A_true.T) > 0).astype(float)

    Z0 = sample_scm(W, args.n_obs, rng, noise); M0 = np.zeros_like(Z0)
    out = {}
    for policy in ["active", "random", "passive"]:
        Z, M = [Z0.copy()], [M0.copy()]
        n_done = np.zeros(K, dtype=int)
        curve, budget = [], []
        prng = np.random.default_rng(7000 + seed)
        for r in range(args.rounds + 1):
            cons, gaps = score_gap_orient(np.concatenate(Z, 0), np.concatenate(M, 0),
                                          K, skel, args.restart, args.outer, args.inner)
            curve.append(shd(A_true, cons))
            budget.append(int(n_done.sum()))
            if r == args.rounds:
                break
            if policy == "passive":
                Znew = sample_scm(W, args.n_int, rng, noise); Mnew = np.zeros_like(Znew)
            else:
                if policy == "active":
                    sc = node_ambiguity(gaps, K, args.gap_thr)
                    sc[n_done >= args.cap] = -1.0
                    if sc.max() > 0:
                        tgt = int(np.argmax(sc))
                    else:
                        avail = [k for k in range(K) if n_done[k] < args.cap]
                        tgt = int(prng.choice(avail)) if avail else int(prng.integers(K))
                else:
                    avail = [k for k in range(K) if n_done[k] < args.cap]
                    tgt = int(prng.choice(avail)) if avail else int(prng.integers(K))
                Znew = sample_scm(W, args.n_int, rng, noise, do_node=tgt)
                Mnew = np.zeros_like(Znew); Mnew[:, tgt] = 1.0
                n_done[tgt] += 1
            Z.append(Znew); M.append(Mnew)
        out[policy] = {"shd": curve, "budget": budget}
    return out, int(A_true.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--p_edge", type=float, default=0.4)
    ap.add_argument("--n_obs", type=int, default=3000)
    ap.add_argument("--n_int", type=int, default=600)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--cap", type=int, default=1)
    ap.add_argument("--restart", type=int, default=2)
    ap.add_argument("--outer", type=int, default=10)
    ap.add_argument("--inner", type=int, default=80)
    ap.add_argument("--gap_thr", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--given_skeleton", action="store_true")
    ap.add_argument("--out", default="min_static_v2_result.json")
    args = ap.parse_args()

    t0 = time.time()
    all_seeds, n_edges = [], []
    for s in range(args.seeds):
        res, ne = run_one_seed(s, args)
        all_seeds.append(res); n_edges.append(ne)
        print(f"[seed {s}] edges={ne} "
              f"active={res['active']['shd']} random={res['random']['shd']} "
              f"passive={res['passive']['shd']}", flush=True)

    agg = {}
    for policy in ["active", "random", "passive"]:
        arr = np.array([r[policy]["shd"] for r in all_seeds], dtype=float)
        agg[policy] = {
            "mean_shd": arr.mean(0).tolist(),
            "budget": all_seeds[0][policy]["budget"],
            "frac_solved": (arr[:, -1] == 0).mean().item(),
            "mean_budget_to_solve": float(np.mean([
                (np.argmax(row == 0) if (row == 0).any() else len(row)) for row in arr])),
        }
    summary = {"args": vars(args), "mean_edges": float(np.mean(n_edges)),
               "elapsed_s": round(time.time() - t0, 1), "agg": agg}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==== AGGREGATE (mean orientation-SHD vs round) ====")
    print(f"mean #edges={summary['mean_edges']:.1f}  elapsed {summary['elapsed_s']}s")
    for policy in ["active", "random", "passive"]:
        a = agg[policy]
        ms = " ".join(f"{v:.2f}" for v in a["mean_shd"])
        print(f"{policy:8s} budget={a['budget']}  SHD=[{ms}]  "
              f"solved@end={a['frac_solved']:.2f}  budget2solve={a['mean_budget_to_solve']:.2f}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
