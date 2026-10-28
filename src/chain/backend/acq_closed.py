"""
acq_closed.py — Closed-form observational CPDAG + interventional orientation.

Frozen graph contract:
  * observational PC returns one object: skeleton + separating sets;
  * unshielded colliders are read from those separating sets and sound Meek
    R1--R4 closure produces a CPDAG only when collider information is complete,
    unambiguous, and has a consistent DAG extension;
  * interventions may orient only edges that are still undirected in that
    observational CPDAG; observationally compelled arrows are immutable;
  * after each accepted do-arrow, sound Meek closure gives an interventional
    essential graph; only a fully directed acyclic result is called a DAG.

The former skeleton -> direct-do-all-edges engine is retained only behind the
explicit ``legacy_skeleton=True`` ablation flag.

For an undirected edge (i,j), the do-criterion uses marginal dependence inside
the single-target do(i) subset. If z_i and z_j remain dependent, infer i->j; if
the dependence disappears, infer j->i. Two-sided data must show the
cross-threshold keep/break pattern; otherwise the edge remains undirected.

ACQUISITION = the agent's motive = pick the node that resolves the most
still-undirected edges. Policies (also the baseline matrix):
  random, roundrobin, maxedge, coverage(ours), eig(true), hb(Hauser-Buhlmann),
  oracle.

  coverage(ours): # undirected edges incident to k (myopic count).
  eig(true):      real expected information gain = entropy of the incident-edge
                  orientation pattern under the belief over acyclic orientations
                  (acq_design.eig_scores). Discounts already-forced edges.
  hb:             Hauser-Buhlmann-style design = expected #edges oriented after
                  do(k) + conservative acyclicity closure (acq_design.hb_scores).
NOTE: `eig` and `coverage` are now DISTINCT branches (previously aliased).

Fair benchmark: graphs are rejection-sampled to have >= min_rev reversible edges
(make_scm_reversible), so passive cannot solve them and active has work to do.

Run:  python acq_closed.py --K 7 --seeds 30 --rounds 8 --out acq_closed_k7.json
"""
import argparse, hashlib, json, os, time, warnings
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np
from scipy.stats import norm

# Public demo adapter: defer synthetic-benchmark-only imports. Those helpers
# depend on PyTorch, which is not needed for graph refitting on saved readouts.
def make_scm_reversible(*args, **kwargs):
    from min_static_v2 import make_scm_reversible as impl
    return impl(*args, **kwargs)

def cpdag_reversible_count(*args, **kwargs):
    from min_static_v2 import cpdag_reversible_count as impl
    return impl(*args, **kwargs)

def sample_scm(*args, **kwargs):
    from min_static_v2 import sample_scm as impl
    return impl(*args, **kwargs)
from acq_design import eig_scores, hb_scores
from orientation_stats import (
    ENGINE_NAMES as STATISTICAL_ORIENTATION_ENGINES,
    orientation_evidence as statistical_orientation_evidence,
)
from acc_protocol import (
    ROLE_ADAPTER,
    ROLE_EVAL,
    ROLE_GRAPH,
    make_acc_role_split,
)
from direct_edge_evidence import (
    EFFECT,
    EQUIVALENT_ZERO,
    PAIR_ABSENT,
    PAIR_DIRECT,
    PAIR_UNKNOWN,
    consensus_pair_evidence,
    effect_evidence,
    pair_direct_evidence,
)
from representation_safety import SAFE, audit_representation_safety

POLICIES = ["random", "roundrobin", "maxedge", "coverage", "eig", "hb", "oracle"]
ORIENTATION_ENGINES = ("direct_do",) + STATISTICAL_ORIENTATION_ENGINES


def _array_sha256(value):
    arr = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _canonical_sha256(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _acc_role_view_sha256(Z, M, U, row_ids, roles, row_mask, role):
    """Bind one physical ACC role view without serializing raw outcomes."""
    row_mask = np.asarray(row_mask, dtype=bool)
    payload = {
        "protocol": "acc_crossrole_evidence_view_v1",
        "role": str(role),
        "Z_sha256": _array_sha256(np.asarray(Z)[row_mask]),
        "M_sha256": _array_sha256(np.asarray(M)[row_mask]),
        "U_sha256": (
            "missing-explicit-intervention-values" if U is None else
            _array_sha256(np.asarray(U)[row_mask])),
        "row_ids_sha256": _canonical_sha256([
            row_id for row_id, keep in zip(row_ids, row_mask) if keep]),
        "roles_sha256": _array_sha256(np.asarray(roles)[row_mask]),
    }
    return _canonical_sha256(payload)


def _manifest_core(manifest):
    return {
        key: value for key, value in dict(manifest).items()
        if key not in ("manifest_sha256", "manifest_path")
    }


def _validate_preaction_manifests(
        manifests, Z, M, U, row_ids, roles, access_ledger,
        require_persisted_receipts=True):
    """Rebuild the D_A/D_G command view from pre-action receipts.

    The access-ledger self-hash is not an authority boundary.  A topology
    mutation is authorized only after this consumer independently reloads each
    append-only receipt and reconstructs row IDs, roles, intervention masks,
    and commands in exactly the materialized order.
    """
    if not isinstance(manifests, list) or not manifests:
        return False, ["verified_preaction_manifests_missing"]
    K = int(M.shape[1])
    expected_ids, expected_roles = [], []
    expected_M, expected_U, manifest_hashes = [], [], []
    reasons = []
    for index, manifest in enumerate(manifests):
        if not isinstance(manifest, dict):
            reasons.append(f"manifest_{index}_not_a_mapping")
            continue
        core = _manifest_core(manifest)
        manifest_sha = str(manifest.get("manifest_sha256", ""))
        if manifest.get("protocol") \
                != "acc_preoutcome_intervention_commitment_v1" \
                or _canonical_sha256(core) != manifest_sha:
            reasons.append(f"manifest_{index}_content_hash_mismatch")
            continue
        if not bool(manifest.get("command_recorded_before_execution", False)) \
                or not bool(manifest.get(
                    "assignment_committed_before_outcome", False)) \
                or bool(manifest.get("contains_outcome", True)):
            reasons.append(f"manifest_{index}_not_preaction_only")
            continue
        path = manifest.get("manifest_path")
        if require_persisted_receipts:
            if not isinstance(path, str) or not os.path.isabs(path) \
                    or not os.path.isfile(path):
                reasons.append(f"manifest_{index}_persisted_receipt_missing")
                continue
            try:
                with open(path) as handle:
                    persisted = json.load(handle)
            except (OSError, ValueError):
                reasons.append(f"manifest_{index}_persisted_receipt_unreadable")
                continue
            if _manifest_core(persisted) != core \
                    or persisted.get("manifest_sha256") != manifest_sha:
                reasons.append(f"manifest_{index}_persisted_receipt_mismatch")
                continue
        ids = [str(item) for item in manifest.get("row_ids", ())]
        block_roles = np.asarray(manifest.get("roles", ()), dtype="<U3")
        command = np.asarray(manifest.get("command_values", ()), dtype=float)
        target = int(manifest.get("target", -1))
        n_rows = int(manifest.get("n_rows", -1))
        if target < 0 or target >= K or len(ids) != n_rows \
                or block_roles.shape != (n_rows,) \
                or command.shape != (n_rows,) \
                or not set(block_roles.tolist()).issubset(
                    {ROLE_ADAPTER, ROLE_GRAPH, ROLE_EVAL}) \
                or _array_sha256(command) \
                != manifest.get("command_sha256"):
            reasons.append(f"manifest_{index}_row_or_command_contract_invalid")
            continue
        block_M = np.zeros((n_rows, K), dtype=float)
        block_M[:, target] = 1.0
        block_U = np.full((n_rows, K), np.nan, dtype=float)
        block_U[:, target] = command
        keep = block_roles != ROLE_EVAL
        expected_ids.extend(
            item for item, selected in zip(ids, keep) if selected)
        expected_roles.extend(block_roles[keep].tolist())
        expected_M.append(block_M[keep])
        expected_U.append(block_U[keep])
        manifest_hashes.append(manifest_sha)
    if reasons:
        return False, reasons
    rebuilt_M = np.concatenate(expected_M, axis=0)
    rebuilt_U = np.concatenate(expected_U, axis=0)
    if expected_ids != list(row_ids) \
            or expected_roles != roles.tolist() \
            or not np.array_equal(rebuilt_M, M) \
            or not np.array_equal(rebuilt_U, U, equal_nan=True):
        reasons.append("manifest_reconstruction_disagrees_with_materialized_view")
    if manifest_hashes != list(access_ledger.get(
            "preoutcome_commitment_hashes", ())):
        reasons.append("manifest_hash_sequence_disagrees_with_access_ledger")
    return not reasons, reasons


def _edge_key(i, j):
    return (int(min(i, j)), int(max(i, j)))


def _undirected_edges(skel, directed):
    K = skel.shape[0]
    return [
        (i, j)
        for i in range(K)
        for j in range(i + 1, K)
        if (skel[i, j] > 0.5 or skel[j, i] > 0.5)
        and directed[i, j] < 0.5
        and directed[j, i] < 0.5
    ]


@dataclass
class EssentialGraph:
    """Auditable CPDAG / interventional-essential-graph representation."""

    skeleton: np.ndarray
    directed: np.ndarray
    undirected: list
    graph_type: str
    valid: bool
    source: str
    sepsets: dict = field(default_factory=dict)
    colliders: list = field(default_factory=list)
    ambiguous_triples: list = field(default_factory=list)
    failure_reasons: list = field(default_factory=list)
    collider_complete: bool = False
    collider_consistent: bool = False
    ci_search_complete: bool = False
    meek_applied: bool = False
    provenance: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def copy(self):
        return EssentialGraph(
            skeleton=self.skeleton.copy(),
            directed=self.directed.copy(),
            undirected=list(self.undirected),
            graph_type=self.graph_type,
            valid=bool(self.valid),
            source=self.source,
            sepsets={k: list(v) for k, v in self.sepsets.items()},
            colliders=list(self.colliders),
            ambiguous_triples=list(self.ambiguous_triples),
            failure_reasons=list(self.failure_reasons),
            collider_complete=bool(self.collider_complete),
            collider_consistent=bool(self.collider_consistent),
            ci_search_complete=bool(self.ci_search_complete),
            meek_applied=bool(self.meek_applied),
            provenance=dict(self.provenance),
            metadata=dict(self.metadata),
        )

    def to_dict(self, include_graph=True):
        sepsets = [
            {
                "pair": [int(i), int(j)],
                "sets": [list(map(int, S)) for S in sets_],
            }
            for (i, j), sets_ in sorted(self.sepsets.items())
        ]
        provenance = [
            {
                "edge": [int(i), int(j)],
                "source": str(source),
            }
            for (i, j), source in sorted(self.provenance.items())
        ]
        out = {
            "graph_type": self.graph_type,
            "valid": bool(self.valid),
            "source": self.source,
            "collider_complete": bool(self.collider_complete),
            "collider_consistent": bool(self.collider_consistent),
            "ci_search_complete": bool(self.ci_search_complete),
            "meek_applied": bool(self.meek_applied),
            "colliders": [list(map(int, t)) for t in self.colliders],
            "ambiguous_triples": [
                list(map(int, t)) for t in self.ambiguous_triples
            ],
            "failure_reasons": list(self.failure_reasons),
            "sepsets": sepsets,
            "provenance": provenance,
            "n_directed": int((self.directed > 0.5).sum()),
            "n_undirected": int(len(self.undirected)),
            "metadata": dict(self.metadata),
        }
        if include_graph:
            out.update({
                "skeleton": self.skeleton.astype(int).tolist(),
                "directed": self.directed.astype(int).tolist(),
                "undirected": [list(map(int, e)) for e in self.undirected],
            })
        return out


# --------------------------- CI test (closed form) ---------------------------
def _pcorr(Z, i, j, S, rows=None):
    """Partial correlation of i,j given set S (optionally on a row subset), via
    the precision-matrix formula on the [i,j]+S correlation submatrix. Works
    entirely on the O(1) correlation matrix (no large-number matmul) -> stable."""
    A = Z if rows is None else Z[rows]
    if A.shape[0] < len(S) + 4:
        return 0.0
    cols = [i, j] + list(S)
    sub = A[:, cols].astype(float)
    sub = sub - sub.mean(0)
    sd = sub.std(0); sd[sd < 1e-8] = 1.0
    sub = sub / sd
    if len(S) == 0:
        c = np.corrcoef(sub[:, 0], sub[:, 1])[0, 1]
        return 0.0 if np.isnan(c) else float(c)
    C = np.corrcoef(sub, rowvar=False) + 1e-3 * np.eye(len(cols))
    P = np.linalg.pinv(C)
    denom = np.sqrt(P[0, 0] * P[1, 1])
    if denom < 1e-12:
        return 0.0
    pc = -P[0, 1] / denom
    return 0.0 if np.isnan(pc) else float(pc)


def _ci_independent(Z, i, j, S, alpha, min_eff):
    """Linear-Gaussian Fisher-z CI decision used by the PC stage."""
    r = float(np.clip(_pcorr(Z, i, j, S), -0.9999, 0.9999))
    if abs(r) < min_eff:
        return True
    n = len(Z)
    z = 0.5 * np.log((1.0 + r) / (1.0 - r))
    stat = np.sqrt(max(n - len(S) - 3, 1)) * abs(z)
    return bool(2.0 * (1.0 - norm.cdf(stat)) > alpha)


def pc_skeleton_sepsets(Z, K, alpha=0.01, max_order=None, min_eff=0.03):
    """PC-stable skeleton phase with all minimal-order separating sets.

    The adjacency snapshot is frozen within each conditioning order. For an
    edge removed at order ``l``, every independent candidate set of size ``l``
    from either endpoint's PC adjacency is retained. Keeping all such sets lets
    the conservative collider step distinguish unambiguous colliders,
    unambiguous non-colliders, and finite-sample conflicts.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[1] != K:
        raise ValueError(f"expected Z shape [N,{K}], got {Z.shape}")
    max_order = K - 2 if max_order is None else int(max_order)
    max_order = max(0, min(max_order, K - 2))
    skel = np.ones((K, K), dtype=float) - np.eye(K)
    adj = {i: set(range(K)) - {i} for i in range(K)}
    sepsets = {}
    removal_order = {}
    n_tests = 0

    for order in range(max_order + 1):
        snapshot = {i: set(v) for i, v in adj.items()}
        edges = [
            (i, j)
            for i in range(K)
            for j in range(i + 1, K)
            if skel[i, j] > 0.5
        ]
        removals = {}
        for i, j in edges:
            candidates = set()
            for pool in (snapshot[i] - {j}, snapshot[j] - {i}):
                if len(pool) >= order:
                    candidates.update(combinations(sorted(pool), order))
            independent = []
            for S_tuple in sorted(candidates):
                n_tests += 1
                if _ci_independent(
                        Z, i, j, set(S_tuple), alpha=alpha,
                        min_eff=min_eff):
                    independent.append(tuple(map(int, S_tuple)))
            if independent:
                removals[(i, j)] = sorted(set(independent))

        for (i, j), sets_ in removals.items():
            skel[i, j] = skel[j, i] = 0.0
            adj[i].discard(j)
            adj[j].discard(i)
            sepsets[(i, j)] = sets_
            removal_order[(i, j)] = int(order)

        # Standard PC can stop once no remaining adjacency supports order+1.
        next_order = order + 1
        if not any(
            len(adj[i] - {j}) >= next_order
            or len(adj[j] - {i}) >= next_order
            for i in range(K)
            for j in range(i + 1, K)
            if skel[i, j] > 0.5
        ):
            break

    next_order = max_order + 1
    search_complete = not any(
        len(adj[i] - {j}) >= next_order
        or len(adj[j] - {i}) >= next_order
        for i in range(K)
        for j in range(i + 1, K)
        if skel[i, j] > 0.5
    )

    Zc = Z - Z.mean(0, keepdims=True)
    cov = (Zc.T @ Zc) / max(len(Zc), 1)
    prec = np.linalg.pinv(cov + 1e-6 * np.eye(K))
    d = np.sqrt(np.clip(np.diag(prec), 1e-12, None))
    strength = np.abs(-prec / np.outer(d, d))
    np.fill_diagonal(strength, 0.0)
    report = {
        "alpha": float(alpha),
        "min_eff": float(min_eff),
        "max_order": int(max_order),
        "n_ci_tests": int(n_tests),
        "search_complete": bool(search_complete),
        "removal_order": {
            f"{i}-{j}": int(order)
            for (i, j), order in sorted(removal_order.items())
        },
    }
    return skel, strength, sepsets, report


def sepsets_for_fixed_skeleton(
        Z, skel, alpha=0.01, max_order=None, min_eff=0.03):
    """Compatibility bridge: estimate sepsets for an externally supplied skeleton.

    This is not the default learned-PC contract and is labelled as such. It is
    useful for old visualization/evaluation callers while they migrate: unlike
    the removed legacy engine, it still forbids direct-do orientation of every
    skeleton edge.
    """
    Z = np.asarray(Z, dtype=float)
    skel = ((np.asarray(skel) + np.asarray(skel).T) > 0).astype(float)
    K = skel.shape[0]
    max_order = K - 2 if max_order is None else int(max_order)
    max_order = max(0, min(max_order, K - 2))
    sepsets = {}
    incomplete_pairs = []
    n_tests = 0
    for i in range(K):
        for j in range(i + 1, K):
            if skel[i, j] > 0.5:
                continue
            common = [
                k for k in range(K)
                if skel[i, k] > 0.5 and skel[j, k] > 0.5
            ]
            if not common:
                continue
            found = []
            others = [k for k in range(K) if k not in (i, j)]
            for order in range(max_order + 1):
                for S_tuple in combinations(others, order):
                    n_tests += 1
                    if _ci_independent(
                            Z, i, j, set(S_tuple), alpha=alpha,
                            min_eff=min_eff):
                        found.append(tuple(map(int, S_tuple)))
                if found:
                    break
            if found:
                sepsets[(i, j)] = sorted(set(found))
            else:
                incomplete_pairs.append((i, j))
    report = {
        "alpha": float(alpha),
        "min_eff": float(min_eff),
        "max_order": int(max_order),
        "n_ci_tests": int(n_tests),
        "search_complete": not incomplete_pairs and max_order >= K - 2,
        "incomplete_pairs": [list(e) for e in incomplete_pairs],
        "compatibility_mode": True,
    }
    return sepsets, report


# --------------------------- graph representation ---------------------------
def _has_directed_path(directed, source, target):
    """Whether `source` reaches `target` through currently directed edges."""
    if source == target:
        return True
    seen = {source}
    stack = [source]
    while stack:
        u = stack.pop()
        for v in np.where(directed[u] > 0.5)[0]:
            v = int(v)
            if v == target:
                return True
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def _orient_acyclic(directed, source, target):
    """Add source->target iff it does not conflict or create a directed cycle."""
    if directed[source, target] > 0.5:
        return True
    if directed[target, source] > 0.5:
        return False
    if _has_directed_path(directed, target, source):
        return False
    directed[source, target] = 1.0
    return True


def _is_acyclic(directed):
    K = directed.shape[0]
    indeg = (directed > 0.5).sum(0).astype(int)
    avail = [k for k in range(K) if indeg[k] == 0]
    n_seen = 0
    while avail:
        u = avail.pop()
        n_seen += 1
        for v in np.where(directed[u] > 0.5)[0]:
            v = int(v)
            indeg[v] -= 1
            if indeg[v] == 0:
                avail.append(v)
    return n_seen == K


def graph_kind(directed, undirected):
    """Name the returned graph object without conflating skeleton/PDAG/DAG."""
    n_directed = int((np.asarray(directed) > 0.5).sum())
    if undirected:
        return "skeleton" if n_directed == 0 else "PDAG"
    return "DAG" if _is_acyclic(np.asarray(directed)) else "INVALID"


def pdag_shd(true_A, skel, directed):
    """Pairwise SHD for a PDAG estimate.

    A missing/extra adjacency, a wrong direction, and an unresolved true edge
    each cost one. This prevents an undirected false-positive skeleton edge from
    disappearing merely because `directed` stores only the directed part.
    """
    K = true_A.shape[0]
    err = 0
    for i in range(K):
        for j in range(i + 1, K):
            true_ij = true_A[i, j] > 0.5
            true_ji = true_A[j, i] > 0.5
            pred_edge = skel[i, j] > 0.5 or skel[j, i] > 0.5
            if not (true_ij or true_ji):
                err += int(pred_edge)
            elif not pred_edge:
                err += 1
            elif true_ij:
                err += int(not (directed[i, j] > 0.5 and directed[j, i] < 0.5))
            else:
                err += int(not (directed[j, i] > 0.5 and directed[i, j] < 0.5))
    return err


# --------------------------- CPDAG construction and Meek closure ---------------------------
def _adjacent(skel, i, j):
    return bool(skel[i, j] > 0.5 or skel[j, i] > 0.5)


def _is_undirected(skel, directed, i, j):
    return (
        _adjacent(skel, i, j)
        and directed[i, j] < 0.5
        and directed[j, i] < 0.5
    )


def consistent_dag_extension(skel, directed):
    """Dor--Tarsi-style consistent extension test for a PDAG.

    Repeatedly remove a node with no outgoing arrow whose undirected neighbors
    form a clique and are adjacent to all its directed parents. Orienting those
    neighbors into the removed node introduces neither a new unshielded
    collider nor a directed cycle.
    """
    skel = ((np.asarray(skel) + np.asarray(skel).T) > 0).astype(float)
    directed = (np.asarray(directed) > 0.5).astype(float)
    K = skel.shape[0]
    if np.any((directed > 0.5) & (directed.T > 0.5)) \
            or not _is_acyclic(directed):
        return False, None
    extension = directed.copy()
    remaining = set(range(K))
    while remaining:
        chosen = None
        for x in sorted(remaining):
            children = [
                y for y in remaining - {x}
                if extension[x, y] > 0.5
            ]
            if children:
                continue
            parents = [
                y for y in remaining - {x}
                if extension[y, x] > 0.5
            ]
            und = [
                y for y in remaining - {x}
                if _is_undirected(skel, extension, x, y)
            ]
            und_clique = all(
                _adjacent(skel, u, v)
                for idx, u in enumerate(und)
                for v in und[idx + 1:]
            )
            parent_compatible = all(
                _adjacent(skel, u, p)
                for u in und
                for p in parents
            )
            if und_clique and parent_compatible:
                chosen = x
                for u in und:
                    extension[u, x] = 1.0
                break
        if chosen is None:
            return False, None
        remaining.remove(chosen)
    return _is_acyclic(extension), extension


def _meek_rule_hits(directed, skel, a, b):
    """Return sound Meek R1--R4 premises supporting a->b for undirected a-b."""
    if not _is_undirected(skel, directed, a, b):
        return []
    K = skel.shape[0]
    hits = []

    # R1: c->a, a-b, c not adjacent b  => a->b.
    if any(
        directed[c, a] > 0.5 and not _adjacent(skel, c, b)
        for c in range(K) if c not in (a, b)
    ):
        hits.append("R1")

    # R2: a->c->b, a-b => a->b.
    if any(
        directed[a, c] > 0.5 and directed[c, b] > 0.5
        for c in range(K) if c not in (a, b)
    ):
        hits.append("R2")

    # R3: d-a-c, d->b<-c, c not adjacent d => a->b.
    r3 = False
    for c in range(K):
        if c in (a, b) or not _is_undirected(skel, directed, a, c) \
                or directed[c, b] < 0.5:
            continue
        for d in range(c + 1, K):
            if d in (a, b) or not _is_undirected(skel, directed, a, d) \
                    or directed[d, b] < 0.5:
                continue
            if not _adjacent(skel, c, d):
                r3 = True
                break
        if r3:
            break
    if r3:
        hits.append("R3")

    # R4: d-a-c, d->c->b, b not adjacent d => a->b.
    r4 = False
    for c in range(K):
        if c in (a, b) or not _is_undirected(skel, directed, a, c) \
                or directed[c, b] < 0.5:
            continue
        for d in range(K):
            if d in (a, b, c) \
                    or not _is_undirected(skel, directed, a, d):
                continue
            if directed[d, c] > 0.5 and not _adjacent(skel, b, d):
                r4 = True
                break
        if r4:
            break
    if r4:
        hits.append("R4")
    return hits


def meek_closure(directed, skel, require_consistent_extension=True,
                 protected_undirected_edges=None):
    """Apply sound Meek R1--R4 to a valid PDAG until a fixed point.

    All proposals in an iteration are collected before mutation. Opposite
    proposals for one edge are treated as a consistency failure rather than
    resolved by iteration order.
    """
    directed = np.asarray(directed)
    skel = ((np.asarray(skel) + np.asarray(skel).T) > 0).astype(float)
    protected = set()
    if protected_undirected_edges is not None:
        for edge in protected_undirected_edges:
            if type(edge) not in (tuple, list) or len(edge) != 2:
                return False, [], ["malformed_protected_undirected_edge"]
            i, j = edge
            if (type(i) is not int or type(j) is not int or i == j or
                    i < 0 or j < 0 or i >= skel.shape[0] or j >= skel.shape[0] or
                    not _adjacent(skel, i, j)):
                return False, [], ["invalid_protected_undirected_edge"]
            protected.add(_edge_key(i, j))
    if np.any((directed > 0.5) & (directed.T > 0.5)) \
            or not _is_acyclic(directed):
        return False, [], ["input_has_conflict_or_cycle"]
    if any(not _is_undirected(skel, directed, i, j) for i, j in protected):
        return False, [], ["protected_edge_was_oriented"]
    if require_consistent_extension:
        ok, _ = consistent_dag_extension(skel, directed)
        if not ok:
            return False, [], ["input_has_no_consistent_dag_extension"]

    oriented = []
    while True:
        proposals = {}
        for i, j in _undirected_edges(skel, directed):
            if (i, j) in protected:
                continue
            for a, b in ((i, j), (j, i)):
                rules = _meek_rule_hits(directed, skel, a, b)
                if rules:
                    proposals.setdefault((i, j), []).append(
                        ((a, b), tuple(rules)))
        if not proposals:
            break
        for edge, proposals_for_edge in proposals.items():
            directions = {p[0] for p in proposals_for_edge}
            if len(directions) != 1:
                return False, oriented, [
                    f"conflicting_meek_proposals_on_{edge[0]}-{edge[1]}"
                ]
        changed = False
        for edge in sorted(proposals):
            (a, b), rules = proposals[edge][0]
            if _orient_acyclic(directed, a, b):
                oriented.append((int(a), int(b), "+".join(rules)))
                changed = True
            else:
                return False, oriented, [
                    f"meek_cycle_or_conflict_on_{a}->{b}"
                ]
        if not changed:
            break

    if require_consistent_extension:
        ok, _ = consistent_dag_extension(skel, directed)
        if not ok:
            return False, oriented, ["closure_has_no_consistent_dag_extension"]
    return True, oriented, []


def _cpdag_from_colliders(
        skel, colliders, source, sepsets=None, ci_search_complete=True,
        collider_ready=True, metadata=None, ambiguous_triples=None):
    skel = ((np.asarray(skel) + np.asarray(skel).T) > 0).astype(float)
    K = skel.shape[0]
    directed = np.zeros((K, K), dtype=float)
    provenance = {}
    failure_reasons = []
    normalized_colliders = sorted(set(tuple(map(int, row)) for row in colliders))
    ambiguous = set() if ambiguous_triples is None else {
        tuple(map(int, row)) for row in ambiguous_triples
    }
    proposals = {}
    proposal_sources = {}
    for i, k, j in normalized_colliders:
        if len({i, k, j}) != 3 or not _adjacent(skel, i, k) \
                or not _adjacent(skel, j, k) or _adjacent(skel, i, j):
            failure_reasons.append(f"invalid_collider_{i}->{k}<-{j}")
            continue
        for a, b in ((i, k), (j, k)):
            edge = _edge_key(a, b)
            proposals.setdefault(edge, set()).add((a, b))
            proposal_sources.setdefault(edge, {}).setdefault((a, b), set()).add(
                (i, k, j))
    conflict_edges = {edge for edge, directions in proposals.items()
                      if len(directions) != 1}
    quarantined = set()
    protected = set()
    if conflict_edges:
        # Finite-sample collider claims can demand both directions of one
        # skeleton edge.  Choosing either direction is order-dependent, while
        # rejecting the adjacency discards supported structure.  Conservatively
        # quarantine every atomic collider touching the conflict component and
        # keep all of its legs undirected for later interventions.
        for edge in conflict_edges:
            for sources in proposal_sources[edge].values():
                quarantined.update(sources)
        changed = True
        while changed:
            changed = False
            for i, k, j in normalized_colliders:
                legs = {_edge_key(i, k), _edge_key(j, k)}
                if (i, k, j) in quarantined:
                    before = len(protected)
                    protected.update(legs)
                    changed |= len(protected) != before
                elif legs & protected:
                    quarantined.add((i, k, j))
                    changed = True
        ambiguous.update(quarantined)
        proposals = {}
        for i, k, j in normalized_colliders:
            if (i, k, j) in quarantined:
                continue
            for a, b in ((i, k), (j, k)):
                proposals.setdefault(_edge_key(a, b), set()).add((a, b))
        if any(len(directions) != 1 for directions in proposals.values()):
            failure_reasons.append("conflict_quarantine_did_not_converge")
        metadata = {} if metadata is None else dict(metadata)
        metadata["conservative_collider_conflict_audit"] = {
            "protocol": "conservative_collider_conflict_quarantine_v1",
            "conflicting_edges": [list(edge) for edge in sorted(conflict_edges)],
            "quarantined_triples": [list(row) for row in sorted(quarantined)],
            "protected_undirected_edges": [list(edge) for edge in sorted(protected)],
            "thresholds_changed": False,
            "truth_intervention_OOD_consumed": False,
        }
    if failure_reasons or not ci_search_complete or not collider_ready:
        if not ci_search_complete:
            failure_reasons.append("ci_search_incomplete")
        if not collider_ready:
            failure_reasons.append("collider_set_incomplete_or_ambiguous")
        return EssentialGraph(
            skeleton=skel,
            directed=directed,
            undirected=_undirected_edges(skel, directed),
            graph_type="INVALID_OBSERVATIONAL_PDG",
            valid=False,
            source=source,
            sepsets={} if sepsets is None else sepsets,
            colliders=[row for row in normalized_colliders if row not in quarantined],
            ambiguous_triples=sorted(ambiguous),
            failure_reasons=failure_reasons,
            collider_complete=bool(ci_search_complete and collider_ready),
            collider_consistent=False,
            ci_search_complete=bool(ci_search_complete),
            metadata={} if metadata is None else dict(metadata),
        )

    for directions in proposals.values():
        a, b = next(iter(directions))
        if not _orient_acyclic(directed, a, b):
            failure_reasons.append(f"collider_cycle_or_conflict_{a}->{b}")
        provenance[(int(a), int(b))] = "observational_collider"
    if failure_reasons:
        return EssentialGraph(
            skeleton=skel,
            directed=np.zeros((K, K), dtype=float),
            undirected=_undirected_edges(skel, np.zeros((K, K))),
            graph_type="INVALID_OBSERVATIONAL_PDG",
            valid=False,
            source=source,
            sepsets={} if sepsets is None else sepsets,
            colliders=[row for row in normalized_colliders if row not in quarantined],
            ambiguous_triples=sorted(ambiguous),
            failure_reasons=failure_reasons,
            collider_complete=True,
            collider_consistent=False,
            ci_search_complete=True,
            metadata={} if metadata is None else dict(metadata),
        )

    ok, meek_edges, meek_failures = meek_closure(
        directed, skel, require_consistent_extension=True,
        protected_undirected_edges=sorted(protected))
    globally_demoted = (
        not ok
        and meek_failures in (
            ["input_has_no_consistent_dag_extension"],
            ["closure_has_no_consistent_dag_extension"],
        )
    )
    if globally_demoted:
        # A finite-sample set of individually unambiguous collider marks can be
        # jointly incompatible with every DAG extension that preserves exactly
        # those v-structures.  It is not scientifically justified to select a
        # subset using iteration order, nor to discard the learned skeleton.
        # Under the least-commitment prior, withdraw *all* directions and pass
        # the observational skeleton downstream as an explicitly non-CPDAG
        # partial prior.  Later interventions, rather than observational truth
        # or a changed CI threshold, must resolve every protected edge.
        withdrawn_edges = [
            [int(i), int(j)]
            for i, j in sorted(zip(*np.where(directed > 0.5)))
        ]
        globally_ambiguous = sorted(ambiguous | set(normalized_colliders))
        directed = np.zeros((K, K), dtype=float)
        protected = set(_undirected_edges(skel, directed))
        conflict_free = not bool(np.any(
            (directed > 0.5) & (directed.T > 0.5)))
        acyclic = bool(_is_acyclic(directed))
        all_uncertain_protected = (
            set(_undirected_edges(skel, directed)) == protected)
        if not (conflict_free and acyclic and all_uncertain_protected):
            raise RuntimeError(
                "least-commitment skeleton prior structural audit failed")
        metadata = {} if metadata is None else dict(metadata)
        metadata["least_commitment_global_demotion_audit"] = {
            "protocol": "least_commitment_global_demotion_v1",
            "trigger_failures": list(meek_failures),
            "withdrawn_directed_edges": withdrawn_edges,
            "demoted_collider_triples": [
                list(row) for row in globally_ambiguous
            ],
            "protected_undirected_edges": [
                list(edge) for edge in sorted(protected)
            ],
            "requires_no_new_vstructure_extension": False,
            "partial_directed_marks_conflict_free": conflict_free,
            "partial_directed_marks_acyclic": acyclic,
            "all_uncertain_edges_protected": all_uncertain_protected,
            "cpdag_claimed": False,
            "thresholds_changed": False,
            "truth_intervention_OOD_consumed": False,
        }
        return EssentialGraph(
            skeleton=skel,
            directed=directed,
            undirected=sorted(protected),
            graph_type="LEAST_COMMITMENT_SKELETON_PRIOR",
            valid=True,
            source=source,
            sepsets={} if sepsets is None else sepsets,
            colliders=[],
            ambiguous_triples=globally_ambiguous,
            failure_reasons=[],
            collider_complete=False,
            collider_consistent=False,
            ci_search_complete=True,
            meek_applied=False,
            provenance={},
            metadata=metadata,
        )
    if not ok:
        failure_reasons.extend(meek_failures)
    for a, b, rule in meek_edges:
        provenance[(a, b)] = f"observational_meek_{rule}"
    return EssentialGraph(
        skeleton=skel,
        directed=directed,
        undirected=_undirected_edges(skel, directed),
        graph_type=("CONSERVATIVE_PDG" if ok and ambiguous else
                    ("CPDAG" if ok else "INVALID_OBSERVATIONAL_PDG")),
        valid=bool(ok),
        source=source,
        sepsets={} if sepsets is None else sepsets,
        colliders=[row for row in normalized_colliders if row not in quarantined],
        ambiguous_triples=sorted(ambiguous),
        failure_reasons=failure_reasons,
        collider_complete=not bool(ambiguous),
        collider_consistent=bool(ok),
        ci_search_complete=True,
        meek_applied=bool(ok),
        provenance=provenance,
        metadata={} if metadata is None else dict(metadata),
    )


def cpdag_from_skeleton_sepsets(
        skel, sepsets, ci_search_complete=True, source="learned_pc",
        metadata=None):
    """Orient exactly the unambiguous collider set encoded by PC sepsets."""
    skel = ((np.asarray(skel) + np.asarray(skel).T) > 0).astype(float)
    K = skel.shape[0]
    colliders = []
    ambiguous = []
    incomplete = []
    for k in range(K):
        nbr = [i for i in range(K) if _adjacent(skel, i, k)]
        for idx, i in enumerate(nbr):
            for j in nbr[idx + 1:]:
                if _adjacent(skel, i, j):
                    continue
                sets_ = sepsets.get(_edge_key(i, j), [])
                triple = (int(min(i, j)), int(k), int(max(i, j)))
                if not sets_:
                    incomplete.append(triple)
                    continue
                membership = [k in set(S) for S in sets_]
                if not any(membership):
                    colliders.append(triple)
                elif not all(membership):
                    ambiguous.append(triple)

    collider_ready = bool(not incomplete and not ambiguous)
    state = _cpdag_from_colliders(
        skel,
        sorted(set(colliders)),
        source=source,
        sepsets=sepsets,
        ci_search_complete=bool(ci_search_complete),
        collider_ready=collider_ready,
        metadata=metadata,
    )
    all_ambiguous = sorted(set(state.ambiguous_triples) | set(ambiguous))
    state.ambiguous_triples = all_ambiguous
    if incomplete:
        state.failure_reasons.extend(
            f"missing_sepset_for_{i}-{j}_via_{k}"
            for i, k, j in sorted(set(incomplete))
        )
    if ambiguous:
        state.failure_reasons.extend(
            f"ambiguous_collider_{i}-{k}-{j}"
            for i, k, j in sorted(set(ambiguous))
        )
    state.collider_complete = bool(
        ci_search_complete and collider_ready and not all_ambiguous)
    state.collider_consistent = bool(
        state.valid and state.graph_type == "CPDAG" and not all_ambiguous)
    return state


def learn_observational_cpdag(
        Z, K, alpha=0.01, max_order=None, min_eff=0.03,
        source="learned_pc"):
    skel, strength, sepsets, report = pc_skeleton_sepsets(
        Z, K, alpha=alpha, max_order=max_order, min_eff=min_eff)
    report["edge_strength"] = strength.tolist()
    return cpdag_from_skeleton_sepsets(
        skel,
        sepsets,
        ci_search_complete=report["search_complete"],
        source=source,
        metadata={"pc": report},
    )


def true_cpdag_from_dag(A, source="diagnostic_upper_bound_given_cpdag"):
    """Exact observational CPDAG used only by an explicitly labelled upper bound."""
    A = (np.asarray(A) > 0.5).astype(float)
    K = A.shape[0]
    skel = ((A + A.T) > 0).astype(float)
    colliders = []
    for k in range(K):
        parents = [i for i in range(K) if A[i, k] > 0.5]
        for idx, i in enumerate(parents):
            for j in parents[idx + 1:]:
                if not _adjacent(skel, i, j):
                    colliders.append((int(min(i, j)), int(k), int(max(i, j))))
    state = _cpdag_from_colliders(
        skel,
        sorted(set(colliders)),
        source=source,
        sepsets={},
        ci_search_complete=True,
        metadata={
            "diagnostic_upper_bound": True,
            "uses_true_dag_only_to_construct_cpdag": True,
        },
    )
    return state


def observational_pdag(Z, skel, K, alpha=0.05, min_eff=0.05,
                       max_order=None, vstructures=None,
                       collider_complete=False):
    """Compatibility API for explicitly supplied collider information."""
    _ = (Z, alpha, min_eff, max_order)
    if not collider_complete:
        return np.zeros((K, K))
    state = _cpdag_from_colliders(
        skel, [] if vstructures is None else vstructures,
        source="explicit_trusted_colliders",
        ci_search_complete=True,
    )
    if not state.valid:
        raise ValueError("; ".join(state.failure_reasons))
    return state.directed


def observational_cpdag(*args, **kwargs):
    """Compatibility alias for :func:`observational_pdag`."""
    return observational_pdag(*args, **kwargs)


def _meek(directed, skel, K, allow_r1=False):
    """Legacy closure callback: R2-only unless validity is explicitly asserted."""
    _ = K
    if allow_r1:
        ok, _, failures = meek_closure(
            directed, skel, require_consistent_extension=True)
        if not ok:
            raise ValueError("; ".join(failures))
        return
    changed = True
    while changed:
        changed = False
        for i, j in _undirected_edges(skel, directed):
            for a, b in ((i, j), (j, i)):
                if any(
                    directed[a, k] > 0.5 and directed[k, b] > 0.5
                    for k in range(K) if k not in (a, b)
                ):
                    if _orient_acyclic(directed, a, b):
                        changed = True
                    break


def _meek_complete(directed, skel, K):
    """Full valid-PDAG closure callback for HB scoring."""
    _ = K
    ok, _, failures = meek_closure(
        directed, skel, require_consistent_extension=True)
    if not ok:
        raise ValueError("; ".join(failures))


# --------------------------- Stage 2: do-criterion orientation ---------------------------
def _dep_poly(Z, i, j, S, rows):
    """Max |corr| over {r, r^2} residual-feature pairs within `rows` (same
    nonlinear dependence measure as the poly CI test): a do-effect that
    shifts the child's spread or a non-monotone readout (Gaussian bump)
    is visible to this measure but not to plain correlation."""
    Zr = Z[rows]
    def resid(k):
        y = Zr[:, k]
        if not S:
            return y - y.mean()
        X = Zr[:, sorted(S)]
        F = np.concatenate([X, X ** 2], axis=1)
        F = (F - F.mean(0)) / (F.std(0) + 1e-9)
        F = np.concatenate([np.ones((len(F), 1)), F], 1)
        w = np.linalg.solve(F.T @ F + 1e-3 * np.eye(F.shape[1]), F.T @ y)
        return y - F @ w
    ri, rj = resid(i), resid(j)
    fi = [(ri - ri.mean()) / (ri.std() + 1e-9)]
    fj = [(rj - rj.mean()) / (rj.std() + 1e-9)]
    fi.append((fi[0] ** 2 - 1.0) / (np.std(fi[0] ** 2) + 1e-9))
    fj.append((fj[0] ** 2 - 1.0) / (np.std(fj[0] ** 2) + 1e-9))
    return max(abs(float(np.corrcoef(a, b)[0, 1])) for a in fi for b in fj)


def orient_with_do(Z, M, i, j, nbr, thr=0.12, margin=0.0, cond=None,
                   dep="linear"):
    """Orient a SKELETON edge (i,j) [known direct edge] from interventional data.
    Use MARGINAL dependence within single-target do(.) data. In do(i) data i is
    exogenous: if i->j, j follows i (dependence above ``thr``); if j->i, doing i
    detaches it from j (dependence below ``thr``). With both do(i) and do(j),
    one score must be above and the other below threshold, with separation at
    least ``margin``. Returns +1 (i->j), -1 (j->i), or 0 (insufficient/conflicting
    evidence). ``nbr`` is retained for call compatibility but is not used.

    margin>0: 置信裕度。证据不够清晰(两方向差距 < margin,或单向证据落在阈值附近)
    时返回 0(保持未定向),避免把噪声/对撞边硬定成错方向。"""
    _ = nbr
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if cond:
        warnings.warn(
            "cond is ignored: production do-orientation uses marginal "
            "single-target dependence and no root/ground-truth controls",
            UserWarning,
            stacklevel=2,
        )
    single_target = np.isclose(M.sum(1), 1.0)
    do_i = np.where((M[:, i] == 1) & single_target)[0]
    do_j = np.where((M[:, j] == 1) & single_target)[0]
    S = set()
    if dep == "poly":
        e_ij = _dep_poly(Z, i, j, S, do_i) if do_i.size > 30 else None
        e_ji = _dep_poly(Z, j, i, S, do_j) if do_j.size > 30 else None
    else:
        e_ij = abs(_pcorr(Z, i, j, S, rows=do_i)) if do_i.size > 30 else None
        e_ji = abs(_pcorr(Z, j, i, S, rows=do_j)) if do_j.size > 30 else None
    if e_ij is not None and e_ji is not None:
        if e_ij > thr and e_ji < thr and e_ij - e_ji >= margin:
            return 1
        if e_ji > thr and e_ij < thr and e_ji - e_ij >= margin:
            return -1
        return 0                                    # both survive/break or are too close
    if e_ij is not None:
        if abs(e_ij - thr) < margin:               # 单向证据落在阈值附近 -> 不定向
            return 0
        return 1 if e_ij > thr else -1
    if e_ji is not None:
        if abs(e_ji - thr) < margin:
            return 0
        return -1 if e_ji > thr else 1
    return 0


def _single_target_rows(M, node):
    single_target = np.isclose(np.asarray(M).sum(1), 1.0)
    return np.where((np.asarray(M)[:, node] > 0.5) & single_target)[0]


def edge_orientation_evidence(Z, M, i, j, engine="direct_do", config=None,
                              do_thr=0.12, do_margin=0.0, dep="linear"):
    """Return a deployable edge decision together with auditable evidence.

    No engine in this function can access graph truth, root identity, or an
    orientation hint.  ``direct_do`` exactly preserves the historical marginal
    correlation rule.  The other engines are two-sided, cross-fitted/bootstrap
    tests implemented in :mod:`orientation_stats`.
    """
    if engine not in ORIENTATION_ENGINES:
        raise ValueError(
            f"unknown orientation engine {engine!r}; "
            f"choose from {ORIENTATION_ENGINES}"
        )
    config = {} if config is None else dict(config)
    if engine != "direct_do":
        return statistical_orientation_evidence(
            np.asarray(Z, dtype=float), np.asarray(M), i, j, engine, config)

    thr = float(config.get("do_thr", do_thr))
    margin = float(config.get("do_margin", do_margin))
    dep_name = config.get("dep", dep)
    rows_i = _single_target_rows(M, i)
    rows_j = _single_target_rows(M, j)
    if dep_name == "poly":
        score_i = _dep_poly(Z, i, j, set(), rows_i) if len(rows_i) > 30 else None
        score_j = _dep_poly(Z, j, i, set(), rows_j) if len(rows_j) > 30 else None
    else:
        score_i = abs(_pcorr(Z, i, j, set(), rows=rows_i)) \
            if len(rows_i) > 30 else None
        score_j = abs(_pcorr(Z, j, i, set(), rows=rows_j)) \
            if len(rows_j) > 30 else None
    decision = orient_with_do(
        Z, M, i, j, None, thr=thr, margin=margin, dep=dep_name)
    return {
        "engine": "direct_do",
        "edge": [int(i), int(j)],
        "orientation": int(decision),
        "orientation_source": "marginal_single_target_dependence",
        "uses_meek": False,
        "n_do_i": int(len(rows_i)),
        "n_do_j": int(len(rows_j)),
        "score_i_to_j": None if score_i is None else float(score_i),
        "score_j_to_i": None if score_j is None else float(score_j),
        "config": {
            "do_thr": thr,
            "do_margin": margin,
            "dep": dep_name,
        },
        "abstain_reason": None if decision else "threshold_or_margin_ambiguous",
    }


def discover(
        Z, M, skel, K, alpha, min_eff, do_margin=0.0, cond=None,
        dep="linear", roots=None, do_thr=0.12, engine="direct_do",
        engine_config=None, return_evidence=False, initial_graph=None,
        max_order=None, legacy_skeleton=False, return_graph=False):
    """Update an observational CPDAG with single-target intervention evidence.

    The default path never starts from a bare skeleton. Callers should pass the
    :class:`EssentialGraph` returned by :func:`learn_observational_cpdag` or
    :func:`true_cpdag_from_dag`. For migration compatibility, if
    ``initial_graph`` is omitted, separating sets are estimated from the
    observational rows for the supplied skeleton before any do-edge is tested.
    The historical skeleton -> direct-do-all-edges behavior requires the
    explicit ``legacy_skeleton=True`` ablation flag.
    """
    Z = np.asarray(Z, dtype=float)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)   # standardize (partial corr scale-invariant; avoids overflow on long chains)
    M = np.asarray(M)
    if cond:
        warnings.warn(
            "discover(cond=...) is deprecated and ignored; no root controls "
            "enter the do-criterion",
            UserWarning,
            stacklevel=2,
        )
    if roots:
        warnings.warn(
            "discover(roots=...) is deprecated and ignored; root truth is not "
            "an orientation premise",
            UserWarning,
            stacklevel=2,
        )
    if engine not in ORIENTATION_ENGINES:
        raise ValueError(
            f"unknown orientation engine {engine!r}; "
            f"choose from {ORIENTATION_ENGINES}"
        )

    if initial_graph is not None:
        if not isinstance(initial_graph, EssentialGraph):
            raise TypeError("initial_graph must be an EssentialGraph")
        current = initial_graph.copy()
        if current.skeleton.shape != (K, K):
            raise ValueError("initial_graph dimension does not match K")
    elif legacy_skeleton:
        skel_legacy = (
            (np.asarray(skel) + np.asarray(skel).T) > 0
        ).astype(float)
        directed_legacy = np.zeros((K, K), dtype=float)
        current = EssentialGraph(
            skeleton=skel_legacy,
            directed=directed_legacy,
            undirected=_undirected_edges(skel_legacy, directed_legacy),
            graph_type="LEGACY_SKELETON_PDG",
            valid=True,
            source="legacy_skeleton_direct_do_ablation",
            failure_reasons=[
                "legacy_ablation_has_no_observational_vstructures_or_cpdag"
            ],
            collider_complete=False,
            collider_consistent=False,
            ci_search_complete=False,
            metadata={"legacy_ablation": True},
        )
    else:
        warnings.warn(
            "discover(initial_graph=None) is a migration compatibility path: "
            "it estimates sepsets for the supplied skeleton but is not a "
            "jointly learned PC CPDAG and must not be labelled given-CPDAG",
            UserWarning,
            stacklevel=2,
        )
        obs_rows = np.isclose(M.sum(1), 0.0)
        Zobs = Z[obs_rows]
        if len(Zobs) < K + 4:
            raise ValueError("insufficient observational rows to construct CPDAG")
        sepsets, sep_report = sepsets_for_fixed_skeleton(
            Zobs,
            skel,
            alpha=alpha,
            max_order=(K - 2 if max_order is None else max_order),
            min_eff=min_eff,
        )
        current = cpdag_from_skeleton_sepsets(
            skel,
            sepsets,
            ci_search_complete=sep_report["search_complete"],
            source="compat_supplied_skeleton_plus_observational_sepsets",
            metadata={"sepset_search": sep_report},
        )

    immutable_observational = current.directed.copy()
    evidence = []
    # A valid bare skeleton is NOT a CPDAG: unknown colliders are not absent
    # colliders. Only a fully certified observational state permits Meek R1-4.
    closure_allowed = bool(current.valid and current.collider_complete and
                           current.collider_consistent and current.ci_search_complete
                           and not legacy_skeleton)
    current.metadata['orientation_update'] = 'batch_fail_closed_v2'
    current.metadata['meek_closure_allowed'] = closure_allowed
    if current.valid:
        proposals = []
        for i, j in list(current.undirected):
            if not _is_undirected(
                    current.skeleton, current.directed, i, j):
                continue
            ev = edge_orientation_evidence(
                Z, M, i, j, engine=engine, config=engine_config,
                do_thr=do_thr, do_margin=do_margin, dep=dep,
            )
            r = int(ev["orientation"])
            if r:
                proposals.append((i, j) if r == 1 else (j, i))
            ev["accepted"] = False
            ev["immutable_observational_edge"] = False
            ev["meek_propagated"] = []
            evidence.append(ev)
        # Score every unresolved edge against the same state before mutation.
        # Reject an inconsistent batch in full, never pick a first edge by ID.
        trial = current.directed.copy()
        for a, b in proposals:
            trial[a, b] = 1.0
        failures, propagated = [], []
        if np.any((trial > .5) & (trial.T > .5)) or not _is_acyclic(trial):
            failures = ['direct_evidence_batch_cycle_or_conflict']
        if not failures and proposals and closure_allowed:
            def collider_set(d):
                return {(min(a,c), b, max(a,c)) for b in range(K)
                        for a in range(K) for c in range(a+1,K)
                        if d[a,b]>.5 and d[c,b]>.5 and current.skeleton[a,c]<.5}
            if collider_set(trial) != collider_set(immutable_observational):
                failures = ['direct_evidence_conflicts_with_certified_colliders']
            else:
                ok, propagated, failures = meek_closure(
                    trial, current.skeleton, require_consistent_extension=True)
                if not ok and not failures:
                    failures = ['inconsistent_batch_closure']
        accepted = bool(proposals and not failures)
        if accepted:
            current.directed = trial
            for edge in proposals:
                current.provenance[edge] = f'intervention_{engine}'
            for a, b, rule in propagated:
                current.provenance[(a,b)] = f'interventional_meek_{rule}'
            current.meek_applied = bool(current.meek_applied or propagated)
        current.metadata['batch_rejection_reasons'] = failures
        current.metadata['batch_meek_propagated'] = [
            {'edge': [a,b], 'rule': rule} for a,b,rule in propagated] if accepted else []
        for ev in evidence:
            if ev['orientation']:
                ev['accepted'] = accepted
                if not accepted:
                    ev['rejected_reason'] = failures
    else:
        evidence.append({
            "engine": engine,
            "orientation": 0,
            "accepted": False,
            "abstain_reason": "invalid_observational_cpdag",
            "failure_reasons": list(current.failure_reasons),
        })

    if np.any(
            (immutable_observational > 0.5)
            & (current.directed < 0.5)):
        raise RuntimeError("observationally compelled arrow was overwritten")
    current.undirected = _undirected_edges(
        current.skeleton, current.directed)
    has_intervention = bool(np.any(np.isclose(M.sum(1), 1.0)))
    if legacy_skeleton:
        current.graph_type = (
            "DAG" if not current.undirected and _is_acyclic(current.directed)
            else "LEGACY_SKELETON_PDG"
        )
    elif not current.valid:
        current.graph_type = "INVALID_OBSERVATIONAL_PDG"
    elif not current.undirected and _is_acyclic(current.directed):
        current.graph_type = "DAG"
    elif not closure_allowed:
        current.graph_type = 'INTERVENTION_PDG' if has_intervention else 'SKELETON_PDG'
    elif has_intervention:
        current.graph_type = "INTERVENTIONAL_ESSENTIAL_GRAPH"
    else:
        current.graph_type = "CPDAG"
    current.metadata["immutable_observational_arrows"] = int(
        (immutable_observational > 0.5).sum())
    current.metadata["orientation_engine"] = engine
    current.metadata["legacy_skeleton"] = bool(legacy_skeleton)

    if return_graph:
        return current, evidence
    if return_evidence:
        return current.directed, current.undirected, evidence
    return current.directed, current.undirected


# --------------------------- ACC graph-repair sidecar ---------------------------
def _acc_candidate_blockers(skeleton, i, j):
    """Return the outcome-independent blocker universe for diagnostic v0.

    A local prior-neighbour rule can silently omit the true mediator when the
    visual prior is exactly what is wrong.  Until a sound possible-descendant
    criterion is implemented, v0 searches the complete observed universe and
    relies on explicit residual-variation gates to avoid deterministic
    over-conditioning.  This is intentionally conservative and exponential in
    ``max_order``; ACC currently targets small causal concept graphs.
    """
    K = np.asarray(skeleton).shape[0]
    return [int(k) for k in range(K) if k not in (i, j)]


def _acc_singleton_order_certificate(Z, M, U, source, mediator, target,
                                     row_mask, direct_cfg, seed_salt):
    """Certify source < mediator < target from hard-do order evidence.

    The provisional CPDAG is never sufficient to authorize a deletion.  A
    singleton blocker must itself be interventionally ordered between the two
    endpoints, while the endpoint response becomes equivalent to zero only
    after conditioning on that frozen mediator.
    """
    if U is None:
        return {
            "protocol": "acc_interventional_singleton_order_v1",
            "ordered_triple": [int(source), int(mediator), int(target)],
            "state": "UNKNOWN",
            "reason": "explicit_preaction_commands_required",
            "claims": {},
            "formal_eligible": False,
        }
    claims = {
        "source_to_mediator": (source, mediator, (), EFFECT),
        "mediator_to_target": (mediator, target, (), EFFECT),
        "mediator_to_source": (mediator, source, (), EQUIVALENT_ZERO),
        "target_to_mediator": (target, mediator, (), EQUIVALENT_ZERO),
        "source_to_target": (source, target, (), EFFECT),
        "target_to_source": (target, source, (), EQUIVALENT_ZERO),
        "source_to_target_given_mediator": (
            source, target, (mediator,), EQUIVALENT_ZERO),
    }
    evidence, passed = {}, True
    for offset, (name, (a, b, conditioning, expected)) in enumerate(
            claims.items()):
        rec = effect_evidence(
            Z,
            M,
            a,
            b,
            conditioning_set=conditioning,
            row_mask=row_mask,
            intervention_values=U,
            config=direct_cfg,
            seed_salt=int(seed_salt) + offset,
        ).to_dict()
        rec["expected_state"] = expected
        rec["claim_passed"] = rec["state"] == expected
        evidence[name] = rec
        passed = passed and rec["claim_passed"]
    return {
        "protocol": "acc_interventional_singleton_order_v1",
        "ordered_triple": [int(source), int(mediator), int(target)],
        "state": "ADMISSIBLE" if passed else "UNKNOWN",
        "claims": evidence,
        "formal_eligible": False,
    }


def _acc_propose_singleton_blocker(
        fold_pair, proposer_role, proposer_view_sha256, direct_config_sha256):
    """Propose a unique ordered singleton from one discovery fold.

    This step never certifies its own proposal.  A different, physically
    materialized role must test the frozen ordered triple below.
    """
    i, j = map(int, fold_pair["pair"])
    ordered_candidates = []
    for side, source, target in (("forward", i, j), ("reverse", j, i)):
        for separator in fold_pair[side].get("separating_sets", ()):
            if len(separator) == 1:
                ordered_candidates.append((source, int(separator[0]), target))
    unique = sorted(set(ordered_candidates))
    payload = {
        "protocol": "acc_singleton_blocker_proposal_v3",
        "proposer_role": str(proposer_role),
        "proposer_view_sha256": str(proposer_view_sha256),
        "direct_config_sha256": str(direct_config_sha256),
        "fold_pair_sha256": _canonical_sha256(fold_pair),
        "formal_eligible": False,
    }
    if len(unique) == 1:
        payload.update({
            "ordered_triple": list(unique[0]),
            "state": "PROPOSED",
            "candidate_count": 1,
        })
    else:
        payload.update({
            "ordered_triple": None,
            "state": "UNKNOWN",
            "candidate_count": len(unique),
            "reason": (
                "no_singleton_blocker_proposed" if not unique else
                "multiple_singleton_blockers_require_multiplicity_control"),
        })
    payload["proposal_sha256"] = _canonical_sha256(payload)
    return payload


def _acc_certify_singleton_blocker(
        proposal, expected_proposer_role, certifier_role,
        certifier_view_sha256, Z, M, U, row_mask, direct_cfg, seed_salt):
    """Certify a frozen proposal on the opposite materialized role."""
    proposal_core = {
        key: value for key, value in proposal.items()
        if key != "proposal_sha256"
    }
    proposal_hash_valid = (
        proposal.get("protocol") == "acc_singleton_blocker_proposal_v3"
        and proposal.get("proposer_role") == expected_proposer_role
        and _canonical_sha256(proposal_core)
        == proposal.get("proposal_sha256")
        and proposal.get("direct_config_sha256")
        == _canonical_sha256(direct_cfg)
    )
    if not proposal_hash_valid \
            or proposal.get("state") != "PROPOSED" \
            or proposal.get("ordered_triple") is None:
        cert = {
            "protocol": "acc_crossrole_singleton_certificate_v3",
            "ordered_triple": proposal.get("ordered_triple"),
            "state": "UNKNOWN",
            "reason": (
                "proposal_binding_invalid" if not proposal_hash_valid else
                "opposite_role_received_no_unique_frozen_proposal"),
            "formal_eligible": False,
        }
    else:
        cert = _acc_singleton_order_certificate(
            Z, M, U, *proposal["ordered_triple"], row_mask,
            direct_cfg, seed_salt)
        cert["protocol"] = "acc_crossrole_singleton_certificate_v3"
    cert.update({
        "proposal_protocol": proposal.get("protocol"),
        "proposal_sha256": proposal.get("proposal_sha256"),
        "expected_proposer_role": str(expected_proposer_role),
        "certifier_role": str(certifier_role),
        "certifier_view_sha256": str(certifier_view_sha256),
        "direct_config_sha256": _canonical_sha256(direct_cfg),
    })
    cert["certificate_sha256"] = _canonical_sha256(cert)
    return cert


_ACC_LEAST_COMMITMENT_GRAPH_TYPES = {
    "ACC_LEAST_COMMITMENT_PARTIAL_DAG_PRIOR",
    "ACC_LEAST_COMMITMENT_DAG_PRIOR",
}
_ACC_LEAST_COMMITMENT_SOURCE = (
    "D0_visual_learned_pc+least_commitment_conflict_normalizer_v1"
)


def _acc_initial_prior_semantics(initial_graph):
    """Validate special prior semantics before ACC consumes any outcomes.

    ``EssentialGraph.undirected`` historically means an undirected CPDAG edge.
    The least-commitment normalizer intentionally uses the same storage slot
    for circle-circle endpoints, which do *not* assert a Markov-equivalence
    class.  Treating those circles as CPDAG edges would incorrectly require a
    consistent extension with no additional v-structures.  This gate binds the
    normalizer receipt and makes the different semantics explicit.
    """
    graph_type = str(initial_graph.graph_type)
    source = str(initial_graph.source)
    normalized_marker = (
        graph_type in _ACC_LEAST_COMMITMENT_GRAPH_TYPES
        or source == _ACC_LEAST_COMMITMENT_SOURCE
        or isinstance(initial_graph.metadata.get(
            "least_commitment_normalizer"), dict)
    )
    if not normalized_marker:
        return {
            "kind": "essential_graph",
            "circle_endpoints": False,
            "certified_cpdag": graph_type == "CPDAG",
            "normalizer_core_audit_sha256": None,
        }
    if graph_type not in _ACC_LEAST_COMMITMENT_GRAPH_TYPES \
            or source != _ACC_LEAST_COMMITMENT_SOURCE \
            or "CPDAG" in graph_type.upper():
        raise ValueError(
            "least-commitment prior must use its explicit non-CPDAG type/source")
    normalizer = initial_graph.metadata.get("least_commitment_normalizer")
    if not isinstance(normalizer, dict):
        raise ValueError("least-commitment normalizer receipt is missing")
    core = {key: value for key, value in normalizer.items()
            if key != "core_audit_sha256"}
    if normalizer.get("protocol") \
            != "acc_least_commitment_conflict_normalizer_v1" \
            or _canonical_sha256(core) \
            != normalizer.get("core_audit_sha256"):
        raise ValueError("least-commitment normalizer receipt hash mismatch")
    for field in ("config_sha256", "code_sha256", "original_graph_sha256"):
        value = normalizer.get(field)
        if not isinstance(value, str) or len(value) != 64 \
                or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"invalid least-commitment receipt field {field}")
    structural = normalizer.get("structural_validation")
    if not isinstance(structural, dict) \
            or structural.get("state") != "PASS" \
            or structural.get("skeleton_preserved_exactly") is not True \
            or structural.get("opposite_direction_conflict_absent") is not True \
            or structural.get("directed_subgraph_acyclic") is not True \
            or structural.get("completion_orientation_not_materialized") is not True \
            or normalizer.get("true_graph_used") is not False \
            or normalizer.get("interventions_used") is not False \
            or normalizer.get("OOD_used") is not False \
            or normalizer.get("formal_eligible") is not False:
        raise ValueError("least-commitment structural/firewall receipt is invalid")
    skeleton = np.asarray(initial_graph.skeleton, dtype=float)
    directed = np.asarray(initial_graph.directed, dtype=float)
    if skeleton.ndim != 2 or skeleton.shape[0] != skeleton.shape[1] \
            or directed.shape != skeleton.shape \
            or not np.all(np.isfinite(skeleton)) \
            or not np.all(np.isfinite(directed)) \
            or not np.array_equal(skeleton, skeleton.T) \
            or not np.all(np.diag(skeleton) == 0) \
            or not set(np.unique(skeleton)).issubset({0.0, 1.0}) \
            or not set(np.unique(directed)).issubset({0.0, 1.0}) \
            or np.any((directed > 0.5) & (skeleton < 0.5)) \
            or np.any((directed > 0.5) & (directed.T > 0.5)) \
            or not _is_acyclic(directed):
        raise ValueError("least-commitment partial-DAG matrices are malformed")
    if normalizer.get("normalized_skeleton_sha256") != _array_sha256(skeleton) \
            or normalizer.get("normalized_directed_sha256") \
            != _array_sha256(directed):
        raise ValueError("least-commitment graph hashes do not match receipt")
    unresolved = _undirected_edges(skeleton, directed)
    if sorted(map(tuple, initial_graph.undirected)) != sorted(unresolved):
        raise ValueError("least-commitment circle endpoint index mismatch")
    expected_type = (
        "ACC_LEAST_COMMITMENT_PARTIAL_DAG_PRIOR" if unresolved
        else "ACC_LEAST_COMMITMENT_DAG_PRIOR"
    )
    if graph_type != expected_type \
            or initial_graph.metadata.get("circle_endpoint_semantics") \
            != "unknown_orientation; not a CPDAG equivalence claim" \
            or initial_graph.metadata.get("certified_cpdag", False):
        raise ValueError("least-commitment endpoint semantics are mislabeled")
    ledger = normalizer.get("demotion_ledger")
    skeleton_edges = {
        (i, j) for i in range(len(skeleton)) for j in range(i + 1, len(skeleton))
        if skeleton[i, j] > 0.5
    }
    if not isinstance(ledger, list) or {
            tuple(map(int, item.get("edge", ())))
            for item in ledger if isinstance(item, dict)
    } != skeleton_edges or len(ledger) != len(skeleton_edges):
        raise ValueError("least-commitment demotion ledger does not cover skeleton")
    for item in ledger:
        i, j = map(int, item["edge"])
        after = str(item.get("after"))
        actual = (
            f"{i}->{j}" if directed[i, j] > 0.5 else
            f"{j}->{i}" if directed[j, i] > 0.5 else
            "circle-circle"
        )
        if after != actual:
            raise ValueError("least-commitment edge receipt disagrees with graph")
    return {
        "kind": "least_commitment_partial_dag",
        "circle_endpoints": bool(unresolved),
        "certified_cpdag": False,
        "normalizer_core_audit_sha256": normalizer["core_audit_sha256"],
    }


def _acc_validate_circle_partial_dag(skeleton, directed):
    failures = []
    if not np.array_equal(skeleton, skeleton.T):
        failures.append("circle_prior_skeleton_not_symmetric")
    if np.any((directed > 0.5) & (skeleton < 0.5)):
        failures.append("circle_prior_direction_outside_skeleton")
    if np.any((directed > 0.5) & (directed.T > 0.5)):
        failures.append("circle_prior_opposite_directions")
    if not _is_acyclic(directed):
        failures.append("circle_prior_directed_cycle")
    return not failures, failures


def _acc_apply_consensus(initial_graph, consensus, locked_edges,
                         allow_direct_edge_mutation=False,
                         initial_prior_semantics=None):
    """Materialize one repair round; evidence, not mutation history, is truth."""
    skeleton = initial_graph.skeleton.copy().astype(float)
    locked = {(int(a), int(b)) for a, b in locked_edges}
    changed_pairs = []
    conflicts = []
    adjacency_changed = False

    # First revise adjacency only.  Directions are rebuilt below so an edge
    # deletion cannot leave stale collider/Meek consequences in place.
    for rec in consensus:
        i, j = map(int, rec["pair"])
        state = rec["state"]
        direction = rec.get("direction")
        locked_here = [(a, b) for a, b in locked
                       if {a, b} == {i, j}]
        if state == PAIR_UNKNOWN:
            continue
        if locked_here:
            compatible = state == PAIR_DIRECT and tuple(direction) in locked
            if not compatible:
                rec["state"] = PAIR_UNKNOWN
                rec["direction"] = None
                rec["reason"] = "conflicts_with_protocol_locked_edge"
                conflicts.append([i, j])
                continue
        before_adjacent = bool(
            skeleton[i, j] > 0.5 or skeleton[j, i] > 0.5)
        mutation_authorized = bool(rec.get("mutation_authorized", False))
        if state == PAIR_ABSENT and mutation_authorized:
            skeleton[i, j] = skeleton[j, i] = 0.0
        elif state == PAIR_DIRECT and mutation_authorized \
                and allow_direct_edge_mutation:
            skeleton[i, j] = skeleton[j, i] = 1.0
        after_adjacent = bool(
            skeleton[i, j] > 0.5 or skeleton[j, i] > 0.5)
        if before_adjacent != after_adjacent:
            adjacency_changed = True
            changed_pairs.append([i, j])

    if adjacency_changed:
        directed = np.zeros_like(initial_graph.directed, dtype=float)
        provenance = {}
    else:
        directed = initial_graph.directed.copy().astype(float)
        provenance = {
            edge: f"provisional_{source}"
            for edge, source in initial_graph.provenance.items()
        }

    for source, target in sorted(locked):
        if skeleton[source, target] < 0.5 and skeleton[target, source] < 0.5:
            conflicts.append(list(_edge_key(source, target)))
            continue
        directed[source, target] = 1.0
        directed[target, source] = 0.0
        provenance[(source, target)] = "protocol_locked"

    for rec in consensus:
        if rec["state"] != PAIR_DIRECT \
                or not rec.get("mutation_authorized", False) \
                or not allow_direct_edge_mutation:
            continue
        i, j = map(int, rec["pair"])
        source, target = map(int, rec["direction"])
        before = (bool(directed[i, j] > 0.5),
                  bool(directed[j, i] > 0.5))
        directed[i, j] = directed[j, i] = 0.0
        directed[source, target] = 1.0
        provenance.pop((target, source), None)
        provenance[(source, target)] = "acc_crossfold_direct"
        after = (bool(directed[i, j] > 0.5),
                 bool(directed[j, i] > 0.5))
        if before != after and [i, j] not in changed_pairs:
            changed_pairs.append([i, j])

    if np.any((directed > 0.5) & (directed.T > 0.5)) \
            or not _is_acyclic(directed):
        # Conservative v0: retract every ACC orientation participating in this
        # round rather than selecting a convenient edge using graph truth.
        skeleton = initial_graph.skeleton.copy().astype(float)
        directed = initial_graph.directed.copy().astype(float)
        provenance = dict(initial_graph.provenance)
        for rec in consensus:
            if rec["pair"] in changed_pairs:
                rec["state"] = PAIR_UNKNOWN
                rec["direction"] = None
                rec["reason"] = "global_cycle_conflict_retracted"
        conflicts.extend(changed_pairs)
        changed_pairs = []
        adjacency_changed = False

    prior_kind = (initial_prior_semantics or {}).get(
        "kind", "essential_graph")
    circle_semantics = prior_kind == "least_commitment_partial_dag"
    if circle_semantics:
        # Circle endpoints mean unknown orientation, not a CPDAG equivalence
        # class.  Applying Meek closure here could guess directions by silently
        # imposing the no-new-v-structure CPDAG completion criterion.
        ok, failures = _acc_validate_circle_partial_dag(skeleton, directed)
        propagated = []
    else:
        ok, propagated, failures = meek_closure(
            directed, skeleton, require_consistent_extension=True)
        if not ok:
            skeleton = initial_graph.skeleton.copy().astype(float)
            directed = initial_graph.directed.copy().astype(float)
            provenance = dict(initial_graph.provenance)
            propagated = []
            for rec in consensus:
                if rec["pair"] in changed_pairs:
                    rec["state"] = PAIR_UNKNOWN
                    rec["direction"] = None
                    rec["reason"] = "inconsistent_extension_retracted"
            conflicts.extend(changed_pairs)
            changed_pairs = []
            adjacency_changed = False
            ok, propagated, failures = meek_closure(
                directed, skeleton, require_consistent_extension=True)
    for a, b, rule in propagated:
        provenance[(int(a), int(b))] = f"acc_derived_meek_{rule}"

    graph = EssentialGraph(
        skeleton=skeleton,
        directed=directed,
        undirected=_undirected_edges(skeleton, directed),
        graph_type=(
            ("ACC_REPAIRED_LEAST_COMMITMENT_DAG"
             if not _undirected_edges(skeleton, directed)
             else "ACC_REPAIRED_LEAST_COMMITMENT_CIRCLE_PRIOR")
            if ok and circle_semantics else
            "ACC_DEVELOPMENT_DAG"
            if ok and not _undirected_edges(skeleton, directed)
            else "ACC_REPAIRED_PDG" if ok else "INVALID_ACC_PDG"
        ),
        valid=bool(ok),
        source="acc_graph_only_v1",
        sepsets=(
            {} if adjacency_changed
            else {k: list(v) for k, v in initial_graph.sepsets.items()}
        ),
        colliders=[] if adjacency_changed else list(initial_graph.colliders),
        ambiguous_triples=(
            [] if adjacency_changed else list(initial_graph.ambiguous_triples)
        ),
        failure_reasons=list(failures),
        collider_complete=bool(
            initial_graph.collider_complete and not adjacency_changed),
        collider_consistent=bool(ok and not adjacency_changed),
        ci_search_complete=bool(
            initial_graph.ci_search_complete and not adjacency_changed),
        meek_applied=bool(propagated),
        provenance=provenance,
        metadata={
            "acc_phase": "graph_only",
            "initial_graph_source": initial_graph.source,
            "observational_edges_are_provisional": True,
            "adjacency_changed": bool(adjacency_changed),
            "stale_observational_orientations_discarded": bool(
                adjacency_changed),
            "protocol_locked_edges": [list(e) for e in sorted(locked)],
            "allow_direct_edge_mutation": bool(allow_direct_edge_mutation),
            "mutation_scope": (
                "crossrole_certified_shortcut_deletion_only"
                if not allow_direct_edge_mutation else
                "experimental_direct_and_absent_mutation"
            ),
            "changed_pairs": changed_pairs,
            "conflict_pairs": sorted(set(map(tuple, conflicts))),
            "initial_prior_semantics": prior_kind,
            "circle_endpoints_are_not_cpdag_edges": bool(circle_semantics),
            "meek_skipped_for_circle_endpoints": bool(circle_semantics),
            "certified_cpdag": False if circle_semantics else None,
        },
    )
    return graph


def discover_acc(Z, M, initial_graph, row_ids, intervention_values=None,
                 config=None, locked_edges=(), role_assignments=None,
                 role_split_audit=None):
    """Fail-closed ACC v0 graph-only repair with sealed evaluation rows.

    This is a sidecar API: the historical :func:`discover` path remains byte
    compatible.  All pair decisions are made independently on ``D_A`` and
    ``D_G`` and accepted only on exact agreement.  ``D_E`` is never indexed by
    the statistical discovery code.
    """
    if not isinstance(initial_graph, EssentialGraph):
        raise TypeError("initial_graph must be an EssentialGraph")
    if not initial_graph.valid:
        raise ValueError("ACC requires a valid prior graph object")
    initial_prior_semantics = _acc_initial_prior_semantics(initial_graph)
    Z = np.asarray(Z, dtype=float)
    M = np.asarray(M, dtype=float)
    if Z.ndim != 2 or M.shape != Z.shape \
            or Z.shape[1] != initial_graph.skeleton.shape[0]:
        raise ValueError("Z, M, and initial_graph dimensions disagree")
    cfg = {} if config is None else dict(config)
    command_binding_valid = False
    command_binding_reasons = []
    for required_gate in (
            "require_complete_blocker_search",
            "require_admissible_blocker",
            "require_representation_safety",
            "require_persisted_preaction_receipts"):
        if cfg.get(required_gate, True) is not True:
            raise ValueError(f"ACC v0 cannot disable {required_gate}")
    row_ids = [str(x) for x in row_ids]
    if len(row_ids) != len(Z) or len(set(row_ids)) != len(row_ids):
        raise ValueError("row_ids must be unique and have one id per row")
    U = None
    if intervention_values is not None:
        U = np.asarray(intervention_values, dtype=float)
        if U.shape != Z.shape:
            raise ValueError("intervention_values must have the same shape as Z")

    if role_assignments is None:
        split = make_acc_role_split(
            row_ids,
            M,
            seed=int(cfg.get("split_seed", 0)),
            fractions=tuple(cfg.get("split_fractions", (0.4, 0.4, 0.2))),
            require_explicit_ids=bool(cfg.get("require_explicit_ids", True)),
        )
        split_audit = split.audit_dict()
        keep = ~split.mask(ROLE_EVAL)
        roles = split.roles[keep].copy()
        # Materialize discovery-only arrays before any statistical module can
        # inspect values.  In particular, non-finite or adversarial D_E values
        # cannot alter a discovery code path.
        Z = Z[keep].copy()
        M = M[keep].copy()
        U = None if U is None else U[keep].copy()
        row_ids = [row_id for row_id, selected in zip(row_ids, keep) if selected]
        command_binding_reasons.append(
            "internal_role_split_has_no_preoutcome_content_commitment")
    else:
        roles = np.asarray(role_assignments, dtype="<U3")
        if roles.shape != (len(Z),):
            raise ValueError("role_assignments must have one role per row")
        if not set(roles.tolist()).issubset({ROLE_ADAPTER, ROLE_GRAPH}):
            raise ValueError(
                "preassigned ACC input must be a materialized D_A/D_G view; "
                "D_E outcomes are forbidden")
        if role_split_audit is None:
            raise ValueError(
                "role_split_audit is required with preassigned materialized roles")
        split_audit = dict(role_split_audit)
        access_ledger = split_audit.get("access_ledger")
        if not isinstance(access_ledger, dict):
            raise ValueError(
                "preassigned ACC input requires a content-bound access ledger")
        if not bool(split_audit.get("materialized_input_excludes_D_E", False)) \
                or int(access_ledger.get("firewall_violation_count", -1)) != 0 \
                or int(access_ledger.get("pre_lock_D_E_access_count", -1)) != 0:
            raise ValueError("preassigned ACC access ledger does not seal D_E")
        access_core = {k: v for k, v in access_ledger.items()
                       if k != "ledger_sha256"}
        expected_access_sha = hashlib.sha256(json.dumps(
            access_core, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if expected_access_sha != access_ledger.get("ledger_sha256"):
            raise ValueError("preassigned ACC access ledger hash mismatch")
        materialized = access_ledger.get("materialized_D_A_D_G_content")
        if not isinstance(materialized, dict):
            raise ValueError(
                "preassigned ACC ledger must bind materialized D_A/D_G content")
        if U is None:
            raise ValueError(
                "preassigned ACC content binding requires intervention values")
        actual_materialized = {
            "Z_sha256": _array_sha256(Z),
            "M_sha256": _array_sha256(M),
            "U_sha256": _array_sha256(U),
            "row_ids_sha256": _canonical_sha256(row_ids),
            "roles_sha256": _array_sha256(roles),
        }
        actual_materialized["content_sha256"] = _canonical_sha256(
            actual_materialized)
        if actual_materialized != materialized:
            raise ValueError(
                "preassigned ACC materialized content hash mismatch")
        commitments = tuple(access_ledger.get(
            "preoutcome_commitment_hashes", ()))
        manifest_valid, manifest_reasons = _validate_preaction_manifests(
            split_audit.get("verified_preaction_manifests"),
            Z,
            M,
            U,
            row_ids,
            roles,
            access_ledger,
            require_persisted_receipts=True,
        )
        command_binding_valid = bool(
            split_audit.get("assignment_committed_before_outcome", False)
            and commitments
            and access_ledger.get("protocol") == "acc_role_access_ledger_v1"
            and access_ledger.get("stage")
            == "graph_discovery_and_representation_safety"
            and str(cfg.get("command_provenance", ""))
            == "agent_issued_pre_execution_arm_do_v1"
            and manifest_valid
        )
        if not command_binding_valid:
            command_binding_reasons.append(
                "preaction_command_provenance_or_commitment_missing")
            command_binding_reasons.extend(manifest_reasons)
    fold_masks = (
        ("A_to_G", roles == ROLE_ADAPTER),
        ("G_to_A", roles == ROLE_GRAPH),
    )
    role_split_sha256 = str(split_audit.get(
        "protocol_sha256", "missing-role-protocol-sha"))
    K = Z.shape[1]
    direct_cfg = dict(cfg.get("direct_evidence", {}))
    requested_order = direct_cfg.get("max_order", "complete")
    if requested_order == "complete":
        direct_cfg["max_order"] = max(0, K - 2)
        blocker_search_scope = "complete_observed_universe"
    else:
        direct_cfg["max_order"] = int(requested_order)
        blocker_search_scope = (
            "complete_observed_universe"
            if int(requested_order) >= max(0, K - 2)
            else "truncated_observed_universe"
        )
    require_complete_search = bool(
        cfg.get("require_complete_blocker_search", True))
    fold_results = []
    for fold_index, (_, fold_mask) in enumerate(fold_masks):
        records = {}
        for i in range(K):
            for j in range(i + 1, K):
                blockers = _acc_candidate_blockers(
                    initial_graph.skeleton, i, j)
                rec = pair_direct_evidence(
                    Z, M, i, j,
                    candidate_nodes=blockers,
                    row_mask=fold_mask,
                    intervention_values=U,
                    config=direct_cfg,
                    seed_salt=1009 * fold_index,
                )
                records[(i, j)] = rec
        fold_results.append(records)

    consensus = []
    for pair in sorted(fold_results[0]):
        consensus.append(consensus_pair_evidence(
            fold_results[0][pair], fold_results[1][pair]))
    if require_complete_search:
        for rec in consensus:
            if rec["state"] == PAIR_UNKNOWN:
                continue
            incomplete = False
            for fold in rec["folds"].values():
                for side in ("forward", "reverse"):
                    ordered = fold[side]
                    # A marginal equivalence result is complete for the claim
                    # "no detectable ancestry".  Any positive ancestry claim
                    # must survive the full observed blocker universe before
                    # it can alter topology or direction.
                    if ordered["ancestry"]["state"] == "EFFECT" \
                            and not ordered["search_complete"]:
                        incomplete = True
            if incomplete:
                rec["pre_completeness_state"] = rec["state"]
                rec["pre_completeness_direction"] = rec.get("direction")
                rec["state"] = PAIR_UNKNOWN
                rec["direction"] = None
                rec["reason"] = "complete_blocker_search_required"

    require_admissible_blocker = bool(
        cfg.get("require_admissible_blocker", True))
    blocker_protocol = str(cfg.get(
        "blocker_admissibility_protocol",
        "interventional_singleton_crossrole_v2"))
    if blocker_protocol != "interventional_singleton_crossrole_v2":
        raise ValueError("unsupported ACC blocker admissibility protocol")
    if require_admissible_blocker:
        fold_mask_by_name = {name: mask for name, mask in fold_masks}
        direct_config_sha256 = _canonical_sha256(direct_cfg)
        role_view_sha256 = {
            ROLE_ADAPTER: _acc_role_view_sha256(
                Z, M, U, row_ids, roles,
                fold_mask_by_name["A_to_G"], ROLE_ADAPTER),
            ROLE_GRAPH: _acc_role_view_sha256(
                Z, M, U, row_ids, roles,
                fold_mask_by_name["G_to_A"], ROLE_GRAPH),
        }
        for rec in consensus:
            if rec["state"] != PAIR_ABSENT:
                continue
            proposal_A = _acc_propose_singleton_blocker(
                rec["folds"]["D_A"], ROLE_ADAPTER,
                role_view_sha256[ROLE_ADAPTER], direct_config_sha256)
            proposal_G = _acc_propose_singleton_blocker(
                rec["folds"]["D_G"], ROLE_GRAPH,
                role_view_sha256[ROLE_GRAPH], direct_config_sha256)
            certificate_A_on_G = _acc_certify_singleton_blocker(
                proposal_A, ROLE_ADAPTER, ROLE_GRAPH,
                role_view_sha256[ROLE_GRAPH], Z, M, U,
                fold_mask_by_name["G_to_A"], direct_cfg, 5003)
            certificate_G_on_A = _acc_certify_singleton_blocker(
                proposal_G, ROLE_GRAPH, ROLE_ADAPTER,
                role_view_sha256[ROLE_ADAPTER], Z, M, U,
                fold_mask_by_name["A_to_G"], direct_cfg, 7001)
            rec["blocker_proposal_D_A"] = proposal_A
            rec["blocker_proposal_D_G"] = proposal_G
            rec["blocker_certificate_D_A_on_D_G"] = certificate_A_on_G
            rec["blocker_certificate_D_G_on_D_A"] = certificate_G_on_A
            same = (
                proposal_A["state"] == "PROPOSED"
                and proposal_G["state"] == "PROPOSED"
                and proposal_A["ordered_triple"]
                == proposal_G["ordered_triple"]
                and certificate_A_on_G["state"] == "ADMISSIBLE"
                and certificate_G_on_A["state"] == "ADMISSIBLE"
                and certificate_A_on_G["ordered_triple"]
                == proposal_A["ordered_triple"]
                and certificate_G_on_A["ordered_triple"]
                == proposal_G["ordered_triple"]
                and certificate_A_on_G["proposal_sha256"]
                == proposal_A["proposal_sha256"]
                and certificate_G_on_A["proposal_sha256"]
                == proposal_G["proposal_sha256"]
                and certificate_A_on_G["direct_config_sha256"]
                == direct_config_sha256
                and certificate_G_on_A["direct_config_sha256"]
                == direct_config_sha256
            )
            if same:
                rec["admissible_singleton_blocker"] = \
                    proposal_A["ordered_triple"]
            else:
                rec["pre_admissibility_state"] = rec["state"]
                rec["state"] = PAIR_UNKNOWN
                rec["direction"] = None
                rec["reason"] = (
                    "interventional_singleton_blocker_not_cross_certified")

    safety_audit = None
    require_safety = bool(cfg.get("require_representation_safety", True))
    if require_safety and U is not None:
        safety_audit = audit_representation_safety(
            Z,
            M,
            U,
            row_ids,
            config={
                "split_seed": int(cfg.get("split_seed", 0)),
                "split_fractions": tuple(cfg.get(
                    "split_fractions", (0.4, 0.4, 0.2))),
                "require_explicit_ids": bool(cfg.get(
                    "require_explicit_ids", True)),
                "test": dict(cfg.get("representation_safety", {})),
            },
            role_assignments=roles,
            role_split_audit=split_audit,
        )
        safety_by_order = {
            tuple(rec["ordered_pair"]): rec for rec in safety_audit["evidence"]
        }
        for rec in consensus:
            i, j = map(int, rec["pair"])
            gate = {
                f"{i}->{j}": safety_by_order[(i, j)]["state"],
                f"{j}->{i}": safety_by_order[(j, i)]["state"],
            }
            rec["representation_safety_gate"] = gate
            if any(state != SAFE for state in gate.values()):
                rec["pre_safety_state"] = rec["state"]
                rec["pre_safety_direction"] = rec.get("direction")
                rec["state"] = PAIR_UNKNOWN
                rec["direction"] = None
                rec["reason"] = "representation_safety_not_certified"
                continue
            triple = rec.get("admissible_singleton_blocker")
            if triple is not None:
                triple_gate = {}
                for target in triple:
                    for source in triple:
                        if source == target:
                            continue
                        triple_gate[f"{source}->{target}"] = \
                            safety_by_order[(source, target)]["state"]
                rec["blocker_representation_safety_gate"] = triple_gate
                if any(state != SAFE for state in triple_gate.values()):
                    rec["pre_safety_state"] = rec["state"]
                    rec["state"] = PAIR_UNKNOWN
                    rec["direction"] = None
                    rec["reason"] = (
                        "blocker_representation_safety_not_certified")
    elif require_safety:
        for rec in consensus:
            rec["representation_safety_gate"] = {
                "state": PAIR_UNKNOWN,
                "reason": "explicit_intervention_values_required",
            }
            rec["pre_safety_state"] = rec["state"]
            rec["pre_safety_direction"] = rec.get("direction")
            rec["state"] = PAIR_UNKNOWN
            rec["direction"] = None
            rec["reason"] = "representation_safety_not_testable_without_commands"

    allow_direct_edge_mutation = bool(
        cfg.get("allow_direct_edge_mutation", False))
    if allow_direct_edge_mutation:
        raise ValueError(
            "ACC v0 permits only cross-role certified shortcut deletion; "
            "direct-edge addition or reversal is diagnostic-only")
    for rec in consensus:
        rec["mutation_authorized"] = False
        rec["mutation_scope"] = "diagnostic_only"
        if not command_binding_valid and rec["state"] != PAIR_UNKNOWN:
            rec["pre_command_binding_state"] = rec["state"]
            rec["pre_command_binding_direction"] = rec.get("direction")
            rec["state"] = PAIR_UNKNOWN
            rec["direction"] = None
            rec["reason"] = "preaction_command_binding_not_certified"
        elif rec["state"] == PAIR_ABSENT \
                and rec.get("admissible_singleton_blocker") is not None:
            rec["mutation_authorized"] = True
            rec["mutation_scope"] = (
                "crossrole_certified_shortcut_deletion_only")
        elif rec["state"] == PAIR_DIRECT:
            rec["mutation_scope"] = (
                "direct_edge_addition_or_reversal_diagnostic_only")
    graph = _acc_apply_consensus(
        initial_graph,
        consensus,
        locked_edges,
        allow_direct_edge_mutation=allow_direct_edge_mutation,
        initial_prior_semantics=initial_prior_semantics,
    )
    diagnostic_prior = bool(
        initial_graph.metadata.get("diagnostic_upper_bound", False)
        or "diagnostic_upper_bound" in str(initial_graph.source)
    )
    state_counts = {
        state: sum(rec["state"] == state for rec in consensus)
        for state in (PAIR_DIRECT, PAIR_ABSENT, PAIR_UNKNOWN)
    }
    acquisition_pairs = [
        list(map(int, rec["pair"])) for rec in consensus
        if rec["state"] == PAIR_UNKNOWN
        or (rec["state"] == PAIR_DIRECT
            and not allow_direct_edge_mutation)
    ]
    acquisition_target_scores = np.zeros(K, dtype=float)
    for i, j in acquisition_pairs:
        acquisition_target_scores[i] += 1.0
        acquisition_target_scores[j] += 1.0
    if safety_audit is not None:
        for rec in safety_audit["evidence"]:
            if rec["state"] != SAFE:
                acquisition_target_scores[int(rec["ordered_pair"][1])] += 2.0
    command_provenance = str(cfg.get(
        "command_provenance", "unbound_explicit_values"))
    access_ledger = split_audit.get("access_ledger", {})
    ledger_payload = {
        "protocol": "acc_evidence_ledger_v1",
        "role_split_sha256": role_split_sha256,
        "initial_graph": initial_graph.to_dict(),
        "initial_prior_semantics": initial_prior_semantics,
        "direct_evidence_config": direct_cfg,
        "require_complete_blocker_search": require_complete_search,
        "require_admissible_blocker": require_admissible_blocker,
        "blocker_admissibility_protocol": blocker_protocol,
        "command_binding_valid": command_binding_valid,
        "command_binding_reasons": command_binding_reasons,
        "allow_direct_edge_mutation": allow_direct_edge_mutation,
        "pair_evidence": consensus,
        "representation_safety_audit": safety_audit,
        "locked_edges": [list(e) for e in sorted(
            (int(a), int(b)) for a, b in locked_edges)],
        "command_provenance": command_provenance,
        "access_ledger_sha256": access_ledger.get("ledger_sha256"),
        "preoutcome_commitment_hashes": list(
            access_ledger.get("preoutcome_commitment_hashes", ())),
    }
    evidence_ledger_sha256 = hashlib.sha256(json.dumps(
        ledger_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    eligibility_reasons = [
        "development_phase_graph_only_v1",
        "observed_blocker_enumeration_not_causal_admissibility_proof",
        "singleton_order_certificate_assumes_acyclic_faithful_sufficient_SCM",
        "no_familywise_error_control",
        "unique_singleton_crossrole_proposal_certification_only",
        "no_discovery_closed_external_D_E_evaluator",
        "no_downstream_no_harm_gate",
        "D0_encoder_uses_ground_truth_latent_anchor",
    ]
    graph.metadata.update({
        "role_split": split_audit,
        "direct_evidence_config": direct_cfg,
        "blocker_search_scope": blocker_search_scope,
        "require_complete_blocker_search": require_complete_search,
        "require_admissible_blocker": require_admissible_blocker,
        "blocker_admissibility_protocol": blocker_protocol,
        "command_binding_valid": command_binding_valid,
        "command_binding_reasons": command_binding_reasons,
        "allow_direct_edge_mutation": allow_direct_edge_mutation,
        "graph_mutation_scope":
            "crossrole_certified_shortcut_deletion_only",
        "state_counts": state_counts,
        "acquisition_pairs": acquisition_pairs,
        "acquisition_target_scores": acquisition_target_scores.tolist(),
        "representation_safety_required": require_safety,
        "command_provenance": command_provenance,
        "evidence_ledger_protocol": "acc_evidence_ledger_v1",
        "evidence_ledger_sha256": evidence_ledger_sha256,
        "evaluation_rows_consumed_by_discovery": 0,
        "materialized_input_roles": [ROLE_ADAPTER, ROLE_GRAPH],
        "materialized_input_excludes_D_E": True,
        "D_E_capability_boundary": "discovery_and_safety_modules_only",
        "external_D_E_process_isolation": False,
        "access_ledger_sha256": access_ledger.get("ledger_sha256"),
        "preoutcome_commitment_hashes": list(
            access_ledger.get("preoutcome_commitment_hashes", ())),
        "initial_prior_semantics": initial_prior_semantics,
        "representation_claim_scope":
            "supervised_D0_anchor_not_pixel_only_active_calibration",
        "diagnostic_prior": diagnostic_prior,
        "eligibility_status": "DEVELOPMENT_ONLY",
        "eligibility_reasons": eligibility_reasons,
        "formal_eligible": False,
    })
    audit = {
        "protocol": "acc_graph_only_v1",
        "phase": "graph_only",
        "role_split": split_audit,
        "direct_evidence_config": direct_cfg,
        "pair_evidence": consensus,
        "state_counts": state_counts,
        "blocker_search_scope": blocker_search_scope,
        "require_complete_blocker_search": require_complete_search,
        "require_admissible_blocker": require_admissible_blocker,
        "blocker_admissibility_protocol": blocker_protocol,
        "command_binding_valid": command_binding_valid,
        "command_binding_reasons": command_binding_reasons,
        "allow_direct_edge_mutation": allow_direct_edge_mutation,
        "graph_mutation_scope":
            "crossrole_certified_shortcut_deletion_only",
        "acquisition_pairs": acquisition_pairs,
        "acquisition_target_scores": acquisition_target_scores.tolist(),
        "representation_safety_required": require_safety,
        "representation_safety_audit": safety_audit,
        "command_provenance": command_provenance,
        "evidence_ledger_protocol": "acc_evidence_ledger_v1",
        "evidence_ledger_sha256": evidence_ledger_sha256,
        "evaluation_rows_consumed_by_discovery": 0,
        "materialized_input_roles": [ROLE_ADAPTER, ROLE_GRAPH],
        "materialized_input_excludes_D_E": True,
        "D_E_capability_boundary": "discovery_and_safety_modules_only",
        "external_D_E_process_isolation": False,
        "access_ledger_sha256": access_ledger.get("ledger_sha256"),
        "preoutcome_commitment_hashes": list(
            access_ledger.get("preoutcome_commitment_hashes", ())),
        "initial_prior_semantics": initial_prior_semantics,
        "representation_claim_scope":
            "supervised_D0_anchor_not_pixel_only_active_calibration",
        "graph": graph.to_dict(),
        "diagnostic_prior": diagnostic_prior,
        "eligibility_status": "DEVELOPMENT_ONLY",
        "eligibility_reasons": eligibility_reasons,
        "formal_eligible": False,
    }
    return graph, audit


def select_acc_target(audit, n_done, cap, K, prng, allowed_targets=None):
    """Choose the intervention target that resolves most ACC uncertainty.

    Representation-safety uncertainty has weight two because it specifically
    requires ``do(target)``; unresolved adjacency/direction contributes one to
    each endpoint.  The function consumes only the auditable ACC state and
    budget counters—never graph truth or downstream/OOD outcomes.
    """
    scores = np.asarray(audit.get("acquisition_target_scores", []), dtype=float)
    n_done = np.asarray(n_done, dtype=int)
    if scores.shape != (int(K),) or n_done.shape != (int(K),):
        raise ValueError("ACC target scores and n_done must both have shape (K,)")
    allowed = set(range(int(K))) if allowed_targets is None else {
        int(k) for k in allowed_targets
    }
    if any(k < 0 or k >= int(K) for k in allowed):
        raise ValueError("allowed_targets contains an invalid node")
    available = np.asarray([
        k for k in range(int(K))
        if k in allowed and n_done[k] < int(cap)
    ], dtype=int)
    if available.size == 0:
        return None
    best = float(np.max(scores[available]))
    tied = available[np.isclose(scores[available], best)]
    return int(prng.choice(tied))


# --------------------------- acquisition (the motive) ---------------------------
def select_target(policy, undirected, cons, oracle_A, n_done, cap, K, prng, rr_ptr):
    """Choose a target; only the explicitly labelled oracle may read oracle_A."""
    avail = [k for k in range(K) if n_done[k] < cap]
    if not avail:
        return int(prng.integers(K)), rr_ptr
    if policy == "random":
        return int(prng.choice(avail)), rr_ptr
    if policy == "roundrobin":
        for off in range(K):
            k = (rr_ptr + off) % K
            if k in avail:
                return k, (k + 1) % K
        return avail[0], rr_ptr
    if policy == "maxedge":
        if undirected:
            i, j = undirected[int(prng.integers(len(undirected)))]
            cand = [x for x in (i, j) if x in avail]
            if cand:
                return int(prng.choice(cand)), rr_ptr
        return int(prng.choice(avail)), rr_ptr
    if policy == "coverage":
        score = np.zeros(K)
        for (i, j) in undirected:
            score[i] += 1.0; score[j] += 1.0   # both count incident undirected edges
        score[[k for k in range(K) if k not in avail]] = -1.0
        if score.max() > 0:
            return int(np.argmax(score)), rr_ptr
        return int(prng.choice(avail)), rr_ptr
    if policy == "eig":
        score = eig_scores(cons, undirected, K, avail, prng)
        if score.max() > 0:
            return int(np.argmax(score)), rr_ptr
        return int(prng.choice(avail)), rr_ptr
    if policy == "hb":
        skel = np.zeros((K, K))
        for a in range(K):
            for b in range(K):
                if cons[a, b] > 0.5:
                    skel[a, b] = skel[b, a] = 1.0
        for (i, j) in undirected:
            skel[i, j] = skel[j, i] = 1.0
        score = hb_scores(
            cons, undirected, K, avail, prng, _meek_complete, skel)
        if score.max() > 0:
            return int(np.argmax(score)), rr_ptr
        return int(prng.choice(avail)), rr_ptr
    if policy == "oracle":
        if oracle_A is None:
            raise ValueError("oracle policy requires oracle_A; deployable policies do not")
        score = np.zeros(K)
        for a in range(K):
            for b in range(K):
                if oracle_A[a, b] > 0 \
                        and not (cons[a, b] > 0.5 and cons[b, a] < 0.5):
                    score[a] += 1; score[b] += 1
        score[[k for k in range(K) if k not in avail]] = -1.0
        if score.max() > 0:
            return int(np.argmax(score)), rr_ptr
        return int(prng.choice(avail)), rr_ptr
    raise ValueError(policy)


def run_one_seed(seed, args):
    K = args.K
    A_true, W, noise, rng = make_scm_reversible(seed, K, args.p_edge, min_rev=args.min_rev)
    skel = ((A_true + A_true.T) > 0).astype(float)
    legacy_skeleton = bool(getattr(args, "legacy_skeleton", False))
    obs_graph = None if legacy_skeleton else true_cpdag_from_dag(A_true)
    Z0 = sample_scm(W, args.n_obs, rng, noise); M0 = np.zeros_like(Z0)
    out = {
        "n_rev": (
            cpdag_reversible_count(A_true)
            if legacy_skeleton else len(obs_graph.undirected)
        ),
        "orientation_engine": args.orientation_engine,
        "orientation_config": args.orientation_config_data,
        "mode": (
            "legacy_skeleton_direct_do_ablation"
            if legacy_skeleton
            else "diagnostic_upper_bound_given_cpdag"
        ),
        "diagnostic_upper_bound": not legacy_skeleton,
        "observational_graph": (
            None if obs_graph is None else obs_graph.to_dict()
        ),
        "graph_contract": "CPDAG -> interventional essential graph -> DAG",
    }
    for policy in POLICIES + ["passive"]:
        Z, M = [Z0.copy()], [M0.copy()]
        n_done = np.zeros(K, dtype=int); rr_ptr = 0
        curve, budget, unresolved_curve, kind_curve = [], [], [], []
        prng = np.random.default_rng(9000 + seed)
        for r in range(args.rounds + 1):
            graph, _ = discover(
                np.concatenate(Z, 0),
                np.concatenate(M, 0),
                skel,
                K,
                args.alpha,
                args.min_eff,
                do_margin=args.do_margin,
                do_thr=args.do_thr,
                engine=args.orientation_engine,
                engine_config=args.orientation_config_data,
                initial_graph=obs_graph,
                legacy_skeleton=legacy_skeleton,
                return_graph=True,
            )
            cons, undirected = graph.directed, graph.undirected
            curve.append(pdag_shd(A_true, graph.skeleton, cons))
            budget.append(int(n_done.sum()))
            unresolved_curve.append(len(undirected))
            kind_curve.append(graph.graph_type)
            if r == args.rounds:
                break
            if not undirected:
                Znew = sample_scm(W, args.n_int, rng, noise)
                do_k = None
            elif not graph.valid and not legacy_skeleton:
                Znew = sample_scm(W, args.n_int, rng, noise)
                do_k = None
            elif policy == "passive":
                Znew = sample_scm(W, args.n_int, rng, noise); do_k = None
            else:
                oracle_A = A_true if policy == "oracle" else None
                do_k, rr_ptr = select_target(policy, undirected, cons, oracle_A,
                                             n_done, args.cap, K, prng, rr_ptr)
                Znew = sample_scm(W, args.n_int, rng, noise, do_node=do_k)
            Mnew = np.zeros_like(Znew)
            if do_k is not None:
                Mnew[:, do_k] = 1.0; n_done[do_k] += 1
            Z.append(Znew); M.append(Mnew)
        out[policy] = {
            "shd": curve,
            "budget": budget,
            "n_undirected": unresolved_curve,
            "graph_kind": kind_curve,
            "final_graph": graph.to_dict(),
        }
    return out, int(A_true.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--p_edge", type=float, default=0.4)
    ap.add_argument("--n_obs", type=int, default=3000)
    ap.add_argument("--n_int", type=int, default=600)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--cap", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min_eff", type=float, default=0.05)
    ap.add_argument("--do_thr", type=float, default=0.12,
                    help="legacy direct_do marginal-correlation threshold")
    ap.add_argument("--do_margin", type=float, default=0.0,
                    help="legacy direct_do score/threshold margin")
    ap.add_argument("--orientation_engine", default="direct_do",
                    choices=ORIENTATION_ENGINES)
    ap.add_argument("--orientation_config", default=None,
                    help="JSON file containing the selected engine configuration")
    ap.add_argument(
        "--legacy_skeleton",
        action="store_true",
        help="legacy ablation: skip observational CPDAG and orient bare skeleton",
    )
    ap.add_argument("--min_rev", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default="acq_closed_result.json")
    args = ap.parse_args()
    if args.orientation_config:
        with open(args.orientation_config) as f:
            loaded_config = json.load(f)
        args.orientation_config_data = loaded_config.get(
            "engines", {}).get(args.orientation_engine, loaded_config)
    else:
        args.orientation_config_data = {}

    t0 = time.time()
    all_seeds, n_edges = [], []
    for s in range(args.seeds):
        res, ne = run_one_seed(s, args)
        all_seeds.append(res); n_edges.append(ne)
        msg = " ".join(f"{p[:4]}={res[p]['shd'][-1]}" for p in POLICIES)
        print(f"[seed {s}] edges={ne} rev={res['n_rev']} passiveEnd={res['passive']['shd'][-1]}  {msg}", flush=True)

    need = [i for i in range(args.seeds) if all_seeds[i]["passive"]["shd"][-1] > 0]

    def agg_over(idxs):
        d = {}
        for policy in POLICIES + ["passive"]:
            arr = np.array([all_seeds[i][policy]["shd"] for i in idxs], dtype=float)
            d[policy] = {
                "mean_shd": arr.mean(0).tolist(),
                "budget": all_seeds[0][policy]["budget"],
                "frac_solved": (arr[:, -1] == 0).mean().item(),
                "mean_budget_to_solve": float(np.mean([
                    (np.argmax(r == 0) if (r == 0).any() else len(r)) for r in arr])),
            }
        return d

    args_out = {k: v for k, v in vars(args).items()
                if k != "orientation_config_data"}
    summary = {"args": args_out, "mean_edges": float(np.mean(n_edges)),
               "n_seeds": args.seeds, "n_need_intervention": len(need),
               "elapsed_s": round(time.time() - t0, 1),
               "orientation_engine": args.orientation_engine,
               "orientation_config": args.orientation_config_data,
               "mode": (
                   "legacy_skeleton_direct_do_ablation"
                   if args.legacy_skeleton
                   else "diagnostic_upper_bound_given_cpdag"
               ),
               "graph_contract":
                   "CPDAG -> interventional essential graph -> DAG",
               "agg_all": agg_over(list(range(args.seeds))),
               "agg_need": agg_over(need) if need else None}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n==== ACQUISITION (closed-form, K={args.K}) ====")
    print(f"seeds={args.seeds}, need-intervention={len(need)}, elapsed {summary['elapsed_s']}s")
    for tag, key in [("ALL", "agg_all"), ("NEED-intervention", "agg_need")]:
        if summary[key] is None:
            continue
        print(f"\n-- {tag} seeds --")
        for policy in POLICIES + ["passive"]:
            a = summary[key][policy]
            ms = " ".join(f"{v:.2f}" for v in a["mean_shd"])
            print(f"  {policy:10s} SHD=[{ms}]  solved={a['frac_solved']:.2f}  "
                  f"b2solve={a['mean_budget_to_solve']:.2f}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
