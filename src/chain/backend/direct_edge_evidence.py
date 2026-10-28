"""Conservative intervention evidence for direct-edge repair.

This module deliberately separates three questions that the historical AICE
orientation rule conflated:

* does ``do(i)`` change ``j`` (ancestry evidence)?
* does that response survive pre-registered blocking sets (directness)?
* is there enough evidence to decide at all (UNKNOWN rather than a false null)?

The implementation is intentionally graph-only.  It consumes learned latents,
an intervention-target mask, and optional intervention command values.  It has
no ground-truth graph, semantic variable name, OOD metric, or downstream score
argument.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations

import numpy as np


EFFECT = "EFFECT"
EQUIVALENT_ZERO = "EQUIVALENT_ZERO"
UNKNOWN = "UNKNOWN"

DIRECT = "DIRECT"
NOT_DIRECT = "NOT_DIRECT"

PAIR_DIRECT = "DIRECT"
PAIR_ABSENT = "ABSENT"
PAIR_UNKNOWN = "UNKNOWN"


DEFAULT_CONFIG = {
    "confidence": 0.95,
    "bootstrap": 300,
    "min_rows": 80,
    "min_source_sd_ratio": 0.15,
    "min_residual_source_sd_ratio": 0.08,
    "min_residual_response_sd_ratio": 0.08,
    "tau_zero": 0.05,
    "tau_present": 0.12,
    "max_order": 2,
    "allow_marginal_absence": False,
}


@dataclass(frozen=True)
class EffectEvidence:
    state: str
    score: float | None
    ci: tuple | None
    n_rows: int
    conditioning_set: tuple
    source_sd_ratio: float
    residual_source_sd_ratio: float
    residual_response_sd_ratio: float
    reason: str | None

    def to_dict(self):
        return {
            "state": self.state,
            "score": self.score,
            "ci": None if self.ci is None else list(self.ci),
            "n_rows": int(self.n_rows),
            "conditioning_set": list(self.conditioning_set),
            "source_sd_ratio": float(self.source_sd_ratio),
            "residual_source_sd_ratio": float(self.residual_source_sd_ratio),
            "residual_response_sd_ratio": float(
                self.residual_response_sd_ratio),
            "reason": self.reason,
        }


def resolve_config(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if not 0.0 < float(cfg["confidence"]) < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if int(cfg["bootstrap"]) < 20:
        raise ValueError("bootstrap must be at least 20")
    if int(cfg["min_rows"]) < 8:
        raise ValueError("min_rows must be at least 8")
    if not 0.0 <= float(cfg["tau_zero"]) < float(cfg["tau_present"]) < 1.0:
        raise ValueError("require 0 <= tau_zero < tau_present < 1")
    if int(cfg["max_order"]) < 0:
        raise ValueError("max_order must be non-negative")
    return cfg


def _single_target_rows(M, node, row_mask=None):
    M = np.asarray(M, dtype=float)
    single = np.isclose(M.sum(1), 1.0)
    keep = (M[:, int(node)] > 0.5) & single
    if row_mask is not None:
        row_mask = np.asarray(row_mask, dtype=bool)
        if row_mask.shape != (len(M),):
            raise ValueError("row_mask has the wrong shape")
        keep &= row_mask
    return np.flatnonzero(keep)


def _residualize(y, X):
    y = np.asarray(y, dtype=float).reshape(-1)
    if X is None or np.asarray(X).size == 0:
        return y - y.mean()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    design = np.column_stack([np.ones(len(X)), X])
    ridge = 1e-5 * np.eye(design.shape[1])
    ridge[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + ridge, design.T @ y)
    return y - design @ coef


def _absolute_partial_correlation(source, response, controls):
    rs = _residualize(source, controls)
    ry = _residualize(response, controls)
    if np.std(rs) < 1e-10 or np.std(ry) < 1e-10:
        return 0.0
    value = np.corrcoef(rs, ry)[0, 1]
    return 0.0 if not np.isfinite(value) else abs(float(value))


def _bootstrap_ci(source, response, controls, confidence, n_boot, seed):
    rng = np.random.default_rng(int(seed))
    n = len(source)
    draws = np.empty(int(n_boot), dtype=float)
    for b in range(len(draws)):
        idx = rng.integers(0, n, n)
        c = None if controls is None else controls[idx]
        draws[b] = _absolute_partial_correlation(
            source[idx], response[idx], c)
    q = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(draws, q)), float(np.quantile(draws, 1.0 - q))


def _seed(source, target, conditioning_set, n_rows, salt=0):
    payload = (
        f"acc-direct-v1|{int(source)}|{int(target)}|"
        f"{','.join(map(str, conditioning_set))}|{int(n_rows)}|{int(salt)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def effect_evidence(Z, M, source, target, conditioning_set=(),
                    row_mask=None, intervention_values=None, config=None,
                    seed_salt=0):
    """Return an effect/equivalence/unknown decision for one ordered pair."""
    cfg = resolve_config(config)
    Z = np.asarray(Z, dtype=float)
    M = np.asarray(M, dtype=float)
    if Z.ndim != 2 or M.shape != Z.shape or not np.all(np.isfinite(Z)):
        raise ValueError("Z and M must be finite, same-shaped two-dimensional arrays")
    source, target = int(source), int(target)
    if source == target or not (0 <= source < Z.shape[1]) \
            or not (0 <= target < Z.shape[1]):
        raise ValueError("source and target must be distinct valid columns")
    S = tuple(sorted({int(k) for k in conditioning_set}))
    if source in S or target in S or any(k < 0 or k >= Z.shape[1] for k in S):
        raise ValueError("conditioning_set must exclude endpoints and use valid columns")
    rows = _single_target_rows(M, source, row_mask=row_mask)
    if len(rows) < int(cfg["min_rows"]):
        return EffectEvidence(
            UNKNOWN, None, None, len(rows), S, 0.0, 0.0, 0.0,
            "insufficient_single_target_rows",
        )

    if intervention_values is None:
        command = Z[:, source]
        command_source = "source_latent_proxy"
    else:
        command = np.asarray(intervention_values, dtype=float)
        if command.shape != Z.shape:
            raise ValueError("intervention_values must have the same shape as Z and M")
        command = command[:, source]
        command_source = "explicit_intervention_value"
    x = np.asarray(command[rows], dtype=float)
    y = Z[rows, target]
    controls = None if not S else Z[np.ix_(rows, S)]
    # Variation gates are part of model selection.  Their reference scale must
    # therefore use the same discovery fold; consulting all rows would leak
    # the sealed D_E distribution even though D_E never entered the effect
    # estimate itself.
    eligible = np.ones(len(Z), dtype=bool) if row_mask is None \
        else np.asarray(row_mask, dtype=bool)
    finite_eligible = eligible & np.isfinite(command)
    global_sd = float(np.std(command[finite_eligible]))
    source_sd_ratio = float(np.std(x) / max(global_sd, 1e-8))
    if not np.all(np.isfinite(x)) or source_sd_ratio < float(cfg["min_source_sd_ratio"]):
        return EffectEvidence(
            UNKNOWN, None, None, len(rows), S, source_sd_ratio, 0.0, 0.0,
            "insufficient_intervention_variation",
        )

    residual_x = _residualize(x, controls)
    residual_ratio = float(np.std(residual_x) / max(np.std(x), 1e-8))
    residual_y = _residualize(y, controls)
    response_ratio = float(np.std(residual_y) / max(np.std(y), 1e-8))
    if residual_ratio < float(cfg["min_residual_source_sd_ratio"]):
        return EffectEvidence(
            UNKNOWN, None, None, len(rows), S, source_sd_ratio,
            residual_ratio, response_ratio,
            "insufficient_conditional_source_variation",
        )
    if response_ratio < float(cfg["min_residual_response_sd_ratio"]):
        return EffectEvidence(
            UNKNOWN, None, None, len(rows), S, source_sd_ratio,
            residual_ratio, response_ratio,
            "conditioning_nearly_determines_response",
        )
    score = _absolute_partial_correlation(x, y, controls)
    ci = _bootstrap_ci(
        x, y, controls, cfg["confidence"], cfg["bootstrap"],
        _seed(source, target, S, len(rows), seed_salt),
    )
    if ci[0] > float(cfg["tau_present"]):
        state, reason = EFFECT, None
    elif ci[1] < float(cfg["tau_zero"]):
        state, reason = EQUIVALENT_ZERO, None
    else:
        state, reason = UNKNOWN, "effect_ci_in_indifference_region"
    ev = EffectEvidence(
        state, float(score), tuple(map(float, ci)), len(rows), S,
        source_sd_ratio, residual_ratio, response_ratio, reason,
    )
    # Keep this provenance in a frozen dataclass without widening its public
    # decision surface; callers record it at the ordered-edge level.
    _ = command_source
    return ev


def enumerate_conditioning_sets(K, source, target, candidate_nodes=None,
                                max_order=2):
    if candidate_nodes is None:
        candidate_nodes = [k for k in range(int(K)) if k not in (source, target)]
    nodes = sorted({int(k) for k in candidate_nodes if k not in (source, target)})
    order = min(int(max_order), len(nodes))
    return [tuple(S) for size in range(1, order + 1)
            for S in combinations(nodes, size)]


def ordered_direct_evidence(Z, M, source, target, candidate_nodes=None,
                            row_mask=None, intervention_values=None,
                            config=None, seed_salt=0):
    """Distinguish marginal ancestry from conditional directness.

    A conditional null is accepted only by an explicit equivalence interval.
    A merely non-significant or under-powered test remains ``UNKNOWN``.
    """
    cfg = resolve_config(config)
    marginal = effect_evidence(
        Z, M, source, target, (), row_mask=row_mask,
        intervention_values=intervention_values, config=cfg,
        seed_salt=seed_salt,
    )
    out = {
        "ordered_edge": [int(source), int(target)],
        "ancestry": marginal.to_dict(),
        "direct_state": UNKNOWN,
        "conditioning_evidence": [],
        "search_complete": False,
        "command_source": (
            "source_latent_proxy" if intervention_values is None
            else "explicit_intervention_value"
        ),
    }
    if marginal.state == EQUIVALENT_ZERO:
        out.update({
            "direct_state": NOT_DIRECT,
            "reason": "no_ancestral_response_by_equivalence",
            "search_complete": True,
        })
        return out
    if marginal.state != EFFECT:
        out["reason"] = "ancestry_unknown"
        return out

    all_nodes = [k for k in range(Z.shape[1]) if k not in (source, target)]
    candidates = all_nodes if candidate_nodes is None else sorted(set(candidate_nodes))
    sets_ = enumerate_conditioning_sets(
        Z.shape[1], source, target, candidates, cfg["max_order"])
    full_candidate_universe = set(candidates) == set(all_nodes)
    search_complete = (
        full_candidate_universe
        and int(cfg["max_order"]) >= len(candidates)
    )
    records = [effect_evidence(
        Z, M, source, target, S, row_mask=row_mask,
        intervention_values=intervention_values, config=cfg,
        seed_salt=seed_salt,
    ) for S in sets_]
    out["conditioning_evidence"] = [rec.to_dict() for rec in records]
    out["search_complete"] = bool(search_complete)
    separators = [rec for rec in records if rec.state == EQUIVALENT_ZERO]
    if separators:
        out.update({
            "direct_state": NOT_DIRECT,
            "reason": "ancestral_response_blocked_by_equivalent_zero_set",
            "separating_sets": [list(rec.conditioning_set) for rec in separators],
        })
    elif (not records and search_complete) or (
            search_complete and all(rec.state == EFFECT for rec in records)):
        out.update({
            "direct_state": DIRECT,
            "reason": "response_survives_complete_blocking_search",
        })
    else:
        out.update({
            "direct_state": UNKNOWN,
            "reason": (
                "blocking_search_incomplete" if not search_complete
                else "conditional_evidence_unknown"
            ),
        })
    return out


def pair_direct_evidence(Z, M, i, j, candidate_nodes=None, row_mask=None,
                         intervention_values=None, config=None, seed_salt=0):
    cfg = resolve_config(config)
    forward = ordered_direct_evidence(
        Z, M, i, j, candidate_nodes=candidate_nodes, row_mask=row_mask,
        intervention_values=intervention_values, config=cfg,
        seed_salt=seed_salt,
    )
    reverse = ordered_direct_evidence(
        Z, M, j, i, candidate_nodes=candidate_nodes, row_mask=row_mask,
        intervention_values=intervention_values, config=cfg,
        seed_salt=seed_salt + 1,
    )
    fi, rj = forward["direct_state"], reverse["direct_state"]
    if fi == DIRECT and rj == NOT_DIRECT:
        state, direction = PAIR_DIRECT, [int(i), int(j)]
    elif rj == DIRECT and fi == NOT_DIRECT:
        state, direction = PAIR_DIRECT, [int(j), int(i)]
    elif fi == NOT_DIRECT and rj == NOT_DIRECT:
        has_conditional_separator = bool(
            forward.get("separating_sets") or reverse.get("separating_sets"))
        if has_conditional_separator or bool(cfg["allow_marginal_absence"]):
            state, direction = PAIR_ABSENT, None
        else:
            state, direction = PAIR_UNKNOWN, None
    else:
        state, direction = PAIR_UNKNOWN, None
    return {
        "pair": [int(min(i, j)), int(max(i, j))],
        "state": state,
        "direction": direction,
        "forward": forward,
        "reverse": reverse,
    }


def consensus_pair_evidence(first, second):
    """Accept only exact cross-fold agreement; conflicts fail to UNKNOWN."""
    same_state = first["state"] == second["state"]
    same_direction = first.get("direction") == second.get("direction")
    accepted = bool(same_state and same_direction)
    separator_agreement = True
    if accepted and first["state"] == PAIR_ABSENT:
        for side in ("forward", "reverse"):
            a = {tuple(S) for S in first[side].get("separating_sets", [])}
            b = {tuple(S) for S in second[side].get("separating_sets", [])}
            marginal_a = (
                first[side]["ancestry"]["state"] == EQUIVALENT_ZERO)
            marginal_b = (
                second[side]["ancestry"]["state"] == EQUIVALENT_ZERO)
            if not ((marginal_a and marginal_b) or (a & b)):
                separator_agreement = False
        accepted = bool(separator_agreement)
    return {
        "pair": list(first["pair"]),
        "state": first["state"] if accepted else PAIR_UNKNOWN,
        "direction": first.get("direction") if accepted else None,
        "fold_agreement": accepted,
        "separator_agreement": bool(separator_agreement),
        "folds": {"D_A": first, "D_G": second},
        "reason": None if accepted else (
            "cross_fold_separator_disagreement"
            if same_state and same_direction and not separator_agreement
            else "cross_fold_disagreement"
        ),
    }
