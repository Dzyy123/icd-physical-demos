"""Hard-intervention residual test for representation leakage.

Under ``do(j)``, incoming causal edges to variable ``j`` are cut.  After the
learned coordinate ``z_j`` is predicted from the externally recorded command
``U_j``, dependence of the residual on another coordinate ``z_i`` is evidence
of representation crosstalk rather than a legitimate ``i -> j`` mechanism.

This diagnostic is a gate, not an adapter objective.  It intentionally refuses
to run without explicit intervention command values: using ``z_j`` itself as
its command would trivially regress away the leakage being tested.
"""

from __future__ import annotations

import hashlib

import numpy as np

from acc_protocol import (
    ROLE_ADAPTER,
    ROLE_GRAPH,
    discovery_fold_masks,
    make_acc_role_split,
)


SAFE = "SAFE"
UNSAFE = "UNSAFE"
UNKNOWN = "UNKNOWN"


DEFAULT_CONFIG = {
    "confidence": 0.95,
    "bootstrap": 300,
    "folds": 5,
    "min_rows": 100,
    "tau_zero": 0.05,
    "tau_present": 0.12,
    "ridge": 1e-4,
    "max_negligible_residual_sd_ratio": 1e-4,
}


def _resolve_config(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    numeric = (
        "confidence", "bootstrap", "folds", "min_rows", "tau_zero",
        "tau_present", "ridge", "max_negligible_residual_sd_ratio",
    )
    if not all(np.isfinite(float(cfg[key])) for key in numeric):
        raise ValueError("all representation-safety parameters must be finite")
    if not 0.5 < float(cfg["confidence"]) < 1.0:
        raise ValueError("confidence must lie in (0.5, 1)")
    if float(cfg["folds"]) != int(cfg["folds"]):
        raise ValueError("folds must be an integer")
    if int(cfg["folds"]) < 2:
        raise ValueError("folds must be at least two")
    if float(cfg["bootstrap"]) != int(cfg["bootstrap"]):
        raise ValueError("bootstrap must be an integer")
    if int(cfg["bootstrap"]) < 20:
        raise ValueError("bootstrap must be at least 20")
    if float(cfg["min_rows"]) != int(cfg["min_rows"]) \
            or int(cfg["min_rows"]) < 4:
        raise ValueError("min_rows must be an integer of at least four")
    if int(cfg["folds"]) > int(cfg["min_rows"]) // 2:
        raise ValueError("folds is too large for the minimum support")
    if not 0.0 <= float(cfg["tau_zero"]) < float(cfg["tau_present"]) < 1.0:
        raise ValueError("require 0 <= tau_zero < tau_present < 1")
    if not 0.0 < float(cfg["ridge"]) <= 1.0:
        raise ValueError("ridge must lie in (0, 1]")
    if not 0.0 <= float(cfg["max_negligible_residual_sd_ratio"]) <= 0.01:
        raise ValueError(
            "max_negligible_residual_sd_ratio must lie in [0, 0.01]")
    return cfg


def _seed(source, target, n_rows, salt):
    payload = f"acc-safety-v1|{source}|{target}|{n_rows}|{salt}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _single_target_rows(M, node, row_mask):
    single = np.isclose(M.sum(1), 1.0)
    keep = single & (M[:, int(node)] > 0.5)
    if row_mask is not None:
        keep &= np.asarray(row_mask, dtype=bool)
    return np.flatnonzero(keep)


def _features(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    mu, sd = float(x.mean()), float(x.std())
    z = (x - mu) / max(sd, 1e-8)
    return np.column_stack([np.ones(len(z)), z, z * z])


def _fit_predict(x_train, y_train, x_eval, ridge):
    X = _features(x_train)
    Xe = _features_with_reference(x_eval, x_train)
    penalty = float(ridge) * np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(X.T @ X + penalty, X.T @ y_train)
    return Xe @ coef


def _features_with_reference(x, reference):
    x = np.asarray(x, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    z = (x - reference.mean()) / max(reference.std(), 1e-8)
    return np.column_stack([np.ones(len(z)), z, z * z])


def _crossfit_residual(command, learned_target, folds, ridge, seed):
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(command))
    assignments = np.empty(len(command), dtype=int)
    assignments[order] = np.arange(len(command)) % int(folds)
    residual = np.empty(len(command), dtype=float)
    for fold in range(int(folds)):
        test = assignments == fold
        train = ~test
        residual[test] = learned_target[test] - _fit_predict(
            command[train], learned_target[train], command[test], ridge)
    return residual


def _residualize_on_command(x, command):
    return x - _fit_predict(command, x, command, ridge=1e-5)


def _abs_corr(x, y):
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    value = np.corrcoef(x, y)[0, 1]
    return 0.0 if not np.isfinite(value) else abs(float(value))


def residual_safety_evidence(Z, M, intervention_values, source, target,
                             row_mask=None, config=None, seed_salt=0):
    cfg = _resolve_config(config)
    Z = np.asarray(Z, dtype=float)
    M = np.asarray(M, dtype=float)
    U = np.asarray(intervention_values, dtype=float)
    if Z.ndim != 2 or M.shape != Z.shape or U.shape != Z.shape:
        raise ValueError("Z, M, and intervention_values must have equal 2D shapes")
    if not np.all(np.isfinite(Z)):
        raise ValueError("Z contains non-finite values")
    source, target = int(source), int(target)
    if source == target:
        raise ValueError("source and target must differ")
    rows = _single_target_rows(M, target, row_mask)
    out = {
        "ordered_pair": [source, target],
        "test_environment": f"do({target})",
        "state": UNKNOWN,
        "n_rows": int(len(rows)),
        "score": None,
        "ci": None,
        "uses_explicit_intervention_values": True,
    }
    if len(rows) < int(cfg["min_rows"]):
        out["reason"] = "insufficient_single_target_rows"
        return out
    command = U[rows, target]
    if not np.all(np.isfinite(command)) or np.std(command) < 1e-8:
        out["reason"] = "missing_or_constant_intervention_command"
        return out
    learned_target = Z[rows, target]
    other = Z[rows, source]
    salt = _seed(source, target, len(rows), seed_salt)
    target_residual = _crossfit_residual(
        command, learned_target, cfg["folds"], cfg["ridge"], salt)
    other_residual = _residualize_on_command(other, command)
    target_residual_ratio = float(
        np.std(target_residual) / max(np.std(learned_target), 1e-8))
    if target_residual_ratio < float(cfg["max_negligible_residual_sd_ratio"]):
        out.update({
            "state": SAFE,
            "score": 0.0,
            "ci": [0.0, 0.0],
            "reason": "target_residual_negligible",
            "target_residual_sd": float(np.std(target_residual)),
            "target_residual_sd_ratio": target_residual_ratio,
            "other_residual_sd": float(np.std(other_residual)),
            "config": dict(cfg),
        })
        return out
    score = _abs_corr(target_residual, other_residual)
    rng = np.random.default_rng(salt + 1)
    draws = np.empty(int(cfg["bootstrap"]), dtype=float)
    for b in range(len(draws)):
        idx = rng.integers(0, len(rows), len(rows))
        draws[b] = _abs_corr(target_residual[idx], other_residual[idx])
    q = (1.0 - float(cfg["confidence"])) / 2.0
    ci = (float(np.quantile(draws, q)),
          float(np.quantile(draws, 1.0 - q)))
    if ci[0] > float(cfg["tau_present"]):
        state, reason = UNSAFE, None
    elif ci[1] < float(cfg["tau_zero"]):
        state, reason = SAFE, None
    else:
        state, reason = UNKNOWN, "safety_ci_in_indifference_region"
    out.update({
        "state": state,
        "score": float(score),
        "ci": list(ci),
        "reason": reason,
        "target_residual_sd": float(np.std(target_residual)),
        "target_residual_sd_ratio": target_residual_ratio,
        "other_residual_sd": float(np.std(other_residual)),
        "config": dict(cfg),
    })
    return out


def consensus_safety_evidence(first, second):
    agree = first["state"] == second["state"]
    state = first["state"] if agree else UNKNOWN
    return {
        "ordered_pair": list(first["ordered_pair"]),
        "state": state,
        "fold_agreement": bool(agree),
        "folds": {"D_A": first, "D_G": second},
        "reason": None if agree else "cross_fold_disagreement",
    }


def audit_representation_safety(Z, M, intervention_values, row_ids,
                                config=None, role_assignments=None,
                                role_split_audit=None):
    """Run the safety gate on D_A/D_G only and return an auditable report."""
    cfg = {} if config is None else dict(config)
    if role_assignments is None:
        split = make_acc_role_split(
            row_ids,
            M,
            seed=int(cfg.get("split_seed", 0)),
            fractions=tuple(cfg.get("split_fractions", (0.4, 0.4, 0.2))),
            require_explicit_ids=bool(cfg.get("require_explicit_ids", True)),
        )
        fold_masks = discovery_fold_masks(split)
        split_audit = split.audit_dict()
    else:
        roles = np.asarray(role_assignments)
        if roles.shape != (len(np.asarray(M)),):
            raise ValueError("role_assignments must have one role per row")
        if not set(roles.tolist()).issubset({ROLE_ADAPTER, ROLE_GRAPH}):
            raise ValueError(
                "materialized safety view may contain only D_A/D_G rows")
        fold_masks = (
            ("A_to_G", roles == ROLE_ADAPTER),
            ("G_to_A", roles == ROLE_GRAPH),
        )
        split_audit = dict(role_split_audit or {
            "protocol": "preassigned_materialized_roles",
            "counts": {
                ROLE_ADAPTER: int(np.sum(roles == ROLE_ADAPTER)),
                ROLE_GRAPH: int(np.sum(roles == ROLE_GRAPH)),
            },
            "evaluation_role_sealed": True,
        })
    test_cfg = dict(cfg.get("test", {}))
    fold_records = []
    for fold_index, (_, mask) in enumerate(fold_masks):
        records = {}
        for source in range(np.asarray(Z).shape[1]):
            for target in range(np.asarray(Z).shape[1]):
                if source == target:
                    continue
                records[(source, target)] = residual_safety_evidence(
                    Z, M, intervention_values, source, target,
                    row_mask=mask, config=test_cfg,
                    seed_salt=1009 * fold_index,
                )
        fold_records.append(records)
    evidence = [
        consensus_safety_evidence(
            fold_records[0][pair], fold_records[1][pair])
        for pair in sorted(fold_records[0])
    ]
    counts = {
        state: sum(rec["state"] == state for rec in evidence)
        for state in (SAFE, UNSAFE, UNKNOWN)
    }
    return {
        "protocol": "acc_hard_do_representation_safety_v1",
        "role_split": split_audit,
        "evidence": evidence,
        "state_counts": counts,
        "evaluation_rows_consumed": 0,
        "input_roles": [ROLE_ADAPTER, ROLE_GRAPH],
        "materialized_input_excludes_D_E": True,
        "eligibility_status": "DEVELOPMENT_ONLY",
        "eligibility_reasons": [
            "diagnostic_quadratic_nuisance_model",
            "no_hard_do_provenance_hash_binding",
            "no_multiplicity_control",
        ],
        "formal_eligible": False,
    }
