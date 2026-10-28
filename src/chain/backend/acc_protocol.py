"""Fail-closed row-role protocol for Active Causal Calibration (ACC).

The role of every episode is fixed from its identifier and intervention target
before outcomes are inspected.  ``D_A`` and ``D_G`` are symmetric discovery
folds; ``D_E`` is a sealed evaluation fold and must never enter discovery or
adapter fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


ROLE_ADAPTER = "D_A"
ROLE_GRAPH = "D_G"
ROLE_EVAL = "D_E"
ROLES = (ROLE_ADAPTER, ROLE_GRAPH, ROLE_EVAL)


@dataclass(frozen=True)
class ACCRoleSplit:
    roles: np.ndarray
    protocol_sha256: str
    formal_eligible: bool
    counts: dict
    fractions: tuple
    seed: int
    explicit_ids: bool

    def mask(self, role):
        if role not in ROLES:
            raise ValueError(f"unknown ACC role {role!r}")
        return self.roles == role

    def audit_dict(self):
        return {
            "protocol": "acc_role_split_v1",
            "protocol_sha256": self.protocol_sha256,
            "formal_eligible": bool(self.formal_eligible),
            "role_protocol_valid": bool(self.explicit_ids),
            "eligibility_status": "DEVELOPMENT_ONLY",
            "counts": {k: int(v) for k, v in self.counts.items()},
            "fractions": [float(x) for x in self.fractions],
            "seed": int(self.seed),
            "role_names": list(ROLES),
            "evaluation_role_sealed": True,
        }


def _target_signature(mask_row):
    active = np.flatnonzero(np.asarray(mask_row, dtype=float) > 0.5)
    if active.size == 0:
        return "obs"
    return "do:" + ",".join(str(int(k)) for k in active)


def make_acc_role_split(row_ids, intervention_mask, seed=0,
                        fractions=(0.4, 0.4, 0.2),
                        require_explicit_ids=True):
    """Assign deterministic, target-stratified roles without reading outcomes.

    ``row_ids`` must be stable episode identifiers for a formally eligible
    split.  Passing ``None`` is allowed only for diagnostic use and marks the
    result ineligible rather than silently inventing formal provenance.
    """
    M = np.asarray(intervention_mask, dtype=float)
    if M.ndim != 2 or not np.all(np.isfinite(M)):
        raise ValueError("intervention_mask must be a finite two-dimensional array")
    fractions = tuple(float(x) for x in fractions)
    if len(fractions) != 3 or any(x <= 0.0 for x in fractions) \
            or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must be three positive values summing to one")

    explicit_ids = row_ids is not None
    if row_ids is None:
        if require_explicit_ids:
            raise ValueError("explicit stable row_ids are required for formal ACC")
        ids = [f"diagnostic-row-{i}" for i in range(len(M))]
    else:
        ids = [str(x) for x in row_ids]
    if len(ids) != len(M):
        raise ValueError("row_ids and intervention_mask must have equal length")
    if len(set(ids)) != len(ids):
        raise ValueError("row_ids must be unique")

    bounds = np.cumsum(fractions)
    roles = np.empty(len(M), dtype="<U3")
    assignments = []
    for idx, (row_id, mask_row) in enumerate(zip(ids, M)):
        target = _target_signature(mask_row)
        payload = f"acc-role-v1|{int(seed)}|{target}|{row_id}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        value = int.from_bytes(digest[:8], "big") / float(2**64)
        role = ROLES[0] if value < bounds[0] else (
            ROLES[1] if value < bounds[1] else ROLES[2])
        roles[idx] = role
        assignments.append((row_id, target, role))

    audit_payload = {
        "protocol": "acc_role_split_v1",
        "seed": int(seed),
        "fractions": fractions,
        "assignments": assignments,
    }
    protocol_sha = hashlib.sha256(json.dumps(
        audit_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    counts = {role: int(np.sum(roles == role)) for role in ROLES}
    return ACCRoleSplit(
        roles=roles,
        protocol_sha256=protocol_sha,
        # Stable IDs are necessary, but nowhere near sufficient, for formal
        # eligibility (which also needs frozen hashes, multiplicity control,
        # per-target power, external evaluation, and downstream no-harm).
        formal_eligible=False,
        counts=counts,
        fractions=fractions,
        seed=int(seed),
        explicit_ids=bool(explicit_ids),
    )


def discovery_fold_masks(split):
    """Return the two symmetric certification folds; never return ``D_E``."""
    if not isinstance(split, ACCRoleSplit):
        raise TypeError("split must be an ACCRoleSplit")
    return (
        ("A_to_G", split.mask(ROLE_ADAPTER)),
        ("G_to_A", split.mask(ROLE_GRAPH)),
    )
