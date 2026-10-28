"""Deployable statistical tests for orienting a known skeleton edge.

The functions in this module only consume learned variables ``Z`` and the
observed intervention-target mask ``M``.  They deliberately have no graph-truth
or root-hint argument.  Every non-legacy engine requires both single-target
``do(i)`` and ``do(j)`` samples and abstains when the target has too little
variation or the bootstrap evidence is ambiguous.

Available engines
-----------------
``cf_regression``
    Type-aware cross-fitted held-out loss improvement inside each intervention
    environment.  Continuous responses use nonlinear ridge regression and
    clearly bimodal responses use logistic regression.
``invariant``
    Fits each candidate mechanism on observational rows and scores its held-out
    transfer to the corresponding parent-intervention environment.
``hsic``
    Non-parametric normalized HSIC contrast with RBF/equality kernels.
``consensus``
    Orients only when at least two engines agree and none disagrees.

All randomization is deterministically derived from the edge indices and sample
counts, so repeated discovery calls on identical data return identical results.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np


ENGINE_NAMES = ("cf_regression", "invariant", "hsic", "consensus")


DEFAULT_CONFIGS = {
    "cf_regression": {
        "confidence": 0.95,
        "bootstrap": 200,
        "folds": 5,
        "min_rows": 80,
        "min_target_sd": 0.15,
        "min_gain": 0.005,
        "min_contrast": 0.005,
        "max_reverse_gain": 0.15,
        "binary_r2": 0.82,
        "ridge": 0.05,
    },
    "invariant": {
        "confidence": 0.95,
        "bootstrap": 200,
        "folds": 5,
        "min_rows": 80,
        "min_obs_rows": 250,
        "min_target_sd": 0.15,
        "min_gain": 0.005,
        "min_contrast": 0.005,
        "max_reverse_gain": 0.15,
        "min_support_overlap": 0.65,
        "binary_r2": 0.82,
        "ridge": 0.05,
    },
    "hsic": {
        "confidence": 0.95,
        "bootstrap": 160,
        "bootstrap_rows": 160,
        "min_rows": 80,
        "min_target_sd": 0.15,
        "min_gain": 0.015,
        "min_contrast": 0.005,
        "max_reverse_gain": 0.20,
        "binary_r2": 0.82,
    },
    "consensus": {
        "members": ["cf_regression", "invariant", "hsic"],
        "min_agree": 2,
        "veto_conflict": True,
    },
}


def resolve_config(engine, config=None):
    """Return a deep-merged, JSON-serializable engine configuration."""
    if engine not in ENGINE_NAMES:
        raise ValueError(
            f"unknown statistical orientation engine {engine!r}; "
            f"choose from {ENGINE_NAMES}"
        )
    out = deepcopy(DEFAULT_CONFIGS[engine])
    if config:
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key].update(value)
            else:
                out[key] = value
    return out


def _seed(i, j, n, salt):
    return int(
        (104729 * (int(i) + 1) + 130363 * (int(j) + 1)
         + 433 * int(n) + 1009 * int(salt)) % (2**32 - 1)
    )


def _single_target_rows(M, node):
    single = np.isclose(np.asarray(M, dtype=float).sum(1), 1.0)
    return np.where((np.asarray(M)[:, int(node)] > 0.5) & single)[0]


def _observational_rows(M):
    return np.where(np.isclose(np.asarray(M, dtype=float).sum(1), 0.0))[0]


def _finite_1d(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def _bimodal_spec(y, min_r2=0.82):
    """Infer a binary/logistic response from two well-separated modes.

    This is an unsupervised type decision on the learned response itself.  It
    does not assume that any named physical variable is binary.
    """
    y = _finite_1d(y)
    if y.size < 40 or float(np.std(y)) < 1e-8:
        return {
            "kind": "continuous",
            "binary_r2": 0.0,
            "threshold": None,
        }
    c = np.quantile(y, [0.25, 0.75]).astype(float)
    if abs(c[1] - c[0]) < 1e-8:
        return {
            "kind": "continuous",
            "binary_r2": 0.0,
            "threshold": None,
        }
    labels = np.zeros(y.size, dtype=int)
    for _ in range(30):
        new = (np.abs(y - c[1]) < np.abs(y - c[0])).astype(int)
        if new.min() == new.max():
            break
        c_new = np.array([y[new == k].mean() for k in (0, 1)])
        if np.allclose(c, c_new, atol=1e-10, rtol=0):
            labels = new
            c = c_new
            break
        labels, c = new, c_new
    order = np.argsort(c)
    c = c[order]
    labels = (y > float(c.mean())).astype(int)
    frac = float(min(labels.mean(), 1.0 - labels.mean()))
    total = float(np.sum((y - y.mean()) ** 2))
    within = float(sum(np.sum((y[labels == k] - c[k]) ** 2)
                       for k in (0, 1)))
    r2 = max(0.0, 1.0 - within / max(total, 1e-12))
    kind = "binary" if frac >= 0.08 and r2 >= float(min_r2) else "continuous"
    return {
        "kind": kind,
        "binary_r2": float(r2),
        "threshold": float(c.mean()) if kind == "binary" else None,
        "minority_fraction": frac,
    }


def infer_variable_type(y, binary_r2=0.82):
    """Public, deployable learned-variable type inference."""
    return _bimodal_spec(y, min_r2=binary_r2)


def _standardize_from_train(x_train, x_eval):
    x_train = np.asarray(x_train, dtype=float).reshape(-1)
    x_eval = np.asarray(x_eval, dtype=float).reshape(-1)
    mu = float(np.mean(x_train))
    sd = float(np.std(x_train))
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    ztr = np.clip((x_train - mu) / sd, -6.0, 6.0)
    zev = np.clip((x_eval - mu) / sd, -6.0, 6.0)
    return ztr, zev, mu, sd


def _feature_matrices(x_train, x_eval):
    ztr, zev, mu, sd = _standardize_from_train(x_train, x_eval)
    hinges = np.quantile(ztr, [0.2, 0.4, 0.6, 0.8])

    def make(z):
        cols = [
            np.ones_like(z),
            z,
            z**2,
            z**3,
            np.abs(z),
            np.tanh(z),
        ]
        cols.extend(np.maximum(z - h, 0.0) for h in hinges)
        return np.column_stack(cols)

    return make(ztr), make(zev), {
        "x_mean": mu,
        "x_sd": sd,
        "hinges": hinges.tolist(),
    }


def _ridge_predict(X, y, Xe, lam):
    penalty = np.sqrt(float(lam)) * np.eye(X.shape[1])
    penalty[0, 0] = 1e-4
    Xa = np.vstack([X, penalty])
    ya = np.concatenate([np.asarray(y, dtype=float),
                         np.zeros(X.shape[1], dtype=float)])
    w = np.linalg.lstsq(Xa, ya, rcond=None)[0]
    with np.errstate(all="ignore"):
        pred = Xe @ w
    if not np.all(np.isfinite(pred)):
        return np.full(len(Xe), float(np.mean(y)))
    return pred


def _logistic_predict(X, y, Xe, lam):
    y = np.asarray(y, dtype=float)
    p0 = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
    w = np.zeros(X.shape[1], dtype=float)
    w[0] = np.log(p0 / (1.0 - p0))
    reg = float(lam) * np.eye(X.shape[1])
    reg[0, 0] = 1e-8
    for _ in range(40):
        with np.errstate(all="ignore"):
            eta = np.clip(X @ w, -25.0, 25.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        weight = np.clip(p * (1.0 - p), 1e-4, None)
        with np.errstate(all="ignore"):
            grad = X.T @ (p - y) + reg @ w
            hess = X.T @ (weight[:, None] * X) + reg
        if not np.all(np.isfinite(grad)) or not np.all(np.isfinite(hess)):
            return np.full(len(Xe), p0)
        step = np.linalg.pinv(hess) @ grad
        w -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    with np.errstate(all="ignore"):
        eta_eval = np.clip(Xe @ w, -25.0, 25.0)
    if not np.all(np.isfinite(eta_eval)):
        return np.full(len(Xe), p0)
    return 1.0 / (1.0 + np.exp(-eta_eval))


def _fit_and_losses(x_train, y_train, x_eval, y_eval, response, ridge):
    X, Xe, meta = _feature_matrices(x_train, x_eval)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    y_eval = np.asarray(y_eval, dtype=float).reshape(-1)
    if response["kind"] == "binary":
        threshold = float(response["threshold"])
        yt = (y_train > threshold).astype(float)
        ye = (y_eval > threshold).astype(float)
        pred = np.clip(_logistic_predict(X, yt, Xe, ridge), 1e-6, 1 - 1e-6)
        base_p = float(np.clip(yt.mean(), 1e-6, 1 - 1e-6))
        model_loss = -(ye * np.log(pred) + (1.0 - ye) * np.log(1.0 - pred))
        base_loss = -(ye * np.log(base_p)
                      + (1.0 - ye) * np.log(1.0 - base_p))
    else:
        ym = float(np.mean(y_train))
        ys = float(np.std(y_train))
        if not np.isfinite(ys) or ys < 1e-8:
            ys = 1.0
        yt = (y_train - ym) / ys
        ye = (y_eval - ym) / ys
        pred = _ridge_predict(X, yt, Xe, ridge)
        model_loss = (ye - pred) ** 2
        base_loss = ye**2
        meta.update({"y_mean": ym, "y_sd": ys})
    return np.asarray(base_loss), np.asarray(model_loss), meta


def _gain_record(base_loss, model_loss, response, extra=None):
    base_loss = np.asarray(base_loss, dtype=float)
    model_loss = np.asarray(model_loss, dtype=float)
    scale = float(np.mean(base_loss))
    if not np.isfinite(scale) or scale < 1e-10:
        scale = 1.0
    contrib = (base_loss - model_loss) / scale
    out = {
        "gain": float(np.mean(contrib)),
        "base_loss": float(np.mean(base_loss)),
        "model_loss": float(np.mean(model_loss)),
        "response_type": response["kind"],
        "binary_r2": float(response.get("binary_r2", 0.0)),
        "n": int(contrib.size),
        "_contrib": contrib,
    }
    if extra:
        out.update(extra)
    return out


def _crossfit_gain(x, y, folds, ridge, binary_r2, seed):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = len(x)
    response = _bimodal_spec(y, min_r2=binary_r2)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold_ids = np.empty(n, dtype=int)
    fold_ids[order] = np.arange(n) % max(2, min(int(folds), n // 20))
    base = np.zeros(n, dtype=float)
    model = np.zeros(n, dtype=float)
    for fold in np.unique(fold_ids):
        te = fold_ids == fold
        tr = ~te
        b, m, _ = _fit_and_losses(
            x[tr], y[tr], x[te], y[te], response, ridge)
        base[te], model[te] = b, m
    return _gain_record(base, model, response)


def _transfer_gain(x_train, y_train, x_eval, y_eval, ridge, binary_r2):
    x_train = np.asarray(x_train, dtype=float).reshape(-1)
    x_eval = np.asarray(x_eval, dtype=float).reshape(-1)
    response = _bimodal_spec(y_train, min_r2=binary_r2)
    base, model, _ = _fit_and_losses(
        x_train, y_train, x_eval, y_eval, response, ridge)
    lo, hi = np.quantile(x_train, [0.01, 0.99])
    overlap = float(np.mean((x_eval >= lo) & (x_eval <= hi)))
    return _gain_record(
        base, model, response,
        extra={"support_overlap": overlap},
    )


def _bootstrap_ci(contrib, confidence, n_boot, seed):
    contrib = np.asarray(contrib, dtype=float).reshape(-1)
    rng = np.random.default_rng(seed)
    if contrib.size == 0:
        return [float("nan"), float("nan")]
    draws = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        draws[b] = float(np.mean(
            contrib[rng.integers(0, contrib.size, contrib.size)]))
    q = (1.0 - float(confidence)) / 2.0
    return [float(np.quantile(draws, q)),
            float(np.quantile(draws, 1.0 - q))]


def _contrast_ci(left, right, confidence, n_boot, seed):
    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_boot), dtype=float)
    for k in range(int(n_boot)):
        am = float(np.mean(a[rng.integers(0, a.size, a.size)]))
        bm = float(np.mean(b[rng.integers(0, b.size, b.size)]))
        draws[k] = am - bm
    q = (1.0 - float(confidence)) / 2.0
    return [float(np.quantile(draws, q)),
            float(np.quantile(draws, 1.0 - q))]


def _variation_record(x_do, x_pool):
    x_do = _finite_1d(x_do)
    x_pool = _finite_1d(x_pool)
    if x_do.size == 0:
        return {
            "n": 0,
            "sd": 0.0,
            "sd_ratio": 0.0,
            "q05": 0.0,
            "q95": 0.0,
            "range90": 0.0,
        }
    pool_sd = float(np.std(x_pool))
    do_sd = float(np.std(x_do))
    ratio = do_sd / max(pool_sd, 1e-8)
    q05, q95 = np.quantile(x_do, [0.05, 0.95]) if x_do.size else (0.0, 0.0)
    return {
        "n": int(x_do.size),
        "sd": do_sd,
        "sd_ratio": float(ratio),
        "q05": float(q05),
        "q95": float(q95),
        "range90": float(q95 - q05),
    }


def _clean_gain(rec):
    out = {k: v for k, v in rec.items() if not k.startswith("_")}
    return out


def _decide_contrast(left, right, cfg, seed):
    n_boot = int(cfg["bootstrap"])
    confidence = float(cfg["confidence"])
    left_ci = _bootstrap_ci(
        left["_contrib"], confidence, n_boot, seed + 1)
    right_ci = _bootstrap_ci(
        right["_contrib"], confidence, n_boot, seed + 2)
    diff_ci = _contrast_ci(
        left["_contrib"], right["_contrib"],
        confidence, n_boot, seed + 3,
    )
    left["ci"] = left_ci
    right["ci"] = right_ci
    max_reverse = cfg.get("max_reverse_gain")
    max_reverse = np.inf if max_reverse is None else float(max_reverse)
    min_gain = float(cfg["min_gain"])
    min_contrast = float(cfg["min_contrast"])
    pos = (
        left_ci[0] > min_gain
        and diff_ci[0] > min_contrast
        and float(right["gain"]) < max_reverse
    )
    neg = (
        right_ci[0] > min_gain
        and diff_ci[1] < -min_contrast
        and float(left["gain"]) < max_reverse
    )
    orientation = 1 if pos and not neg else -1 if neg and not pos else 0
    return orientation, diff_ci


def _base_evidence(engine, i, j, rows_i, rows_j, cfg, var_i, var_j):
    return {
        "engine": engine,
        "edge": [int(i), int(j)],
        "orientation": 0,
        "orientation_source": "direct_statistical_test",
        "uses_meek": False,
        "n_do_i": int(len(rows_i)),
        "n_do_j": int(len(rows_j)),
        "target_variation_i": var_i,
        "target_variation_j": var_j,
        "config": deepcopy(cfg),
    }


def _check_rows_and_variation(out, cfg):
    if out["n_do_i"] < int(cfg["min_rows"]) \
            or out["n_do_j"] < int(cfg["min_rows"]):
        out["abstain_reason"] = "insufficient_two_sided_samples"
        return False
    if out["target_variation_i"]["sd_ratio"] < float(cfg["min_target_sd"]) \
            or out["target_variation_j"]["sd_ratio"] < float(cfg["min_target_sd"]):
        out["abstain_reason"] = "insufficient_intervention_variation"
        return False
    return True


def _cf_regression_evidence(Z, M, i, j, cfg):
    rows_i = _single_target_rows(M, i)
    rows_j = _single_target_rows(M, j)
    var_i = _variation_record(Z[rows_i, i], Z[:, i])
    var_j = _variation_record(Z[rows_j, j], Z[:, j])
    out = _base_evidence(
        "cf_regression", i, j, rows_i, rows_j, cfg, var_i, var_j)
    if not _check_rows_and_variation(out, cfg):
        return out
    salt = _seed(i, j, len(Z), 11)
    left = _crossfit_gain(
        Z[rows_i, i], Z[rows_i, j], cfg["folds"], cfg["ridge"],
        cfg["binary_r2"], salt + 1,
    )
    right = _crossfit_gain(
        Z[rows_j, j], Z[rows_j, i], cfg["folds"], cfg["ridge"],
        cfg["binary_r2"], salt + 2,
    )
    orientation, contrast_ci = _decide_contrast(left, right, cfg, salt + 10)
    out.update({
        "orientation": int(orientation),
        "score_i_to_j": _clean_gain(left),
        "score_j_to_i": _clean_gain(right),
        "contrast_i_minus_j": float(left["gain"] - right["gain"]),
        "contrast_ci": contrast_ci,
    })
    if orientation == 0:
        out["abstain_reason"] = "bootstrap_or_effect_ambiguous"
    return out


def _invariant_evidence(Z, M, i, j, cfg):
    rows_i = _single_target_rows(M, i)
    rows_j = _single_target_rows(M, j)
    obs = _observational_rows(M)
    var_i = _variation_record(Z[rows_i, i], Z[:, i])
    var_j = _variation_record(Z[rows_j, j], Z[:, j])
    out = _base_evidence(
        "invariant", i, j, rows_i, rows_j, cfg, var_i, var_j)
    out["n_observational"] = int(len(obs))
    if len(obs) < int(cfg["min_obs_rows"]):
        out["abstain_reason"] = "insufficient_observational_samples"
        return out
    if not _check_rows_and_variation(out, cfg):
        return out
    left = _transfer_gain(
        Z[obs, i], Z[obs, j], Z[rows_i, i], Z[rows_i, j],
        cfg["ridge"], cfg["binary_r2"],
    )
    right = _transfer_gain(
        Z[obs, j], Z[obs, i], Z[rows_j, j], Z[rows_j, i],
        cfg["ridge"], cfg["binary_r2"],
    )
    if min(left["support_overlap"], right["support_overlap"]) \
            < float(cfg["min_support_overlap"]):
        out.update({
            "score_i_to_j": _clean_gain(left),
            "score_j_to_i": _clean_gain(right),
            "abstain_reason": "insufficient_observational_support_overlap",
        })
        return out
    salt = _seed(i, j, len(Z), 23)
    orientation, contrast_ci = _decide_contrast(left, right, cfg, salt)
    out.update({
        "orientation": int(orientation),
        "score_i_to_j": _clean_gain(left),
        "score_j_to_i": _clean_gain(right),
        "contrast_i_minus_j": float(left["gain"] - right["gain"]),
        "contrast_ci": contrast_ci,
    })
    if orientation == 0:
        out["abstain_reason"] = "bootstrap_or_effect_ambiguous"
    return out


def _kernel_1d(x, binary_r2):
    x = np.asarray(x, dtype=float).reshape(-1)
    spec = _bimodal_spec(x, min_r2=binary_r2)
    if spec["kind"] == "binary":
        labels = x > float(spec["threshold"])
        return (labels[:, None] == labels[None, :]).astype(float), spec
    z = (x - x.mean()) / (x.std() + 1e-8)
    probe = z if len(z) <= 240 else z[np.linspace(
        0, len(z) - 1, 240, dtype=int)]
    dist = np.abs(probe[:, None] - probe[None, :])
    positive = dist[dist > 1e-8]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    K = np.exp(-0.5 * ((z[:, None] - z[None, :]) / max(
        bandwidth, 1e-4)) ** 2)
    spec = dict(spec)
    spec["bandwidth"] = bandwidth
    return K, spec


def _center_kernel(K):
    return K - K.mean(0, keepdims=True) - K.mean(1, keepdims=True) + K.mean()


def _normalized_hsic(K, L):
    Kc, Lc = _center_kernel(K), _center_kernel(L)
    denom = float(np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc)))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(Kc * Lc) / denom)


def _hsic_record(x, y, cfg, seed):
    K, x_type = _kernel_1d(x, cfg["binary_r2"])
    L, y_type = _kernel_1d(y, cfg["binary_r2"])
    point = _normalized_hsic(K, L)
    n = len(K)
    bsize = min(n, int(cfg["bootstrap_rows"]))
    rng = np.random.default_rng(seed)
    draws = np.empty(int(cfg["bootstrap"]), dtype=float)
    for b in range(len(draws)):
        idx = rng.integers(0, n, bsize)
        draws[b] = _normalized_hsic(K[np.ix_(idx, idx)],
                                    L[np.ix_(idx, idx)])
    q = (1.0 - float(cfg["confidence"])) / 2.0
    ci = [float(np.quantile(draws, q)),
          float(np.quantile(draws, 1.0 - q))]
    return {
        "gain": point,
        "ci": ci,
        "n": int(n),
        "input_type": x_type["kind"],
        "response_type": y_type["kind"],
        "input_binary_r2": float(x_type.get("binary_r2", 0.0)),
        "response_binary_r2": float(y_type.get("binary_r2", 0.0)),
        "_draws": draws,
    }


def _hsic_evidence(Z, M, i, j, cfg):
    rows_i = _single_target_rows(M, i)
    rows_j = _single_target_rows(M, j)
    var_i = _variation_record(Z[rows_i, i], Z[:, i])
    var_j = _variation_record(Z[rows_j, j], Z[:, j])
    out = _base_evidence("hsic", i, j, rows_i, rows_j, cfg, var_i, var_j)
    if not _check_rows_and_variation(out, cfg):
        return out
    salt = _seed(i, j, len(Z), 37)
    left = _hsic_record(Z[rows_i, i], Z[rows_i, j], cfg, salt + 1)
    right = _hsic_record(Z[rows_j, j], Z[rows_j, i], cfg, salt + 2)
    diff = left["_draws"] - right["_draws"]
    q = (1.0 - float(cfg["confidence"])) / 2.0
    diff_ci = [float(np.quantile(diff, q)),
               float(np.quantile(diff, 1.0 - q))]
    max_reverse = cfg.get("max_reverse_gain")
    max_reverse = np.inf if max_reverse is None else float(max_reverse)
    pos = (
        left["ci"][0] > float(cfg["min_gain"])
        and diff_ci[0] > float(cfg["min_contrast"])
        and right["gain"] < max_reverse
    )
    neg = (
        right["ci"][0] > float(cfg["min_gain"])
        and diff_ci[1] < -float(cfg["min_contrast"])
        and left["gain"] < max_reverse
    )
    orientation = 1 if pos and not neg else -1 if neg and not pos else 0
    out.update({
        "orientation": int(orientation),
        "score_i_to_j": {k: v for k, v in left.items()
                         if not k.startswith("_")},
        "score_j_to_i": {k: v for k, v in right.items()
                         if not k.startswith("_")},
        "contrast_i_minus_j": float(left["gain"] - right["gain"]),
        "contrast_ci": diff_ci,
    })
    if orientation == 0:
        out["abstain_reason"] = "bootstrap_or_effect_ambiguous"
    return out


def orientation_evidence(Z, M, i, j, engine, config=None):
    """Return orientation and full evidence for one known skeleton edge.

    ``orientation`` is +1 for ``i -> j``, -1 for ``j -> i``, and 0 for a
    conservative abstention.
    """
    Z = np.asarray(Z, dtype=float)
    M = np.asarray(M, dtype=float)
    if Z.ndim != 2 or M.shape != Z.shape:
        raise ValueError("Z and M must be same-shaped two-dimensional arrays")
    if i == j or not (0 <= int(i) < Z.shape[1]) \
            or not (0 <= int(j) < Z.shape[1]):
        raise ValueError("i and j must be distinct valid column indices")
    if not np.all(np.isfinite(Z)):
        raise ValueError("Z contains non-finite values")
    cfg = resolve_config(engine, config)
    if engine == "cf_regression":
        return _cf_regression_evidence(Z, M, int(i), int(j), cfg)
    if engine == "invariant":
        return _invariant_evidence(Z, M, int(i), int(j), cfg)
    if engine == "hsic":
        return _hsic_evidence(Z, M, int(i), int(j), cfg)

    members = list(cfg["members"])
    records = {}
    votes = []
    member_cfg = {} if config is None else config.get("member_configs", {})
    for member in members:
        if member == "consensus":
            raise ValueError("consensus cannot recursively contain itself")
        rec = orientation_evidence(
            Z, M, i, j, member, config=member_cfg.get(member))
        records[member] = rec
        votes.append(int(rec["orientation"]))
    nonzero = [v for v in votes if v != 0]
    signs = set(nonzero)
    orientation = 0
    if len(nonzero) >= int(cfg["min_agree"]):
        if len(signs) == 1:
            orientation = nonzero[0]
        elif not bool(cfg.get("veto_conflict", True)):
            orientation = 1 if sum(nonzero) > 0 else -1 if sum(nonzero) < 0 else 0
    return {
        "engine": "consensus",
        "edge": [int(i), int(j)],
        "orientation": int(orientation),
        "orientation_source": "direct_statistical_test_consensus",
        "uses_meek": False,
        "votes": {member: int(records[member]["orientation"])
                  for member in members},
        "members": records,
        "config": cfg,
        "abstain_reason": None if orientation else (
            "member_conflict" if len(signs) > 1 else "insufficient_member_agreement"
        ),
    }
