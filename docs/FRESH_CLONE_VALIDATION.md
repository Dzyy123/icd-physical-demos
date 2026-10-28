# Fresh-clone README acceptance — 2026-10-28

**PASS:** all 13 acceptance steps below exited successfully on a new anonymous
clone of code commit `eb52515172f18301d6962e2271031ffa77968e0a`. This report is
published in a later documentation-only commit; it does not claim the report
itself existed in the tested code commit.

The run took place on a Linux x86_64 CPU host, from 04:14:42 to 04:18:54 UTC.
No GPU, private research checkout, checkpoint, calibration coefficients, or
training-image bank was used. This is a portability/README acceptance check,
not an independent experiment-integrity audit.

## Clean environment

Python 3.10.12; NumPy 2.2.6; SciPy 1.15.3; Pillow 11.3.0; PyBullet 3.2.7;
pip 26.2.2. The new venv had both system-site and user-site packages disabled.
Git's global configuration and credential helper were disabled. Pip configuration
and cache were disabled, with downloads from public PyPI only. `PYTHONPATH`
was empty, `PYTHONNOUSERSITE=1`, and `CUDA_VISIBLE_DEVICES` was empty.

The official PyPA bootstrap was fetched over HTTPS. Its URL and SHA-256,
full dependency versions, commands, exit codes and replay results are recorded
in [the machine-readable report](FRESH_CLONE_VALIDATION.json).

## Failures found and fixed

1. **Original README installation failed.** The initial anonymous clone of
   `75f7415b8094b79fe1561d02f857e83b46a3a770` succeeded (exit 0), but
   `python3 -m venv .venv` exited 1 because this Python distribution omits
   `ensurepip`. The retained error starts: “The virtual environment was not
   created successfully because ensurepip is not available.”
   The new `setup_env.py` creates an isolated venv without relying on ensurepip,
   bootstraps pip inside it, installs the pinned requirements, and runs pip check.
   It requires neither sudo nor an existing research environment.

2. **The first bootstrap draft failed on that partially created venv.**
   Its default interpreter-copy mode encountered `shutil.SameFileError` because
   the existing venv interpreter was already a symlink to the system Python.
   The draft's numeric exit code was not separately captured; its traceback is
   retained and is not counted as a passed test. The published fix explicitly
   uses interpreter symlinks on POSIX. Recovery in the partial venv then exited 0.
   Crucially, the acceptance run below did not reuse that recovered environment:
   it cloned the public fix into a different new directory and installed again.

The original failure logs and the recovery logs were retained. Public reporting
omits private host paths and account identifiers.

## README results

All commands used the new venv. The environment row is an additional isolation
assertion, and pip-check is the final dependency consistency check.

| Step | Exit code | Seconds |
|---|---:|---:|
| clone | 0 | 7.325 |
| setup | 0 | 42.030 |
| environment | 0 | 0.065 |
| verify | 0 | 0.115 |
| report-chain | 0 | 0.064 |
| report-fork | 0 | 0.064 |
| film-chain | 0 | 2.819 |
| film-fork | 0 | 2.018 |
| refit-chain | 0 | 0.515 |
| refit-fork | 0 | 2.618 |
| simulate-chain | 0 | 98.073 |
| simulate-fork | 0 | 95.917 |
| pip-check | 0 | 0.215 |

The manifest verified 144 public files. Both HTML reports were produced. Both
public film downloads passed their archived SHA-256 checks. Chain refit matched
the initial graph, selected targets and both updates; fork refit matched all
21 states, scores, retained candidate pools and next-target choices.

Both requested physics operations realized their target, settled, had no
post-release pose resets, and passed the minimum-contact-distance assertion.
The 128×128 model-input pixel byte hashes were also identical in this environment:

| Replay | Input-pixel SHA-256 | Exact match |
|---|---|---|
| Chain, round 1, row 0 | `ef0058ca0955bfd72d889584d0569c62f7d71fb0b1a062dce7035eb1cbb33cfb` | yes |
| Fork, round 18, row 0 | `536246fa8ae2462b8a7c967e917e2bb8f5907b53f4045ae811f217abf057af2e` | yes |

This is an exact-pixel result for these two recorded jobs on this environment,
not a claim that all jobs or all platforms are byte-identical.

## Retained non-blocking warning

Normal chain refit exits 0 but NumPy reports `invalid value encountered in
divide` when legacy `_pcorr` calls `corrcoef` on a constant intervention
endpoint (`src/chain/backend/acq_closed.py`, line 329). An additional
`-W error::RuntimeWarning` diagnostic intentionally exits 1 at that call.
That diagnostic is recorded separately from the successful README acceptance.

In normal execution, legacy `_pcorr` converts the undefined unconditioned
correlation to zero; the already-present `variation_guard` then rejects the
constant endpoint's do-score as inadmissible (`None`). The guard is applied
before the final orientation is used. All archived graph and target assertions
still pass. No historical algorithm code was changed and the warning was not
silently suppressed.

## Reproduction boundary

The public release supports **physics replay plus graph refit from saved visual
readouts**. It does not rerun the frozen visual encoder or ACC inference:
complete pixels → ACC → graph reproduction still requires private weights and
calibration assets. Fresh physical replay and exact endpoint pixels do not
establish an independent graph-learning replicate or generalization performance.

