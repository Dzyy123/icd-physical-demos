# Export and portable adapters

`provenance/source-manifest.json` records original snapshots and the first
sanitized export. `MANIFEST.json` identifies the final public files. Original
private path prefixes are replaced by symbolic identifiers, not live server
addresses. No credentials or model parameter files are part of the release.

Portable additions:

- `demo.py` supplies explicit physics, saved-readout refit, trace HTML and public
  film-download commands. No model artifact is silently substituted.
- `src/fork/robot_repaired.py` ports the archived 5 mm F1-hold correction to a
  relative import. The original repair source remains in `archive/`.
- `src/chain/backend/acq_closed.py` defers three synthetic-SCM benchmark imports.
  This avoids importing PyTorch for the saved-readout graph command; observational
  tests, targeting, thresholds and orientation rules are unchanged. The historical
  synthetic benchmark utilities are not public demo entry points.

`validate_remote.py` is an internal release smoke-test harness. It requires a
Linux CPU environment and does not train models, modify earlier runs, or access
private checkpoints. The completed test record is `docs/VALIDATION.json`.
