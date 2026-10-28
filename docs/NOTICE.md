# Third-party software and licensing

The public entry points depend on NumPy, SciPy, Pillow and PyBullet, installed
from their upstream distributions. Their respective licenses continue to apply.
The `pybullet_data` Panda model is loaded at runtime; no robot mesh or URDF is
vendored here. Video encoding, if added by a user, requires a separate FFmpeg
installation and its own applicable license.

The extracted graph backend and physics adapters are project source snapshots;
their source hashes and any export changes are recorded in `provenance/`.
Some backend modules also contain historical research functions unused by these
entry points, including oracle/test policies. The public refit command does not
select those policies or load a ground-truth graph. Their presence does not mean
they were used by the released runs.

The authors have authorized public distribution of this demo subset. No broad
software license or rights to the private model checkpoints are implied. Ask
the authors about reuse licensing; third-party rights remain with their owners.
