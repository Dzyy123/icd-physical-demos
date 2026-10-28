# Intervene, Calibrate, Discover — physical demos

Code and recorded evidence for two robot-contact simulations:

1. **Seven-node chain:** pusher → four dominoes → bell → lamp.
2. **Revisable fork:** two physical branches, repeated interventions, and withdrawal of incorrect graph hypotheses.

[Project website](https://dzyy123.github.io/intervene-calibrate-discover/) ·
[English paper](https://dzyy123.github.io/intervene-calibrate-discover/assets/paper.pdf)

This is a standalone **demo-code release**, not the full research repository.
Model checkpoints, learned calibration coefficients and training-image banks
remain private. The commands below explicitly distinguish recorded playback,
graph recomputation from saved visual readouts, and fresh physics simulation.

## Install

Python 3.10 or 3.11 on Linux is recommended. The historical simulator used
PyBullet 3.2.7. No GPU is needed for the public commands.

```sh
git clone https://github.com/Dzyy123/icd-physical-demos.git
cd icd-physical-demos
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py verify
```

## 1. Inspect every graph update (no model or simulator required)

```sh
python demo.py report --scene chain --out outputs/chain.html
python demo.py report --scene fork --out outputs/fork.html
```

Open the HTML files in a browser. These display every saved graph state, not
a new learning run. Arrows are hypotheses; they are not colored using truth.
The published films include the physical scene and captions:

```sh
python demo.py film --scene chain --out outputs/chain.mp4
python demo.py film --scene fork --out outputs/fork.mp4
```

Only the already-public videos are downloaded; checksums are checked.

## 2. Recompute the graph-learning stages

```sh
python demo.py refit --scene chain --out outputs/chain-refit.json
python demo.py refit --scene fork --out outputs/fork-refit.json
```

Inputs are the **recorded binary outputs of the frozen visual encoder and
calibration**, not simulator labels. Chain recomputation checks the observational
graph, selected targets, and both graph updates. Fork recomputation checks all
21 states (initial + 20 rounds), scores, retained candidate pools and next-target
choices. It fails if these disagree with the archived run.

This verifies graph computation on the acquired data. It does **not** regenerate
RGB, rerun encoder inference, constitute an independent replicate, or measure
generalization. Mechanism reference graphs are not inputs to this command.

## 3. Replay a recorded physical operation

Use a new output directory for each invocation:

```sh
python demo.py simulate --scene chain --round 1 --row 0 --out outputs/chain-hold
python demo.py simulate --scene fork --round 18 --row 0 --out outputs/fork-hold
```

The job's original target, value, seed and nuisance context are loaded from its
launch record. The robot performs the operation using PyBullet contact dynamics.
Outputs include endpoint RGB, captured frames, physical measurements and a check
against the recorded 128×128 model-input image hash. Pixel identity can depend
on platform/library versions; physical checks and exact-hash results are reported
separately. No hidden simulator labels enter a graph learner in this command:
it is a **physics replay only**.

For the fork, rounds 1–12 use the original frozen controller; rounds 13–20 use
the disclosed 5 mm F1-hold correction. Row indices refer to complete batch jobs,
not only the pair chosen for the short film. High-resolution contact simulation
can take minutes per operation on CPU.

## What the two demonstrations establish

| | Chain | Revisable fork |
|---|---|---|
| Variables | Seven supervised, node-aligned binary event states | Same seven node identities; different branch geometry |
| Observation images | 512 | 24 |
| Graph-stage data | 2 batches × 64 images | 20 batches × 8 images |
| Graph backend | PC-type skeleton, direct-do orientation, Meek rules, endpoint-variation guard | Truncated interventional BDeu-style score, exact top-32 DAG search, predictive-JS targeting |
| Calibration | 280 separate command-affine calibration images | Reused 280-image calibration + 64-image P-coordinate repair |
| Budget qualification | Representative paired replays summarize full batches | 12 initial rounds, then 8 additional rounds requested after inspecting the remaining error |
| Reported result | Six directed links match the design-level event reference | Final six links match at rounds 18–20; intermediate errors and reversals retained |

The fork backend is an **exploratory extension**, not the paper's original PC/Meek
algorithm. Neither film establishes a baseline advantage, OOD robustness, or a
general calibration benefit. They are not unsupervised discovery of unknown
node identities. The lamp is an explicit contact-sensor relay, not simulated
acoustics. See [protocol and development history](docs/PROTOCOL.md).

## Records and reproducibility boundary

- `records/chain/`, `records/fork/`: all accepted graph states, decisions, launch
  settings, readout arrays and per-round summaries.
- `physical-audits.jsonl.gz`: all saved per-row physical audit records of the
  released graph acquisitions, including independent trial identifiers.
- `records/development-history.zip`: available JSON/log records for listed prior
  development attempts, failed continuation, and repair checks. The exact list
  and exclusions are in `records/development-index.json`.
- `provenance/`: original source/record hashes and export mapping. Private
  filesystem prefixes are replaced by symbolic `private-runs/` references;
  they are identifiers, not downloadable paths.
- `archive/`: original pixel-training, collection and inference orchestration
  scripts for inspection. These are **not** the portable entry points and require
  private model assets and the original environment. Public commands are in
  `demo.py`; no unavailable checkpoint is downloaded automatically.
- `docs/VALIDATION.json`: release smoke/recomputation checks, separate from
  experimental integrity certification.

Raw image banks and checkpoints are not distributed. Consequently this release
does not claim a weight-free rerun of the complete pixels → ACC → graph pipeline.
It provides executable physical demos and reproducible graph updates on their
actual saved visual readouts. Complete records here means the declared acquisition
and graph histories, not all large binary artifacts from the research project.

No independent cross-model integrity audit is claimed for these demonstrations.

## Attribution and terms

Paper: *Intervene, Calibrate, Discover: Learning Causal Structure from Pixels*,
Ziyi Ding, Xiaodi Huang, Xianxin Lai, Haijier Chen, Zhang Zhang, Xiao-Ping Zhang,
and Jiayu Chen.

Python dependencies retain their upstream licenses. Panda robot assets are
loaded from the separately installed `pybullet_data` distribution, not copied
into this repository. No new blanket license is granted by this source release;
contact the authors for reuse permissions. See [third-party notice](docs/NOTICE.md).
