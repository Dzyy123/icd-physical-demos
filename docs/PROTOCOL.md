# Protocol, costs, and development history

## Learning vs display

The encoder was trained with state labels anchoring known node identities.
Graph learning consumes its frozen, calibrated image readouts and intervention
identities. Simulator state labels and graph references are separate evaluation
objects. In the published fork film, green/red means agreement/disagreement with
the reference **after the run**, not model-known correctness or confidence.
Candidate-pool weights are relative scores, not calibrated posterior confidence.

The chain uses controlled clean contexts for intervention pairs, whereas its
observations randomize backgrounds and root activation. Its orientation evidence
is conditional on those controlled setups, not an unrestricted population effect.
The revisable fork uses fresh observations and interventions from the same
context generator. A pair shares context and is not independent replication.

## Chain

The visual bank contains 392 training, 56 validation and 112 readout-test images.
The encoder uses seed 42 and a fixed 100-epoch final checkpoint. Calibration uses
280 additional images. PC starts from 512 observations. Coverage selection chooses
F1 and then P, each with 64 images. The first batch is reused only after checking
its commands, contexts, seeds and controller match; the endpoint-variation guard
changes admissibility of constant-readout evidence, not the collected outcomes.

Four representative paired replay endpoints were internally checked against
acquired RGB. The bell-hold segment is a separate physical check with the graph
frozen. The film camera is wider than the fixed 128×128 inference camera.

Earlier pusher versions could travel beyond the visible rail. The physical fix
uses rail end stops and a U-shaped carriage, not post-release pose resets. A
56-call check passed travel/contact checks but had three bell-hold settling
failures; those are retained in the development archive. This is development
evidence, not a collection of independent successful benchmark seeds.

## Fork: initial run and readout repair

Initial controller pilots exposed invalid F1 holds. Alternative grasp corrections
were tested on fixed diagnostic cases; failed versions were retained. A complete
12-round attempt then exposed P readout errors in new grasp/background contexts.
The P coordinate alone was corrected using 64 newly acquired calibration images
with train/validation separation and an unchanged 0.5 threshold. The encoder and
other six coordinates stayed fixed. New graph-data seeds were set before the
next acquisition. This is supervised interface development, not autonomous
discovery of new variable identities.

The accepted initial run uses 24 observations and 12 batches of four pairs
(96 intervention images). Its reference SHD is non-monotonic and ends at 1.

## Fork: continuation after inspecting the result

After the remaining error was observed, eight more batches were requested.
Thus 20 rounds were **not** the original pre-registered budget. In the first
continuation, round 16 stopped before graph update because a held F1 reached
10.222635 degrees, exceeding the unchanged 10-degree criterion.

Two fixed-priority offsets were checked on the failed pair and a clean pair
(8 calls). A 5 mm inward correction was selected by physical checks, not graph
accuracy. The entire continuation restarted from round 12 with the same declared
rounds 13–20 and seed schedule under the repaired controller. Encoder, ACC,
graph score and acquisition policy were unchanged. The failed 32-image attempt
was retained, not selectively pooled into the successful run.

In the accepted trace, round 14 replaces F3→F2 with another false edge F1→F2;
round 18 removes it, and rounds 19–20 remain unchanged. Accepted learning data:
24 observation + 160 intervention images. Additional continuation acquisition
cost: 64 accepted + 32 failed-attempt + 8 repair-check calls = 104, excluding
display-only replays and earlier development/calibration costs.

## Editing of the 65-second film

All structural revision rounds are retained: 1, 2, 3, 4, 5, 6, 14 and 18.
Repeated/unchanged trials are condensed with notices. Each displayed pair was
selected from setup information before outcomes (clean context preferred), while
graph updates use the complete four-pair batch. Published videos are available
from the project website; this repository's report command exposes all states.

This is one exploratory trajectory, not a convergence guarantee, an OOD test,
or a fair matched-baseline comparison. Claiming any of those would require
additional experiments with fixed budgets and common evaluation permissions.
