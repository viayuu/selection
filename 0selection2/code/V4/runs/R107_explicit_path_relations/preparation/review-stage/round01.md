# R107 Focused Prelaunch Review, Round 01

2026-10-24 Asia/Shanghai. One compact engineering round; no Astra or secondary
reviewer invoked. No source edits, training, GPU commands, Git, or new literature.
Only the three review-stage files are reviewer-owned. No Git diff was requested
from the environment: current sources were compared with their direct R105/R84
implementation dependencies instead, respecting the explicit no-Git restriction.

## Findings By Priority

**Blocking P0/P1: none found in the reviewed implementation.** No demonstrated
high-impact indexing, label-input, gradient, padding, checkpoint restore, or
TEST/LIB selection-lock defect. This is a bounded review, not a proof of absence.

**P3, nonblocking nomenclature warning:** `r107_model.py:41` names a field
`depot_walk_billed_travel`, but `r107_model.py:202-205` computes a physical
depot-anchored local walk, with O omitting only the appended final return.
For example, an O path customer->depot->customer still includes its physical
customer-to-depot edge. Do not present this as a solver-certified billed length,
free-depot-edge rule, or native feasibility test. `MODEL_PREP.md:39-40` already
limits its meaning. Prefer an explicit physical-walk description in reporting;
no numerical or experimental-semantic change is requested.

The initial 128/512 cap concern is superseded, not an outstanding blocker:
current `r107_experiment.py:49-59` and `r107_launch.py:28` use the main-selected
common **256/1024, eval128**, with logical1024 unchanged. Do not silently choose
other caps per arm or at runtime. Final admin/source synchronization and the
reported mock plotting-shape repair remain main-owned work in progress, not
new demonstrated model defects or reasons to open an additional audit round.

## Focused Evidence

- Reviewer CPU run: 43 tests PASS across model, experiment, launch and reporting.
  This run preceded final cap/admin edits; it is not a final frozen-source test
  receipt. Main subsequently reports model8 PASS and runner4 PASS.
- Independent CPU spot checks: all 100 valid directed paths matched scalar
  matrix lookups across distinct batch matrices. `r107_model.py:195-197`
  explicitly expands source indices to the target shape; no broadcast defect.
- Six arm/problem cases (TSP/CVRP/ATSP) passed added-width padding and mixed-row
  versus separately cropped inference checks, with unchanged input tensors.
  This checks LIB-style padding, not real LIB GPU capacity or efficacy.
- Original/new gradients, complete shared fresh tensors, inactive A columns,
  label perturbation isolation, and strict restore are covered by the model
  tests. A's 4,096 inactive composed weights and nonmatched functional freedom
  are explicit in `r107_model.py:66-78,392-395` and `MODEL_PREP.md:50-60`.
- TW uses depot-time0 local-walk arithmetic, waiting/service, no depot zero
  deadline and no implicit reset (`r107_model.py:118-123,184-200`). Signed demand
  extrema are not physical onboard-load or classical-B feasibility assertions.
- Original two-forward CE/comparison/risk/RDrop and row-weighted logical1024
  accumulation are retained (`r76_experiment.py:164-181`,
  `r84_experiment.py:97-117`). VAL selects the whole model; strict factory,
  capture binding and TEST authorization retain the unique selected state
  (`r107_experiment.py:209-267,298-317`). No TEST/LIB tuning path was found.
- Read, did not run, main's `kernel_witness/B_larger256_1024/complete.json`:
  B largest-TRAIN two-dropout backward PASS, zero optimizer steps; maxima
  allocated 18,856.46 MiB and reserved 21,618 MiB. Model SHA matches this review.
  Thus capacity is no longer wholly unmeasured. This is a synthetic-gradient
  witness for four TRAIN shapes, not full-objective/all18/LIB capacity proof.

## Overall Judgment

**Provisional engineering readiness: 8/10; no blocking source fix requested.**
The fixed matched pair is reasonable to launch after main completes its already
ongoing admin/report synchronization and final freeze. This review does not set
formal ready or certify the unfinished final snapshot. No formal training was
started by this reviewer, and R106/resources were not touched.

The five-task TRAIN-only descriptive ridge bridge remains negative. This pair
is a falsification experiment for learned connected composition, not a certified
green mechanism gate. Negative summary proxies do not prove learned paths must
fail; no coefficient/feature search or efficacy score is recommended. Round 01
is complete; no further review round absent an actual critical defect.
