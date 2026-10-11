# R60: expanded real partial supervision

Date: 2026-10-11. Performance round 2/100. Status: ready PLAN ONLY;
same-family/provisional. Implementation and execution belong to Sol. This plan
does not authorize this planner to run training, tests, solvers or commits.

## Locked names and implementation ownership

- ROOT: `code/V4/runs/R60_expanded_partial_supervision` from the repository root,
  or `runs/R60_expanded_partial_supervision` relative to `code/V4`.
- A run directory: `A_replay_seed2`; B run directory: `B_expanded_seed2`.
- Volta owns `code/V4/r60_data.py`: unchanged original TRAIN/validation data,
  paired auxiliary inputs and actual-size schedule, registry pool identities,
  FP64-derived pair targets/tie masks, and shared original-TRAIN geometry stats.
- Pascal owns `code/V4/r60_experiment.py`: matched scratch initialization,
  the combined two-plus-one-forward update, A-controlled common LR/horizon,
  independent validation-cost checkpoint selection, resume and result output.
- Auxiliary labels/identities stay outside forward inputs; the data/runner
  handoff preserves the full legal pool, not a two-candidate model input.
- This design has no identified blocking issue. Implementation may proceed;
  GPU execution still requires the parent's separate authorization. No new
  review round, old implementation verification or CPU replay is a prerequisite.

## R59 decision and mechanism

R59 results are published in commit `53656d519`. R59 is finished, not an interim
negative: A's validation-cost-selected epoch 32
has native Top1 48.5277778% and actual regret 0.9470817925%; B's selected epoch 16
has 48.0333333% and 0.9623450753%. Both are below the R45A reference of
49.1166667% / 0.9260483583%. B failed the precommitted +3pp versus R45A, +2pp
versus A and regret guards. No test was accessed. This rejects this transfer
recipe, not pretrained representations in general.

R60 tests whether substantially more independent, observed solver comparisons
improve instance-level winner discrimination in the existing full-pool model.
Unlike soft self-training, the new targets are actual two-method costs, not
predictions of a roughly 49%-accurate selector. Unlike R53, this is not a local
binary specialist or a top-two gate on the original 10k examples.

## Fixed scope and available data

- LEGACY deployment scenario only. Preserve all original 18 tasks, 22 global
  solver identities, 128 legal deployments, original 10k TRAIN/task and original
  validation data/pools. R58 remains deferred; B-domain mismatch remains open.
- Existing expansion root: `/public/home/shiys/训练数据_每类10万`.
  `标签配方与校准记录.csv` and `标签运行进度.csv` record all 30 RELD_MOEL/RELD_MTL
  columns across the 15 MVRP tasks as complete: 90k additions/column, 90k route
  checks/column, no anomalous shard metadata or pending/abnormal replay records.
  Thus 1.35 million additional paired TRAIN observations are recorded available,
  irrespective of completion of other methods' full-pool tables.
- Recorded recipes: corresponding native ReLD epoch-5000 MOEL/MTL weights,
  all N customer starts, augmentation 1, sample size 1, greedy, FP32. All 30
  columns have stratified historical replay agreement and matching recorded
  calibration/current recipes. Agreement uses 4 or 8 historical examples/column,
  not full historical certification; some numerical differences are nonzero.
  Existing source-provenance limitations remain disclosed, as described in
  `标签配方与校准说明.md`. Do not present this as a repaired common B benchmark.
- Use only these two existing expanded cost columns. Exclude the old 10k prefix
  from B's extra examples. Join each cost pair to its existing input identity;
  use actual input node counts/masks, not index-derived size or other metadata.
  Do not use the other 89 historically mismatching deployments, inferred full
  winners, synthetic costs, new labels, teacher labels or held-out answers.
- Expanded data have an existing data-only no-overlap report predating the
  authorized VRPLTW repair. Preserve that scope/repair disclosure; do not claim
  a fresh no-overlap audit or start one for this round. Use the existing repaired
  input and corresponding recorded pair costs together.

## Two arms, one unchanged architecture

A = oldTRAIN pair replay; B = expanded real pair supervision.

Both start FROM SCRATCH with identical seed-2 R45A handcrafted-solver
`ScoreDifferenceSelector` parameters and geometry statistics fitted ONLY on
original TRAIN. Reuse `r45_experiment.parameters('A')` and its initialization
path, not a trained R45 checkpoint. All parameters are trainable. No native
R59 encoder, transferred weights, code embeddings, extra expert, feature changes
or geometric augmentation. Keep original raw/local geometry and task fields.

Every original and auxiliary forward receives the task's FULL original legal
solver pool and original input features. Only the auxiliary loss observes two
solver columns. Inference remains the original full-pool maximin score-difference
ranking with existing global-ID tie breaking and one selected solver/instance.
No costs, targets, input identities or routes enter a model forward.

## Locked objective and single-update semantics

For each original batch, retain the EXACT R43 two-dropout-forward objective:
`single_objective = .35*CE + .35*comparison + .02*risk`, with the existing
true3/predicted3/all-pairs comparison weights, risk scale/clamp and symmetric
R-Drop. Preserve its original three-epoch consistency ramp to weight .35.

On each MVRP original step only, add ONE dropout forward for an auxiliary batch
of the same sample count. Extract `pair_margin[:, MOEL, MTL] = u_MOEL-u_MTL`
using registry solver identities and each batch's actual pool ordering.
Define target `y = 1[c_MOEL < c_MTL]` from original FP64 costs, before any cast;
exact FP64 ties are excluded from the auxiliary loss. Average
`BCEWithLogits(pair_margin, y)` over valid non-tied rows; an all-tied batch has
zero auxiliary contribution. Do not add CE, risk, oracle-normalized targets,
full-pool ranking targets, top-three focus, R-Drop or a winner for extra rows.

The step loss is the unchanged original two-forward loss PLUS
`.35 * auxiliary_pair_BCE`; .35 is fixed from epoch 1, with no coefficient scan.
TSP/CVRP/ATSP have no auxiliary forward or loss. Combine losses before ONE
optimizer update, with the existing AMP behavior, unscale and clip norm 1.
An AMP retry repeats the same original/auxiliary batches without advancing
either cursor; count only successful optimizer updates. Never call a helper
that already steps the optimizer and then perform a second auxiliary update.

## Matched data exposure and training schedule

- Seed 2; batch 128 INCLUDING the 16-example tails; no dropped rows. Use the
  identical original `paired_schedule` for A/B, covering all 10k/task/epoch.
- B: per MVRP task, deterministically shuffle its 90k new rows, consume 10k per
  original epoch, and reshuffle after each nine-epoch traversal. Auxiliary
  batch counts follow original step counts, including tails. Every new example
  is therefore covered within nine original epochs; do not run nine times as
  many full-pool updates or wait for active label queues.
- Construct one B auxiliary plan from the actual inputs. For each planned row,
  A independently samples a labeled oldTRAIN row from the SAME task and actual
  customer-count bucket, with replacement. Match the per-step real-size vector,
  batch/tail sizes, padding and full legal pool across arms. This is a size-
  matched replay control, not cost/rank-balanced sampling or feature matching.
  B otherwise keeps the natural expanded input distribution. Exclude exact
  cost ties only from loss, not by redrawing examples or changing schedules.
- Identical scratch initialization, original batch order, auxiliary task/size
  plan, optimizer, dropout settings and successful update counts. Keep model
  and data-schedule RNGs separate so preparation does not alter initialization.
- AdamW LR 1e-4, WD 1e-4, dropout .1, clip 1, original three-epoch LR warmup.
  Preserve CostController constants: LR patience 3, factor .5, minimum epoch
  15, stopping patience 8, effective cost improvement .001 percentage point,
  maximum epoch 40.
- To make TOTAL compute/update matching real, run A first. Its prescribed
  validation controller determines the LR trace and stopping epoch E (15--40)
  for BOTH arms. B follows that trace through exactly E epochs, rather than
  stopping independently or extending after seeing its validation results.
  B still independently selects its own strict validation-cost-best checkpoint.
  This shared, replay-controlled schedule is an explicit R60 protocol choice;
  it is not a claim that R45/R59 used identical realized A/B stopping times.

There are 1,422 successful full-pool optimizer updates/original epoch, including
1,185 MVRP auxiliary forwards and 150k auxiliary example exposures. Both arms
finish with exactly `1,422 * E` successful updates. No architecture, LR, loss,
sampling or seed search follows within this round.

## Validation decision and reporting

Each arm selects its checkpoint by STRICT minimum original full-18 validation
`macro_vs_sbs_pct`; equal values retain the earlier checkpoint. Do not substitute
best Top1, pair accuracy, a chosen task subset or a later handpicked epoch.

Advance B only if its selected checkpoint meets ALL of:
- Native equal-task-macro Top1 >= 52.1166667% (R45A +3 percentage points).
- Native equal-task-macro Top1 >= selected A +2 percentage points.
- Actual equal-task-macro regret <= BOTH selected A and the R45A reference
  0.9260483583%, using unrounded metrics for decisions.

Use exact reference values 0.49116666666666664 for R45A Top1 and
0.9260483583286789 for its percentage-valued actual regret. Partial improvement
below this screen is reported as such, not advancement or attainment of 70%.

NO TEST ACCESS OR EVALUATION in R60, even on advancement. Report original full-18
native Top1, actual regret, macro_vs_sbs_pct, mean cost, family/per-task metrics,
validation curves, selected epochs, successful updates and actual runtimes.
Auxiliary pair loss/accuracy and tie counts are diagnostics, not substitutes for
full-pool success. Report existing recipe/provenance and B-semantic limitations.

B beating matched replay on the full pool supports usefulness of additional
real comparisons under this fixed recipe; one seed is a direction screen, not
statistical replication. Pair gains without full-pool gains mean the new signal
did not transfer to the actual objective. Failure does not prove that more
complete compatible labels, other supervision or the 70% goal are impossible.

## Minimal execution handoff

Sol owns implementation, cost-path/shard/mmap engineering and execution. Use
existing completed columns without touching active queues or creating costs.
Only a brief column/identity/count/FP64/real-size check and ONE real combined
MVRP update are needed before immediate A -> B training; preserve the identical
scratch start after that check. No broad tests, solver replay, new audits,
extra experiment matrix or serial review ceremony is requested.

Planning estimate: 4--6 assigned RTX3090 GPU-hours for both arms through at most
40 epochs plus evaluation; not a cutoff or measured R60 timing. The existing
R45 optimization reference is about 114 seconds/epoch, and the third forward
adds approximately 42% to forward-count workload, not 9x optimizer updates.
Data preparation/I/O and actual backward/runtime overhead remain unmeasured;
record them and first-epoch throughput during training, without a separate
timing pilot. No GPU-hour limit. Only available assigned UUID
`GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d` on gpu03 may be used.

Use easynco, tmux, existing offline W&B, and the existing epoch-atomic resume
pattern; resume both original and auxiliary cursors/RNGs consistently. No GPU,
test, replay or implementation action was performed to write this plan.
One commit/push for the completed R60 version including the negative or positive
result, not intermediary commits or a new mandatory prelaunch review.

## Reviewer memory update

Settled: R59 final results failed its advancement screen without test access.
Existing records support 30 completed matched ReLD columns and 1.35M additional
paired TRAIN examples. R60 now locks scratch R45A full-pool learning with fixed
.35 real pair supervision versus matched oldTRAIN pair replay, not self-training
or a binary expert. Shared replay-controlled LR/horizon preserves equal updates.
ROOT, A_replay_seed2/B_expanded_seed2 and Volta data/Pascal runner ownership are
locked above. Necessary design review found no blocking issue; GPU remains
separately unauthorized. No implementation verification or replay was performed.
Open: actual implementation, execution and whether partial pair information
improves full-18 native winner selection. No gain or 70% achievement is claimed.
Boundary: legacy scenario, unresolved B mismatch, R58 deferred, no test access,
no new labels, no search branches. Plan readiness only; same-family/provisional.
