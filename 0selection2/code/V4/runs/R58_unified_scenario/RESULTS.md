# R58: Unified scenario_v2

**Status: not complete. No new model result is claimed.**

All128 ORIGINAL-recipe deployment preflights passed. The original full fresh-solve recipe is
SKIPPED because its size-weighted label-only projection is181.637GPUh, above
the user's100GPUh cap. A bounded train-only throughput pilot has now completed;
no complete label generation, scenario release or training has started.
See`PREFLIGHT_RESULTS.md` and`BUDGET.md` for the frozen evidence and limitations.

- Original-recipe qualified deployment preflights: 128/128. These do NOT qualify changed production code.
- Full independently checked scenario published: False.
- Full baseline training completed: False.
- Locked checkpoint test completed: False.

## Locked semantic decisions
- Classical backhaul, separate delivery/pickup capacity, delivery-before-pickup within each route.
- Pure pickup routes are allowed. No implicit finite depot deadline.
- L limits travel distance, not waiting/service. Open routes exclude depot return distance/time.
- Fixed original weights with explicitly NEW adapted deployments, not historical reproduction.
- Every saved route must pass a solver-independent simulation before a cost can be published.
- All old instances, labels, results and solver source trees remain unchanged.

## Completion gates
All candidate preflights must pass before deployment lock; all54 original train/val/test datasets
must have complete feasible cost vectors before training. Missing or invalid rows stop publication;
they do not remove difficult instances or change per-instance candidate eligibility.

See task_contract.json, deployment_preflight.csv, preflight/*.json, and pipeline_state.json.
The provisional timing sum includes only probed deployments and is not a total-budget guarantee.

## Bounded Throughput Pilot

The three RF-family models were tested with original orientation, augmentation1,
all customer starts and fixed weights/FP32. All36 representative phases passed:
432 timed routes and252 warmups passed independent checking; concurrent paid
costs and physical tours exactly matched their serial controls. This is NOT
quality equivalence between augmentation1 and the original augmentation8.

Four20-step diffusion deployments also ran successfully. T2T actually performed
one10-step guided rewrite with the original respective checkpoint and ratio;
it was not an initialization-only substitute. Actual call witnesses are saved.

Cumulative pilot controller GPU wall was1486.465seconds (0.413h), not the
projected full-solve hours. The representative ordinary-W4 transfer projection
still gives102.709label hours BEFORE overhead or training; it is not a passing
whole-roster budget. Failed initial pilot attempts are preserved and diagnosed
in `throughput_pilot/REPORT.md`.

One final precommitted TRAIN-only timing refinement is planned within the SAME
cumulative3GPUh pilot cap. MPS tools are installed, but no daemon was started by
the pilot. A conditional execution-only MPS4 comparison may be measured only
with private node-local directories, the designated UUID, no system/GPU mode
changes and exact route/cost checks. No acceleration is assumed.

Original evidence is preserved in `original_recipe/` and commits0dd2862c8 and
2ba1cd5d2. Revised production code must obtain new qualifications and a complete
sub100GPUh plan, including spent work, startup/checking/IO, baseline/evaluation
and contingency, before deployment lock or full labels. All18 tasks and all
original candidates remain required. No selector score or70% result is claimed.
