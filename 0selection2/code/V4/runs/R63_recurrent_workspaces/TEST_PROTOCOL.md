# R63 formal test protocol

User amendment recorded 2026-10-12, before either R63 arm finishes training
and before any R63 test inference. This replaces the original PLAN.md
no-test restriction only. The live training recipe is unchanged.

## Training budget amendment

At2026-10-12T12:19:24+08:00 the user requests stopping B after epoch15 and
complete validation. A has already completed40 epochs. budget_amendment.json
records this change; the original model, first15 batch/LR prefix and objectives
remain unchanged. Report A40/B15 as unequal budgets, not a matched-horizon
architecture experiment. Common-epoch curves may compare their first15 epochs.

## Checkpoint selection

- Preserve completed A40; finish B15 and complete its validation/finalization.
- For each arm, lock its single strict validation-native-macro-Top1 best
  checkpoint; exact ties retain the earlier epoch, as already implemented.
- Record the two checkpoint paths and validation results in
  final_test_lock.json before test inference.
- The recommended R63 arm is the one with higher validation Top1; an exact
  A/B tie selects A. Test results cannot change this recommendation.
- R60B is a historical reference with its already selected epoch39 best.pt,
  not a new search over historical checkpoints. R45A uses its existing official
  full18 test results when available.

## One test evaluation per locked model

Evaluate R63A, R63B and the locked R60B reference once each on the complete
original18-task test split, with each task's original full legal candidate pool.
Use native winners, greedy single-solver selection, eval mode, no augmentation,
all instances including tail batches, and original raw costs accumulated in
FP64. Restore geometry preprocessing from the corresponding checkpoint;
never fit it on test inputs.

Test prediction artifacts are retained so tables can be computed from the
same inference outputs. No test-driven checkpoint replacement, decision-rule
change, candidate change, task-specific model assembly or training restart.
No additional solver runs, new labels or R58 operations are permitted.

## Reporting and continuation

Report validation and test Top1/2/3, mean cost and actual regret for ALL,
the task families and every task. Mark historical cost-selected references
separately from R63's prospectively Top1-selected models. Disclose selector
latency and unresolved legacy backhaul semantics without changing the pools.

The user target remains native Top1 at least70% on the full18-task single-solver
task. Validation alone is not proof of achieving the target. Costs and A/B
mechanism contrasts are reported, not added vetoes on the Top1 target.
Subsequent experiment design uses training/validation evidence; this test
evaluation does not authorize tuning to individual test errors.
