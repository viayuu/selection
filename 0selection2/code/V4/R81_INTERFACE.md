# R81B method-conditional cross-instance AUC integration

R81B is the new run identifier; its proposal alias is R80B. Do not overwrite or
restart the existing R80 batch512 run. This worker implements only the objective
and small CPU tests, not training, GPU allocation, launch, or Git submission.

## Public API

Module: `code.V4.method_auc_objective`.

```python
from code.V4.method_auc_objective import (
    AUC_POS_KEY, AUC_NEG_KEY, CachedAUCObjective,
    method_auc_targets, train_eligible_methods, utility_update,
)

# Once per problem, from ORIGINAL native FP64 TRAIN costs, never cached FP32.
targets = method_auc_targets(native_train_costs_fp64, train_solver_mask)
eligible = train_eligible_methods(targets[AUC_POS_KEY], targets[AUC_NEG_KEY])

objective = CachedAUCObjective(selector)  # Existing ScoreDifferenceSelector.

# Gather the bool targets using the exact TRAIN row indices of this batch.
# Either put them in batch[AUC_POS_KEY]/batch[AUC_NEG_KEY], or pass pos=/neg=.
# `eligible` is fixed for this problem's entire TRAIN split, aligned to pool_ids.
metrics, retries = utility_update(
    objective, optimizer, scaler, batch, micro_cap,
    consistency_weight=original_r77_kl_weight,
    successful_update=successful + 1,
    train_eligible=eligible,
    pos=batch_pos, neg=batch_neg,
    check_replay=False,
)
# Advance counters, data cursor, LR schedule, etc. ONLY after successful return.
successful += 1
```

- `method_auc_targets(costs, mask=None)` accepts an FP64 Torch tensor or NumPy
  array shaped `[N, K]`, and returns `{AUC_POS_KEY: bool[N,K],
  AUC_NEG_KEY: bool[N,K]}`. Padding is never a label. Native `ind` is untouched.
- `train_eligible_methods(pos, neg)` returns `bool[K]`: has at least one P and N
  in the complete TRAIN split. Save this axis/mask in the run protocol. Never
  derive it from a batch, VAL, TEST, a zero-selection subset, or the model.
- Each logical batch is from ONE problem with a shared candidate column axis.
  `pool_ids` may be `[K]` or identical rows `[B,K]`. A common candidate permutation
  must also permute utilities, masks, labels and the TRAIN eligible mask. Per-row
  different candidate axes are rejected rather than silently pairing methods.
- All new label keys are explicitly removed before EVERY selector forward;
  the old `pairwise_objective.inputs_only` does not know these keys. Call
  `auc_inputs_only(batch)` for any additional runner-owned forward containing
  these private labels, or keep labels outside the batch entirely.
- `CachedAUCObjective(selector, auc_weight=.35, ramp_updates=3726)` adds no model
  parameters. `auc_weight=0.` exists only for equivalence tests/future A; this
  task does not authorize launching A.
- `utility_update(..., check_replay=False, max_retries=20,
  deadline_check=lambda: None)` returns `(metrics_dict, retry_count)`.
- Metrics: `loss`, `base_loss`, `auc_loss` (unweighted two-view mean),
  `auxiliary_loss` (lambda times AUC), `auc_weight`, `ce`, `cmp`, `risk`, `kl`,
  `true3`, `pred3`, `all_pairs`, `grad_norm` (before clip).
- Reference/debug helpers: `method_competitive_scores(utility, mask)`,
  `method_auc_loss(utility, mask, pos, neg, train_eligible)`,
  `objective.views(batch)`, `objective.base(first, second, batch,
  consistency_weight)`, and `objective.full_loss(u1, u2, batch,
  consistency_weight=..., successful_update=..., train_eligible=...,
  pos=None, neg=None)` returning `(loss, parts)`.

## Locked semantics

For each method m, P contains unique exact FP64 minima; N contains legal costs
strictly greater than that row's minimum. On a tied-minimum row, each tied method
is excluded for itself, but other strict losers still enter their N sets. Do not
reuse FP32 argmin or native CE `ind` to construct these masks.

`a_m = u_m - logsumexp(other legal u)` is a competitive score, NOT a calibrated
probability. Invalid slots and singleton legal pools have neutral finite score
zero; the empty-opponent reduction is made finite BEFORE logsumexp to avoid NaN
gradients. Homogeneous single-method problems have no eligible methods and zero
AUC. Original maximin inference and native-label CE remain unchanged.

Each eligible method contributes the mean of `softplus(a_neg - a_pos)` over ALL
P*N pairs in the logical batch. Sum those method means, then divide by the fixed
TRAIN eligible count. A batch missing P or N contributes zero for that method;
do NOT dynamically renormalize. An empty TRAIN eligible set yields zero AUC.

The two dropout views have their AUC losses computed separately then averaged.
Lambda is `.35 * min(1, successful_update / 3726)` with the one-based target
successful update index; a retry keeps the same index. Base loss reuses
`pairwise_objective.single_objective`, `predicted_top3`, and R42 `symmetric_kl`;
the existing runner supplies the unchanged original R77 KL ramp.

The cache pass is sequential micro1/view1/view2, micro2/view1/view2, etc., under
no_grad; score-head dropout also executes. Cache only utilities and per-micro RNG,
concatenate `[B,K]` leaves, compute FULL logical-batch base+AUC gradients, and
replay both views with their saved RNG. VJP gradients already include every
global mean and lambda: no further micro weighting. One scaler/unscale, global
clip at 1, and optimizer step cover the logical batch. Restore the first pass's
final RNG. Nonfinite loss/gradients lower the scaler and retry the entire SAME
batch without stepping; exhaustion raises. No trainable feature/cache persists
across optimizer updates. This helper targets the existing single-device runner,
not DDP or modules with forward-mutated training buffers such as BatchNorm.

## CPU verification command

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest code.V4.test_r81_auc -v
```

## Completion / CPU evidence

Core implementation and focused CPU verification are complete. The command above
passes 7 tests (about 1 second of test execution). In particular:

- Lambda-zero uses the original `RDropObjective` with the SAME sequential paired
  dropout forwards, identical micro partition `3+3+1`, and identical initial RNG;
  checks true loss, all parameter gradients, clipping, final parameters and RNG.
- Positive-lambda uses a real logical batch of 1024 with four 256-row microbatches.
  Each micro has zero within-micro AUC pairs, whereas the full batch has nonzero
  AUC. The reference RETAINS all micrographs, averages the two full-batch view AUC
  losses, and matches cached true loss and every parameter gradient/update.
- Other checks cover exact FP64 ties/strict losers, direction, literal loss and
  derivatives, per-instance shifts, candidate permutations, padding/singletons,
  fixed TRAIN eligible denominator, encoder/score-path gradients, new label
  stripping, exact dropout replay, and whole-batch overflow retry without stepping.
- Two existing R43 CPU checks also pass: `test_shared_initialization_and_old_interface`
  and `test_permutation_and_padding`, including ExplicitPairSelector compatibility.

Parameter comparisons use `rtol=3e-5, atol=3e-7`; full-batch versus weighted
microbatch FP32 part means use `rtol=2e-7, atol=3e-7`. Scalar `loss` is always the
actual full utility-leaf objective, never a VJP injection surrogate.

No training, GPU operation, queue launch, or Git submission was performed by this
worker. The runner/queue owners control readiness and launch. This document is
not a completed training result or independent launch authorization.
