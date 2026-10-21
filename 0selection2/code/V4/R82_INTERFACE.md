# R82B Context-Matched Relative-Regret Ranking

Only `B_context_matched_rank_seed2` is queued under
`runs/R82_context_matched_ranking`. R80 and R81 are separate experiments.
Alias: `Requested_R80B_context_matched_residual_ranking`.

## Objective API

```python
from code.V4.context_matched_objective import (
    CMR_GROUP_KEY, CMR_REGRET_KEY, CachedCMRObjective, utility_update,
)

objective = CachedCMRObjective(selector, cmr_weight=.10, ramp_updates=3726)
parts, retries = utility_update(
    objective, optimizer, scaler, batch, micro_cap,
    consistency_weight=original_r77_kl_weight,
    successful_update=successful + 1,
    groups=batch_group_ids, regrets=batch_regrets_fp64,
)
```

- `groups`: int64 `[B]`, one problem per batch, matching only identical group IDs.
- `regrets`: original-FP64-derived relative regret, float64 `[B,K]`.
- Private batch keys are `r82_context_group` and `r82_relative_regret`.
  Keep labels outside the model; strip both keys before every selector forward.
- All original losses reuse the R81/R77 implementation without changing inference.
- The new term uses ALL unequal-regret unordered pairs inside a group. Its
  denominator is the number of unequal pairs, NOT the sum of their weights.
- Average weighted pair losses within group, average nonempty groups within
  method, then average methods with valid comparisons in this logical batch.
  Unlike R81, there is no fixed TRAIN-eligible-method denominator.
- Two dropout views are evaluated separately and averaged. Lambda is
  `.10 * min(1, successful_update / 3726)` on successful updates.
- `CachedCMRObjective.full_loss(u1, u2, batch, *, consistency_weight,
  successful_update, groups=None, regrets=None)` returns `(loss, parts)`.
- `context_matched_loss(utility, mask, regrets, groups)` returns `(loss, stats)`.
- `utility_update` uses full-1024 utility leaves and exact micro RNG replay,
  one scaler/clip/optimizer update, and whole-batch AMP retry. No feature queue.
- Metrics retain all base parts plus `base_loss`, `cmr_loss`, `auxiliary_loss`,
  `cmr_weight`, `cmr_effective_methods`, `cmr_effective_groups`, `cmr_pair_count`,
  `cmr_matched_instance_pairs`, and `cmr_row_coverage`.

## TRAIN Context API

`code.V4.r82_context` provides:

- `relative_regret(costs_fp64)` -> float64 `[N,K]`.
- `fit_context(metadata, problem, bins=10)` -> `(int64[N] groups, spec)`.
- `apply_context(metadata, spec)` -> int64 `[N]`, unseen TRAIN combinations `-1`.
- `prepare_training_context(args)` -> preparation receipt. Args have `root`,
  `prepared_root` (R77), `data_root`, `cache_root`. Only TRAIN is read.

Actual metadata fields are `scale`, `distribution`, and `capacity_type` in each
`input_metadata.npz`; no information is guessed from instance indices.
Ten log-size quantile bins are fitted per problem on TRAIN only. Duplicate
cutpoints are merged; equal physical sizes are never split arbitrarily.
TSP/ATSP omit capacity type; the other problems use it exactly as recorded.

Each `support/PROBLEM.npz` contains `groups`, `regrets`, `group_means`, `pool`,
`pool_ids`, and cache-row `source_indices`. `support/manifest.json` has per-problem
`path`, `rows`, `pool`, `pool_ids`, `native_costs_sha256`, `native_winner_sha256`,
`context` (fitted spec), `groups_sha256`, `regrets_sha256`, and `methods`.
Preparation records source-cache provenance separately from the runtime local
cache path. Frozen native costs/winners verify row alignment on runtime open.

The TRAIN group mean is saved for interpretation. Within a single group,
`(r_i - mean_g) - (r_j - mean_g) == r_i - r_j`, so loss comparisons use original
FP64 regrets directly, avoiding unnecessary cancellation. Group means never
enter the network forward.

`prepared/pair_coverage.json` and `.csv` summarize eight unchanged natural
batch1024 draws per problem, seed2. This is a small TRAIN-only coverage check,
not a parameter search, heldout analysis, or additional training.

## Locked Protocol

R77 fresh seed2 initialization; original data/model/loss/inference; batch1024,
49680 successful updates, warmup3726, LR1e-4 cosine to1e-6, WD1e-4, dropout.1,
clip1. Full VAL every1242 updates; strict minimum full-VAL macro_vs_sbs_pct
selects one checkpoint, then complete TEST and both LIBs. No R82A training;
R77B is historical, not a matched newly trained A/B control.

## Focused CPU Evidence

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest
code.V4.test_r82_cmr -v` passes five tests. Checks include the literal hierarchical
loss, small-gap weighting, exact FP64 near-ties, permutations/padding, TRAIN-only
quantile groups, original-loss/gradient equivalence at lambda zero, and a true
1024-row (4x256) retained-graph reference with comparisons exclusively crossing
microbatch boundaries. Dropout replay, final RNG and all parameter gradients agree.

The real TRAIN preparation sampled eight unchanged natural batches per problem.
All 144 batches have 100% row coverage by at least one unequal-regret,
same-context pair. TSP/ATSP have 10 contexts each, CVRP 20, and each MVRP 40.
This is coverage evidence, not model accuracy or a claim of useful generalization.
