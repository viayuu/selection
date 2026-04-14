# Dataset and Label Planning Report V2

**Topic**: build the real full-scale supervised dataset-and-label release for a unified cross-problem solver selector  
**Date**: 2026-04-14  
**This file is new**: it does not overwrite `DATASET_LABEL_PLAN_REPORT.md` or the previous `dataset-plan-logs/`

## Executive Summary

This V2 plan replaces the old idea of fixing everything at `n=100` and also removes the old `smoke / coverage-first` staging. The new goal is to directly design the **actual 10k-per-problem dataset release** and the **actual initialization-only label-running schedule** that will feed the later supervised-learning work.

The key changes are:

1. `TSP` and base `CVRP` directly follow the `NSS` varying-size philosophy rather than a fixed-size protocol.
2. Other problems also become **multi-scale**, but in a **solver-aware** way rather than forcing the same size range everywhere.
3. The real labeling unit is not just a paper name like `MatNet` or `DeepACO`; it is a **solver entry**:
   - `method + checkpoint + decoding regime`
4. The real feasible set is therefore instance-dependent:
   - problem-compatible
   - attribute-compatible
   - scale/checkpoint-compatible
5. The full label campaign remains **initialization only**:
   - no iterative improvement
   - no search refinement
   - no local search stage

The most important consequence is that later supervised learning should consume a label table built from:

- a unified instance pool
- an instance-level feasible mask
- per-instance per-solver initialization scores

rather than from a fixed-size single-head classification target.

## What Changed From V1

The previous dataset plan had two assumptions that should now be dropped:

1. `n=100` as the first canonical size
2. `smoke dataset / smoke labeling / coverage audit` as the main next step

This V2 document changes both:

- no problem family is forced to be single-scale
- the document now describes the **actual first full label release**

## Core Principles

### 1. Every problem contributes one 10k dataset

The unit required by the user remains:

- one dataset per problem
- `10,000` instances each

For the current scope this means:

- `tsp`
- `cvrp`
- `atsp`
- `pctsp`
- `16` CVRP variants

### 2. Size diversity is now mandatory

For this release, size diversity is not optional.

The reference point is `NSS`:

- its synthetic data is not fixed at one size
- the converted `EasyNCO` manifest shows `TSP/CVRP` train data covering scales `50` to `499`
- the original `NSS` generator samples variable size in `[50, 500)`

So the right lesson from `NSS` is:

- do not build the whole program around a single size
- let scale variation become part of the dataset definition itself

### 3. But scale diversity should be solver-aware

Not every problem family has the same pretrained checkpoint coverage as `NSS TSP/CVRP`.

So V2 uses two styles:

1. **NSS-style continuous varying-size**
   - used where platform support is already strong
2. **solver-aware multi-scale buckets**
   - used where the available checkpoints are narrower

This keeps the spirit of `NSS` without pretending every problem has the same maturity as `TSP/CVRP`.

### 4. EasyNCO stays the center of gravity

The platform should remain the main execution path for:

- instance generation
- dataset storage
- method execution
- per-instance result export

External dataset design ideas can be borrowed, but the actual construction and labeling should stay as close as possible to `EasyNCO`.

### 5. Labeling is initialization-only

For this release:

- keep each solver at initialization stage only
- do not run iterative refinement
- do not run local-search improvement loops
- do not mix init-only labels with full-search labels

This means the target is:

- the quality of the initial solution emitted by each solver entry

## Problem-Wise Dataset Construction

## A. TSP

### Data source

Use the existing converted `NSS` varying dataset directly:

- `EasyNCO/data/datasets/nss_varying/nss_tsptrain_varying.pt`

This already matches the user request and already has:

- `10,000` instances
- scale diversity
- fixed-scale sidecar shards for graph/static methods

### Scale protocol

Keep the `NSS` size policy as-is:

- effective scale range: `50` to `499`

### Distribution protocol

Do not regenerate. Reuse the `NSS` source.

Operationally, this is the canonical example for the remaining problem families:

- one unified varying dataset file
- metadata recording scale/distribution
- fixed-by-scale derivative files for methods that require them

## B. Base CVRP

### Data source

Use the existing converted `NSS` varying dataset directly:

- `EasyNCO/data/datasets/nss_varying/nss_cvrptrain_varying.pt`

### Scale protocol

Keep the `NSS` size policy as-is:

- effective scale range: `50` to `499`

### Distribution / capacity protocol

Do not regenerate.

The original `NSS` generator already uses variable problem size and varying capacity logic, so base `CVRP` should simply inherit it.

## C. The 16 CVRP Variants

### Variant set

Use the standard `16`-variant family already aligned with the `MVRP` literature and `EasyNCO`:

1. `CVRP`
2. `OVRP`
3. `VRPB`
4. `VRPL`
5. `VRPTW`
6. `OVRPTW`
7. `OVRPB`
8. `OVRPL`
9. `VRPBL`
10. `VRPBTW`
11. `VRPLTW`
12. `OVRPBL`
13. `OVRPBTW`
14. `OVRPLTW`
15. `VRPBLTW`
16. `OVRPBLTW`

This family is standard in `RouteFinder / RL4CO MTVRP` style work.

### Generator base

Use:

- `EasyNCO/data/MVRPGenerator.py`

but extend it so the output becomes `varying-style` instead of single-style.

### Size protocol

Do **not** fix `n=100`.

Use a solver-aware multi-scale protocol:

- `30%` at `n=50`
- `45%` at `n=100`
- `25%` at `n=200`

So each 10k dataset becomes:

- `3,000` instances at `50`
- `4,500` instances at `100`
- `2,500` instances at `200`

Why this choice:

- it gives real size diversity
- it is much closer to current `MTPOMO / MVMoE` checkpoint reality than forcing `50-499`
- it keeps labeling practical inside the current platform

### Geometry protocol

For each scale bucket, sample the coordinate geometry from the same family of distributions already used in `TSP/CVRP` generation utilities:

- `uniform`: `20%`
- `gaussian_mixture`: `20%`
- `cluster`: `15%`
- `gaussian`: `10%`
- `uniform_rectangle`: `10%`
- `diagonal`: `10%`
- `explosion`: `7.5%`
- `implosion`: `7.5%`

This should be implemented by extending `MVRPGenerator` to reuse the same coordinate-generation helpers already used by `TSPGenerator.py` and `CVRPGenerator.py`.

### Constraint protocol

Do not invent a new variant semantics.

Reuse the current `MVRPGenerator` logic for:

- open-route flags
- backhaul flags
- route-length limit
- time windows
- service times

The extension should mainly improve:

- size diversity
- geometry diversity
- metadata recording

## D. ATSP

### Generator base

Use:

- `EasyNCO/data/ATSPGenerator.py`

but extend it beyond the current single random-matrix style.

### Size protocol

Use solver-aware exact scales:

- `25%` at `n=20`
- `35%` at `n=50`
- `40%` at `n=100`

So each 10k dataset becomes:

- `2,500` instances at `20`
- `3,500` instances at `50`
- `4,000` instances at `100`

This choice is directly aligned with the current local ATSP checkpoint ecosystem:

- `MatNet`: `20/50/100`
- `MatPOENet`: mixed checkpoint but still ATSP-centered
- `GLOP`: flexible ATSP checkpoint

### Instance-family protocol

Split the ATSP dataset into three structural sub-families:

- `40%` random metric-closure matrices
- `35%` Euclidean asymmetric skew matrices
- `25%` clustered asymmetric skew matrices

Implementation principle:

1. keep the current random metric-closure generator as one branch
2. add Euclidean-derived asymmetric generators that start from coordinates and then build directional costs
3. record the sub-family type in metadata

This gives more meaningful diversity than only drawing one kind of random cost matrix.

## E. PCTSP

### Generator base

Use:

- `EasyNCO/data/PCTSPGenerator.py`

and extend it to become both geometry-diverse and regime-diverse.

### Size protocol

Use a two-scale design:

- `35%` at `n=100`
- `65%` at `n=500`

So each 10k dataset becomes:

- `3,500` instances at `100`
- `6,500` instances at `500`

Why not a larger continuous interval here:

- the current local `PCTSP` solver pool is much narrower than `TSP/CVRP`
- `DeepACO` and `GLOP` have overlapping support centered around the larger scale regime
- forcing many scales with almost no overlapping solver coverage would weaken the usefulness of the labels

### Geometry protocol

Use the same geometry family as for Euclidean routing:

- `uniform`
- `gaussian_mixture`
- `cluster`
- `gaussian`
- `uniform_rectangle`
- `diagonal`
- `explosion`
- `implosion`

### Prize / penalty regimes

Within each scale bucket, split into three regimes:

- `40%` standard
- `30%` high-prize / low-penalty
- `30%` low-prize / high-penalty

This is important because later solver preference may depend as much on prize/penalty structure as on node geometry.

## Unified Varying-Dataset Format

All newly generated non-NSS datasets should imitate the `NSS varying` style and include:

- one main `.pt` file per problem dataset
- one manifest JSON file
- one fixed-by-scale directory

Recommended metadata fields:

- `problem`
- `problem_variant`
- `scale_list`
- `distribution_list`
- `generator_family_list`
- `regime_list`
- problem-specific tensors
- optional `capacity_list` or other scalar attributes when relevant

Recommended storage pattern:

- `EasyNCO/data/datasets/unified_varying/<problem>_train10k_varying.pt`
- `EasyNCO/data/datasets/unified_varying/graph_fixed_by_scale/<problem>/...`
- `EasyNCO/data/datasets/unified_varying/<problem>_manifest.json`

This makes the new datasets behave like `NSS` data inside the platform.

## Real Label-Running Plan

## 1. The candidate unit is a solver entry, not just a solver name

The correct labeling unit is:

- `solver entry = method + checkpoint + decoding regime`

This follows the same logic as `NSS`, where different pretrained variants of the same family can behave like different candidates.

Examples:

- `matnet_atsp20`
- `matnet_atsp50`
- `matnet_atsp100`
- `deepaco_pctsp100`
- `deepaco_pctsp500`
- `pointerformer_tsp50`
- `pointerformer_tsp100`
- `pointerformer_tsp200`
- `pointerformer_tsp500`

This is the cleanest way to handle:

- scale-conditioned availability
- unequal checkpoint support
- output imbalance

## 2. First full label release: platform-native init-only solver pools

The first label release should prioritize methods that are already EasyNCO-native and compatible with init-only evaluation.

### TSP

Recommended first-wave pool:

- `Pointerformer`
- `INVIT`
- `ELG`
- `LEHD`
- `ICAM`
- `LIH`
- `DACT`
- `UDC`
- `OMNI`
- `DIFUSCO`
- `T2T`
- `GLOP`

Optional pure-init baselines:

- `INSERTIONInitialization(random)`
- `INSERTIONInitialization(nearest)`
- `INSERTIONInitialization(farthest)`
- `INSERTIONInitialization(median)`
- `INSERTIONInitialization(regret)`

Not part of the first init-only release:

- `AM`, `POMO`
  - no ready checkpoint in current local platform state
- `DeepACO@TSP`
  - current note says init-only does not emit a usable solution
- `HTSP`
  - current local checkpoint situation does not match this scale program well
- `LKH / EAX / OR-Tools / CPLEX / Gurobi`
  - these are better treated as full classical solvers, not as the init-only label release

### Base CVRP

Recommended first-wave pool:

- `INVIT`
- `ELG`
- `LEHD`
- `ICAM`
- `LIH`
- `DACT`
- `UDC`
- `OMNI`
- `DeepACO`
- `GLOP`

Optional pure-init baselines:

- `INSERTIONInitialization(random)`
- `INSERTIONInitialization(nearest)`
- `INSERTIONInitialization(farthest)`
- `INSERTIONInitialization(median)`

Not part of the first init-only release:

- `AM`, `POMO`
  - no ready checkpoint
- `NLNS`
  - not a pure initialization path
- `HGS / PyVRP / OR-Tools / CPLEX / Gurobi`
  - not part of the init-only release

### ATSP

Recommended first-wave pool:

- `MatNet`
- `MatPOENet`
- `GLOP`

Checkpoint-specific entries should be used for the scale buckets where relevant.

### PCTSP

Recommended first-wave pool:

- `DeepACO`
- `GLOP`

Again, checkpoint-specific entries should be tracked explicitly.

### 16 CVRP Variants

Recommended first-wave pool:

- `MTPOMO`
- `MVMoE`

These two methods define the current core label pool for the variant family.

## 3. Feasible mask definition

The feasible mask should no longer mean only:

- does this method support this problem in principle

It should mean:

1. method structurally supports the problem family
2. method structurally supports the attributes of this variant
3. a checkpoint/config exists for the instance scale bucket
4. the run completed successfully on that shard

So the actual label table should store an **instance-level feasible mask**.

## 4. Label config strategy

For the full release, create label-only settings or runtime overrides so that every label run satisfies:

- `mode=test`
- `iteration = NoIteration`
- per-instance export enabled

Two acceptable implementation styles:

1. create `settings/<method>_label_initonly.yaml`
2. reuse the original settings file and override:
   - `settings.iteration._target_=EasyNCO.neural_solvers.pipeline.NoIteration`

The second option is lighter and closer to how the current platform is already used.

## 5. Real run unit

The real run unit should be:

- one `solver entry`
- on one `dataset shard`

not:

- one solver on a mixed dataset with no scale bookkeeping

For dynamic methods, a full varying dataset may still be runnable in one shot.

For scale-sensitive or graph/static methods, run by fixed-scale shard and then merge.

This is already exactly how `run_nss_eval_suite.py` handles graph methods like `DIFUSCO` and `T2T`.

## 6. Real launch pattern

### TSP / CVRP on NSS varying data

For methods that can read the whole varying dataset directly, use:

```bash
conda activate easynco_zhoucl
cd /public/home/zhoucl/shiys/EasyNCO
python eval.py \
  settings=invit_settings \
  mode=test \
  model=invit \
  problem=tsp \
  cuda=[0] \
  batch_size=128 \
  episodes=10000 \
  test_data_path=nss_varying/nss_tsptrain_varying.pt \
  varying_data_params.scale_range=[50,500] \
  varying_data_params.distribution_list=null \
  settings.iteration._target_=EasyNCO.neural_solvers.pipeline.NoIteration \
  dir=results/label_v2/invit_tsp/TSPtrain_full
```

For graph/static methods, use the fixed-scale shard pattern:

```bash
python eval.py \
  settings=t2t_settings \
  mode=test \
  model=t2t \
  problem=tsp \
  scale=100 \
  cuda=[0] \
  batch_size=32 \
  episodes=<count_of_this_scale_shard> \
  test_data_path=nss_varying/graph_fixed_by_scale/TSPtrain/TSPtrain_scale_100.pt \
  settings.iteration._target_=EasyNCO.neural_solvers.pipeline.NoIteration \
  dir=results/label_v2/t2t_tsp/TSPtrain/scale_100
```

### ATSP

```bash
python eval.py \
  settings=matnet_settings \
  mode=test \
  model=matnet \
  problem=atsp \
  scale=50 \
  cuda=[0] \
  batch_size=64 \
  episodes=<count_of_scale_50_shard> \
  test_data_path=unified_varying/graph_fixed_by_scale/atsp/atsp_scale_50.pt \
  settings.test_loader.model_filename=matnet_atsp50.ckpt \
  settings.iteration._target_=EasyNCO.neural_solvers.pipeline.NoIteration \
  dir=results/label_v2/matnet_atsp/scale_50
```

### PCTSP

```bash
python eval.py \
  settings=glop_settings \
  mode=test \
  model=glop \
  problem=pctsp \
  scale=500 \
  cuda=[0] \
  batch_size=32 \
  episodes=<count_of_scale_500_shard> \
  test_data_path=unified_varying/graph_fixed_by_scale/pctsp/pctsp_scale_500.pt \
  settings.test_loader.model_filename=glop_policy_pctsp_500.pt \
  settings.iteration._target_=EasyNCO.neural_solvers.pipeline.NoIteration \
  dir=results/label_v2/glop_pctsp/scale_500
```

### 16 CVRP Variants

```bash
python eval.py \
  settings=mvmoe_settings \
  mode=test \
  model=mvmoe \
  problem=OVRPBLTW \
  scale=100 \
  cuda=[0] \
  batch_size=64 \
  episodes=<count_of_scale_100_shard> \
  test_data_path=unified_varying/graph_fixed_by_scale/ovrpbltw/ovrpbltw_scale_100.pt \
  settings.iteration._target_=EasyNCO.neural_solvers.pipeline.NoIteration \
  dir=results/label_v2/mvmoe_ovrpbltw/scale_100
```

`eval.py` already has the special handling needed for `mtpomo / mvmoe` and already exports per-instance results through `ARREINFORCELightning`.

## 7. Scheduling the full release

This release should be scheduled in four actual blocks rather than through tiny tests:

### Block A. Generate all datasets

Generate all new 10k datasets first:

- `16` CVRP variant datasets
- `atsp`
- `pctsp`

At the same time, emit:

- main varying `.pt`
- manifest JSON
- fixed-by-scale shards

### Block B. Run TSP + base CVRP labels

These two should run first because:

- they already have the strongest dataset base
- they have the richest solver zoo
- their outputs will define the merge format for the rest

### Block C. Run ATSP + PCTSP labels

These are smaller solver pools but they are crucial for the unified cross-problem target.

### Block D. Run the 16 CVRP-variant labels

This is operationally the widest block because the same two core methods must be run across many variants and scale shards.

## 8. Result merge

Each run already exports:

- `instance_results.jsonl`

After all runs finish, merge in two levels.

### Level 1. Solver-entry-level merge

For methods run by multiple shards:

- concatenate shard outputs
- restore global instance indices
- write one dataset-level file per solver entry

### Level 2. Final supervision-table merge

For each instance, collect:

- `instance_id`
- `problem`
- `problem_variant`
- `scale`
- `distribution`
- `generator_family`
- `regime`
- `feasible_solver_entries`
- `no_aug_score` per solver entry
- `aug_score` per solver entry
- `run_status` per solver entry
- `best_solver_entry_by_aug_score`
- `rank_of_each_solver_entry`
- `relative_gap_to_best_observed`

The primary label should default to:

- `best_solver_entry_by_aug_score`

while still preserving:

- `no_aug_score`
- the full ranking

This leaves room for later training either as:

- top-1 classification
- pairwise ranking
- listwise ranking

## 9. What V2 explicitly does not do

This V2 plan intentionally does **not** center the next step on:

- smoke datasets
- 200-instance probes
- tiny dry-run labels
- an abstract coverage-only audit before data generation

Those can still happen informally if needed during execution, but they are no longer the official plan.

The official plan is the **full dataset + full label release**.

## Final Recommendation

The correct next move is:

1. keep `TSP/CVRP` on the existing `NSS varying` datasets
2. build solver-aware varying datasets for `ATSP`, `PCTSP`, and the `16` CVRP variants
3. treat `solver entry` rather than raw method name as the label candidate unit
4. launch the first real full init-only label release directly in `EasyNCO`

That is the cleanest data foundation for the later unified supervised selector.

## Main Sources

Local sources:

- `/public/home/zhoucl/shiys/平台方法统计统计.md`
- `/public/home/zhoucl/shiys/EasyNCO/data/datasets/nss_varying/nss_manifest.json`
- `/public/home/zhoucl/shiys/9nss论文/neural-solver-selection/datasets/data_config.yml`
- `/public/home/zhoucl/shiys/9nss论文/neural-solver-selection/datasets/generate_data.py`
- `/public/home/zhoucl/shiys/EasyNCO/data/TSPGenerator.py`
- `/public/home/zhoucl/shiys/EasyNCO/data/CVRPGenerator.py`
- `/public/home/zhoucl/shiys/EasyNCO/data/MVRPGenerator.py`
- `/public/home/zhoucl/shiys/EasyNCO/data/ATSPGenerator.py`
- `/public/home/zhoucl/shiys/EasyNCO/data/PCTSPGenerator.py`
- `/public/home/zhoucl/shiys/EasyNCO/eval.py`
- `/public/home/zhoucl/shiys/EasyNCO/phases/rl/ar_reinforce.py`
- `/public/home/zhoucl/shiys/EasyNCO/run_nss_eval_suite.py`

External references:

- NSS: https://arxiv.org/abs/2410.09693
- RouteFinder: https://arxiv.org/abs/2406.15007
- RL4CO routing docs: https://rl4co.readthedocs.io/en/stable/docs/content/api/envs/routing/
