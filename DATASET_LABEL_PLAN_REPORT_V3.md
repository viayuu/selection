# Dataset and Label Planning Report V3

**Topic**: executable zero-code-change dataset and label plan for a unified cross-problem init-only solver selector  
**Date**: 2026-04-14  
**This file is new**: it does not overwrite `DATASET_LABEL_PLAN_REPORT.md` or `DATASET_LABEL_PLAN_REPORT_V2.md`

## Executive Summary

This V3 plan is the corrected version under the user's hard constraint:

- do **not** modify `EasyNCO` code
- do **not** build the plan around smoke tests
- do **not** fall back to coverage-only auditing
- do build a **real multi-scale label campaign**
- do call existing `EasyNCO` code directly for generation and evaluation

The most important correction relative to V2 is:

- V2 still implicitly assumed generator extensions
- V3 assumes **zero EasyNCO code changes**

So the official executable strategy becomes:

1. reuse `NSS` varying datasets directly for `TSP` and base `CVRP`
2. generate all other datasets by calling the existing `EasyNCO` generators from external scripts
3. store those datasets as **fixed-scale shard files + one manifest**, not as new custom loader formats
4. run labels only through existing `EasyNCO/eval.py` or `EasyNCO/run_nss_eval_suite.py`
5. keep labels strictly **initialization only**

This leads to one hard but important conclusion:

- `PCTSP` cannot join the **first selector label release** under strict `init-only`, because the current local method pool collapses to a single feasible method after applying the `NoIteration` constraint

So the first true selector-label release should focus on:

- `TSP`
- `CVRP`
- `ATSP`
- `16` CVRP variants

while `PCTSP` is treated as:

- a dataset-generation track
- and optionally a single-method label archive

## Hard Constraints

This V3 plan treats the following as non-negotiable:

1. no edits under `EasyNCO/`
2. no new loaders inside `EasyNCO`
3. no patching generator internals
4. no fake preliminary stage as the official plan
5. labels must come from actual batch runs

## What Existing EasyNCO Already Gives Us

From the local codebase:

### 1. Data generation entrypoints already exist

The following generators already exist and can be called directly:

- `TSPGenerator`
- `CVRPGenerator`
- `ATSPGenerator`
- `PCTSPGenerator`
- `MVRPGenerator`

through:

- `EasyNCO.data.main.generate_data`

### 2. Real eval entrypoints already exist

The actual label-running entrypoints already exist:

- `EasyNCO/eval.py`
- `EasyNCO/run_nss_eval_suite.py`

### 3. Per-instance export already exists

`ARREINFORCELightning` already exports:

- `instance_results.jsonl`

So we do not need new export code.

### 4. NSS varying support already exists

`TSP/CVRP` already have:

- converted `NSS` varying files
- manifest metadata
- fixed-scale graph shards

### 5. The platform already supports init-only overrides

For many methods, init-only evaluation is already achieved by:

- `settings.iteration._target_=EasyNCO.neural_solvers.pipeline.NoIteration`

## Hard Feasibility Findings

These findings matter because they change which problems can enter the first real label release.

### A. TSP and base CVRP are ready for a true varying-size release

Reason:

- `NSS` varying datasets already exist locally
- `run_nss_eval_suite.py` already supports them
- graph/static methods are already handled by fixed-scale shard splitting

### B. ATSP is ready for a true multi-scale release without code changes

Reason:

- `ATSPGenerator` already exists
- `MatNet` has local checkpoints at `20 / 50 / 100`
- `GLOP` has a local ATSP policy
- `MatPOENet` has a local checkpoint

So ATSP can enter the first real selector-label release.

### C. The 16 CVRP variants are ready for a true multi-scale release without code changes

Reason:

- `MVRPGenerator` already exists
- `MTPOMO` and `MVMoE` already have local checkpoints
- local checkpoint overlap is strongest at `50 / 100`

So the variants can enter the first real selector-label release.

### D. PCTSP is **not** ready for the first real selector-label release under strict init-only

Reason:

- local PCTSP pool is effectively `DeepACO` and `GLOP`
- `DeepACO` does **not** support eval-time `NoIteration` cleanly
- after enforcing strict init-only, only `GLOP` remains feasible

That means:

- PCTSP can still have data
- PCTSP can still have a single-method archive
- but PCTSP should **not** be treated as part of the first selector label release

This is a real execution constraint, not a modeling preference.

## Dataset Construction Plan Without Modifying EasyNCO

## A. Representation Rule

For every non-NSS problem family, the official dataset object is:

- a directory of fixed-scale shard files
- plus one manifest file outside `EasyNCO` logic

This is the key trick that lets us avoid code changes.

Instead of inventing a new varying loader, we do this:

1. call existing `EasyNCO` generator at one fixed scale
2. save the output in the file format that current loaders already accept
3. repeat for several scales
4. keep a manifest that says these shards together form the 10k dataset

So the data is multi-scale at the **dataset-program level**, even if each eval job still reads one fixed-scale shard.

## B. TSP

### Official dataset source

Use directly:

- `EasyNCO/data/datasets/nss_varying/nss_tsptrain_varying.pt`

### Size diversity

Already provided by `NSS`:

- effective size range `50-499`

### Additional data generation

None.

## C. Base CVRP

### Official dataset source

Use directly:

- `EasyNCO/data/datasets/nss_varying/nss_cvrptrain_varying.pt`

### Size diversity

Already provided by `NSS`:

- effective size range `50-499`

### Additional data generation

None.

## D. ATSP

### Generation rule

Call existing `ATSPGenerator` externally at several fixed scales.

### Official scale mix

Create a 10k dataset using:

- `2,500` instances at `n=20`
- `3,500` instances at `n=50`
- `4,000` instances at `n=100`

### File layout

Recommended shard layout:

- `EasyNCO/data/datasets/offline_init_v3/atsp/atsp20_nums2500.pt`
- `EasyNCO/data/datasets/offline_init_v3/atsp/atsp50_nums3500.pt`
- `EasyNCO/data/datasets/offline_init_v3/atsp/atsp100_nums4000.pt`
- `EasyNCO/data/datasets/offline_init_v3/atsp/manifest.json`

### Why this is valid under zero-code-change

- `ATSPGenerator` already returns tensors that the current ATSP loader accepts
- `eval.py` can already read those `.pt` tensors directly

## E. The 16 CVRP Variants

### Variant set

Use the current standard family:

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

### Generation rule

Call existing `MVRPGenerator` externally, one variant at a time.

### Official scale mix

Because the local overlapping checkpoints are at `50` and `100`, the first real release should use:

- `5,000` instances at `n=50`
- `5,000` instances at `n=100`

This still satisfies the user's requirement of size diversity while maximizing real label overlap.

### File layout

For each variant:

- `EasyNCO/data/datasets/offline_init_v3/mvrp/<variant_lower>/<variant_lower>50_nums5000.pkl`
- `EasyNCO/data/datasets/offline_init_v3/mvrp/<variant_lower>/<variant_lower>100_nums5000.pkl`
- `EasyNCO/data/datasets/offline_init_v3/mvrp/<variant_lower>/manifest.json`

### Format rule

Save exactly the file formats already accepted by `customized_vrp_loader`:

- `.pkl`
- tuple layout matching the variant type

This format is already exemplified in:

- `EasyNCO/data/datasets/offline_init_v1/mvrp/`
- `9Reld/final/reld-nco/Multi-Task/data/`

### What kind of diversity we truly get here

Under zero-code-change, MVRP variant diversity comes from:

- scale diversity
- variant diversity
- route/open/backhaul/length/TW attribute diversity
- random seed diversity

but **not** from multiple geometry families like Gaussian vs cluster, because current `MVRPGenerator` does not expose those modes.

This limitation should be stated explicitly.

## F. PCTSP

### Data generation rule

Call existing `PCTSPGenerator` externally at several fixed scales.

### Recommended data-only scale mix

If the user still wants a 10k PCTSP dataset prepared now, use:

- `2,000` instances at `n=20`
- `2,000` instances at `n=100`
- `6,000` instances at `n=500`

Why this mix:

- it gives real size diversity
- it stays close to existing local checkpoint support
- it emphasizes `500`, the only scale where the local PCTSP method pool overlaps best

### File layout

- `EasyNCO/data/datasets/offline_init_v3/pctsp/pctsp20_nums2000.pt`
- `EasyNCO/data/datasets/offline_init_v3/pctsp/pctsp100_nums2000.pt`
- `EasyNCO/data/datasets/offline_init_v3/pctsp/pctsp500_nums6000.pt`
- `EasyNCO/data/datasets/offline_init_v3/pctsp/manifest.json`

### Real status

Prepare the dataset now if desired, but do **not** count it as part of the first selector label release.

## Real Label-Running Plan

## 1. Label Unit

The label candidate unit is:

- `solver entry = method + checkpoint + decoding regime`

This stays true in V3.

## 2. First Real Selector-Label Release

The official first release should include only problems with real multi-method init-only competition:

- `TSP`
- `CVRP`
- `ATSP`
- `16` CVRP variants

It should exclude:

- `PCTSP`

from the selector release itself.

## 3. TSP Label Plan

### Dataset

- `nss_tsptrain_varying.pt`

### Methods that can be batch-run immediately

Using discovered local base runs, the ready pool is:

- `pointerformer`
- `invit`
- `elg`
- `lehd`
- `icam`
- `lih`
- `dact`
- `udc`
- `omni`
- `difusco`
- `t2t`
- `glop`

### Official execution path

Use existing:

- `EasyNCO/run_nss_eval_suite.py`

This is the cleanest true batch path because it already supports:

- varying `NSS` datasets
- graph/static scale splitting
- per-instance export

### Actual launch pattern

```bash
cd /public/home/zhoucl/shiys/EasyNCO
PYTHONPATH=/public/home/zhoucl/shiys python run_nss_eval_suite.py \
  --cuda [0] \
  --datasets TSPtrain
```

If needed, restrict methods explicitly:

```bash
PYTHONPATH=/public/home/zhoucl/shiys python run_nss_eval_suite.py \
  --cuda [0] \
  --datasets TSPtrain \
  --methods pointerformer,invit,elg,lehd,icam,lih,dact,udc,omni,difusco,t2t,glop
```

## 4. Base CVRP Label Plan

### Dataset

- `nss_cvrptrain_varying.pt`

### Methods that can be batch-run immediately

Using discovered local base runs, the ready pool is:

- `invit`
- `elg`
- `lehd`
- `icam`
- `lih`
- `dact`
- `udc`
- `omni`

`glop_cvrp` should only be added if a stable base run is first fixed externally.

### Official execution path

Also use:

- `EasyNCO/run_nss_eval_suite.py`

### Actual launch pattern

```bash
cd /public/home/zhoucl/shiys/EasyNCO
PYTHONPATH=/public/home/zhoucl/shiys python run_nss_eval_suite.py \
  --cuda [0] \
  --datasets CVRPtrain \
  --methods invit,elg,lehd,icam,lih,dact,udc,omni
```

## 5. ATSP Label Plan

### Dataset shards

- `atsp20_nums2500.pt`
- `atsp50_nums3500.pt`
- `atsp100_nums4000.pt`

### Official method pool

- `matnet`
- `matpoenet`
- `glop`

### Execution rule

Run one job per:

- method
- scale shard

So the full ATSP release is:

- `3 methods x 3 scales = 9 jobs`

### Command templates

`MatNet`:

```bash
python eval.py \
  settings=matnet_settings \
  mode=test \
  model=matnet \
  problem=atsp \
  scale=50 \
  batch_size=4 \
  episodes=3500 \
  decoder_strategy=greedy \
  cuda=[0] \
  test_data_path=offline_init_v3/atsp/atsp50_nums3500.pt \
  settings.test_loader.model_dirpath=pretrained/matnet_pretrain \
  settings.test_loader.model_filename=matnet_atsp50.ckpt \
  dir=results/label_v3/matnet_atsp/scale_50
```

`MatPOENet`:

```bash
python eval.py \
  settings=matpoenet_settings \
  mode=test \
  model=matpoenet \
  problem=atsp \
  scale=100 \
  batch_size=4 \
  episodes=4000 \
  decoder_strategy=greedy \
  cuda=[0] \
  test_data_path=offline_init_v3/atsp/atsp100_nums4000.pt \
  dir=results/label_v3/matpoenet_atsp/scale_100
```

`GLOP` init-only:

```bash
python eval.py \
  settings=glop_settings \
  mode=test \
  model=glop \
  problem=atsp \
  scale=100 \
  batch_size=1 \
  episodes=4000 \
  decoder_strategy=greedy \
  cuda=[0] \
  test_data_path=offline_init_v3/atsp/atsp100_nums4000.pt \
  settings.iteration._target_=EasyNCO.neural_solvers.pipeline.NoIteration \
  settings.test_loader.model_dirpath=pretrained/glop \
  settings.test_loader.model_filename=glop_policy_atsp.pt \
  dir=results/label_v3/glop_atsp/scale_100
```

## 6. The 16 CVRP-Variant Label Plan

### Dataset shards

For each variant:

- one `50`-scale shard with `5,000` instances
- one `100`-scale shard with `5,000` instances

### Official method pool

- `mtpomo`
- `mvmoe`

### Checkpoint preparation

Before running labels, unpack:

- `EasyNCO/pretrained/mtpomo/mtpomo.zip`
- `EasyNCO/pretrained/mvmoe/mvmoe.zip`

### Execution rule

Run one job per:

- variant
- method
- scale

So the full release is:

- `16 variants x 2 methods x 2 scales = 64 jobs`

### Command templates

`MTPOMO`:

```bash
python eval.py \
  settings=mtpomo_settings \
  mode=test \
  model=mtpomo \
  problem=OVRP \
  scale=50 \
  batch_size=64 \
  episodes=5000 \
  decoder_strategy=greedy \
  cuda=[0] \
  test_data_path=offline_init_v3/mvrp/ovrp/ovrp50_nums5000.pkl \
  settings.test_loader.model_dirpath=pretrained/mtpomo/mtpomo \
  settings.test_loader.model_filename=mtpomo_mvrp_50.ckpt \
  dir=results/label_v3/mtpomo_ovrp/scale_50
```

`MVMoE`:

```bash
python eval.py \
  settings=mvmoe_settings \
  mode=test \
  model=mvmoe \
  problem=OVRP \
  scale=100 \
  batch_size=64 \
  episodes=5000 \
  decoder_strategy=greedy \
  cuda=[0] \
  test_data_path=offline_init_v3/mvrp/ovrp/ovrp100_nums5000.pkl \
  settings.test_loader.model_dirpath=pretrained/mvmoe/mvmoe \
  settings.test_loader.model_filename=mvmoe_mvrp100.ckpt \
  dir=results/label_v3/mvmoe_ovrp/scale_100
```

## 7. PCTSP Reality Check

### Data

PCTSP data can be generated now via:

- `PCTSPGenerator`

### Labels

Under strict init-only:

- `DeepACO` is not feasible
- `GLOP` remains feasible

So the honest plan is:

- **do not** include PCTSP in the first selector label release
- optionally run a `GLOP`-only archive for later use

If PCTSP must be included in selector training, then one of the following assumptions must change:

1. relax strict init-only
2. allow non-EasyNCO methods
3. allow EasyNCO code modification

## Merge Plan

Every job already writes:

- `instance_results.jsonl`

The merge should happen outside `EasyNCO` in two stages.

### Stage 1. Shard-level merge

For each solver entry:

- concatenate all shard outputs
- attach shard metadata:
  - problem
  - scale
  - dataset shard path

### Stage 2. Problem-level label table

For each instance, build:

- `instance_id`
- `problem`
- `problem_variant`
- `scale`
- `solver_entry`
- `no_aug_score`
- `aug_score`
- `status`

Then compute:

- `feasible_mask`
- `best_solver_entry`
- per-instance ranks
- relative regret to best observed solver entry

## Final Recommendation

Under the user's actual constraints, the correct official plan is:

1. `TSP/CVRP`: use existing `NSS` varying datasets and batch-run labels with `run_nss_eval_suite.py`
2. `ATSP`: generate fixed-scale shards by calling `ATSPGenerator` externally and label them with `eval.py`
3. `16` CVRP variants: generate fixed-scale shards by calling `MVRPGenerator` externally and label them with `eval.py`
4. `PCTSP`: prepare data if needed, but keep it out of the first selector label release under strict init-only

This is the first plan that is both:

- faithful to the user's multi-scale requirement
- and executable without modifying `EasyNCO`

## Main Sources

Local sources:

- `/public/home/zhoucl/shiys/EasyNCO/data/main.py`
- `/public/home/zhoucl/shiys/EasyNCO/data/TSPGenerator.py`
- `/public/home/zhoucl/shiys/EasyNCO/data/CVRPGenerator.py`
- `/public/home/zhoucl/shiys/EasyNCO/data/ATSPGenerator.py`
- `/public/home/zhoucl/shiys/EasyNCO/data/PCTSPGenerator.py`
- `/public/home/zhoucl/shiys/EasyNCO/data/MVRPGenerator.py`
- `/public/home/zhoucl/shiys/EasyNCO/eval.py`
- `/public/home/zhoucl/shiys/EasyNCO/phases/rl/ar_reinforce.py`
- `/public/home/zhoucl/shiys/EasyNCO/run_nss_eval_suite.py`
- `/public/home/zhoucl/shiys/EasyNCO/neural_solvers/methods/deepaco/initialization.py`
- `/public/home/zhoucl/shiys/EasyNCO/neural_solvers/methods/deepaco/iteration.py`
- `/public/home/zhoucl/shiys/EasyNCO/data/datasets/nss_varying/nss_manifest.json`
- `/public/home/zhoucl/shiys/9nss论文/neural-solver-selection/datasets/generate_data.py`
- `/public/home/zhoucl/shiys/平台方法统计统计.md`
- `/public/home/zhoucl/shiys/9Reld/final/reld-nco/Multi-Task/data/`
