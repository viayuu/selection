# Dataset and Label Planning Report

**Topic**: Build supervised datasets and initialization-only labels for a unified cross-problem solver selector  
**Date**: 2026-04-14  
**This file is new**: it does **not** replace `IDEA_REPORT.md` or anything under `refine-logs/`

## Executive Summary

The immediate bottleneck for the next research stage is **not model design**, but **dataset and label construction**. The right next step is to build a unified dataset-and-label pipeline that uses `EasyNCO` as much as possible, keeps `TSP/CVRP` aligned with `NSS`, and extends to `16` CVRP variants, `PCTSP`, and `ATSP` with sufficiently diverse instance generation.

The recommended plan is:

1. Reuse **NSS synthetic datasets** for `TSP` and base `CVRP`.
2. Use `EasyNCO` generators as the base for all other problems.
3. Keep the **first unified dataset at scale 100** to maximize checkpoint and method coverage.
4. Emphasize **distribution diversity**, not scale diversity, in the first dataset release.
5. Run **initialization-only labels** using `EasyNCO/eval.py` and export per-instance results to `instance_results.jsonl`.
6. Build the first label release around a **high-overlap solver zoo**, rather than the full solver list in one shot.

## Problem Scope

### 1. Problems included in the first supervised dataset program

- `tsp`
- `cvrp`
- `16` CVRP-family variants
- `pctsp`
- `atsp`

### 2. Canonical first scale

The first dataset program should fix:

- **problem size = 100**

Reason:

- `NSS` already has strong `100`-scale synthetic data for `TSP/CVRP`
- many `EasyNCO` pretrained checkpoints and settings are centered around `100`
- `ATSP`, `PCTSP`, and `MVRP`-style variants are much easier to align at this size
- this avoids exploding the labeling budget before the pipeline is stable

Scale generalization can be a later extension.

## Dataset Construction Plan

## A. TSP and base CVRP

### Requirement from user

- `TSP` and `CVRP` datasets should directly use the `NSS` paper datasets

### Plan

Use the already prepared varying NSS datasets under `EasyNCO`:

- `EasyNCO/data/datasets/nss_varying/nss_tsptrain_varying.pt`
- `EasyNCO/data/datasets/nss_varying/nss_tspval_varying.pt`
- `EasyNCO/data/datasets/nss_varying/nss_tsptest_varying.pt`
- `EasyNCO/data/datasets/nss_varying/nss_cvrptrain_varying.pt`
- `EasyNCO/data/datasets/nss_varying/nss_cvrpval_varying.pt`
- `EasyNCO/data/datasets/nss_varying/nss_cvrptest_varying.pt`

### Why this is the right choice

- directly satisfies the user's requirement
- preserves comparability with the older `NSS`-related work
- avoids unnecessary regeneration for the two most established tasks

## B. The 16 CVRP-family variants

### Recommended variant set

Use the standard `16`-variant family consistent with `MVMoE / RouteFinder / RL4CO MTVRP`:

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

These match the standard capacity/open/backhaul/length/time-window combination space that current multi-problem routing literature already uses.

### Base generator choice

Use `EasyNCO/data/MVRPGenerator.py` as the starting point, because it already supports:

- open-route
- backhaul
- route-length limit
- time windows
- combination variants

### Necessary enhancement

The current `MVRPGenerator` is structurally useful, but its random generation is too narrow:

- it currently samples coordinates essentially in one style
- for the new paper, this is not diverse enough

So the recommended implementation is:

- **extend** `MVRPGenerator` rather than replace it
- let it reuse `EasyNCO.data.data_utils` coordinate distributions already used by `TSP/CVRP`

### Distribution design for each variant

For each of the `16` variants, generate **10,000** instances at size `100` with a mixed coordinate distribution:

- `40%` uniform
- `20%` cluster
- `15%` gaussian
- `10%` uniform_rectangle
- `10%` explosion
- `5%` implosion

Rationale:

- uniform keeps compatibility with classic routing settings
- cluster and gaussian increase structure diversity
- rectangle and diagonal-like anisotropy stress geometry bias
- explosion / implosion introduce more extreme spatial patterns

### Constraint generation policy

Keep the constraint-generation logic in `MVRPGenerator` aligned with current cross-problem routing practice:

- demand scaling consistent with current generator
- backhaul ratio reuse current defaults unless manually changed
- time-window generation reuse current MVRP / RouteFinder-style logic
- route length limit reuse current MVRP defaults first

Do **not** redesign all variant constraints from scratch in v1. Reuse the current generator logic as much as possible.

## C. PCTSP

### Base generator choice

Use `EasyNCO/data/PCTSPGenerator.py` as the base.

### Why it is not enough as-is

The current generator is useful, but currently too narrow:

- coordinates are sampled in a single simple way
- prize / penalty generation lacks regime diversity

### Recommended enhancement

Extend `PCTSPGenerator` to support the same coordinate distribution menu as `TSPGenerator`:

- uniform
- cluster
- gaussian
- uniform_rectangle
- explosion
- implosion

### Prize / penalty regimes

For the first release of 10,000 instances at size 100, split into three regimes:

- `40%` standard regime
  - keep current generator logic
- `30%` high-prize / low-penalty regime
  - easier to collect more nodes
- `30%` low-prize / high-penalty regime
  - harder tradeoff, more selective tours

This is important because a single prize/penalty regime would make selector behavior too narrow.

## D. ATSP

### Base generator choice

Use `EasyNCO/data/ATSPGenerator.py` as the base.

### Why it needs extension

The current generator matches the standard random asymmetric-matrix style and is likely close to `MatNet`-style synthetic data, but it is still too distributionally narrow for the new selector setting.

### Recommended first dataset composition

Generate `10,000` ATSP instances at size `100` with three sub-families:

- `50%` random metric-closure matrices
  - directly use current `ATSPGenerator`
- `25%` Euclidean-asymmetric skew instances
  - start from random coordinates, then build directional asymmetric costs
- `25%` clustered-asymmetric instances
  - same as above, but coordinates sampled from clustered distributions

### Implementation principle

Do this by adding a **thin extension** under `EasyNCO/data`, not by bypassing the platform.

The new ATSP extension should still output `.pt` files in the same style expected by current `EasyNCO` loaders.

## Dataset Size Policy

### User requirement

- every problem dataset should contain `10,000` instances

### Recommended interpretation

Treat `10,000` as the **main training pool** for each problem family.

For the first implementation:

- main train dataset: `10,000`
- optional validation/test sets:
  - either derived later from the same protocol
  - or kept at `1,000 / 1,000` for problems where evaluation parity with NSS is useful

This keeps the user requirement intact while preserving room for proper validation later.

## Labeling Plan

## 1. Labeling principle

The labels should be generated by:

- running candidate methods through `EasyNCO`
- **initialization only**
- **no iteration**

So the target supervised signal is:

- how good a solver's **initial solution** is on each instance

## 2. Problem-method coverage source

Use:

- `/public/home/zhoucl/shiys/平台方法统计统计.md`

as the primary coverage reference.

### Important consequence

The first label run should **not** try to force every method onto every problem.

Instead:

- build a coverage matrix first
- only run supported `(problem, method)` pairs
- emit `feasible_mask` in the final label dataset

## 3. Labeling engine

Use `EasyNCO/eval.py` for all feasible neural methods and platform-integrated heuristics whenever possible.

### Existing support that is already useful

- `eval.py` instantiates env + policy + initialization + iteration using settings files
- `ARREINFORCELightning` already exports:
  - `instance_results.jsonl`
- `NoIteration` is already supported in the pipeline

This is exactly what the labeling pipeline needs.

## 4. Initialization-only enforcement

There are two cases:

### Case A. Methods whose settings already use `NoIteration`

These can be used almost directly.

### Case B. Methods whose default settings include iteration

For labeling, create **label-specific settings copies** rather than mutating the original training settings in place.

Recommended pattern:

- `settings/<method>_label_initonly.yaml`

In these files:

- force `iteration` to `EasyNCO.neural_solvers.pipeline.NoIteration`
- or set `max_steps: 0` if the method internally respects this cleanly

### Why this is important

It keeps:

- training settings clean
- label generation reproducible
- "initialization-only" semantics explicit

## 5. Output format for labels

For each `(problem, instance, method)` record, store at least:

- problem name
- dataset split / source
- global instance index
- method name
- feasible flag
- no-aug score
- aug score
- optional runtime
- optional instance name / source name

Then merge by instance into a joint supervision file containing:

- list of feasible methods
- method-to-score dict
- best method
- ranks
- regret to oracle-in-zoo

## 6. First-phase label zoo recommendation

Do **not** start with the largest possible zoo.

Start with methods that satisfy at least one of:

- good coverage across target problems
- existing checkpoints
- stable `EasyNCO` inference path

### Suggested first-wave method groups

- `TSP/CVRP` shared group:
  - `ELG`
  - `LEHD`
  - `ICAM`
  - `INVIT`
  - `OMNI`
  - insertion heuristics where platform support is clean
- `ATSP`:
  - `MatNet`
  - `MatPOENet`
  - `GLOP`
- `PCTSP`:
  - `DeepACO`
  - `GLOP`
- `16` CVRP variants:
  - `MTPOMO`
  - `MVMoE`
  - optionally `DeepACO` / `GLOP` only where truly supported by the platform path

### Methods to defer in v1

Defer methods that currently conflict with the init-only labeling goal or stable solution extraction:

- methods marked in `平台方法统计统计.md` as "只初始化效果差" can still be retained as weak baselines later, but do not need to block v1
- methods whose current init-only path does not produce a clean solution / score export should not be forced in the first labeling release

## Recommended Execution Order

### Stage A: coverage and config audit

1. build a CSV/Markdown matrix:
   - rows = problems
   - columns = methods
   - cells = supported / unsupported / uncertain
2. map each supported pair to:
   - settings file
   - checkpoint
   - whether iteration must be disabled

### Stage B: dataset smoke generation

Generate tiny pilot datasets first:

- `ATSP`: 200
- `PCTSP`: 200
- `4` representative CVRP variants: 200 each

Verify:

- format
- loading
- naming
- split reproducibility

### Stage C: label smoke run

Run 2-3 representative methods per problem on the smoke datasets.

Verify:

- `eval.py` works end-to-end
- `instance_results.jsonl` is emitted correctly
- per-instance score extraction is stable

### Stage D: full 10k generation

Only after smoke tests pass:

- generate the full `10,000` per problem
- run labeling in tmux-backed batches

## Key Risks

### Risk 1: dataset diversity is too weak

Mitigation:

- diversify distributions early
- not just uniform coordinates

### Risk 2: ATSP synthetic data is too artificial

Mitigation:

- do not rely only on random matrix closure
- add structured asymmetric families

### Risk 3: full solver zoo is too expensive or brittle

Mitigation:

- build the first release around a high-overlap mini-zoo
- expand later

### Risk 4: init-only semantics differ across methods

Mitigation:

- enforce dedicated label-only config files
- audit each method's initialization output path

## Final Recommendation

The most practical and research-aligned next move is:

1. keep `TSP/CVRP` on `NSS`
2. generate `ATSP/PCTSP/16-CVRP-variants` with `EasyNCO`-centric extensions
3. freeze size `100`
4. prioritize distribution diversity
5. run a small init-only label smoke test before any full 10k job

## Key References

- Local:
  - `/public/home/zhoucl/shiys/平台方法统计统计.md`
  - `/public/home/zhoucl/shiys/EasyNCO/data/TSPGenerator.py`
  - `/public/home/zhoucl/shiys/EasyNCO/data/CVRPGenerator.py`
  - `/public/home/zhoucl/shiys/EasyNCO/data/ATSPGenerator.py`
  - `/public/home/zhoucl/shiys/EasyNCO/data/PCTSPGenerator.py`
  - `/public/home/zhoucl/shiys/EasyNCO/data/MVRPGenerator.py`
  - `/public/home/zhoucl/shiys/EasyNCO/eval.py`
  - `/public/home/zhoucl/shiys/EasyNCO/phases/rl/ar_reinforce.py`
- External:
  - NSS: https://arxiv.org/abs/2410.09693
  - RouteFinder: https://arxiv.org/abs/2406.15007
  - RL4CO MTVRP docs: https://rl4co.readthedocs.io/en/stable/docs/content/api/envs/routing/
  - MMoE: https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/
  - STAR: https://arxiv.org/abs/2101.11427
