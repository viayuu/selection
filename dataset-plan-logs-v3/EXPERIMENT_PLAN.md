# Dataset and Label Experiment Plan V3

**Goal**: run the actual first multi-scale init-only label campaign with zero changes to `EasyNCO`.

## 1. Workstream A: Prepare Real Datasets

## A1. TSP / CVRP

No generation required.

Use directly:

- `nss_tsptrain_varying.pt`
- `nss_cvrptrain_varying.pt`

## A2. ATSP

Generate three shard files by calling `ATSPGenerator` externally:

- `20`: `2,500`
- `50`: `3,500`
- `100`: `4,000`

Save:

- `.pt` tensors
- one manifest JSON

## A3. 16 CVRP variants

For each variant, generate two shard files by calling `MVRPGenerator` externally:

- `50`: `5,000`
- `100`: `5,000`

Save:

- `.pkl` files in the current accepted tuple layout
- one manifest JSON per variant

## A4. PCTSP

Optional data-only preparation:

- `20`: `2,000`
- `100`: `2,000`
- `500`: `6,000`

This is not part of the first selector release.

## 2. Workstream B: Launch Real Labels

## B1. TSP

Use:

- `run_nss_eval_suite.py`

Official pool:

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

## B2. CVRP

Use:

- `run_nss_eval_suite.py`

Official pool:

- `invit`
- `elg`
- `lehd`
- `icam`
- `lih`
- `dact`
- `udc`
- `omni`

## B3. ATSP

Use:

- `eval.py`

Official pool:

- `matnet`
- `matpoenet`
- `glop`

Run count:

- `9` jobs total

## B4. 16 CVRP variants

Use:

- `eval.py`

Official pool:

- `mtpomo`
- `mvmoe`

Run count:

- `64` jobs total

## B5. PCTSP

Do not include in the first selector release.

If needed, run a single-method `glop` archive only.

## 3. Workstream C: Merge Labels

## C1. Shard merge

For each solver entry:

- merge shard `instance_results.jsonl`
- attach shard metadata

## C2. Problem merge

For each problem family:

- build the final supervision table
- compute feasible masks
- compute best entry and ranks

## 4. Completion Criteria

The campaign is complete when:

1. `TSP/CVRP` varying labels are complete
2. `ATSP` 9 jobs are complete
3. `16`-variant `MVRP` 64 jobs are complete
4. merged supervision tables exist for those problem families

## 5. Non-Goals

- changing `EasyNCO`
- making all problems use the same dataset format
- forcing `PCTSP` into the first selector release despite lacking real init-only competition
