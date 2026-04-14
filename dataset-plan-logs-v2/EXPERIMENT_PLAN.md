# Dataset and Label Experiment Plan V2

**Goal**: build the actual 10k-per-problem varying datasets and run the actual full init-only labels inside `EasyNCO`.

## 1. Workstream A: Dataset Generation

## A1. Reuse the NSS datasets directly

For:

- `tsp`
- base `cvrp`

do not regenerate.

Use:

- `EasyNCO/data/datasets/nss_varying/nss_tsptrain_varying.pt`
- `EasyNCO/data/datasets/nss_varying/nss_cvrptrain_varying.pt`

These are already the correct 10k varying datasets.

## A2. Build new varying datasets for the non-NSS problems

Create one 10k dataset each for:

- `atsp`
- `pctsp`
- `OVRP`
- `VRPB`
- `VRPL`
- `VRPTW`
- `OVRPTW`
- `OVRPB`
- `OVRPL`
- `VRPBL`
- `VRPBTW`
- `VRPLTW`
- `OVRPBL`
- `OVRPBTW`
- `OVRPLTW`
- `VRPBLTW`
- `OVRPBLTW`

Recommended file pattern:

- `EasyNCO/data/datasets/unified_varying/<problem>_train10k_varying.pt`

Recommended side products:

- `.../<problem>_manifest.json`
- `.../graph_fixed_by_scale/<problem>/...`

## A3. Per-problem size protocol

### TSP

- reuse `NSS`
- effective sizes: `50-499`

### Base CVRP

- reuse `NSS`
- effective sizes: `50-499`

### 16 CVRP variants

- `3,000` instances at `50`
- `4,500` instances at `100`
- `2,500` instances at `200`

### ATSP

- `2,500` instances at `20`
- `3,500` instances at `50`
- `4,000` instances at `100`

### PCTSP

- `3,500` instances at `100`
- `6,500` instances at `500`

## A4. Per-problem diversity protocol

### Euclidean routing families

For:

- `16` CVRP variants
- `pctsp`

sample geometry using:

- `uniform`: `20%`
- `gaussian_mixture`: `20%`
- `cluster`: `15%`
- `gaussian`: `10%`
- `uniform_rectangle`: `10%`
- `diagonal`: `10%`
- `explosion`: `7.5%`
- `implosion`: `7.5%`

### ATSP

Use three generator branches:

- random metric-closure: `40%`
- Euclidean asymmetric skew: `35%`
- clustered asymmetric skew: `25%`

### PCTSP regimes

Within each scale bucket:

- standard: `40%`
- high-prize / low-penalty: `30%`
- low-prize / high-penalty: `30%`

## A5. Generator code targets

The minimum code targets implied by this plan are:

- extend `EasyNCO/data/MVRPGenerator.py`
- extend `EasyNCO/data/PCTSPGenerator.py`
- extend `EasyNCO/data/ATSPGenerator.py`
- optionally add helper utilities for:
  - fixed-scale shard export
  - manifest export
  - generator-family metadata

## 2. Workstream B: Solver Entry Definition

## B1. Candidate unit

Define the label candidate as:

- `solver entry = method + checkpoint + decoding regime`

This should be recorded explicitly in the label metadata.

## B2. First-wave solver pools

### TSP

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
- insertion heuristics

### Base CVRP

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
- insertion heuristics

### ATSP

- `MatNet`
- `MatPOENet`
- `GLOP`

### PCTSP

- `DeepACO`
- `GLOP`

### 16 CVRP variants

- `MTPOMO`
- `MVMoE`

## B3. Exclusions for the first init-only release

- `AM`, `POMO`
  - no ready local checkpoint
- `NLNS`
  - not pure init
- full classical solvers
  - not part of this init-only release
- `DeepACO@TSP`
  - current local note says init-only does not emit a usable solution

## 3. Workstream C: Full Label Execution

## C1. Universal run rule

Every label job must satisfy:

- `mode=test`
- `iteration = NoIteration`
- per-instance export enabled
- results written to a dedicated output dir

## C2. Dynamic-varying jobs

For methods that can read varying data directly:

- run once on the full varying dataset

Use the pattern already present in:

- `EasyNCO/run_nss_eval_suite.py`

## C3. Fixed-scale shard jobs

For scale-sensitive or graph/static methods:

- run one job per fixed scale shard
- then merge the outputs back to dataset level

This is the required pattern for methods like:

- `DIFUSCO`
- `T2T`

and should also be the default pattern for the new `ATSP / PCTSP / MVRP-variant` datasets.

## C4. Full run order

### Block 1. TSP labels

Run the full TSP pool on the `NSS` TSP varying dataset.

### Block 2. Base CVRP labels

Run the full CVRP pool on the `NSS` CVRP varying dataset.

### Block 3. ATSP labels

Run all ATSP solver entries on the `20 / 50 / 100` shards.

### Block 4. PCTSP labels

Run all PCTSP solver entries on the `100 / 500` shards.

### Block 5. Variant-CVRP labels

For each of the `16` variants, run:

- `MTPOMO`
- `MVMoE`

on the `50 / 100 / 200` shards.

## C5. Job output contract

Each job must produce:

- `instance_results.jsonl`

and preferably also preserve:

- hydra config snapshot
- log file
- checkpoint override info

## 4. Workstream D: Label Merge

## D1. Per-solver-entry merge

For every solver entry:

- merge all shards
- restore global indices
- produce one dataset-level label file

## D2. Final dataset-level supervision table

For each instance, aggregate:

- `instance_id`
- `problem`
- `problem_variant`
- `scale`
- `distribution`
- `generator_family`
- `regime`
- `solver_entry`
- `no_aug_score`
- `aug_score`
- `status`

Then compute:

- `feasible_solver_entries`
- `best_solver_entry_by_aug_score`
- rank list
- gap to best observed score

## D3. Default training target

Store both:

- top-1 best entry
- full ranking signal

so the future model can choose between:

- masked cross-entropy
- pairwise ranking
- listwise ranking

## 5. Workstream E: Completion Criteria

The V2 release is complete when:

1. every target problem has one 10k dataset
2. every dataset has a manifest and fixed-scale shards
3. every selected solver entry has been run init-only on its assigned shards
4. every completed run has exported per-instance results
5. every problem has a merged supervision table with instance-level feasible masks

## 6. Official Next Step After This Plan

The next step is not another abstract planning round.

The next step is to implement:

1. the new varying generators
2. the full label-run scripts
3. the merge scripts for the final supervision tables
