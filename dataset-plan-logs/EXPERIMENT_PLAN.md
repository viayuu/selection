# Dataset and Label Experiment Plan

**Goal**: produce datasets and initialization-only labels that directly support the next supervised-learning phase.

## 1. Workstream A: Coverage Audit

### A1. Freeze the exact problem list

- `tsp`
- `cvrp`
- `atsp`
- `pctsp`
- `16` CVRP variants:
  - `CVRP`
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

### A2. Build problem-method coverage matrix

For each problem, annotate:

- supported methods from `平台方法统计统计.md`
- settings file
- checkpoint existence
- whether init-only is already natural
- whether custom label-only config is needed

### A3. Output

- `dataset-plan-logs/problem_method_coverage.md`
- optional CSV version for scripting

## 2. Workstream B: Dataset Generation

## B1. TSP / CVRP

- do not regenerate
- use `NSS` varying datasets as canonical source

## B2. 16 CVRP variants

### Implementation

- extend `EasyNCO/data/MVRPGenerator.py`
- add coordinate distribution support by reusing existing `data_utils`

### Per-variant count

- `10,000` instances each

### Distribution mix

- uniform: 40%
- cluster: 20%
- gaussian: 15%
- uniform_rectangle: 10%
- explosion: 10%
- implosion: 5%

## B3. PCTSP

### Implementation

- extend `EasyNCO/data/PCTSPGenerator.py`

### Dataset size

- `10,000` instances at `n=100`

### Regimes

- standard prize/penalty: 40%
- high-prize low-penalty: 30%
- low-prize high-penalty: 30%

### Coordinate mix

- same distribution mix as CVRP variants

## B4. ATSP

### Implementation

- keep current `ATSPGenerator` as one sub-family
- add two structured sub-families in `EasyNCO/data`

### Dataset composition

- random metric-closure: 50%
- Euclidean asymmetric skew: 25%
- clustered asymmetric skew: 25%

### Dataset size

- `10,000` instances at `n=100`

## 3. Workstream C: Labeling

## C1. Label config preparation

For each method that needs modification:

- create `settings/<method>_label_initonly.yaml`

### Rules

- set `iteration` to `NoIteration`, or equivalent zero-step behavior
- preserve the original initialization path
- preserve current checkpoint path

## C2. Labeling command pattern

Use `EasyNCO/eval.py` with:

- `mode=test`
- proper `settings=...`
- `problem=...`
- `test_data_path=...`
- label-only config

Expected per-run artifact:

- `instance_results.jsonl`

## C3. Label merge

For each problem dataset, merge all method outputs into:

- one per-instance aggregated label table

Fields:

- instance id
- problem
- feasible methods
- no-aug score per method
- aug score per method
- runtime if available
- best method
- ranks
- normalized regret

## 4. Run Order

### Step 1. coverage table

- required before all else

### Step 2. tiny smoke datasets

- `ATSP`: 200
- `PCTSP`: 200
- `CVRP`: 200
- `4` representative variants: 200 each

### Step 3. tiny smoke labels

- 2-3 methods per problem

### Step 4. full 10k generation

- only after smoke passes

### Step 5. full label generation

- run in tmux
- split by problem and method

## 5. Decision Gates

### Gate 1

If `ATSP` and `PCTSP` generators are not diverse enough after the first extension, pause full labeling and improve generation first.

### Gate 2

If per-instance export fails for a method, remove that method from v1 rather than blocking the entire dataset release.

### Gate 3

If full-zoo labeling becomes too expensive, freeze a smaller high-overlap zoo for the first supervised-learning paper.
