# Claims From Results

## Supported Claims

1. A simple unified supervised selector can work across many routing problems with one shared model.
- Evidence:
  - Full-data `true baseline` reaches `val_macro_problem_oracle_ratio = 1.003591` and `test_macro_problem_oracle_ratio = 1.003554`.
  - It also achieves negative macro gap vs best single on full test: `-0.018162`.
  - The full-test paired bootstrap CI against best single is entirely negative: `[-0.020755, -0.015696]`.

2. The unified selector beats best-single baselines on many standard validation and test splits.
- Evidence:
  - On full test, the `true baseline` has `test_macro_problem_gap_vs_best_single = -0.018162`.
  - Per-problem full-test gains are visible on problems such as `ovrpbl`, `ovrpb`, `vrpbl`, and `cvrp`.

3. Richer conditioning and balancing do not yield a robust full-scale win over the simple unified selector.
- Evidence:
  - At medium scale, `balanced+stats` was slightly better overall.
  - At full scale, `true baseline` is better on `val` and `test`, while `balanced+stats` keeps only a very small benchmark edge.
  - The full-test paired bootstrap CI for `true baseline - balanced+stats` is entirely negative: `[-0.004537, -0.001494]`.

## Partially Supported Claims

1. Problem-balanced training may help in some settings.
- Evidence:
  - Medium-scale experiments showed that `balance only` and `balanced+stats` could help relative to the plain baseline early or on selected splits.
  - However, the finished medium and full comparisons do not support making balancing the headline method contribution.

2. Richer conditioning may interact with balancing.
- Evidence:
  - The bundled variant was strongest on the medium benchmark and remained close overall.
  - But `stats-only` and `balanced+stats` do not provide a stable full-scale gain over the simple baseline, so this should remain an ablation-level observation.

## Unsupported Claims

1. Richer conditioning clearly improves unified cross-problem solver selection.
- Why unsupported:
  - Final full-data results contradict a strong version of this claim.
  - The simpler `true baseline` is better on full `val` and `test`.

2. The current selector solves benchmark / OOD generalization.
- Why unsupported:
  - Both top models remain worse than the best single solver on benchmark `CVRP/LIB`.
  - Full benchmark macro gaps are still positive:
    - `true baseline`: `0.630013`
    - `balanced+stats`: `0.623931`
  - The benchmark paired bootstrap CI for `true baseline - best single` is entirely positive: `[+0.367633, +1.482379]`.

## Safest Paper Narrative

The safest final narrative is:
- one shared unified selector already works surprisingly well across 18 routing problems
- the key ingredients are unified representation, global solver pooling with feasible masking, and masked ranking loss
- richer conditioning and balancing are informative ablations, not the core winning method
- benchmark `CVRP/LIB` shift remains the main open failure mode
- the `CVRP/LIB` gap grows with instance size, so the benchmark limitation can be described concretely rather than vaguely
