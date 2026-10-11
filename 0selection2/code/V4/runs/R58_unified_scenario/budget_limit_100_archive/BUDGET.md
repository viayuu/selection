# R58 Whole-Project Budget

`G = S_actual + L_reference + H63 + 3h + 2h + 0.10 * L_reference < 100h`

Whole-plan evidence complete: True. Decision: `skip_whole_plan_at_or_over_100h`.

Whole reference: ~114.2 GPUh. RF family: `normal4`.

| Component | GPUh |
|---|---:|
| actual_spent | 3.33 |
| remaining_label_reference | 95.28 |
| measured63_session_overhead | 1.109 |
| baseline_preparation_evaluation | 3 |
| otherwise_unaccounted_overhead | 2 |
| label_contingency | 9.528 |

H63 is charged ONCE: removed from the refined reference before adding the three-times measured session overhead. The65 unchanged estimates retain current preflight conservative references. No unmeasured parallel speedup is assumed.


The3h baseline and2h otherwise-unaccounted overhead are fixed allowances, not measured R58 completion times. The10% label contingency and timing references are not confidence bounds or guarantees. Validation/test sizes are estimated from TRAIN.

The preflight-only diagnostics below are NOT a whole-project approval.

# R58 Fresh-Solve Budget

**Budget measurement only. No performance claim or launch authorization.**

Snapshot: 2026-10-10T21:32:35.740884+00:00.

- Current passing deployments: 128/128.
- Timing coverage: 128/128; all-deployment estimate complete: True.
- Current-profile fresh-solve subtotal: ~128 GPUh (>100h).
- Conservative reference for the same subset: ~173 GPUh (>100h); not a proven bound.
- Full projection status: `>100h`. Decision: `skip_full_fresh_solve_projected_over_100h`.
- Fixed counts per deployment: 10000 TRAIN + 1000 validation + 1000 test; 1,536,000 evaluations over the full roster.

## Assumptions And Limits

- Budget measurement, not an experiment or performance claim; no solves or training are launched.
- Only original TRAIN dataset.pkl files are opened. No validation/test inputs or cost files are read.
- Validation/test counts are fixed at 1000 each; their size distributions are ESTIMATED FROM TRAIN.
- Only passing TRAIN profiles matching current implementation, contract and TRAIN input hashes contribute.
- Runtime environments are recorded as measurement provenance, not compared to this CPU reporting process. Deployment lock/release qualification is a separate gate.
- Singletons: linear interpolation of results.seconds by true size (maximum of repeats at each size). The conservative reference uses the larger adjacent endpoint. No extrapolation outside observed sizes.
- RF-family results.seconds are warm timings; first_call_seconds NEVER contributes. Other methods may retain first-call overhead in results.seconds.
- Width16: only a measured full batch at maximum TRAIN size is used. Charge that whole batch for every size-homogeneous batch, including tails; val/test use TRAIN proportions with batch counts rounded up. This is a worst-scale, upper-ish POINT estimate, not a measured mean or a proven upper bound. No speedup is inferred from singleton timings.
- Excludes training, deferred initialization/checkpoint loading, dataset/label IO, and outer CPU route validation. Recorded backend timings can already include internal CPU checks; these are not subtracted.
- Serial solver elapsed time is expressed as single-GPU occupancy hours, not measured kernel-active time. No scaling to other GPUs or parallel speedup is assumed; no new device measurement is made.
- Sparse timings have no confidence interval or hard budget guarantee. Conservative references are not bounds. JSON retains arithmetic precision for audit, not timing accuracy.
- Preflight may still be running: this is a timestamped snapshot. Rerun after final preflight.

## Training Context (Excluded)

Observed existing R45A history: 40 epochs at about 114 seconds of optimization per epoch (about 1.3 GPUh, excluding evaluation). Training under 3h on a 3090 is an expectation, not a measured R58 training budget. This context is NOT added to the solve projection. Even a label-solve estimate <=100h does not certify total R58 work <=100h.

## Original TRAIN Size Distributions

| Problem | Count | True size range | Distinct sizes |
|---|---:|---:|---:|
| TSP | 10000 | 50..499 | 450 |
| CVRP | 10000 | 50..499 | 450 |
| ATSP | 10000 | 20..100 | 81 |
| OVRP | 10000 | 50..100 | 51 |
| VRPB | 10000 | 50..100 | 51 |
| VRPL | 10000 | 50..100 | 51 |
| VRPTW | 10000 | 50..100 | 51 |
| OVRPTW | 10000 | 50..100 | 51 |
| OVRPB | 10000 | 50..100 | 51 |
| OVRPL | 10000 | 50..100 | 51 |
| VRPBL | 10000 | 50..100 | 51 |
| VRPBTW | 10000 | 50..100 | 51 |
| VRPLTW | 10000 | 50..100 | 51 |
| OVRPBL | 10000 | 50..100 | 51 |
| OVRPBTW | 10000 | 50..100 | 51 |
| OVRPLTW | 10000 | 50..100 | 51 |
| VRPBLTW | 10000 | 50..100 | 51 |
| OVRPBLTW | 10000 | 50..100 | 51 |

## Current Profile Estimates

Hours cover all three fixed-count splits; validation/test sizes are estimated from TRAIN. Stale/unpassed/missing profiles have NO current estimate. Batch rows use only one worst-scale measurement.

| Problem | Method | Status / model | Seconds/instance | Fresh GPUh | Conservative reference GPUh |
|---|---|---|---:|---:|---:|
| TSP | BQ | true_size_linear_interpolation | ~1.4 | ~4.56 | ~6.82 |
| TSP | DIFUSCO | true_size_linear_interpolation | ~1.5 | ~5.12 | ~8.16 |
| TSP | DIFUSCO500 | true_size_linear_interpolation | ~1.5 | ~5.15 | ~8.2 |
| TSP | ELG | true_size_linear_interpolation | ~0.5 | ~1.68 | ~2.47 |
| TSP | LEHD | true_size_linear_interpolation | ~1.2 | ~3.84 | ~5.79 |
| TSP | OMNI | true_size_linear_interpolation | ~0.44 | ~1.48 | ~2.38 |
| TSP | T2T | true_size_linear_interpolation | ~3.2 | ~10.8 | ~17.2 |
| TSP | T2T500 | true_size_linear_interpolation | ~3.3 | ~10.8 | ~17.3 |
| CVRP | BQ | true_size_linear_interpolation | ~1.7 | ~5.59 | ~8.24 |
| CVRP | ELG | true_size_linear_interpolation | ~0.95 | ~3.18 | ~4.57 |
| CVRP | ICAM | true_size_linear_interpolation | ~0.38 | ~1.27 | ~1.94 |
| CVRP | LEHD | true_size_linear_interpolation | ~1.2 | ~4.15 | ~5.99 |
| CVRP | MVMOE | true_size_linear_interpolation | ~0.7 | ~2.35 | ~3.42 |
| CVRP | MoSES_CaDA | true_size_linear_interpolation | ~1.1 | ~3.74 | ~5.1 |
| CVRP | MoSES_RF | true_size_linear_interpolation | ~1.1 | ~3.71 | ~5.17 |
| CVRP | OMNI | true_size_linear_interpolation | ~0.33 | ~1.09 | ~1.62 |
| CVRP | RELD_CVRP | true_size_linear_interpolation | ~0.43 | ~1.44 | ~2.3 |
| CVRP | RouteFinder | true_size_linear_interpolation | ~0.68 | ~2.28 | ~3.18 |
| ATSP | GLOP | true_size_linear_interpolation | ~0.00035 | ~0.00118 | ~0.00148 |
| ATSP | ICAM_ATSP | true_size_linear_interpolation | ~0.13 | ~0.426 | ~0.615 |
| ATSP | MATNET | true_size_linear_interpolation | ~0.26 | ~0.85 | ~1.05 |
| ATSP | MATPOENET | true_size_linear_interpolation | ~0.1 | ~0.344 | ~0.473 |
| ATSP | UNICO_MatPOENet | true_size_linear_interpolation | ~0.37 | ~1.23 | ~1.53 |
| OVRP | MTPOMO | worst_scale_batch_upper_ish_point | ~0.012 | ~0.047 | ~0.047 |
| OVRP | MVMOE | worst_scale_batch_upper_ish_point | ~0.022 | ~0.085 | ~0.085 |
| OVRP | MoSES_CaDA | true_size_linear_interpolation | ~0.4 | ~1.32 | ~1.47 |
| OVRP | MoSES_RF | true_size_linear_interpolation | ~0.34 | ~1.13 | ~1.27 |
| OVRP | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.044 | ~0.044 |
| OVRP | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.043 | ~0.043 |
| OVRP | RouteFinder | true_size_linear_interpolation | ~0.21 | ~0.708 | ~0.808 |
| VRPB | MTPOMO | worst_scale_batch_upper_ish_point | ~0.013 | ~0.049 | ~0.049 |
| VRPB | MVMOE | worst_scale_batch_upper_ish_point | ~0.02 | ~0.077 | ~0.077 |
| VRPB | MoSES_CaDA | true_size_linear_interpolation | ~0.37 | ~1.25 | ~1.39 |
| VRPB | MoSES_RF | true_size_linear_interpolation | ~0.34 | ~1.12 | ~1.26 |
| VRPB | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.012 | ~0.045 | ~0.045 |
| VRPB | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.043 | ~0.043 |
| VRPB | RouteFinder | true_size_linear_interpolation | ~0.2 | ~0.653 | ~0.747 |
| VRPL | MTPOMO | worst_scale_batch_upper_ish_point | ~0.012 | ~0.046 | ~0.046 |
| VRPL | MVMOE | worst_scale_batch_upper_ish_point | ~0.02 | ~0.075 | ~0.075 |
| VRPL | MoSES_CaDA | true_size_linear_interpolation | ~0.41 | ~1.37 | ~1.6 |
| VRPL | MoSES_RF | true_size_linear_interpolation | ~0.34 | ~1.14 | ~1.29 |
| VRPL | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.055 | ~0.055 |
| VRPL | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.043 | ~0.043 |
| VRPL | RouteFinder | true_size_linear_interpolation | ~0.2 | ~0.678 | ~0.778 |
| VRPTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.016 | ~0.061 | ~0.061 |
| VRPTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.022 | ~0.086 | ~0.086 |
| VRPTW | MoSES_CaDA | true_size_linear_interpolation | ~0.5 | ~1.66 | ~1.83 |
| VRPTW | MoSES_RF | true_size_linear_interpolation | ~0.37 | ~1.23 | ~1.39 |
| VRPTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.052 | ~0.052 |
| VRPTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.013 | ~0.05 | ~0.05 |
| VRPTW | RouteFinder | true_size_linear_interpolation | ~0.22 | ~0.741 | ~0.84 |
| OVRPTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.017 | ~0.064 | ~0.064 |
| OVRPTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.023 | ~0.09 | ~0.09 |
| OVRPTW | MoSES_CaDA | true_size_linear_interpolation | ~0.43 | ~1.44 | ~1.65 |
| OVRPTW | MoSES_RF | true_size_linear_interpolation | ~0.37 | ~1.25 | ~1.42 |
| OVRPTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.013 | ~0.052 | ~0.052 |
| OVRPTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.013 | ~0.049 | ~0.049 |
| OVRPTW | RouteFinder | true_size_linear_interpolation | ~0.22 | ~0.727 | ~0.823 |
| OVRPB | MTPOMO | worst_scale_batch_upper_ish_point | ~0.013 | ~0.05 | ~0.05 |
| OVRPB | MVMOE | worst_scale_batch_upper_ish_point | ~0.02 | ~0.078 | ~0.078 |
| OVRPB | MoSES_CaDA | true_size_linear_interpolation | ~0.39 | ~1.31 | ~1.45 |
| OVRPB | MoSES_RF | true_size_linear_interpolation | ~0.35 | ~1.15 | ~1.3 |
| OVRPB | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.012 | ~0.046 | ~0.046 |
| OVRPB | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.044 | ~0.044 |
| OVRPB | RouteFinder | true_size_linear_interpolation | ~0.2 | ~0.681 | ~0.78 |
| OVRPL | MTPOMO | worst_scale_batch_upper_ish_point | ~0.012 | ~0.048 | ~0.048 |
| OVRPL | MVMOE | worst_scale_batch_upper_ish_point | ~0.02 | ~0.078 | ~0.078 |
| OVRPL | MoSES_CaDA | true_size_linear_interpolation | ~0.39 | ~1.3 | ~1.45 |
| OVRPL | MoSES_RF | true_size_linear_interpolation | ~0.34 | ~1.15 | ~1.29 |
| OVRPL | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.012 | ~0.045 | ~0.045 |
| OVRPL | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.042 | ~0.042 |
| OVRPL | RouteFinder | true_size_linear_interpolation | ~0.21 | ~0.69 | ~0.792 |
| VRPBL | MTPOMO | worst_scale_batch_upper_ish_point | ~0.012 | ~0.047 | ~0.047 |
| VRPBL | MVMOE | worst_scale_batch_upper_ish_point | ~0.02 | ~0.077 | ~0.077 |
| VRPBL | MoSES_CaDA | true_size_linear_interpolation | ~0.38 | ~1.28 | ~1.43 |
| VRPBL | MoSES_RF | true_size_linear_interpolation | ~0.34 | ~1.15 | ~1.3 |
| VRPBL | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.012 | ~0.047 | ~0.047 |
| VRPBL | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.043 | ~0.043 |
| VRPBL | RouteFinder | true_size_linear_interpolation | ~0.2 | ~0.664 | ~0.767 |
| VRPBTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.016 | ~0.062 | ~0.062 |
| VRPBTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.023 | ~0.087 | ~0.087 |
| VRPBTW | MoSES_CaDA | true_size_linear_interpolation | ~0.43 | ~1.43 | ~1.63 |
| VRPBTW | MoSES_RF | true_size_linear_interpolation | ~0.36 | ~1.19 | ~1.35 |
| VRPBTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.055 | ~0.055 |
| VRPBTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.053 | ~0.053 |
| VRPBTW | RouteFinder | true_size_linear_interpolation | ~0.21 | ~0.705 | ~0.807 |
| VRPLTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.015 | ~0.058 | ~0.058 |
| VRPLTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.022 | ~0.083 | ~0.083 |
| VRPLTW | MoSES_CaDA | true_size_linear_interpolation | ~0.41 | ~1.38 | ~1.55 |
| VRPLTW | MoSES_RF | true_size_linear_interpolation | ~0.36 | ~1.21 | ~1.36 |
| VRPLTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.052 | ~0.052 |
| VRPLTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.013 | ~0.05 | ~0.05 |
| VRPLTW | RouteFinder | true_size_linear_interpolation | ~0.21 | ~0.711 | ~0.805 |
| OVRPBL | MTPOMO | worst_scale_batch_upper_ish_point | ~0.014 | ~0.053 | ~0.053 |
| OVRPBL | MVMOE | worst_scale_batch_upper_ish_point | ~0.02 | ~0.076 | ~0.076 |
| OVRPBL | MoSES_CaDA | true_size_linear_interpolation | ~0.38 | ~1.27 | ~1.41 |
| OVRPBL | MoSES_RF | true_size_linear_interpolation | ~0.35 | ~1.16 | ~1.31 |
| OVRPBL | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.012 | ~0.046 | ~0.046 |
| OVRPBL | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.012 | ~0.044 | ~0.044 |
| OVRPBL | RouteFinder | true_size_linear_interpolation | ~0.2 | ~0.673 | ~0.77 |
| OVRPBTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.016 | ~0.063 | ~0.063 |
| OVRPBTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.022 | ~0.086 | ~0.086 |
| OVRPBTW | MoSES_CaDA | true_size_linear_interpolation | ~0.45 | ~1.5 | ~1.68 |
| OVRPBTW | MoSES_RF | true_size_linear_interpolation | ~0.39 | ~1.3 | ~1.46 |
| OVRPBTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.055 | ~0.055 |
| OVRPBTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.052 | ~0.052 |
| OVRPBTW | RouteFinder | true_size_linear_interpolation | ~0.22 | ~0.727 | ~0.823 |
| OVRPLTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.016 | ~0.062 | ~0.062 |
| OVRPLTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.022 | ~0.087 | ~0.087 |
| OVRPLTW | MoSES_CaDA | true_size_linear_interpolation | ~0.4 | ~1.33 | ~1.48 |
| OVRPLTW | MoSES_RF | true_size_linear_interpolation | ~0.38 | ~1.26 | ~1.4 |
| OVRPLTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.013 | ~0.052 | ~0.052 |
| OVRPLTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.013 | ~0.049 | ~0.049 |
| OVRPLTW | RouteFinder | true_size_linear_interpolation | ~0.21 | ~0.696 | ~0.797 |
| VRPBLTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.017 | ~0.064 | ~0.064 |
| VRPBLTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.023 | ~0.089 | ~0.089 |
| VRPBLTW | MoSES_CaDA | true_size_linear_interpolation | ~0.41 | ~1.38 | ~1.55 |
| VRPBLTW | MoSES_RF | true_size_linear_interpolation | ~0.39 | ~1.3 | ~1.48 |
| VRPBLTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.056 | ~0.056 |
| VRPBLTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.052 | ~0.052 |
| VRPBLTW | RouteFinder | true_size_linear_interpolation | ~0.22 | ~0.726 | ~0.832 |
| OVRPBLTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.016 | ~0.063 | ~0.063 |
| OVRPBLTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.023 | ~0.088 | ~0.088 |
| OVRPBLTW | MoSES_CaDA | true_size_linear_interpolation | ~0.43 | ~1.42 | ~1.58 |
| OVRPBLTW | MoSES_RF | true_size_linear_interpolation | ~0.4 | ~1.32 | ~1.5 |
| OVRPBLTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.054 | ~0.054 |
| OVRPBLTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.053 | ~0.053 |
| OVRPBLTW | RouteFinder | true_size_linear_interpolation | ~0.22 | ~0.747 | ~0.862 |

For batch rows seconds/instance is the maximum-size full-batch observation, before tail charges.

## Missing Coverage

None: every deployment has a current passing profile and a supported timing estimate.

Audit details: sibling `budget.json` contains TRAIN histograms and paths/hashes, accepted timing observations, batch indices, per-split arithmetic, and excluded-profile reasons.

Regenerate (CPU only):

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m code.V4.r58_budget --root /public/home/shiys/0selection2/code/V4/runs/R58_unified_scenario
```
