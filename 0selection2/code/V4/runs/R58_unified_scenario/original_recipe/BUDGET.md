# R58 Fresh-Solve Budget

**Budget measurement only. No performance claim or launch authorization.**

Snapshot: 2026-10-10T17:48:06.965559+00:00.

- Current passing deployments: 128/128.
- Timing coverage: 128/128; all-deployment estimate complete: True.
- Current-profile fresh-solve subtotal: ~182 GPUh (>100h).
- Conservative reference for the same subset: ~258 GPUh (>100h); not a proven bound.
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
| TSP | BQ | true_size_linear_interpolation | ~1.4 | ~4.6 | ~6.87 |
| TSP | DIFUSCO | true_size_linear_interpolation | ~3.7 | ~12.2 | ~19.3 |
| TSP | DIFUSCO500 | true_size_linear_interpolation | ~3.7 | ~12.3 | ~19.4 |
| TSP | ELG | true_size_linear_interpolation | ~0.52 | ~1.75 | ~2.57 |
| TSP | LEHD | true_size_linear_interpolation | ~1.2 | ~3.84 | ~5.78 |
| TSP | OMNI | true_size_linear_interpolation | ~0.44 | ~1.47 | ~2.35 |
| TSP | T2T | true_size_linear_interpolation | ~8.8 | ~29.3 | ~46.6 |
| TSP | T2T500 | true_size_linear_interpolation | ~8.8 | ~29.4 | ~46.7 |
| CVRP | BQ | true_size_linear_interpolation | ~1.7 | ~5.58 | ~8.21 |
| CVRP | ELG | true_size_linear_interpolation | ~0.95 | ~3.16 | ~4.54 |
| CVRP | ICAM | true_size_linear_interpolation | ~0.38 | ~1.26 | ~1.92 |
| CVRP | LEHD | true_size_linear_interpolation | ~1.2 | ~4.06 | ~5.87 |
| CVRP | MVMOE | true_size_linear_interpolation | ~0.71 | ~2.38 | ~3.46 |
| CVRP | MoSES_CaDA | true_size_linear_interpolation | ~1.2 | ~4.01 | ~5.69 |
| CVRP | MoSES_RF | true_size_linear_interpolation | ~1.2 | ~4.13 | ~5.93 |
| CVRP | OMNI | true_size_linear_interpolation | ~0.34 | ~1.12 | ~1.67 |
| CVRP | RELD_CVRP | true_size_linear_interpolation | ~0.42 | ~1.41 | ~2.26 |
| CVRP | RouteFinder | true_size_linear_interpolation | ~0.85 | ~2.84 | ~4.18 |
| ATSP | GLOP | true_size_linear_interpolation | ~0.00035 | ~0.00115 | ~0.00141 |
| ATSP | ICAM_ATSP | true_size_linear_interpolation | ~0.12 | ~0.387 | ~0.539 |
| ATSP | MATNET | true_size_linear_interpolation | ~0.26 | ~0.855 | ~1.06 |
| ATSP | MATPOENET | true_size_linear_interpolation | ~0.1 | ~0.342 | ~0.469 |
| ATSP | UNICO_MatPOENet | true_size_linear_interpolation | ~0.38 | ~1.27 | ~1.56 |
| OVRP | MTPOMO | worst_scale_batch_upper_ish_point | ~0.013 | ~0.048 | ~0.048 |
| OVRP | MVMOE | worst_scale_batch_upper_ish_point | ~0.025 | ~0.097 | ~0.097 |
| OVRP | MoSES_CaDA | true_size_linear_interpolation | ~0.4 | ~1.32 | ~1.48 |
| OVRP | MoSES_RF | true_size_linear_interpolation | ~0.36 | ~1.2 | ~1.35 |
| OVRP | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.044 | ~0.044 |
| OVRP | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.043 | ~0.043 |
| OVRP | RouteFinder | true_size_linear_interpolation | ~0.21 | ~0.705 | ~0.806 |
| VRPB | MTPOMO | worst_scale_batch_upper_ish_point | ~0.013 | ~0.05 | ~0.05 |
| VRPB | MVMOE | worst_scale_batch_upper_ish_point | ~0.02 | ~0.077 | ~0.077 |
| VRPB | MoSES_CaDA | true_size_linear_interpolation | ~0.39 | ~1.29 | ~1.43 |
| VRPB | MoSES_RF | true_size_linear_interpolation | ~0.34 | ~1.14 | ~1.29 |
| VRPB | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.012 | ~0.046 | ~0.046 |
| VRPB | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.043 | ~0.043 |
| VRPB | RouteFinder | true_size_linear_interpolation | ~0.2 | ~0.681 | ~0.78 |
| VRPL | MTPOMO | worst_scale_batch_upper_ish_point | ~0.012 | ~0.046 | ~0.046 |
| VRPL | MVMOE | worst_scale_batch_upper_ish_point | ~0.02 | ~0.076 | ~0.076 |
| VRPL | MoSES_CaDA | true_size_linear_interpolation | ~0.39 | ~1.3 | ~1.45 |
| VRPL | MoSES_RF | true_size_linear_interpolation | ~0.36 | ~1.18 | ~1.34 |
| VRPL | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.044 | ~0.044 |
| VRPL | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.041 | ~0.041 |
| VRPL | RouteFinder | true_size_linear_interpolation | ~0.2 | ~0.675 | ~0.776 |
| VRPTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.016 | ~0.06 | ~0.06 |
| VRPTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.022 | ~0.086 | ~0.086 |
| VRPTW | MoSES_CaDA | true_size_linear_interpolation | ~0.43 | ~1.43 | ~1.6 |
| VRPTW | MoSES_RF | true_size_linear_interpolation | ~0.39 | ~1.3 | ~1.47 |
| VRPTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.013 | ~0.052 | ~0.052 |
| VRPTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.013 | ~0.051 | ~0.051 |
| VRPTW | RouteFinder | true_size_linear_interpolation | ~0.23 | ~0.783 | ~0.892 |
| OVRPTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.016 | ~0.063 | ~0.063 |
| OVRPTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.024 | ~0.092 | ~0.092 |
| OVRPTW | MoSES_CaDA | true_size_linear_interpolation | ~0.46 | ~1.54 | ~1.77 |
| OVRPTW | MoSES_RF | true_size_linear_interpolation | ~0.39 | ~1.31 | ~1.5 |
| OVRPTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.013 | ~0.051 | ~0.051 |
| OVRPTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.013 | ~0.05 | ~0.05 |
| OVRPTW | RouteFinder | true_size_linear_interpolation | ~0.23 | ~0.783 | ~0.895 |
| OVRPB | MTPOMO | worst_scale_batch_upper_ish_point | ~0.012 | ~0.048 | ~0.048 |
| OVRPB | MVMOE | worst_scale_batch_upper_ish_point | ~0.02 | ~0.078 | ~0.078 |
| OVRPB | MoSES_CaDA | true_size_linear_interpolation | ~0.39 | ~1.29 | ~1.43 |
| OVRPB | MoSES_RF | true_size_linear_interpolation | ~0.36 | ~1.19 | ~1.35 |
| OVRPB | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.012 | ~0.047 | ~0.047 |
| OVRPB | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.044 | ~0.044 |
| OVRPB | RouteFinder | true_size_linear_interpolation | ~0.2 | ~0.677 | ~0.777 |
| OVRPL | MTPOMO | worst_scale_batch_upper_ish_point | ~0.012 | ~0.046 | ~0.046 |
| OVRPL | MVMOE | worst_scale_batch_upper_ish_point | ~0.02 | ~0.078 | ~0.078 |
| OVRPL | MoSES_CaDA | true_size_linear_interpolation | ~0.4 | ~1.32 | ~1.47 |
| OVRPL | MoSES_RF | true_size_linear_interpolation | ~0.36 | ~1.2 | ~1.35 |
| OVRPL | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.044 | ~0.044 |
| OVRPL | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.042 | ~0.042 |
| OVRPL | RouteFinder | true_size_linear_interpolation | ~0.22 | ~0.717 | ~0.822 |
| VRPBL | MTPOMO | worst_scale_batch_upper_ish_point | ~0.013 | ~0.048 | ~0.048 |
| VRPBL | MVMOE | worst_scale_batch_upper_ish_point | ~0.02 | ~0.077 | ~0.077 |
| VRPBL | MoSES_CaDA | true_size_linear_interpolation | ~0.38 | ~1.25 | ~1.4 |
| VRPBL | MoSES_RF | true_size_linear_interpolation | ~0.34 | ~1.13 | ~1.28 |
| VRPBL | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.012 | ~0.046 | ~0.046 |
| VRPBL | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.043 | ~0.043 |
| VRPBL | RouteFinder | true_size_linear_interpolation | ~0.2 | ~0.659 | ~0.762 |
| VRPBTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.016 | ~0.06 | ~0.06 |
| VRPBTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.022 | ~0.086 | ~0.086 |
| VRPBTW | MoSES_CaDA | true_size_linear_interpolation | ~0.47 | ~1.58 | ~1.8 |
| VRPBTW | MoSES_RF | true_size_linear_interpolation | ~0.38 | ~1.28 | ~1.45 |
| VRPBTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.055 | ~0.055 |
| VRPBTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.053 | ~0.053 |
| VRPBTW | RouteFinder | true_size_linear_interpolation | ~0.23 | ~0.761 | ~0.871 |
| VRPLTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.015 | ~0.057 | ~0.057 |
| VRPLTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.021 | ~0.083 | ~0.083 |
| VRPLTW | MoSES_CaDA | true_size_linear_interpolation | ~0.42 | ~1.41 | ~1.57 |
| VRPLTW | MoSES_RF | true_size_linear_interpolation | ~0.37 | ~1.24 | ~1.39 |
| VRPLTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.053 | ~0.053 |
| VRPLTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.013 | ~0.049 | ~0.049 |
| VRPLTW | RouteFinder | true_size_linear_interpolation | ~0.23 | ~0.752 | ~0.86 |
| OVRPBL | MTPOMO | worst_scale_batch_upper_ish_point | ~0.014 | ~0.053 | ~0.053 |
| OVRPBL | MVMOE | worst_scale_batch_upper_ish_point | ~0.02 | ~0.076 | ~0.076 |
| OVRPBL | MoSES_CaDA | true_size_linear_interpolation | ~0.39 | ~1.31 | ~1.46 |
| OVRPBL | MoSES_RF | true_size_linear_interpolation | ~0.36 | ~1.19 | ~1.35 |
| OVRPBL | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.012 | ~0.047 | ~0.047 |
| OVRPBL | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.011 | ~0.043 | ~0.043 |
| OVRPBL | RouteFinder | true_size_linear_interpolation | ~0.21 | ~0.695 | ~0.797 |
| OVRPBTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.016 | ~0.063 | ~0.063 |
| OVRPBTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.022 | ~0.085 | ~0.085 |
| OVRPBTW | MoSES_CaDA | true_size_linear_interpolation | ~0.48 | ~1.59 | ~1.8 |
| OVRPBTW | MoSES_RF | true_size_linear_interpolation | ~0.41 | ~1.38 | ~1.56 |
| OVRPBTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.055 | ~0.055 |
| OVRPBTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.054 | ~0.054 |
| OVRPBTW | RouteFinder | true_size_linear_interpolation | ~0.24 | ~0.793 | ~0.91 |
| OVRPLTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.016 | ~0.062 | ~0.062 |
| OVRPLTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.022 | ~0.086 | ~0.086 |
| OVRPLTW | MoSES_CaDA | true_size_linear_interpolation | ~0.43 | ~1.44 | ~1.65 |
| OVRPLTW | MoSES_RF | true_size_linear_interpolation | ~0.38 | ~1.28 | ~1.43 |
| OVRPLTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.053 | ~0.053 |
| OVRPLTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.013 | ~0.05 | ~0.05 |
| OVRPLTW | RouteFinder | true_size_linear_interpolation | ~0.22 | ~0.728 | ~0.827 |
| VRPBLTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.016 | ~0.062 | ~0.062 |
| VRPBLTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.023 | ~0.087 | ~0.087 |
| VRPBLTW | MoSES_CaDA | true_size_linear_interpolation | ~0.43 | ~1.43 | ~1.62 |
| VRPBLTW | MoSES_RF | true_size_linear_interpolation | ~0.4 | ~1.32 | ~1.49 |
| VRPBLTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.054 | ~0.054 |
| VRPBLTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.013 | ~0.052 | ~0.052 |
| VRPBLTW | RouteFinder | true_size_linear_interpolation | ~0.23 | ~0.755 | ~0.87 |
| OVRPBLTW | MTPOMO | worst_scale_batch_upper_ish_point | ~0.017 | ~0.064 | ~0.064 |
| OVRPBLTW | MVMOE | worst_scale_batch_upper_ish_point | ~0.023 | ~0.087 | ~0.087 |
| OVRPBLTW | MoSES_CaDA | true_size_linear_interpolation | ~0.44 | ~1.47 | ~1.67 |
| OVRPBLTW | MoSES_RF | true_size_linear_interpolation | ~0.4 | ~1.32 | ~1.49 |
| OVRPBLTW | RELD_MOEL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.054 | ~0.054 |
| OVRPBLTW | RELD_MTL | worst_scale_batch_upper_ish_point | ~0.014 | ~0.053 | ~0.053 |
| OVRPBLTW | RouteFinder | true_size_linear_interpolation | ~0.23 | ~0.781 | ~0.898 |

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
