# R77 Final Results

Both arms completed49680 successful updates (2760/problem), selected one unique
common-SBS VAL best, and performed exactly one full TEST each. A/B selected
updates are16146/34776. Both TEST/analysis pipelines and both Slurm steps exited
successfully. SUCCESS_RECEIPT.json records both approved4090s idle with no compute
processes at2026-10-14T20:07:24Z; no GPU was reassigned by this runner.

## Historical Reference And Matched Pair

All three rows use the new-only-B single-reld_mt seed2 release and the same FULL
TRAIN-selected SBS solver identities. Native Top1 is strict winner accuracy;
tie-aware accuracy is a separate column in finalcomparison.csv.

| Run | Unique TRAIN/problem | Global batch | Completed updates | Selected epoch / update | TEST Native Top1 % | TEST CE | TEST regret % | TEST vs common TRAIN SBS % |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Historical R76, new data / old batch | 70114 | 9856 | 3744 | 18 / 2592 | 69.4395 | 0.750073 | 0.464144 | -0.140434 |
| Matched R77A, metadata-stratified subset | 10000 | 1024 | 49680 | 13 / 16146 | 70.5691 | 0.719735 | 0.412741 | -0.175897 |
| Matched R77B, full TRAIN | 70114 | 1024 | 49680 | 28 / 34776 | 71.2252 | 0.707470 | 0.394014 | -0.194048 |

Only R77A/B are the strict sample-size pair: same fresh seed2 trainable tensors,
task order,1024 logical batch, successful-update cosine LR,3726-update warmup /
R-Drop ramp,49680-update fixed budget, and common FULL TRAIN SBS selection rule.
A is a winner-independent input-metadata-stratified subset of B; original
VAL/TEST3895/problem are unchanged. Geometry normalization and own SBS are fitted
on each arm's own TRAIN; own SBS is reported, never used for paired selection.

R76 shares architecture/loss, but uses four-rank batch9856, epoch-based control,
3-epoch warmup/ramp (432 updates), validation-plateau LR and early stopping.
It stopped at epoch26/3744 updates. R76 versus either R77 arm changes the
optimization protocol and is NOT a data-only comparison. A's update2484 resume
was this R77 run's own saved checkpoint, not historical weights or a Git commit;
no architecture or optimization formula changed between its early updates and B.

## Common Update Axis And Generalization

At the shared49680-update endpoint, VAL Native Top1 is69.4808%/70.8415% (A/B),
CE0.853683/0.708262, regret0.539679%/0.410186%, and common-SBS
-0.067623%/-0.184136%. Endpoint TRAIN/VAL CE is0.558489/0.853683 for A
(gap0.295193) and0.698277/0.708262 for B (gap0.009985). A shows persistent
TRAIN improvement with VAL deterioration; B shows no persistent analogous
overfitting pattern. These diagnostics never changed the budget, LR or selection.
Own TRAIN populations differ, so compare within-arm trends rather than treating
raw TRAIN metrics as measurements on the same population.

The preselected TEST comparison gives B-minus-A Native Top1+0.6561 percentage
points, CE-0.012264, regret-0.018727 percentage points (about4.54% relative
regret reduction), and common-SBS-0.018151 percentage points. Full TRAIN is not
uniformly better on every task: OVRP and OVRPBLTW have slightly worse regret.
This is one seed2 paired run, not a multi-seed significance or universal claim.

## Locked TEST: All Tasks And Families

| Scope | A Native Top1 % | B Native Top1 % | A regret % | B regret % | A vs common SBS % | B vs common SBS % |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 70.5691 | 71.2252 | 0.412741 | 0.394014 | -0.175897 | -0.194048 |
| Family:TSP | 43.7484 | 44.8267 | 0.443233 | 0.420080 | -0.151192 | -0.175707 |
| Family:CVRP | 45.2118 | 47.3941 | 0.710553 | 0.675836 | -0.570065 | -0.617545 |
| Family:ATSP | 76.3286 | 76.8421 | 0.375485 | 0.373454 | -2.311877 | -2.314272 |
| Family:MVRP | 73.6637 | 74.1994 | 0.393338 | 0.374859 | -0.008867 | -0.025689 |
| TSP | 43.7484 | 44.8267 | 0.443233 | 0.420080 | -0.151192 | -0.175707 |
| CVRP | 45.2118 | 47.3941 | 0.710553 | 0.675836 | -0.570065 | -0.617545 |
| ATSP | 76.3286 | 76.8421 | 0.375485 | 0.373454 | -2.311877 | -2.314272 |
| OVRP | 39.2041 | 40.1540 | 0.642561 | 0.650285 | -0.125179 | -0.115693 |
| VRPB | 80.1797 | 80.2824 | 0.438262 | 0.387539 | 0.024616 | -0.015176 |
| VRPL | 35.4044 | 37.4069 | 0.420315 | 0.400886 | -0.036465 | -0.054778 |
| VRPTW | 80.9499 | 81.0526 | 0.296630 | 0.288880 | -0.000037 | -0.007991 |
| OVRPTW | 84.0308 | 84.0822 | 0.240309 | 0.235977 | 0.000000 | -0.004287 |
| OVRPB | 88.0359 | 88.0616 | 0.301555 | 0.298673 | 0.000547 | -0.002346 |
| OVRPL | 38.4852 | 42.6701 | 0.705133 | 0.616272 | -0.038621 | -0.126441 |
| VRPBL | 79.5635 | 79.8716 | 0.497009 | 0.412951 | 0.062182 | -0.018618 |
| VRPBTW | 78.9217 | 78.9474 | 0.441300 | 0.433076 | -0.008986 | -0.012708 |
| VRPLTW | 79.6149 | 79.7433 | 0.309682 | 0.304999 | -0.002175 | -0.006075 |
| OVRPBL | 89.4480 | 89.6021 | 0.268283 | 0.260486 | 0.002322 | -0.004511 |
| OVRPBTW | 84.8524 | 84.8524 | 0.292365 | 0.291072 | -0.001329 | -0.003108 |
| OVRPLTW | 83.1579 | 83.1836 | 0.274649 | 0.270640 | -0.000424 | -0.004822 |
| VRPBLTW | 79.6149 | 79.6406 | 0.453928 | 0.451376 | -0.008491 | -0.009803 |
| OVRPBLTW | 83.4917 | 83.4403 | 0.318085 | 0.319776 | -0.000966 | 0.001020 |

Each task has3895 TEST rows/arm; MVRP is the macro mean across its15 tasks.
ALL is the18-task macro mean, with70110 total rows/arm. TEST values above are
read from the original single TEST artifacts, never recomputed or used for tuning.

## Canonical Files

- STATUS.json / SUCCESS_RECEIPT.json: final budget, unique locks and GPU release.
- PROCESS_EXIT_RECEIPT.json: observed worker-PID absence and zero step exits.
- finalcomparison.csv:4600 rows, all saved TRAIN/VAL/locked TEST points,23 scopes,
  common/own TRAIN SBS, native accuracy, counts, dataset provenance and update axes.
- COMMON_UPDATE_CURVES.png: common successful updates on the vertical axis.
- selected_train_val_predictions/manifest.json: both final-selected TRAIN/VAL
  NPZ handoffs complete, including pool/IDs/nodes/native costs/indices and hashes.

No extra inference, TRAIN rerun for diagnostics, TEST rerun, solver execution,
rule adjustment or Git commit was performed for this final reporting step.
