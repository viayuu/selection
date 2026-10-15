# Selection Research

The current project is [0selection2](0selection2/README.md). This repository also
contains older working directories; they have not been moved or renamed.

## Current Release: R75-R79 (2026-10-15)

The retained full 18-problem selector is **R77B**, using one validation-selected
checkpoint for VAL, TEST, TSPLIB and CVRPLIB. No R78 candidate was accepted.
R79 provides the completed, new-data-retrained TSP/CVRP NSS comparison.
R77B native TEST Top1 is **71.2252%**, with **0.394014% actual regret**.
NSS new TSP TEST Top1 is **44.3389% native**; historical **43.9538%** uses FP32
argmin labels and is not the same metric. R58 has not released a unified
semantic cost table. R74 was terminated to release GPU memory, retaining disk
artifacts; no goal or solver repair is claimed.

- [Project overview, data version and experiment navigation](0selection2/README.md)
- [R77B full statistics PDF, 22 pages](0selection2/code/V4/runs/R77_sample_size_batch1024/B_full_seed2/full_model_statistics/MODEL_STATISTICS.pdf)
- [R77B selection distributions PDF, 18 pages](0selection2/code/V4/runs/R77_sample_size_batch1024/B_full_seed2/full_model_statistics/selection_distributions.pdf)
- [R78 six-round retrospective PDF, 3 pages](0selection2/code/V4/runs/R78_zero_selection_goal/final_delivery/zero_selection_retrospective/ZERO_SELECTION_RETROSPECTIVE.pdf)
- [R79 NSS selection distributions PDF, 2 pages](0selection2/code/V4/runs/R79_NSS_new_data_batch1024/selection_distribution/selection_distributions.pdf)

This is a source/configuration/report release, not a checkpoint or dataset
distribution. Weights, per-instance prediction arrays, datasets, caches, W&B
state and environments remain on the server. See the project overview for their
locations, metric conventions and the distinction between native and FP32 Top1.
