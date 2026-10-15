# Train-Only Advantage Regions: All 18 Tasks

Fixed diagnostic, not a new competitive selector. All thresholds, smoothing, actions and trees are fitted on training only. Original FP64 costs and native winners use current registry pools; README solver lists are not used. Train/val/test total 216,000 rows; LIB is supplemental. No GPU, neural training, solver reruns, tuning or commits.

## Conditional priors and fixed constants

Relative gap is 100*(selected cost - instance native-pool minimum)/minimum. SBS minimizes TRAIN mean absolute cost; hindsight constant minimizes HELDOUT mean relative gap and is a lower reference among constants only. Size prior uses exact training sizes shrunk to train-quartile bins, then problem-global (pseudocount 20 at each level). An unseen in-range size uses its bin; an out-of-train size uses the global prior.

| Task | Majority val/test Top1 % | Mean-gap val/test Top1 % | Mean-gap val/test gap % | Train SBS val/test gap % | Hindsight constant val/test gap % |
|---|---:|---:|---:|---:|---:|
| TSP | 30.500/30.300 | 19.800/20.400 | 0.863/0.906 | 1.095/1.075 | 1.095/1.075 |
| CVRP | 27.600/26.700 | 22.300/25.300 | 1.047/0.972 | 1.053/0.973 | 1.053/0.973 |
| ATSP | 78.000/77.800 | 77.700/78.000 | 0.318/0.298 | 2.726/2.797 | 2.726/2.797 |
| OVRP | 36.900/34.800 | 37.300/35.600 | 0.673/0.765 | 0.797/0.826 | 0.797/0.826 |
| VRPB | 49.800/48.000 | 50.400/48.300 | 0.776/0.848 | 0.776/0.848 | 0.776/0.848 |
| VRPL | 29.400/29.500 | 29.300/27.500 | 0.503/0.503 | 0.533/0.555 | 0.507/0.533 |
| VRPTW | 24.300/27.800 | 27.200/28.300 | 4.597/4.434 | 6.198/6.380 | 6.061/5.845 |
| OVRPTW | 47.800/44.400 | 47.200/46.800 | 0.939/0.937 | 0.926/0.923 | 0.926/0.923 |
| OVRPB | 54.100/54.600 | 54.500/56.000 | 0.977/1.038 | 0.977/1.038 | 0.977/1.038 |
| OVRPL | 39.400/39.000 | 36.400/39.400 | 0.773/0.669 | 0.807/0.830 | 0.807/0.830 |
| VRPBL | 52.000/51.800 | 52.800/53.700 | 0.812/0.781 | 0.812/0.781 | 0.812/0.781 |
| VRPBTW | 35.700/34.700 | 37.200/35.700 | 4.361/3.880 | 4.361/3.880 | 4.361/3.880 |
| VRPLTW | 31.000/29.600 | 30.400/28.800 | 4.318/4.203 | 6.811/7.097 | 5.902/5.607 |
| OVRPBL | 54.700/54.600 | 55.500/54.400 | 1.053/1.049 | 1.053/1.049 | 1.053/1.049 |
| OVRPBTW | 44.500/46.200 | 45.700/46.600 | 1.278/1.321 | 1.254/1.300 | 1.254/1.300 |
| OVRPLTW | 47.900/49.100 | 49.100/48.800 | 0.892/0.897 | 0.888/0.900 | 0.888/0.900 |
| VRPBLTW | 34.500/34.300 | 34.700/35.200 | 4.074/4.132 | 4.074/4.132 | 4.074/4.132 |
| OVRPBLTW | 48.000/46.400 | 46.600/45.800 | 1.268/1.265 | 1.256/1.195 | 1.256/1.195 |

Equal-task macro (18 tasks, no LIB):

| Policy | Val Top1 % | Test Top1 % | Val gap % | Test gap % |
|---|---:|---:|---:|---:|
| majority | 42.561 | 42.200 | 1.7882 | 1.7400 |
| mean_gap | 41.894 | 41.922 | 1.6401 | 1.6053 |
| global_sbs | 37.767 | 37.828 | 2.0221 | 2.0322 |
| global_mean_gap | 38.739 | 38.750 | 1.9639 | 1.9196 |
| size_bin_majority | 42.694 | 42.111 | 1.7519 | 1.7293 |
| size_bin_mean_gap | 41.739 | 41.506 | 1.6326 | 1.6105 |
| winner_tree_leaf_majority | 46.456 | 46.194 | 1.0372 | 1.0571 |
| gap_tree_leaf_mean_gap | 44.378 | 43.406 | 1.2345 | 1.2508 |
| hindsight_best_constant_relative_gap | 38.778 | 38.728 | 1.9625 | 1.9184 |

## Identical-Region Accuracy/Cost Tradeoff

Every frozen size/feature/tree region has BOTH train-majority and train-minmeanrelativegap decisions. The following uses the SAME label-free four size regions for both actions; their disagreement is the fraction of instances receiving different decisions. Positive gap advantage favors the mean-gap action. Full rule-level supports, uncertainty and tradeoffs are in regions.csv and decision_tradeoffs.csv.

| Task | Val/test disagreement % | Mean-gap minus majority Top1 pp (val/test) | Mean-gap advantage pp (val/test) |
|---|---:|---:|---:|
| TSP | 75.7/75.7 | -9.90/-7.30 | +0.5888/+0.5072 |
| CVRP | 49.9/51.5 | -7.00/-2.80 | +0.5502/+0.7108 |
| ATSP | 24.0/24.0 | -2.70/-4.30 | +0.3942/+0.5645 |
| OVRP | 0.0/0.0 | +0.00/+0.00 | +0.0000/+0.0000 |
| VRPB | 0.0/0.0 | +0.00/+0.00 | +0.0000/+0.0000 |
| VRPL | 25.2/25.2 | +0.20/+1.20 | +0.0255/+0.0429 |
| VRPTW | 48.0/48.0 | +2.90/+3.20 | +0.5295/+0.3296 |
| OVRPTW | 26.0/26.0 | -0.70/-0.90 | +0.0589/-0.0162 |
| OVRPB | 0.0/0.0 | +0.00/+0.00 | +0.0000/+0.0000 |
| OVRPL | 0.0/0.0 | +0.00/+0.00 | +0.0000/+0.0000 |
| VRPBL | 0.0/0.0 | +0.00/+0.00 | +0.0000/+0.0000 |
| VRPBTW | 0.0/0.0 | +0.00/+0.00 | +0.0000/+0.0000 |
| VRPLTW | 0.0/0.0 | +0.00/+0.00 | +0.0000/+0.0000 |
| OVRPBL | 0.0/0.0 | +0.00/+0.00 | +0.0000/+0.0000 |
| OVRPBTW | 0.0/0.0 | +0.00/+0.00 | +0.0000/+0.0000 |
| OVRPLTW | 0.0/0.0 | +0.00/+0.00 | +0.0000/+0.0000 |
| VRPBLTW | 0.0/0.0 | +0.00/+0.00 | +0.0000/+0.0000 |
| OVRPBLTW | 0.0/0.0 | +0.00/+0.00 | +0.0000/+0.0000 |

## Supported Region Trends

637 regions have at least 200 training and 30 validation/test instances; 61 have the same TRAIN/val/test primary-pair advantage sign and simultaneous heldout CIs excluding zero, including 32 label-free input-quantile regions. Overlapping regions are NOT independent discoveries. Primary pair is fixed from the top two TRAIN winner frequencies; positive advantage favors A. Intervals on target-derived TRAIN tree leaves are adaptive/descriptive, not independent estimation.

| Task / region | Pair A vs B | Train/val/test support | Train/val/test advantage pp | Val simultaneous CI | Test simultaneous CI |
|---|---|---:|---:|---:|---:|
| TSP / all:0 | BQ vs T2T500 | 10000/1000/1000 | -0.632/-0.667/-0.580 | [-1.145, -0.189] | [-1.012, -0.147] |
| TSP / size_bin:3 | BQ vs T2T500 | 2480/243/243 | -1.252/-1.510/-1.298 | [-2.644, -0.375] | [-2.240, -0.355] |
| CVRP / all:0 | OMNI vs RELD_CVRP | 10000/1000/1000 | -1.433/-1.253/-1.615 | [-1.664, -0.842] | [-2.068, -1.162] |
| CVRP / size_bin:0 | OMNI vs RELD_CVRP | 2507/247/236 | -1.474/-1.104/-1.434 | [-1.772, -0.435] | [-2.209, -0.659] |
| CVRP / size_bin:1 | OMNI vs RELD_CVRP | 2519/260/274 | -1.089/-0.922/-1.224 | [-1.672, -0.173] | [-1.992, -0.456] |
| CVRP / size_bin:2 | OMNI vs RELD_CVRP | 2482/239/241 | -1.245/-1.299/-1.558 | [-2.192, -0.405] | [-2.561, -0.555] |
| CVRP / size_bin:3 | OMNI vs RELD_CVRP | 2492/254/249 | -1.927/-1.694/-2.272 | [-2.675, -0.713] | [-3.355, -1.189] |
| CVRP / geometry_quantile:1 | OMNI vs RELD_CVRP | 1655/164/159 | -1.570/-0.999/-1.589 | [-1.775, -0.223] | [-2.607, -0.570] |
| CVRP / geometry_quantile:6 | OMNI vs RELD_CVRP | 1530/146/156 | -1.761/-1.536/-2.428 | [-2.797, -0.274] | [-3.843, -1.013] |
| CVRP / geometry_quantile:7 | OMNI vs RELD_CVRP | 962/108/93 | -2.191/-1.908/-2.010 | [-3.532, -0.284] | [-3.774, -0.246] |
| CVRP / demand_quantile:0 | OMNI vs RELD_CVRP | 559/48/41 | -4.349/-3.453/-4.704 | [-5.663, -1.243] | [-7.082, -2.325] |
| CVRP / demand_quantile:2 | OMNI vs RELD_CVRP | 935/90/107 | -3.752/-3.781/-3.924 | [-5.181, -2.382] | [-5.220, -2.629] |
| CVRP / demand_quantile:3 | OMNI vs RELD_CVRP | 1584/170/167 | +0.482/+0.591/+0.506 | [+0.277, +0.906] | [+0.180, +0.833] |
| CVRP / demand_quantile:4 | OMNI vs RELD_CVRP | 1179/119/117 | -3.324/-3.376/-4.032 | [-4.719, -2.032] | [-5.561, -2.504] |
| CVRP / demand_quantile:5 | OMNI vs RELD_CVRP | 1303/120/124 | +0.637/+0.761/+0.777 | [+0.338, +1.185] | [+0.363, +1.190] |
| CVRP / demand_quantile:6 | OMNI vs RELD_CVRP | 2327/233/233 | -2.064/-1.868/-2.426 | [-2.925, -0.812] | [-3.569, -1.284] |
| ATSP / all:0 | UNICO_MatPOENet vs ICAM_ATSP | 10000/1000/1000 | +2.196/+2.446/+2.176 | [+1.293, +3.600] | [+1.071, +3.280] |
| ATSP / size_bin:0 | UNICO_MatPOENet vs ICAM_ATSP | 2604/273/273 | +11.387/+11.959/+10.860 | [+10.253, +13.665] | [+9.297, +12.424] |
| ATSP / size_bin:1 | UNICO_MatPOENet vs ICAM_ATSP | 2476/247/247 | +5.511/+5.503/+5.650 | [+4.357, +6.649] | [+4.360, +6.940] |
| ATSP / size_bin:2 | UNICO_MatPOENet vs ICAM_ATSP | 2460/240/240 | -1.694/-1.963/-1.957 | [-2.881, -1.044] | [-2.875, -1.040] |
| ATSP / size_bin:3 | UNICO_MatPOENet vs ICAM_ATSP | 2460/240/240 | -6.980/-7.112/-7.147 | [-8.036, -6.187] | [-7.996, -6.297] |
| ATSP / geometry_quantile:1 | UNICO_MatPOENet vs ICAM_ATSP | 2601/273/273 | +11.391/+11.959/+10.860 | [+10.253, +13.665] | [+9.297, +12.424] |
| ATSP / geometry_quantile:3 | UNICO_MatPOENet vs ICAM_ATSP | 2088/203/212 | +6.293/+6.162/+6.345 | [+4.941, +7.383] | [+4.993, +7.698] |
| ATSP / geometry_quantile:4 | UNICO_MatPOENet vs ICAM_ATSP | 2149/211/201 | -2.152/-2.378/-2.517 | [-3.295, -1.462] | [-3.452, -1.583] |
| ATSP / geometry_quantile:6 | UNICO_MatPOENet vs ICAM_ATSP | 2460/240/240 | -6.980/-7.112/-7.147 | [-8.036, -6.187] | [-7.996, -6.297] |
| OVRP / size_bin:1 | MoSES_RF vs MoSES_CaDA | 2548/260/260 | +0.530/+0.681/+0.542 | [+0.222, +1.140] | [+0.048, +1.036] |
| OVRPB / all:0 | RELD_MTL vs RELD_MOEL | 10000/1000/1000 | +0.827/+0.878/+0.795 | [+0.354, +1.402] | [+0.234, +1.356] |
| VRPBL / all:0 | RELD_MTL vs RELD_MOEL | 10000/1000/1000 | +0.634/+0.793/+0.752 | [+0.301, +1.285] | [+0.254, +1.250] |
| VRPBTW / all:0 | RELD_MTL vs RELD_MOEL | 10000/1000/1000 | +0.667/+0.594/+0.659 | [+0.085, +1.103] | [+0.182, +1.136] |
| VRPLTW / all:0 | RELD_MTL vs RELD_MOEL | 10000/1000/1000 | +0.482/+0.467/+0.454 | [+0.090, +0.844] | [+0.093, +0.815] |
| OVRPBL / all:0 | RELD_MTL vs RELD_MOEL | 10000/1000/1000 | +0.801/+0.857/+0.730 | [+0.297, +1.417] | [+0.226, +1.234] |
| VRPBLTW / all:0 | RELD_MTL vs RELD_MOEL | 10000/1000/1000 | +0.645/+0.855/+0.648 | [+0.331, +1.378] | [+0.148, +1.147] |

All native method pairs, including rarely winning methods, are retained in pair_statistics.csv. Top3 training-winner pairs and large TRAIN expected-gap pairs are flagged rather than selected using heldout data. independence_baseline.csv gives analytic same-region probability-matching expectations, not a permutation p-value or a performance ceiling.

## Distribution and LIB Extrapolation

KS is ECDF distance, not a p-value. SMD is signed mean change divided by pooled population SD. Any-feature range exceedance is sensitive to feature count; per-feature effects are available in feature_shift.csv. These are empirical distribution contrasts, not inferred generation-distribution labels.

| Task/split | N | Actual size range | Size extrapolation % | Any non-size feature out of train % | Max non-size abs SMD | Max non-size KS |
|---|---:|---:|---:|---:|---:|---:|
| TSP/val | 1000 | 50..499 | 0.0 | 0.1 | 0.068 | 0.055 |
| TSP/test | 1000 | 50..499 | 0.0 | 0.5 | 0.059 | 0.044 |
| CVRP/val | 1000 | 50..499 | 0.0 | 0.8 | 0.073 | 0.054 |
| CVRP/test | 1000 | 50..499 | 0.0 | 0.2 | 0.065 | 0.041 |
| ATSP/val | 1000 | 20..100 | 0.0 | 0.3 | 0.037 | 0.032 |
| ATSP/test | 1000 | 20..100 | 0.0 | 0.2 | 0.044 | 0.031 |
| OVRP/val | 1000 | 50..100 | 0.0 | 0.5 | 0.052 | 0.042 |
| OVRP/test | 1000 | 50..100 | 0.0 | 0.7 | 0.057 | 0.041 |
| VRPB/val | 1000 | 50..100 | 0.0 | 0.6 | 0.051 | 0.037 |
| VRPB/test | 1000 | 50..100 | 0.0 | 0.6 | 0.049 | 0.034 |
| VRPL/val | 1000 | 50..100 | 0.0 | 0.4 | 0.056 | 0.041 |
| VRPL/test | 1000 | 50..100 | 0.0 | 0.5 | 0.083 | 0.037 |
| VRPTW/val | 1000 | 50..100 | 0.0 | 1.4 | 0.074 | 0.048 |
| VRPTW/test | 1000 | 50..100 | 0.0 | 1.0 | 0.085 | 0.044 |
| OVRPTW/val | 1000 | 50..100 | 0.0 | 1.6 | 0.058 | 0.040 |
| OVRPTW/test | 1000 | 50..100 | 0.0 | 1.1 | 0.086 | 0.059 |
| OVRPB/val | 1000 | 50..100 | 0.0 | 0.4 | 0.066 | 0.048 |
| OVRPB/test | 1000 | 50..100 | 0.0 | 0.7 | 0.066 | 0.045 |
| OVRPL/val | 1000 | 50..100 | 0.0 | 0.6 | 0.035 | 0.050 |
| OVRPL/test | 1000 | 50..100 | 0.0 | 0.6 | 0.065 | 0.049 |
| VRPBL/val | 1000 | 50..100 | 0.0 | 0.4 | 0.094 | 0.060 |
| VRPBL/test | 1000 | 50..100 | 0.0 | 0.6 | 0.085 | 0.036 |
| VRPBTW/val | 1000 | 50..100 | 0.0 | 0.3 | 0.091 | 0.053 |
| VRPBTW/test | 1000 | 50..100 | 0.0 | 1.1 | 0.076 | 0.052 |
| VRPLTW/val | 1000 | 50..100 | 0.0 | 1.3 | 0.079 | 0.037 |
| VRPLTW/test | 1000 | 50..100 | 0.0 | 1.0 | 0.095 | 0.045 |
| OVRPBL/val | 1000 | 50..100 | 0.0 | 0.9 | 0.085 | 0.049 |
| OVRPBL/test | 1000 | 50..100 | 0.0 | 0.9 | 0.053 | 0.034 |
| OVRPBTW/val | 1000 | 50..100 | 0.0 | 1.3 | 0.057 | 0.049 |
| OVRPBTW/test | 1000 | 50..100 | 0.0 | 0.8 | 0.067 | 0.050 |
| OVRPLTW/val | 1000 | 50..100 | 0.0 | 0.6 | 0.073 | 0.052 |
| OVRPLTW/test | 1000 | 50..100 | 0.0 | 1.0 | 0.063 | 0.050 |
| VRPBLTW/val | 1000 | 50..100 | 0.0 | 1.1 | 0.051 | 0.043 |
| VRPBLTW/test | 1000 | 50..100 | 0.0 | 0.8 | 0.079 | 0.051 |
| OVRPBLTW/val | 1000 | 50..100 | 0.0 | 1.5 | 0.074 | 0.046 |
| OVRPBLTW/test | 1000 | 50..100 | 0.0 | 0.9 | 0.072 | 0.047 |
| TSP/TSPLIB | 49 | 51..1002 | 14.3 | 87.8 | 1.661 | 0.772 |

TSPLIB: train-fixed majority Top1=28.57%, gap=2.1418%; train-fixed mean-gap Top1=20.41%, gap=0.6820%; train-SBS gap=1.2322%; hindsight-constant gap=1.0522%.

| CVRP/CVRPLIB | 100 | 100..1000 | 32.0 | 52.0 | 1.653 | 0.696 |

CVRPLIB: train-fixed majority Top1=15.00%, gap=2.5155%; train-fixed mean-gap Top1=45.00%, gap=0.5768%; train-SBS gap=0.5768%; hindsight-constant gap=0.5768%.

CVRPLIB geometry/actual size excludes an explicitly duplicated zero-demand first depot; raw_node_counts remains available. Train/test candidate costs and native labels are unchanged.

## Scope and Limits

Small heldout supports (<30) are flagged, never merged. Out-of-train sizes fall back to training global priors; trees still route extrapolated inputs, explicitly flagged in the arrays. Geometry distances are bounded to 128 deterministically coordinate-selected customers; subset nearest-neighbor values are not exact full-instance nearest neighbors when sampled.

A null result for these fixed shallow regions does NOT prove feature insufficiency, label randomness, or conditional independence. The historical solver deployments/contracts are not recertified here. This report makes no architecture or competitive-selector claim. The main agent separately owns frozen-model/error/permutation and R70 score-conditioned residual statistics.

## Files

- methodology.json: fixed definitions, confidence/support caveats, schemas and ownership.
- solver_statistics.csv / size_decisions.csv: all tasks, exact actual sizes, training bins and solver frequencies/gaps.
- baselines.csv: train-fixed priors, every fixed solver, and separately labeled hindsight-constant references.
- regions.csv / decision_tradeoffs.csv / region_transfer.csv: same-region decisions, supports, rules and replicated pair signs.
- pair_statistics.csv / independence_baseline.csv: full native-pool pair coverage and analytic same-region reference.
- feature_shift.csv / shift_summary.csv: train-to-val/test/LIB SMD, KS and train-range effects.
- feature_cache/ / memberships/ / fitted/: per-instance raw statistics, frozen memberships and full train-only rules.
- val_predictions/ / test_predictions/ / LIB_predictions/: original-order aggregation arrays with registry pools.
