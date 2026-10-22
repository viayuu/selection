# R97/A Final Results

## Evidence Boundaries

- One prelocked full18 R97/A context/weight bank; no VAL grid, no task-wise model assembly.
- TabICL v2.0.2 classifier, official v2 public synthetic checkpoint; external pretraining is NOT matched compute.
- Resource preset: n_estimators=1; norm_methods=[none]; full historical 81,000 TRAIN rows/task at natural frequency/order.
- The full-context resource budget is predeclared, not tuned via VAL. No selector SGD or fine-tuned weights.
- Input-only R92 statistics: 357 raw columns, drop only task-TRAIN constants (100-217 retained in this release).
- Historical TRAIN labels are authorized context. Current query labels/winners/costs never enter forward inputs.
- TRAIN self-context accuracy, epochs, SGD learning curves and training loss: N/A; no generalization claim from self labels.
- SBS is fixed on R84 full 81,000-row TRAIN. Oracle is each instance's minimum in its unchanged legal candidate pool.
- ALL is the equal-problem mean across 18 tasks; mixed raw mean_cost is descriptive, not a pooled objective.
- vs_SBS / vs_Oracle = 100*(ratio of mean costs - 1); actual regret = mean_i[100*(selected_i-oracle_i)/oracle_i].
- NSS reuses R79 historical 70,114-row TRAIN weights and saved new90k/LIB predictions; NOT a new 81,000-row retrain.
- R89B is the main historical reference; R84A is secondary. Neither is a newly compute-matched control.
- Retained old VAL/TEST members informed development; this is not a fully independent blind TEST.
- Fewer zero-selected groups are not success: inspect true recovery, lost winners, new false picks, cost and net regret.
- kv_cache=False: every predict re-encodes all 81,000 TRAIN context rows. No cached low-latency claim.
- Fit total_seconds includes context read/verification; fit seconds excludes that loading. Do not add read time twice.
- VAL/TEST use canonical R92 statistics caches. Timings EXCLUDE fresh raw-graph-to-statistics/cache construction.
- LIB native_input_to_statistics includes native loading, geometry and 357 statistics. Recorded sums are not end-to-end latency.

Data: `/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`
Bank SHA256: `0f670be5c6e76a550a9e284daeb2c4f4a7f4b7352ff6712b2bdc13b52612806d`
Public weights: `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`
Package/source: `tabicl==2.0.2` / `ba4bdbc6bb46b9307da3dbc1685109371ec5085f`

Main PDF: [RESULTS.pdf](RESULTS.pdf). Detailed CSV/PDF: `full_model_statistics/`.

## VAL

| Problem | n | Top1 % | Top2 % | Top3 % | mean_cost | SBS cost | Oracle cost | vs_SBS % | vs_Oracle % | actual regret % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 81000 | 71.128395 | 90.732099 | 96.328395 | 11.417186 | 11.427978 | 11.371692 | -0.193229 | 0.394340 | 0.405033 |
| TSP | 4500 | 46.200000 | 72.688889 | 87.355556 | 8.025842 | 8.045354 | 7.990533 | -0.242526 | 0.441881 | 0.429541 |
| CVRP | 4500 | 47.600000 | 68.666667 | 81.600000 | 18.160288 | 18.269377 | 18.064087 | -0.597114 | 0.532555 | 0.721204 |
| ATSP | 4500 | 77.600000 | 95.622222 | 99.311111 | 1.637825 | 1.676905 | 1.632471 | -2.330468 | 0.327991 | 0.325280 |
| OVRP | 4500 | 41.711111 | 72.911111 | 92.488889 | 7.460558 | 7.469579 | 7.415777 | -0.120768 | 0.603871 | 0.616363 |
| VRPB | 4500 | 80.888889 | 97.222222 | 100.000000 | 9.354829 | 9.354876 | 9.318750 | -0.000506 | 0.387167 | 0.395131 |
| VRPL | 4500 | 37.866667 | 65.111111 | 82.800000 | 12.430332 | 12.437959 | 12.383444 | -0.061321 | 0.378637 | 0.384540 |
| VRPTW | 4500 | 79.977778 | 98.911111 | 99.711111 | 18.819325 | 18.819431 | 18.766139 | -0.000561 | 0.283418 | 0.299418 |
| OVRPTW | 4500 | 83.777778 | 99.333333 | 99.666667 | 10.143730 | 10.143730 | 10.116859 | 0.000000 | 0.265602 | 0.262806 |
| OVRPB | 4500 | 88.511111 | 96.977778 | 100.000000 | 6.357254 | 6.357254 | 6.341078 | 0.000000 | 0.255104 | 0.265013 |
| OVRPL | 4500 | 41.600000 | 73.600000 | 91.666667 | 7.437515 | 7.445709 | 7.390416 | -0.110057 | 0.637300 | 0.656749 |
| VRPBL | 4500 | 78.644444 | 96.977778 | 100.000000 | 9.370119 | 9.371523 | 9.328366 | -0.014978 | 0.447589 | 0.451268 |
| VRPBTW | 4500 | 78.488889 | 99.911111 | 100.000000 | 19.949764 | 19.950430 | 19.851084 | -0.003341 | 0.497101 | 0.474922 |
| VRPLTW | 4500 | 78.733333 | 98.644444 | 99.444444 | 18.765896 | 18.765871 | 18.707843 | 0.000135 | 0.310318 | 0.325328 |
| OVRPBL | 4500 | 89.200000 | 97.177778 | 100.000000 | 6.370093 | 6.370093 | 6.352824 | 0.000000 | 0.271836 | 0.283581 |
| OVRPBTW | 4500 | 83.444444 | 99.977778 | 100.000000 | 10.661098 | 10.661098 | 10.623245 | 0.000000 | 0.356326 | 0.330859 |
| OVRPLTW | 4500 | 83.422222 | 99.511111 | 99.866667 | 10.157589 | 10.157423 | 10.129818 | 0.001632 | 0.274147 | 0.276946 |
| VRPBLTW | 4500 | 79.355556 | 99.933333 | 100.000000 | 19.797261 | 19.797044 | 19.706420 | 0.001096 | 0.460970 | 0.447581 |
| OVRPBLTW | 4500 | 83.288889 | 100.000000 | 100.000000 | 10.610025 | 10.609955 | 10.571302 | 0.000658 | 0.366304 | 0.344071 |

## TEST

| Problem | n | Top1 % | Top2 % | Top3 % | mean_cost | SBS cost | Oracle cost | vs_SBS % | vs_Oracle % | actual regret % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 81000 | 71.449383 | 90.661728 | 96.337037 | 11.374047 | 11.385299 | 11.329932 | -0.196943 | 0.386201 | 0.394650 |
| TSP | 4500 | 46.577778 | 71.200000 | 87.377778 | 7.971139 | 7.989621 | 7.937593 | -0.231324 | 0.422619 | 0.414746 |
| CVRP | 4500 | 48.955556 | 69.911111 | 82.422222 | 18.008807 | 18.124983 | 17.918509 | -0.640977 | 0.503935 | 0.668405 |
| ATSP | 4500 | 77.355556 | 96.111111 | 99.533333 | 1.635052 | 1.674207 | 1.629441 | -2.338710 | 0.344348 | 0.342676 |
| OVRP | 4500 | 41.822222 | 72.355556 | 91.977778 | 7.445239 | 7.455750 | 7.400487 | -0.140972 | 0.604715 | 0.614181 |
| VRPB | 4500 | 80.511111 | 97.177778 | 100.000000 | 9.351055 | 9.351810 | 9.315298 | -0.008071 | 0.383853 | 0.388008 |
| VRPL | 4500 | 37.866667 | 64.222222 | 82.133333 | 12.348634 | 12.356138 | 12.300958 | -0.060732 | 0.387576 | 0.395029 |
| VRPTW | 4500 | 80.911111 | 99.244444 | 99.711111 | 18.690158 | 18.690545 | 18.637408 | -0.002074 | 0.283030 | 0.294745 |
| OVRPTW | 4500 | 83.933333 | 99.422222 | 99.800000 | 10.093063 | 10.093063 | 10.068135 | 0.000000 | 0.247590 | 0.238522 |
| OVRPB | 4500 | 88.133333 | 96.733333 | 100.000000 | 6.370826 | 6.370826 | 6.352225 | 0.000000 | 0.292814 | 0.301388 |
| OVRPL | 4500 | 41.977778 | 73.711111 | 91.822222 | 7.463581 | 7.472542 | 7.418033 | -0.119910 | 0.614018 | 0.625518 |
| VRPBL | 4500 | 79.533333 | 96.933333 | 100.000000 | 9.396876 | 9.396379 | 9.355967 | 0.005291 | 0.437253 | 0.440052 |
| VRPBTW | 4500 | 78.866667 | 99.822222 | 100.000000 | 19.665399 | 19.665725 | 19.577666 | -0.001660 | 0.448128 | 0.444958 |
| VRPLTW | 4500 | 79.266667 | 98.911111 | 99.600000 | 18.803045 | 18.803136 | 18.745143 | -0.000485 | 0.308891 | 0.323144 |
| OVRPBL | 4500 | 89.288889 | 96.822222 | 100.000000 | 6.375741 | 6.375741 | 6.359310 | 0.000000 | 0.258375 | 0.269531 |
| OVRPBTW | 4500 | 84.822222 | 100.000000 | 100.000000 | 10.659591 | 10.659591 | 10.624748 | 0.000000 | 0.327942 | 0.299912 |
| OVRPLTW | 4500 | 82.911111 | 99.400000 | 99.688889 | 10.085766 | 10.086112 | 10.059001 | -0.003422 | 0.266088 | 0.266661 |
| VRPBLTW | 4500 | 79.755556 | 99.933333 | 100.000000 | 19.706748 | 19.707068 | 19.613284 | -0.001623 | 0.476534 | 0.454942 |
| OVRPBLTW | 4500 | 83.600000 | 100.000000 | 100.000000 | 10.662116 | 10.662149 | 10.625574 | -0.000306 | 0.343910 | 0.321275 |

## TSPLIB

| Problem | n | Top1 % | Top2 % | Top3 % | mean_cost | SBS cost | Oracle cost | vs_SBS % | vs_Oracle % | actual regret % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TSP | 49 | 32.653061 | 48.979592 | 73.469388 | 8.197021 | 8.240435 | 8.142167 | -0.526841 | 0.673705 | 0.661599 |

## CVRPLIB

| Problem | n | Top1 % | Top2 % | Top3 % | mean_cost | SBS cost | Oracle cost | vs_SBS % | vs_Oracle % | actual regret % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CVRP | 100 | 64.000000 | 80.000000 | 88.000000 | 67.337521 | 67.245364 | 66.945093 | 0.137046 | 0.586194 | 0.590074 |

## NSS Reference

| Dataset | Status | NSS Top1 % | R97 Top1 % | NSS mean_cost | R97 mean_cost |
| --- | --- | --- | --- | --- | --- |
| TSP TEST | complete | 45.888889 | 46.577778 | 7.970401 | 7.971139 |
| CVRP TEST | complete | 45.977778 | 48.955556 | 18.004802 | 18.008807 |
| TSPLIB | complete | 20.408163 | 32.653061 | 8.217907 | 8.197021 |
| CVRPLIB | complete | 70.000000 | 64.000000 | 67.101112 | 67.337521 |

## Historical Full-TEST Comparison

| Baseline | Role | Status | Corrected | Harmed | Top1 delta pp | mean_cost delta | actual regret delta pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R89B | main | complete | 1759 | 1735 | 0.029630 | 0.000576 | 0.005004 |
| R84A | secondary | complete | 1682 | 1509 | 0.213580 | 0.000203 | -0.000197 |

## Fixed36 Target Recovery

| Baseline | Status | True rescued | True lost | Net recovery | New false picks | ALL regret delta pp |
| --- | --- | --- | --- | --- | --- | --- |
| R89B | complete | 30 | 41 | -11 | 41 | 0.005004 |
| R84A | complete | 31 | 10 | 21 | 42 | -0.000197 |

Positive cost/regret deltas are worse. Fixed36 comes from the frozen R84 VAL target list, not R97 TEST.

## Context And Resource Summary

| Task | Context n | Columns | Fit total s | Read/verify s | Predict s | LIB input/statistics s | Recorded sum s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TSP | 81000 | 162 | 6.150084 | 0.380629 | 10.533224 | 1.640518 | 22.511577 |
| CVRP | 81000 | 173 | 6.009129 | 0.361544 | 10.863262 | 0.464489 | 17.336880 |
| ATSP | 81000 | 100 | 2.204676 | 0.174709 | 5.156194 | N/A | 7.360869 |
| OVRP | 81000 | 174 | 4.038595 | 0.239015 | 7.632580 | N/A | 11.671175 |
| VRPB | 81000 | 175 | 4.146444 | 0.263490 | 7.442585 | N/A | 11.589029 |
| VRPL | 81000 | 176 | 3.974886 | 0.285351 | 7.447799 | N/A | 11.422685 |
| VRPTW | 81000 | 214 | 4.440565 | 0.292474 | 8.425300 | N/A | 12.865865 |
| OVRPTW | 81000 | 214 | 4.520239 | 0.293862 | 8.423446 | N/A | 12.943685 |
| OVRPB | 81000 | 175 | 3.926625 | 0.325997 | 7.412123 | N/A | 11.338748 |
| OVRPL | 81000 | 176 | 4.233626 | 0.258818 | 7.427538 | N/A | 11.661164 |
| VRPBL | 81000 | 177 | 4.039629 | 0.258053 | 7.502343 | N/A | 11.541971 |
| VRPBTW | 81000 | 215 | 4.518479 | 0.339944 | 8.414197 | N/A | 12.932676 |
| VRPLTW | 81000 | 216 | 4.519979 | 0.322893 | 8.451313 | N/A | 12.971292 |
| OVRPBL | 81000 | 177 | 3.786840 | 0.264992 | 7.435109 | N/A | 11.221950 |
| OVRPBTW | 81000 | 215 | 4.646063 | 0.337240 | 9.092688 | N/A | 13.738750 |
| OVRPLTW | 81000 | 216 | 5.086643 | 0.417084 | 8.438047 | N/A | 13.524690 |
| VRPBLTW | 81000 | 217 | 4.712048 | 0.376192 | 8.848915 | N/A | 13.560963 |
| OVRPBLTW | 81000 | 217 | 4.746329 | 0.408306 | 8.507774 | N/A | 13.254103 |

Fit total includes context read/verification and the fit call. Read/verify is a component, not added again.
`kv_cache=False`, `context_reencoded=True`: **every prediction re-encodes all 81,000 context rows**.
**VAL/TEST use canonical R92 statistics caches. Fresh raw-graph-to-statistics/cache-build latency is excluded (N/A).**
LIB input/statistics time includes the native loader, geometry and all 357 statistics. This is not full end-to-end latency.
TSP witness fit is also its VAL fit and is counted once. Its separate unlabeled witness query is added once.
Epochs, training loss, TRAIN self-context accuracy, and learning curves: **N/A**.
