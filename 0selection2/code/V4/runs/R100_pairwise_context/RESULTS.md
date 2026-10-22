# R100 Frozen Pairwise Binary Context

Primary delivery: [RESULTS.pdf](RESULTS.pdf).

- One predeclared full18 R100 pairwise-binary frozen TabICL bank. No SGD, fine-tuning, epochs, TRAIN accuracy or VAL aggregation search.
- For every legal pair: ALL historical TRAIN rows in natural frequency/order, including third-method full-pool winners. Only exact native FP64 pair ties are excluded.
- Binary labels: 0 first method wins, 1 second wins. Pair orientation follows ascending stable global IDs; no FP32 reconstruction or full-winner filtering.
- Constant shortcut ONLY for exactly one strict class, or no strict rows (.5 neutral). Support/tie/class counts are identity-bound and reported.
- P[i,j] predicts i beating j conditional on a strict TRAIN relation; P[j,i]=1-P[i,j]. These are NOT full-pool winner probabilities.
- Decision: argmax mean opponent win probability, fixed expected-rank/Borda; ascending global-ID exact ties. No coupling, calibration, heuristic threshold or cost-sensitive claim.
- Formal log-softmax(mean-win) is for descriptive CE/report interoperability only, not calibrated probability or the ranking rule.
- Each nonconstant pair independently fits its strict TRAIN context and predicts the whole named dataset; only one pair classifier is resident. Runtime scales with K*(K-1)/2 contexts.
- Original R97 public classifier weights/runtime: max_classes=10 is sufficient for binary2. n_estimators=1, norm=none, batch_size=1, seed2, AMP, kv_cache=False, GPU-only, Flash-only.
- Immutable R97 input-only statistics and TRAIN-constant drop. Query labels/winners/costs never enter fit/forward; any query base ID in complete TRAIN is rejected BEFORE pair filtering.
- SBS fixed on R84 full81000 TRAIN. Oracle is the per-instance legal-pool minimum. ALL is equal-problem macro; mixed raw costs descriptive only.
- vs_SBS/vs_Oracle are percent ratios of mean costs; actual regret is mean per-instance percent regret. Native/FP64 ind is never replaced by FP32 argmin.
- Same current bank on VAL, full TEST and native LIB. Historical TEST members informed research, not wholly blind. Public pretraining compute is unmatched.
- NSS reuses saved R79 historical70114 TRAIN weights, not a new81000 retrain. Missing stays missing; incompatible saved evidence fails.
- R89B main, R84A secondary; fixed36 targets frozen from R84A VAL. Report rescued/lost true winners, false picks, raw false-pick cost and regret, not zero-count changes alone.
- Per-pair fit/predict times are NESTED in enclosing calls and are not added twice. VAL/TEST use cached inputs: fresh raw-graph statistics/cache-build latency is NOT measured.
- LIB native_input_to_statistics includes native loading/geometry/statistics without query-label loading. Recorded sums are not full end-to-end latency.

| problem | split | n | top1_pct | mean_cost | sbs_cost | oracle_cost | vs_sbs_pct | vs_oracle_pct | actual_regret_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | VAL | 81000 | 71.070370 | 11.416165 | 11.427978 | 11.371692 | -0.201078 | 0.386414 | 0.394268 |
| TSP | VAL | 4500 | 45.600000 | 8.023986 | 8.045354 | 7.990533 | -0.265594 | 0.418656 | 0.409313 |
| CVRP | VAL | 4500 | 46.777778 | 18.147534 | 18.269377 | 18.064087 | -0.666926 | 0.461949 | 0.595322 |
| ATSP | VAL | 4500 | 77.911111 | 1.637724 | 1.676905 | 1.632471 | -2.336494 | 0.321801 | 0.319152 |
| OVRP | VAL | 4500 | 42.044444 | 7.459348 | 7.469579 | 7.415777 | -0.136966 | 0.587554 | 0.597930 |
| VRPB | VAL | 4500 | 80.866667 | 9.354679 | 9.354876 | 9.318750 | -0.002104 | 0.385563 | 0.394564 |
| VRPL | VAL | 4500 | 37.533333 | 12.429751 | 12.437959 | 12.383444 | -0.065994 | 0.373944 | 0.380305 |
| VRPTW | VAL | 4500 | 79.955556 | 18.819473 | 18.819431 | 18.766139 | 0.000223 | 0.284205 | 0.301462 |
| OVRPTW | VAL | 4500 | 83.777778 | 10.143730 | 10.143730 | 10.116859 | 0.000000 | 0.265602 | 0.262806 |
| OVRPB | VAL | 4500 | 88.511111 | 6.357254 | 6.357254 | 6.341078 | 0.000000 | 0.255104 | 0.265013 |
| OVRPL | VAL | 4500 | 41.688889 | 7.436408 | 7.445709 | 7.390416 | -0.124924 | 0.622322 | 0.639050 |
| VRPBL | VAL | 4500 | 78.555556 | 9.370291 | 9.371523 | 9.328366 | -0.013138 | 0.449437 | 0.452938 |
| VRPBTW | VAL | 4500 | 78.422222 | 19.950456 | 19.950430 | 19.851084 | 0.000128 | 0.500588 | 0.477917 |
| VRPLTW | VAL | 4500 | 78.755556 | 18.765968 | 18.765871 | 18.707843 | 0.000518 | 0.310703 | 0.326395 |
| OVRPBL | VAL | 4500 | 89.222222 | 6.369942 | 6.370093 | 6.352824 | -0.002370 | 0.269460 | 0.280748 |
| OVRPBTW | VAL | 4500 | 83.466667 | 10.660992 | 10.661098 | 10.623245 | -0.001001 | 0.355322 | 0.329802 |
| OVRPLTW | VAL | 4500 | 83.444444 | 10.157471 | 10.157423 | 10.129818 | 0.000474 | 0.272986 | 0.276925 |
| VRPBLTW | VAL | 4500 | 79.444444 | 19.796009 | 19.797044 | 19.706420 | -0.005230 | 0.454615 | 0.443949 |
| OVRPBLTW | VAL | 4500 | 83.288889 | 10.609955 | 10.609955 | 10.571302 | 0.000000 | 0.365644 | 0.343234 |
| ALL | TEST | 81000 | 71.316049 | 11.373466 | 11.385299 | 11.329932 | -0.200894 | 0.382209 | 0.389043 |
| TSP | TEST | 4500 | 46.444444 | 7.970024 | 7.989621 | 7.937593 | -0.245278 | 0.408573 | 0.402728 |
| CVRP | TEST | 4500 | 47.577778 | 18.000121 | 18.124983 | 17.918509 | -0.688899 | 0.455461 | 0.590110 |
| ATSP | TEST | 4500 | 77.466667 | 1.635020 | 1.674207 | 1.629441 | -2.340671 | 0.342334 | 0.340713 |
| OVRP | TEST | 4500 | 41.200000 | 7.445250 | 7.455750 | 7.400487 | -0.140823 | 0.604865 | 0.614165 |
| VRPB | TEST | 4500 | 80.511111 | 9.350952 | 9.351810 | 9.315298 | -0.009174 | 0.382745 | 0.386606 |
| VRPL | TEST | 4500 | 37.622222 | 12.347549 | 12.356138 | 12.300958 | -0.069511 | 0.378757 | 0.386561 |
| VRPTW | TEST | 4500 | 80.866667 | 18.690730 | 18.690545 | 18.637408 | 0.000987 | 0.286100 | 0.297267 |
| OVRPTW | TEST | 4500 | 83.933333 | 10.093063 | 10.093063 | 10.068135 | 0.000000 | 0.247590 | 0.238522 |
| OVRPB | TEST | 4500 | 88.133333 | 6.370826 | 6.370826 | 6.352225 | 0.000000 | 0.292814 | 0.301388 |
| OVRPL | TEST | 4500 | 41.955556 | 7.463368 | 7.472542 | 7.418033 | -0.122762 | 0.611145 | 0.621077 |
| VRPBL | TEST | 4500 | 79.533333 | 9.397090 | 9.396379 | 9.355967 | 0.007569 | 0.439541 | 0.443116 |
| VRPBTW | TEST | 4500 | 78.800000 | 19.665560 | 19.665725 | 19.577666 | -0.000839 | 0.448952 | 0.446033 |
| VRPLTW | TEST | 4500 | 79.266667 | 18.803004 | 18.803136 | 18.745143 | -0.000702 | 0.308673 | 0.322482 |
| OVRPBL | TEST | 4500 | 89.288889 | 6.375741 | 6.375741 | 6.359310 | 0.000000 | 0.258375 | 0.269531 |
| OVRPBTW | TEST | 4500 | 84.822222 | 10.659591 | 10.659591 | 10.624748 | 0.000000 | 0.327942 | 0.299912 |
| OVRPLTW | TEST | 4500 | 82.911111 | 10.085766 | 10.086112 | 10.059001 | -0.003422 | 0.266088 | 0.266661 |
| VRPBLTW | TEST | 4500 | 79.755556 | 19.706624 | 19.707068 | 19.613284 | -0.002254 | 0.475901 | 0.454619 |
| OVRPBLTW | TEST | 4500 | 83.600000 | 10.662116 | 10.662149 | 10.625574 | -0.000306 | 0.343910 | 0.321275 |
| TSP | TSPLIB | 49 | 20.408163 | 8.224506 | 8.240435 | 8.142167 | -0.193312 | 1.011260 | 1.037809 |
| CVRP | CVRPLIB | 100 | 59.000000 | 67.423326 | 67.245364 | 66.945093 | 0.264646 | 0.714366 | 0.594214 |

NSS/fixed36/support/timing/provenance: full_model_statistics/.
