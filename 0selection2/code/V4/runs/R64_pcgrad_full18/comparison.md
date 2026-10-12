# R64 Saved Comparison

Official arm B40 was frozen by integer validation counts before TEST. A40=8765/18000, B40=8777/18000; tie rule A. Locked TEST: A8528/18000, B8534/18000. The70% goal remains unachieved.

| Model | Split | Native Top1 % | Top2 % | Top3 % | CE | Mean Cost | vs SBS % | Actual Regret % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R64A | train | 48.5244 | 82.1294 | 92.7939 | 1.086541 | 11.178202 | -1.068215 | 0.950804 |
| R64A | val | 48.6944 | 82.0556 | 92.5778 | 1.094214 | 11.169958 | -1.057617 | 0.938619 |
| R64A | test | 47.3778 | 81.4500 | 92.2167 | 1.102747 | 11.173794 | -0.992692 | 0.981460 |
| R64B | train | 48.6156 | 82.1272 | 92.7806 | 1.085773 | 11.178306 | -1.069022 | 0.949743 |
| R64B | val | 48.7611 | 82.1111 | 92.4889 | 1.094646 | 11.170174 | -1.055205 | 0.940973 |
| R64B | test | 47.4111 | 81.2444 | 92.2500 | 1.103343 | 11.173766 | -0.995764 | 0.977711 |
| R45A | val | 49.1167 | 82.4444 | 92.7611 | 1.089768 | 11.168517 | -1.070041 | 0.926048 |
| R45A | test | 47.9111 | 81.7444 | 92.4333 | 1.101119 | 11.168366 | -1.032003 | 0.943221 |
| R60B | val | 49.3611 | 82.2389 | 92.7222 | 1.092423 | 11.168642 | -1.073544 | 0.922789 |
| R60B | test | 48.2444 | 81.7222 | 92.5111 | 1.102036 | 11.168404 | -1.035671 | 0.940486 |

| Comparison | Split | Native Top1 Delta pp | CE Delta | Mean Cost Delta | Regret Delta pp |
| --- | --- | --- | --- | --- | --- |
| R64B minus R64A | train | 0.091111 | -0.000768 | 0.000104 | -0.001062 |
| R64B minus R64A | val | 0.066667 | 0.000432 | 0.000216 | 0.002354 |
| R64B minus R64A | test | 0.033333 | 0.000596 | -0.000028 | -0.003749 |
| R64A minus R45A | val | -0.422222 | 0.004446 | 0.001442 | 0.012571 |
| R64A minus R45A | test | -0.533333 | 0.001629 | 0.005428 | 0.038239 |
| R64B minus R45A | val | -0.355556 | 0.004878 | 0.001658 | 0.014925 |
| R64B minus R45A | test | -0.500000 | 0.002224 | 0.005400 | 0.034490 |
| R64A minus R60B | val | -0.666667 | 0.001791 | 0.001316 | 0.015830 |
| R64A minus R60B | test | -0.866667 | 0.000711 | 0.005390 | 0.040974 |
| R64B minus R60B | val | -0.600000 | 0.002223 | 0.001532 | 0.018184 |
| R64B minus R60B | test | -0.833333 | 0.001307 | 0.005362 | 0.037225 |

All18 task and four family metrics are in [comparison.csv](comparison.csv) (230 rows). Family deltas are in [comparison_deltas.csv](comparison_deltas.csv). Top1/2/3 are fractions in CSV and percentages above; their deltas are percentage points. Actual regret and vsSBS are already percentages. Full narrative, native task tables, compute, numerical disclosure and provenance are in [RESULTS.md](RESULTS.md). Historical R45A/R60B TEST was reused without reruns or reselection.
