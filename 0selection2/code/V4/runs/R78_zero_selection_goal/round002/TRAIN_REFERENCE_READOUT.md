# Round002: Authorized TRAIN Reference Readout

Historical execution: inconclusive. Research decision: route_closed_negative.
One existing TRAIN-cache readout; zero optimizer updates, new forwards, VAL or TEST reads.
Reference b* is diagnostic only. Original outputs and the PLAN gate are unchanged.

| TRAIN rule | Native Top1 % | Raw macro mean cost | Actual regret % |
|---|---:|---:|---:|
| identity | 69.369645625 | 11.373004840416 | 0.460650092 |
| fitted | 40.849743117 | 11.653491213001 | 2.826220296 |
| reference | 40.849505409 | 11.653492604283 | 2.826224389 |

| Comparison (new minus old) | Changed | Corrected Top1 | Broken Top1 | Cost delta | Regret delta pp |
|---|---:|---:|---:|---:|---:|
| fitted_vs_identity | 748729 | 143169 | 503105 | 0.280486372585 | 2.365570204 |
| reference_vs_identity | 748754 | 143176 | 503115 | 0.280487763868 | 2.36557429618 |
| reference_vs_fitted | 27 | 8 | 11 | 1.39128284429e-06 | 4.09218605826e-06 |

Full-K epsilon=||b-b*||_inf. Fitted top-two margin>2*epsilon certifies an unchanged winner.
This is a descriptive stability bound, not a new acceptance gate.

| Task | Full-K epsilon | Mean fitted margin | Rows margin>2epsilon | Changed | Changed above bound |
|---|---:|---:|---:|---:|---:|
| TSP | 4.41838655e-07 | 0.696554568 | 70114 | 0 | 0 |
| CVRP | 0.000181851601 | 0.297570543 | 70065 | 14 | 0 |
| ATSP | 9.45244283e-09 | 1.60211231 | 70114 | 0 | 0 |
| OVRP | 0.000129339781 | 0.426861294 | 70078 | 8 | 0 |
| VRPB | 2.05600189e-07 | 1.21260123 | 70114 | 0 | 0 |
| VRPL | 4.75062193e-05 | 0.704717588 | 70106 | 3 | 0 |
| VRPTW | 2.84633881e-08 | 1.62154445 | 70114 | 0 | 0 |
| OVRPTW | 9.22526757e-08 | 2.02022057 | 70114 | 0 | 0 |
| OVRPB | 2.71139441e-07 | 1.30473739 | 70114 | 0 | 0 |
| OVRPL | 1.13918022e-05 | 0.41366244 | 70107 | 2 | 0 |
| VRPBL | 5.44658081e-08 | 1.19594901 | 70114 | 0 | 0 |
| VRPBTW | 5.94237015e-09 | 1.57568723 | 70114 | 0 | 0 |
| VRPLTW | 4.18617168e-08 | 1.68073232 | 70114 | 0 | 0 |
| OVRPBL | 1.31781168e-07 | 1.45814808 | 70114 | 0 | 0 |
| OVRPBTW | 5.99572747e-09 | 2.09696943 | 70114 | 0 | 0 |
| OVRPLTW | 9.04383546e-08 | 1.97990258 | 70114 | 0 | 0 |
| VRPBLTW | 1.02661279e-08 | 1.54220504 | 70114 | 0 | 0 |
| OVRPBLTW | 8.31244407e-10 | 1.96325757 | 70114 | 0 | 0 |

| Task | Rule | Native Top1 % | Mean cost | Actual regret % |
|---|---|---:|---:|---:|
| TSP | identity | 41.864677525 | 7.646266166844 | 0.460728838 |
| TSP | fitted | 36.182502781 | 7.648682935475 | 0.491359795 |
| TSP | reference | 36.182502781 | 7.648682935475 | 0.491359795 |
| CVRP | identity | 33.432695325 | 18.060584396135 | 1.418565398 |
| CVRP | fitted | 23.470348290 | 18.103344622534 | 1.576025230 |
| CVRP | reference | 23.467495793 | 18.103372980178 | 1.576127102 |
| ATSP | identity | 74.169210143 | 1.641183333554 | 0.449629035 |
| ATSP | fitted | 70.553669738 | 1.652769878684 | 1.191578606 |
| ATSP | reference | 70.553669738 | 1.652769878684 | 1.191578606 |
| OVRP | identity | 37.595915224 | 7.449545426316 | 0.715544932 |
| OVRP | fitted | 26.204467011 | 7.531907647618 | 1.727793493 |
| OVRP | reference | 26.203040762 | 7.531908554665 | 1.727802935 |
| VRPB | identity | 80.303505719 | 9.357769237550 | 0.416156834 |
| VRPB | fitted | 59.955215791 | 9.443277613556 | 1.351685800 |
| VRPB | reference | 59.955215791 | 9.443277613556 | 1.351685800 |
| VRPL | identity | 33.334284166 | 12.396519049380 | 0.462565780 |
| VRPL | fitted | 24.221981345 | 12.491968681126 | 1.277512091 |
| VRPL | reference | 24.221981345 | 12.491964072317 | 1.277466553 |
| VRPTW | identity | 80.236472031 | 18.815456747965 | 0.311791195 |
| VRPTW | fitted | 28.858715806 | 19.380497469512 | 3.679450190 |
| VRPTW | reference | 28.858715806 | 19.380497469512 | 3.679450190 |
| OVRPTW | identity | 83.864848675 | 10.119165380941 | 0.245813430 |
| OVRPTW | fitted | 29.994009756 | 10.473170026127 | 4.270396298 |
| OVRPTW | reference | 29.994009756 | 10.473170026127 | 4.270396298 |
| OVRPB | identity | 88.316170808 | 6.363213157540 | 0.271242675 |
| OVRPB | fitted | 67.947913398 | 6.439794356282 | 1.375688347 |
| OVRPB | reference | 67.947913398 | 6.439794356282 | 1.375688347 |
| OVRPL | identity | 37.618735203 | 7.451387656372 | 0.731713316 |
| OVRPL | fitted | 28.099951508 | 7.520981286683 | 1.559391941 |
| OVRPL | reference | 28.099951508 | 7.520981673891 | 1.559399824 |
| VRPBL | identity | 79.400690304 | 9.402794261883 | 0.428708035 |
| VRPBL | fitted | 62.165901247 | 9.462229175304 | 1.054626762 |
| VRPBL | reference | 62.165901247 | 9.462229175304 | 1.054626762 |
| VRPBTW | identity | 79.551872665 | 19.740816343707 | 0.444766024 |
| VRPBTW | fitted | 37.409076647 | 20.531447752090 | 5.000669578 |
| VRPBTW | reference | 37.409076647 | 20.531447752090 | 5.000669578 |
| VRPLTW | identity | 80.400490630 | 18.775183100523 | 0.309772451 |
| VRPLTW | fitted | 28.768862139 | 19.328933103829 | 3.628796013 |
| VRPLTW | reference | 28.768862139 | 19.328933103829 | 3.628796013 |
| OVRPBL | identity | 88.581453062 | 6.359180160072 | 0.263211844 |
| OVRPBL | fitted | 68.698120204 | 6.428586352945 | 1.224070425 |
| OVRPBL | reference | 68.698120204 | 6.428586352945 | 1.224070425 |
| OVRPBTW | identity | 83.251561742 | 10.625827688674 | 0.336848744 |
| OVRPBTW | fitted | 37.196565593 | 11.145657536277 | 6.124512092 |
| OVRPBTW | reference | 37.196565593 | 11.145657536277 | 6.124512092 |
| OVRPLTW | identity | 83.773568759 | 10.109913379188 | 0.248052857 |
| OVRPLTW | fitted | 31.103631229 | 10.446010031206 | 4.064821534 |
| OVRPLTW | reference | 31.103631229 | 10.446010031206 | 4.064821534 |
| VRPBLTW | identity | 79.604643866 | 19.771468264366 | 0.440427766 |
| VRPBLTW | fitted | 36.787232222 | 20.590776404846 | 5.172419376 |
| VRPBLTW | reference | 36.787232222 | 20.590776404846 | 5.172419376 |
| OVRPBLTW | identity | 83.352825399 | 10.627813376478 | 0.336162512 |
| OVRPBLTW | fitted | 37.677211399 | 11.142806959920 | 6.101167766 |
| OVRPBLTW | reference | 37.677211399 | 11.142806959920 | 6.101167766 |

Fixed14 fitted arithmetic bounds: Top1 <= 57.405479330%; actual regret >= 2.485069032%.
These bounds hold those14 fitted rules fixed; they are not a claim of fitted/reference prediction equality.
No further fixed-R76 intercept, scale, margin, prior or numerical-precision trial is authorized.
Goal stays active, waiting_R77_final_trainval_distributions; R76 identity remains incumbent.
