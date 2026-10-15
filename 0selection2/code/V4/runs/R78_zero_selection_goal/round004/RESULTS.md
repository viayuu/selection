# R78 Round004 Fixed Endpoint

Outcome: finite_frozen_score_cost_head_trial_closed_negative.
B, actual fitted heads and whole-version B fallback are separate. No taskwise mixture.
CE below is unchanged frozen-B CE, not softmax(-g) or a head-calibration metric.

| Split | Rule | Top1 % | Raw mean cost | Actual regret % | Frozen B CE | Zero/winner pairs |
|---|---|---:|---:|---:|---:|---:|
| train | B | 71.176702703 | 11.365977443942 | 0.393011700 | 0.704272755 | 25 |
| train | fitted | 68.119776364 | 11.378174190314 | 0.479358581 | 0.704272755 | 21 |
| train | fallback_B | 71.176702703 | 11.365977443942 | 0.393011700 | 0.704272755 | 25 |
| val | B | 70.894308943 | 11.395558343544 | 0.401326745 | 0.713641610 | 30 |
| val | fitted | 67.947511054 | 11.407802464375 | 0.492726054 | 0.713641610 | 23 |
| val | fallback_B | 70.894308943 | 11.395558343544 | 0.401326745 | 0.713641610 | 30 |

Gate: {"native_top1_improves": false, "raw_mean_cost_not_worse": false, "actual_regret_not_worse": false}.
Selected whole18 rule: fallback_B.
Successful optimizer updates: 24840; fixed1380 per nondegenerate task, batch1024.
Native method counts and corrected/harmed/wrong-to-wrong FP64 cost changes are saved as CSV.
All18 matrices were locked before the only candidate VAL endpoint.
No TEST/LIB permission, optimum certificate, parameter scan, budget extension or automatic next trial.
Ongoing file goal remains active. Native/legacy app goal statuses are untouched.
