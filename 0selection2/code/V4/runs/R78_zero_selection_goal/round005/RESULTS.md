# R78 Round005 Fixed Equal-Probability Screen

Outcome: fixed_equal_probability_route_closed_negative.
One complete70110-row VAL candidate /18 tasks /chunks1024. No TRAIN readout or optimizer update.
CPU cached-screen runtime: 0.290172s; not real deployment latency.

| Rule | Native Top1 % | Raw mean cost | Actual regret % | CE |
|---|---:|---:|---:|---:|
| B | 70.894308943 | 11.395558343544 | 0.401326745 | 0.713641610 |
| ensemble | 70.753102268 | 11.395610637023 | 0.402752679 | 0.711996996 |
| fallback_B | 70.894308943 | 11.395558343544 | 0.401326745 | 0.713641610 |

Gate: {"native_top1_improves": false, "raw_mean_cost_not_worse": false, "actual_regret_not_worse": false}.
Selected whole18 rule: fallback_B.
CE_E is descriptive -log(normalized probability mixture); B CE is the unchanged locked VAL record.
Changed decisions: {"changed_predictions": 2444, "corrected": 751, "harmed": 850, "wrong_to_wrong": 843, "exact_cost_tie_changes": 0}.
Cold methods: {"B": {"zero_with_winner_pairs": 30, "zero_winner_count": 2648, "zero_winner_coverage_pct": 3.7769219797461133}, "ensemble": {"zero_with_winner_pairs": 33, "zero_winner_count": 3268, "zero_winner_coverage_pct": 4.661246612466124}}.
Per-task/category adverse FP64 cost changes remain visible; no cross-task raw-cost pooling.
Real deployment needs both frozen model forwards/weight sets, about2x network arithmetic; actual latency/memory is unmeasured.
No TEST/LIB, new weight/temperature/logit rule, taskwise rescue, GPU benchmark, model API, training or follow-up screen.
File goal remains active; native and legacy app goals are not queried or changed.
