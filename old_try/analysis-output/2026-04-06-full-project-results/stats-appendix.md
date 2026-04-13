# Statistical Appendix

## Primary Metrics
- Primary metric: `mean_cost`，lower is better。
- Supporting metric: `top1_accuracy`。
- Current branch-specific auxiliary metric: `gap_to_oracle = selected_cost - best_cost`。

## Sample Structure
- Legacy family: unit of analysis = held-out test instance, `n = 3000`。
- Current NSS family: unit of analysis = held-out test instance, `n = 2000`。
- Stage2 tuning family: unit of analysis = held-out test instance, `n = 1500`。
- Early RL branches: no valid repeated-measure protocol for inferential testing; descriptive only。

## Baseline Derivation for Legacy Family
- `single_best_per_problem` arms were re-derived from the full `2l2r alpha0` trace: `{'cvrp': 'icam', 'tsp': 'lehd'}`。
- `single_best_global` arm on the same split was `pointerformer`。

## Descriptive Statistics
### Legacy family
| name                           |   count |   mean_cost |   sd_cost |   ci_low |   ci_high |   top1_accuracy |   top1_ci_low |   top1_ci_high |
|:-------------------------------|--------:|------------:|----------:|---------:|----------:|----------------:|--------------:|---------------:|
| Oracle                         |    3000 |     11.7622 |    4.1969 |  11.6120 |   11.9125 |          1.0000 |        0.9987 |         1.0000 |
| SingleBestPerProblem           |    3000 |     11.8096 |    4.2331 |  11.6581 |   11.9612 |          0.4273 |        0.4097 |         0.4451 |
| 2cmab2 Neural-LinUCB (legacy)  |    3000 |     11.8110 |    4.2371 |  11.6593 |   11.9627 |          0.3033 |        0.2871 |         0.3200 |
| 2cmab2 NeuralUCB-Diag (legacy) |    3000 |     11.8117 |    4.2428 |  11.6598 |   11.9636 |          0.3900 |        0.3727 |         0.4076 |
| 2l2r Best-vs-All alpha0        |    3000 |     11.8117 |    4.2428 |  11.6598 |   11.9636 |          0.3900 |        0.3727 |         0.4076 |
| 2l2r Listwise                  |    3000 |     11.8122 |    4.2423 |  11.6603 |   11.9641 |          0.3557 |        0.3387 |         0.3730 |
| 2l2r Pairwise                  |    3000 |     11.8130 |    4.2416 |  11.6612 |   11.9648 |          0.2753 |        0.2596 |         0.2916 |
| 2l2r Best-vs-All alpha2        |    3000 |     11.8163 |    4.2327 |  11.6647 |   11.9678 |          0.1940 |        0.1802 |         0.2085 |

### Current NSS family
| name                        |   count |   mean_cost |   sd_cost |   ci_low |   ci_high |   top1_accuracy |
|:----------------------------|--------:|------------:|----------:|---------:|----------:|----------------:|
| Oracle NSS                  |    2000 |     13.1191 |  nan      | nan      |  nan      |          1.0000 |
| Neural-LinUCB NSS safe_bs64 |    2000 |     13.2027 |    8.0247 |  12.8508 |   13.5546 |          0.4400 |
| Neural-LinUCB NSS server    |    2000 |     13.2076 |    8.0399 |  12.8551 |   13.5602 |          0.4410 |
| NeuralUCB-Diag NSS          |    2000 |     13.2944 |    8.0195 |  12.9428 |   13.6461 |          0.2785 |
| SingleBestPerProblem NSS    |    2000 |     13.3093 |  nan      | nan      |  nan      |          0.3380 |

### Stage2 tuning family
| tag                        |   mean_cost |   top1_accuracy |
|:---------------------------|------------:|----------------:|
| s2_00_baseline_default     |     11.8502 |          0.2787 |
| s2_02_more_explore         |     11.8504 |          0.3447 |
| s2_04_full_linear_history  |     11.8505 |          0.3607 |
| s2_03_less_explore         |     11.8508 |          0.2127 |
| s2_06_freeze_graph_encoder |     11.8511 |          0.3207 |
| s2_07_low_lr               |     11.8513 |          0.3167 |
| s2_05_autosaea_reward      |     11.8525 |          0.3387 |
| s2_01_less_rep_fit         |     11.8540 |          0.2760 |

## Assumption Checks and Inferential Tests
### Legacy family planned contrasts
| contrast                                                  |    n |   delta_mean_cost | assumption_check    | test                 |    statistic |   p_value |   p_holm |   effect_size |
|:----------------------------------------------------------|-----:|------------------:|:--------------------|:---------------------|-------------:|----------:|---------:|--------------:|
| 2cmab2 Neural-LinUCB (legacy) vs SingleBestPerProblem     | 3000 |            0.0014 | Shapiro p=0         | Wilcoxon signed-rank | 1139133.0000 |    0.3019 |   0.9389 |        0.0256 |
| 2cmab2 NeuralUCB-Diag (legacy) vs SingleBestPerProblem    | 3000 |            0.0021 | Shapiro p=0         | Wilcoxon signed-rank |  540774.5000 |    0.1878 |   0.9389 |       -0.0393 |
| 2l2r Pairwise vs SingleBestPerProblem                     | 3000 |            0.0034 | Shapiro p=4.204e-45 | Wilcoxon signed-rank | 2020278.5000 |    0.8013 |   0.9389 |        0.0054 |
| 2l2r Listwise vs SingleBestPerProblem                     | 3000 |            0.0026 | Shapiro p=2.803e-45 | Wilcoxon signed-rank | 1994997.5000 |    0.3083 |   0.9389 |       -0.0220 |
| 2l2r Best-vs-All alpha0 vs SingleBestPerProblem           | 3000 |            0.0021 | Shapiro p=0         | Wilcoxon signed-rank |  540774.5000 |    0.1878 |   0.9389 |       -0.0393 |
| 2l2r Best-vs-All alpha2 vs SingleBestPerProblem           | 3000 |            0.0066 | Shapiro p=1.195e-36 | Wilcoxon signed-rank | 1716356.5000 |    0.0000 |   0.0000 |        0.1873 |
| 2l2r Best-vs-All alpha0 vs 2cmab2 Neural-LinUCB (legacy)  | 3000 |            0.0007 | Shapiro p=0         | Wilcoxon signed-rank |  751590.0000 |    0.0003 |   0.0015 |       -0.0988 |
| 2l2r Best-vs-All alpha0 vs 2cmab2 NeuralUCB-Diag (legacy) | 3000 |            0.0000 | Shapiro p=1         | paired t-test        |     nan      |  nan      | nan      |        0.0000 |

### Current NSS family planned contrasts
| contrast                                                |    n |   delta_mean_cost | assumption_check   | test                 |   statistic |   p_value |   p_holm |   effect_size |
|:--------------------------------------------------------|-----:|------------------:|:-------------------|:---------------------|------------:|----------:|---------:|--------------:|
| Neural-LinUCB NSS safe_bs64 vs NeuralUCB-Diag NSS       | 2000 |           -0.0917 | Shapiro p=0        | Wilcoxon signed-rank | 165444.0000 |    0.0000 |   0.0000 |       -0.5354 |
| Neural-LinUCB NSS server vs NeuralUCB-Diag NSS          | 2000 |           -0.0868 | Shapiro p=0        | Wilcoxon signed-rank | 169715.0000 |    0.0000 |   0.0000 |       -0.5137 |
| Neural-LinUCB NSS safe_bs64 vs Neural-LinUCB NSS server | 2000 |           -0.0049 | Shapiro p=0        | Wilcoxon signed-rank | 115191.5000 |    0.9628 |   0.9628 |        0.0021 |

### Stage2 tuning focused contrasts
| contrast                                            |    n |   delta_mean_cost | assumption_check    | test                 |   statistic |   p_value |   p_holm |   effect_size |
|:----------------------------------------------------|-----:|------------------:|:--------------------|:---------------------|------------:|----------:|---------:|--------------:|
| s2_00 baseline_default vs s2_04 full_linear_history | 1500 |           -0.0003 | Shapiro p=1.275e-43 | Wilcoxon signed-rank | 150755.5000 |    0.3312 |   0.6623 |        0.0399 |
| s2_00 baseline_default vs s2_02 more_explore        | 1500 |           -0.0003 | Shapiro p=2.522e-44 | Wilcoxon signed-rank | 131456.0000 |    0.3987 |   0.6623 |        0.0359 |

## Historical / Descriptive-Only Branches
### 2cmab legacy text reports
| line                         |   overall_top1 |   overall_mean_cost |   tsp_top1 |   tsp_mean_cost |   cvrp_top1 |   cvrp_mean_cost |
|:-----------------------------|---------------:|--------------------:|-----------:|----------------:|------------:|-----------------:|
| 2cmab Neural-LinUCB step7000 |         0.2217 |             12.0186 |     0.1913 |          7.8779 |      0.2520 |          16.1592 |
| 2cmab Neural-LinUCB final    |         0.2000 |             12.0405 |     0.1560 |          7.8819 |      0.2440 |          16.1990 |
| 2cmab NeuralTS final         |         0.2947 |             11.9855 |     0.2040 |          8.1397 |      0.3853 |          15.8312 |

### 1online smoke runs
| run                          |   train_steps |   train_improvement |   original_improvement |   augmented_improvement |   reward_mean_train |
|:-----------------------------|--------------:|--------------------:|-----------------------:|------------------------:|--------------------:|
| smoke_kroA100                |             1 |              0.0000 |                 0.0000 |                  0.0000 |              0.0000 |
| smoke_kroA100_prefereasynco  |             1 |              0.0000 |                 0.0000 |                  0.0000 |              0.0000 |
| smoke_kroA100_prefereasynco2 |             1 |              0.0000 |                 0.0000 |                  0.0000 |              0.0000 |
| smoke_kroA100_rewardfix      |             1 |              0.0000 |                 0.0000 |                  0.0000 |             -5.4559 |

## Blockers and Limits
- No seed-level repeated runs for any main family -> no seed-wise uncertainty or cross-seed significance.
- Current NSS traces do not store all-arm `true_cost`, so `single_best_per_problem` cannot be paired instance-wise there.
- Benchmark outputs provide only selector trace plus baseline aggregates -> no paired benchmark significance tests.
- `1two_gate` and `1online` are prototype-scale artifacts and should not be overclaimed.

## Interpretation Guardrails
- Observation layer: report exact deltas in `mean_cost` and `top1`.
- Support layer: only the paired tests above count as inferential evidence.
- Boundary layer: any claim about robustness across seeds, datasets, or deployment remains unsupported.
