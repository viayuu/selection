# Solver Selection 双线结果汇总

日期: 2026-04-21

## 口径说明

- 这版继续按 [观测指标.md](/public/home/zhoucl/shiys/0selection/观测指标/观测指标.md) 的口径组织，并尽量保持和 [solver_selection_results_summary_2026-04-19.md](/public/home/zhoucl/shiys/solver_selection_results_summary_2026-04-19.md)、[solver_selection_results_summary_2026-04-20.md](/public/home/zhoucl/shiys/solver_selection_results_summary_2026-04-20.md) 一样的结构。
- `0selection` 这次不再沿用旧 `R18` 主表，而是改用我刚补出的 [R40D test analysis](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R40D_NSS_rr_microstep_seed0/analysis_test_20260421/analysis_test.json) 作为 detailed 长表来源。为支持 `R40D` 这类 NSS checkpoint，我同步修了 [analyze.py](/public/home/zhoucl/shiys/0selection/code/unified_selector/analyze.py) 的模型恢复逻辑，使其能按 checkpoint args 恢复 `encoder_type/use_film/local_head/residual_adapter` 等结构。
- `0selection` 的显著性结论仍以 [R40D test_eval_ep6](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R40D_NSS_rr_microstep_seed0/test_eval_ep6/test_eval.md) 为准：`test macro_top1 = 0.5226`，相对 `R18 = 0.5166`，`Δ = +0.0060 [+0.0001, +0.0118]`，`p(Δ>0)=0.977`。新 detailed analysis 给出的 macro 为 `0.5228 / -0.0283% vs SBS`，和 bootstrap 报告只差 `0.0002`，我这里把二者同时注明。
- `0selection2` 的 detailed 长表仍使用当前最新且最强的正式 test 报表 [R25 analysis_test](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R25_multigen_partition_support_seed2/analysis_test_20260420/analysis_test.json)。
- `0selection2` 在 2026-04-21 之后的 `R26a/R26d/R27a/R29a/R30a/R30b`，大多只有 val 或 smoke 级诊断；它们会写进摘要结论，但不会替换 `R25` 作为 mainline test 主表。对应总整理见 [2026-04-21-post-review-results.md](/public/home/zhoucl/shiys/0selection2/review-stage/2026-04-21-post-review-results.md)。
- 目前 detailed JSON 仍然没有直接物化 `best top2 / best top3` 两列，所以长表继续沿用 `best top1 (oracle)` 这一列，风格和 `04-20` 版保持一致。

## 一页总览

| Line | Setting | top1 | top2 | top3 | macro vs SBS | 备注 |
|---|---|---:|---:|---:|---:|---|
| 0selection | R40D latest test-detailed | 0.5228 | 0.8758 | 0.9858 | -0.0283% | 当前主表；同线 bootstrap 报告 `0.5226`, `Δ vs R18=+0.0060`, first significant positive |
| 0selection | R41B residual adapter test | 0.5197 | - | - | - | `Δ vs R18 = +0.0031 [-0.0010, +0.0072]`, near-positive but not significant |
| 0selection | R42S keep-SBS reranker test | 0.5162 | - | - | - | `Δ vs SBS = +0.0017 [-0.0043, +0.0077]`, non-significant |
| 0selection | R42S_fixpair test | 0.5143 | - | - | - | `Δ vs SBS = -0.0002 [-0.0093, +0.0090]`, family-level theta collapsed to near-SBS |
| 0selection2 | R25 latest test-confirmed | 0.5353 | 0.8848 | 0.9871 | -0.1247% | 当前 mainline 最强正式 test 结果 |
| 0selection2 | R29a-R30b late diagnostics (val) | 0.5378-0.5383 | 0.9074 | 0.9837 | about -0.142% | hidden_winner_mass 固定在 `0.0779`, 仍未把 hidden winners 激活出来 |
| 0selection2 | R30a switchgate smoke-only test | 0.7180 | 0.8057 | 0.8707 | -0.7425% | smoke only, not promoted to mainline |

## 0selection: 当前结果解读

- 这条线在 4 月 21 日发生了一个关键变化：它不再只是 `diagnostic paper` 的旧 base，而是出现了第一条真正跨过 95% 线的正式正结果。对应 reviewer 汇总见 [4.20_R40DE_R41B_R42S_to_reviewer.md](/public/home/zhoucl/shiys/0selection/4.20_R40DE_R41B_R42S_to_reviewer.md) 和 [REVIEW_STATE.json](/public/home/zhoucl/shiys/0selection/review-stage/REVIEW_STATE.json)。
- 当前 detailed 主表来自 `R40D NSS + round_robin`：
  - `macro_top1 = 0.5228`
  - `macro_top2 = 0.8758`
  - `macro_top3 = 0.9858`
  - `macro_vs_sbs = -0.0283%`
  - `macro_vbs_gap_closed = +3.32%`
  - `n_problems_beat_sbs = 13`
  - `n_problems_match_sbs = 11`
- 同一条 run 的 bootstrap 报告给出：`test macro_top1 = 0.5226`，相对 `R18 = 0.5166` 有 `+0.0060` 提升，95% CI `[+0.0001, +0.0118]`，`p(Δ>0)=0.977`。也就是说，这条线现在的核心故事已经从“global null + diagnostic”推进到“修正 blockwise scheduler bug 后，NSS from-scratch 首次显著超过旧 base”。
- 但也要保持克制：
  - 这仍然是 `single-seed` 正结果，reviewer 里明确的下一步还是 multi-seed R40D。
  - `R41B` 只是 near-positive，`R42S/R42S_fixpair` 都还没有形成稳定 improvement story。
  - 所以 `0selection` 现在更像：**已经找到一条真能过线的训练主线，但论文叙事还需要 multi-seed 和 follow-up 来把这条线坐实。**
- 辅助结果也值得一起看：
  - [R41B test_eval](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R41B_R18_residual_adapter_seed0/test_eval/test_eval.md)：残差 adapter `0.5197`，差一点但没过线。
  - [R42S report](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R42S_keepSBS_fullpool_rerank_seed0/report.md)：KEEP_SBS reranker 在若干 hard MVRP 上能赢，但总体仍被 val-to-test 漂移抵消。
  - [R42S_fixpair report](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R42S_fixpair_seed0/report.md)：修正 pairwise sign 后，family-level theta 反而把大部分 MVRP 压回 near-SBS。

### 0selection 详细观测指标报表

详细 JSON 见 [analysis_test.json](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R40D_NSS_rr_microstep_seed0/analysis_test_20260421/analysis_test.json)。下面直接保留核心长表。

# 观测指标报表

Split: **test**  ·  18 problems

## A. Selector top-k vs oracle, and per-method top1 winrate

| Problem | selector top1 | top2 | top3 | best top1 (oracle) | beat_top1 | lose_top1 |
|---|---:|---:|---:|---:|---|---|
| TSP | 0.722 | 0.908 | 0.977 | 0.706 | DACT, DIFUSCO, ELG, GLOP, INVIT, LEHD, LIH, OMNI, T2T, UDC | none |
| CVRP | 0.419 | 0.753 | 0.939 | 0.413 | DACT, ELG, GLOP, ICAM, INVIT, LEHD, OMNI, RELD_CVRP, UDC | none |
| ATSP | 0.701 | 0.996 | 1.000 | 0.701 | GLOP, MATNET | none |
| OVRP | 0.482 | 0.827 | 0.957 | 0.481 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPB | 0.495 | 0.881 | 0.991 | 0.483 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPL | 0.453 | 0.750 | 0.943 | 0.454 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPTW | 0.478 | 0.875 | 1.000 | 0.473 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPTW | 0.499 | 0.888 | 1.000 | 0.480 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPB | 0.557 | 0.934 | 0.991 | 0.561 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPL | 0.497 | 0.827 | 0.957 | 0.494 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPBL | 0.551 | 0.885 | 0.995 | 0.537 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPBTW | 0.481 | 0.861 | 1.000 | 0.488 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPLTW | 0.506 | 0.881 | 1.000 | 0.502 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPBL | 0.572 | 0.933 | 0.996 | 0.546 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPBTW | 0.496 | 0.890 | 1.000 | 0.461 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPLTW | 0.487 | 0.892 | 1.000 | 0.492 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPBLTW | 0.516 | 0.884 | 0.999 | 0.510 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPBLTW | 0.499 | 0.899 | 1.000 | 0.478 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| **MACRO** | **0.5228** | **0.8758** | **0.9858** |  |  |  |

## B. Mean cost - selector vs SBS, and per-method mean cost

| Problem | selector mean_cost | SBS mean_cost (best) | vs_sbs (%) | beat_mean_cost | lose_mean_cost |
|---|---:|---:|---:|---|---|
| TSP | 8.1358 | 8.1579 | -0.271% | DACT, DIFUSCO, ELG, GLOP, INVIT, LEHD, LIH, OMNI, T2T, UDC | none |
| CVRP | 18.3286 | 18.3323 | -0.021% | DACT, ELG, GLOP, ICAM, INVIT, LEHD, OMNI, RELD_CVRP, UDC | none |
| ATSP | 1.7159 | 1.7159 | +0.000% | GLOP, MATNET | none |
| OVRP | 7.6089 | 7.6094 | -0.008% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPB | 9.2919 | 9.2951 | -0.034% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPL | 12.4556 | 12.4568 | -0.010% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPTW | 18.5752 | 18.5741 | +0.006% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPTW | 10.1170 | 10.1233 | -0.062% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPB | 6.2960 | 6.2963 | -0.005% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPL | 7.5938 | 7.5929 | +0.013% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPBL | 9.2507 | 9.2492 | +0.016% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPBTW | 19.5293 | 19.5263 | +0.015% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPLTW | 18.4253 | 18.4269 | -0.009% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPBL | 6.3032 | 6.3068 | -0.057% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPBTW | 10.6065 | 10.6090 | -0.024% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPLTW | 10.1028 | 10.1036 | -0.008% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPBLTW | 19.6732 | 19.6770 | -0.019% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPBLTW | 10.5894 | 10.5928 | -0.032% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| **MACRO vs_sbs** |  |  | **-0.028%** |  |  |

## C. Per-method comparison (mean cost on val/test)

### TSP  (selector top1=0.722, mean_cost=8.1358)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| DACT | 0.000 | 10.1287 | 0 |
| DIFUSCO | 0.056 | 9.2355 | 0 |
| ELG | 0.007 | 16.9025 | 0 |
| GLOP | 0.014 | 8.6876 | 0 |
| INVIT | 0.007 | 8.7046 | 0 |
| LEHD | 0.706 | 8.1579 | 778 |
| LIH | 0.000 | 116.1080 | 0 |
| OMNI | 0.210 | 8.3188 | 222 |
| T2T | 0.000 | 9.4649 | 0 |
| UDC | 0.000 | 13.0844 | 0 |

### CVRP  (selector top1=0.419, mean_cost=18.3286)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| DACT | 0.000 | 22.1375 | 0 |
| ELG | 0.001 | 61.4521 | 0 |
| GLOP | 0.000 | 22.0717 | 0 |
| ICAM | 0.154 | 18.5471 | 39 |
| INVIT | 0.000 | 20.8204 | 0 |
| LEHD | 0.261 | 18.5653 | 37 |
| OMNI | 0.171 | 18.6303 | 0 |
| RELD_CVRP | 0.413 | 18.3323 | 924 |
| UDC | 0.000 | 31.3299 | 0 |

### ATSP  (selector top1=0.701, mean_cost=1.7159)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| GLOP | 0.024 | 1.9203 | 0 |
| MATNET | 0.275 | 2.3139 | 0 |
| MATPOENET | 0.701 | 1.7159 | 1000 |

### OVRP  (selector top1=0.482, mean_cost=7.6089)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.090 | 8.4301 | 0 |
| MVMOE | 0.047 | 9.1483 | 0 |
| RELD_MOEL | 0.382 | 7.6451 | 58 |
| RELD_MTL | 0.481 | 7.6094 | 942 |

### VRPB  (selector top1=0.495, mean_cost=9.2919)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.022 | 10.7014 | 0 |
| MVMOE | 0.059 | 11.5846 | 0 |
| RELD_MOEL | 0.436 | 9.3303 | 269 |
| RELD_MTL | 0.483 | 9.2951 | 731 |

### VRPL  (selector top1=0.453, mean_cost=12.4556)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.127 | 13.3434 | 0 |
| MVMOE | 0.063 | 14.1639 | 0 |
| RELD_MOEL | 0.356 | 12.4897 | 159 |
| RELD_MTL | 0.454 | 12.4568 | 841 |

### VRPTW  (selector top1=0.478, mean_cost=18.5752)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 23.2402 | 0 |
| MVMOE | 0.099 | 20.2432 | 0 |
| RELD_MOEL | 0.428 | 18.6273 | 137 |
| RELD_MTL | 0.473 | 18.5741 | 863 |

### OVRPTW  (selector top1=0.499, mean_cost=10.1170)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 14.5072 | 0 |
| MVMOE | 0.091 | 11.4039 | 0 |
| RELD_MOEL | 0.429 | 10.1432 | 542 |
| RELD_MTL | 0.480 | 10.1233 | 458 |

### OVRPB  (selector top1=0.557, mean_cost=6.2960)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.020 | 7.3061 | 0 |
| MVMOE | 0.014 | 8.1414 | 0 |
| RELD_MOEL | 0.405 | 6.3388 | 253 |
| RELD_MTL | 0.561 | 6.2963 | 747 |

### OVRPL  (selector top1=0.497, mean_cost=7.5938)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.085 | 8.2996 | 0 |
| MVMOE | 0.036 | 8.9026 | 0 |
| RELD_MOEL | 0.385 | 7.6341 | 89 |
| RELD_MTL | 0.494 | 7.5929 | 911 |

### VRPBL  (selector top1=0.551, mean_cost=9.2507)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.013 | 10.6000 | 0 |
| MVMOE | 0.048 | 11.2127 | 0 |
| RELD_MOEL | 0.402 | 9.3095 | 326 |
| RELD_MTL | 0.537 | 9.2492 | 674 |

### VRPBTW  (selector top1=0.481, mean_cost=19.5293)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 24.6857 | 0 |
| MVMOE | 0.118 | 21.3102 | 0 |
| RELD_MOEL | 0.394 | 19.6309 | 79 |
| RELD_MTL | 0.488 | 19.5263 | 921 |

### VRPLTW  (selector top1=0.506, mean_cost=18.4253)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 23.1636 | 0 |
| MVMOE | 0.088 | 20.0123 | 0 |
| RELD_MOEL | 0.410 | 18.4951 | 179 |
| RELD_MTL | 0.502 | 18.4269 | 821 |

### OVRPBL  (selector top1=0.572, mean_cost=6.3032)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.010 | 7.2184 | 0 |
| MVMOE | 0.016 | 7.8678 | 0 |
| RELD_MOEL | 0.428 | 6.3443 | 274 |
| RELD_MTL | 0.546 | 6.3068 | 726 |

### OVRPBTW  (selector top1=0.496, mean_cost=10.6065)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 15.2537 | 0 |
| MVMOE | 0.095 | 11.8587 | 0 |
| RELD_MOEL | 0.444 | 10.6443 | 354 |
| RELD_MTL | 0.461 | 10.6090 | 646 |

### OVRPLTW  (selector top1=0.487, mean_cost=10.1028)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 14.4558 | 0 |
| MVMOE | 0.078 | 11.3122 | 0 |
| RELD_MOEL | 0.430 | 10.1273 | 363 |
| RELD_MTL | 0.492 | 10.1036 | 637 |

### VRPBLTW  (selector top1=0.516, mean_cost=19.6732)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.001 | 24.7936 | 0 |
| MVMOE | 0.107 | 21.5061 | 0 |
| RELD_MOEL | 0.382 | 19.7998 | 89 |
| RELD_MTL | 0.510 | 19.6770 | 911 |

### OVRPBLTW  (selector top1=0.499, mean_cost=10.5894)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 15.2835 | 0 |
| MVMOE | 0.075 | 12.0405 | 0 |
| RELD_MOEL | 0.447 | 10.6280 | 282 |
| RELD_MTL | 0.478 | 10.5928 | 718 |

## D. Arm distribution (how often selector picks each method)

- **TSP**: DACT:0.0% | DIFUSCO:0.0% | ELG:0.0% | GLOP:0.0% | INVIT:0.0% | LEHD:77.8% | LIH:0.0% | OMNI:22.2% | T2T:0.0% | UDC:0.0%
- **CVRP**: DACT:0.0% | ELG:0.0% | GLOP:0.0% | ICAM:3.9% | INVIT:0.0% | LEHD:3.7% | OMNI:0.0% | RELD_CVRP:92.4% | UDC:0.0%
- **ATSP**: GLOP:0.0% | MATNET:0.0% | MATPOENET:100.0%
- **OVRP**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:5.8% | RELD_MTL:94.2%
- **VRPB**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:26.9% | RELD_MTL:73.1%
- **VRPL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:15.9% | RELD_MTL:84.1%
- **VRPTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:13.7% | RELD_MTL:86.3%
- **OVRPTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:54.2% | RELD_MTL:45.8%
- **OVRPB**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:25.3% | RELD_MTL:74.7%
- **OVRPL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:8.9% | RELD_MTL:91.1%
- **VRPBL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:32.6% | RELD_MTL:67.4%
- **VRPBTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:7.9% | RELD_MTL:92.1%
- **VRPLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:17.9% | RELD_MTL:82.1%
- **OVRPBL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:27.4% | RELD_MTL:72.6%
- **OVRPBTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:35.4% | RELD_MTL:64.6%
- **OVRPLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:36.3% | RELD_MTL:63.7%
- **VRPBLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:8.9% | RELD_MTL:91.1%
- **OVRPBLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:28.2% | RELD_MTL:71.8%

### 0selection 图和补充结果入口

- top1 图: [top1_vs_single_methods_test.png](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R40D_NSS_rr_microstep_seed0/analysis_test_20260421/top1_vs_single_methods_test.png)
- mean cost 图: [mean_cost_vs_single_methods_test.png](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R40D_NSS_rr_microstep_seed0/analysis_test_20260421/mean_cost_vs_single_methods_test.png)
- arm 分布图: [arm_distribution_test.png](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R40D_NSS_rr_microstep_seed0/analysis_test_20260421/arm_distribution_test.png)
- 显著性主报告: [test_eval_ep6.md](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R40D_NSS_rr_microstep_seed0/test_eval_ep6/test_eval.md)
- 训练轨迹补充: [eval_epoch6.json](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R40D_NSS_rr_microstep_seed0/eval_epoch6.json)，以及同目录下 `eval_epoch0-8.json`
- 4 月 21 日 reviewer 汇总: [4.20_R40DE_R41B_R42S_to_reviewer.md](/public/home/zhoucl/shiys/0selection/4.20_R40DE_R41B_R42S_to_reviewer.md)

## 0selection2: 当前结果解读

- 这条线目前仍然是 **效果提升主线**。截至现在，真正完成完整 `analysis_test` 的最强正式 run 仍然是 `R25`，而不是 4 月 21 日之后那些 val-only / smoke-only 诊断线。
- 当前主表 `R25` 的关键数字：
  - `macro_top1 = 0.5353`
  - `macro_top2 = 0.8848`
  - `macro_top3 = 0.9871`
  - `macro_vs_sbs = -0.1247%`
  - `macro_vbs_gap_closed = +8.32%`
  - `n_problems_beat_sbs = 13`
  - `n_problems_match_sbs = 8`
  - `macro_final_arm_coverage = 0.5145`
  - `macro_pick_entropy = 0.4432`
  - `macro_hidden_winner_mass = 0.0750`
  - `macro_oracle_support_mass_recall = 0.9250`
- 和更早的 R6 3-seed mean 相比，`R25` 依然是明显更强的 test-confirmed 结果：`0.5353 / 0.8848 / 0.9871 / -0.1247%` 对 `0.5221 / 0.8734 / 0.9833 / -0.0812%`。
- 4 月 21 日之后的新推进并不是没有价值，但性质更偏诊断：
  - [2026-04-21-post-review-results.md](/public/home/zhoucl/shiys/0selection2/review-stage/2026-04-21-post-review-results.md) 里，`R26a` 证明激进 full-pool rerank 可以动起来，但会严重伤底座。
  - `R26d / R27a / R29a / R30a / R30b` 都把模型维持在 near-SBS-safe 区间，但 `hidden_winner_mass` 基本固定在 `0.0779`，没有真正打开 hidden winner collapse。
  - `_smoke_r30a_pairwin_switch` 的 smoke test 虽然出现了 `0.7180 / -0.7425%` 这种非常强的数值，但它仍只是 smoke，不该当成 mainline 结论。
- 所以 `0selection2` 当前最准确的说法还是：**正式 test 主结果依旧是 R25；4 月 21 日之后的工作主要是在定位“为什么 hidden winners 还是出不来”。**

### 0selection2 详细观测指标报表

详细 JSON 见 [analysis_test.json](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R25_multigen_partition_support_seed2/analysis_test_20260420/analysis_test.json)。下面直接保留核心长表。

# 观测指标报表

Split: **test**  ·  18 problems

## A. Selector top-k vs oracle, and per-method top1 winrate

| Problem | selector top1 | top2 | top3 | best top1 (oracle) | beat_top1 | lose_top1 |
|---|---:|---:|---:|---:|---|---|
| TSP | 0.747 | 0.926 | 0.973 | 0.706 | DACT, DIFUSCO, ELG, GLOP, INVIT, LEHD, LIH, OMNI, T2T, UDC | none |
| CVRP | 0.529 | 0.846 | 0.960 | 0.413 | DACT, ELG, GLOP, ICAM, INVIT, LEHD, OMNI, RELD_CVRP, UDC | none |
| ATSP | 0.879 | 0.988 | 1.000 | 0.701 | GLOP, MATNET, MATPOENET | none |
| OVRP | 0.495 | 0.834 | 0.954 | 0.481 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPB | 0.500 | 0.878 | 0.991 | 0.483 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPL | 0.452 | 0.766 | 0.948 | 0.454 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPTW | 0.481 | 0.891 | 1.000 | 0.473 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPTW | 0.458 | 0.885 | 1.000 | 0.480 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPB | 0.540 | 0.937 | 0.991 | 0.561 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPL | 0.486 | 0.833 | 0.960 | 0.494 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPBL | 0.553 | 0.888 | 0.994 | 0.537 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPBTW | 0.497 | 0.870 | 1.000 | 0.488 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPLTW | 0.475 | 0.877 | 1.000 | 0.502 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPBL | 0.551 | 0.931 | 0.997 | 0.546 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPBTW | 0.505 | 0.885 | 1.000 | 0.461 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPLTW | 0.489 | 0.898 | 1.000 | 0.492 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPBLTW | 0.512 | 0.888 | 0.999 | 0.510 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPBLTW | 0.487 | 0.906 | 1.000 | 0.478 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| **MACRO** | **0.5353** | **0.8848** | **0.9871** |  |  |  |

## B. Mean cost - selector vs SBS, and per-method mean cost

| Problem | selector mean_cost | SBS mean_cost (best) | vs_sbs (%) | beat_mean_cost | lose_mean_cost |
|---|---:|---:|---:|---|---|
| TSP | 8.1287 | 8.1579 | -0.357% | DACT, DIFUSCO, ELG, GLOP, INVIT, LEHD, LIH, OMNI, T2T, UDC | none |
| CVRP | 18.2802 | 18.3323 | -0.284% | DACT, ELG, GLOP, ICAM, INVIT, LEHD, OMNI, RELD_CVRP, UDC | none |
| ATSP | 1.6882 | 1.7159 | -1.614% | GLOP, MATNET, MATPOENET | none |
| OVRP | 7.6076 | 7.6094 | -0.024% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPB | 9.2908 | 9.2951 | -0.046% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPL | 12.4543 | 12.4568 | -0.020% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPTW | 18.5728 | 18.5741 | -0.007% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPTW | 10.1282 | 10.1233 | +0.048% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPB | 6.3000 | 6.2963 | +0.059% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPL | 7.5979 | 7.5929 | +0.066% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPBL | 9.2486 | 9.2492 | -0.006% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPBTW | 19.5273 | 19.5263 | +0.005% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPLTW | 18.4298 | 18.4269 | +0.016% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPBL | 6.3059 | 6.3068 | -0.014% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPBTW | 10.6048 | 10.6090 | -0.039% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPLTW | 10.1019 | 10.1036 | -0.016% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| VRPBLTW | 19.6767 | 19.6770 | -0.001% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| OVRPBLTW | 10.5919 | 10.5928 | -0.008% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | none |
| **MACRO vs_sbs** |  |  | **-0.125%** |  |  |

## C. Per-method comparison (mean cost on val/test)

### TSP  (selector top1=0.747, mean_cost=8.1287)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| DACT | 0.000 | 10.1287 | 0 |
| DIFUSCO | 0.056 | 9.2355 | 41 |
| ELG | 0.007 | 16.9025 | 0 |
| GLOP | 0.014 | 8.6876 | 10 |
| INVIT | 0.007 | 8.7046 | 0 |
| LEHD | 0.706 | 8.1579 | 744 |
| LIH | 0.000 | 116.1080 | 0 |
| OMNI | 0.210 | 8.3188 | 205 |
| T2T | 0.000 | 9.4649 | 0 |
| UDC | 0.000 | 13.0844 | 0 |

### CVRP  (selector top1=0.529, mean_cost=18.2802)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| DACT | 0.000 | 22.1375 | 0 |
| ELG | 0.001 | 61.4521 | 0 |
| GLOP | 0.000 | 22.0717 | 0 |
| ICAM | 0.154 | 18.5471 | 160 |
| INVIT | 0.000 | 20.8204 | 0 |
| LEHD | 0.261 | 18.5653 | 56 |
| OMNI | 0.171 | 18.6303 | 269 |
| RELD_CVRP | 0.413 | 18.3323 | 515 |
| UDC | 0.000 | 31.3299 | 0 |

### ATSP  (selector top1=0.879, mean_cost=1.6882)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| GLOP | 0.024 | 1.9203 | 0 |
| MATNET | 0.275 | 2.3139 | 250 |
| MATPOENET | 0.701 | 1.7159 | 750 |

### OVRP  (selector top1=0.495, mean_cost=7.6076)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.090 | 8.4301 | 0 |
| MVMOE | 0.047 | 9.1483 | 0 |
| RELD_MOEL | 0.382 | 7.6451 | 113 |
| RELD_MTL | 0.481 | 7.6094 | 887 |

### VRPB  (selector top1=0.500, mean_cost=9.2908)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.022 | 10.7014 | 0 |
| MVMOE | 0.059 | 11.5846 | 0 |
| RELD_MOEL | 0.436 | 9.3303 | 433 |
| RELD_MTL | 0.483 | 9.2951 | 567 |

### VRPL  (selector top1=0.452, mean_cost=12.4543)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.127 | 13.3434 | 0 |
| MVMOE | 0.063 | 14.1639 | 0 |
| RELD_MOEL | 0.356 | 12.4897 | 300 |
| RELD_MTL | 0.454 | 12.4568 | 700 |

### VRPTW  (selector top1=0.481, mean_cost=18.5728)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 23.2402 | 0 |
| MVMOE | 0.099 | 20.2432 | 0 |
| RELD_MOEL | 0.428 | 18.6273 | 285 |
| RELD_MTL | 0.473 | 18.5741 | 715 |

### OVRPTW  (selector top1=0.458, mean_cost=10.1282)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 14.5072 | 0 |
| MVMOE | 0.091 | 11.4039 | 1 |
| RELD_MOEL | 0.429 | 10.1432 | 516 |
| RELD_MTL | 0.480 | 10.1233 | 483 |

### OVRPB  (selector top1=0.540, mean_cost=6.3000)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.020 | 7.3061 | 0 |
| MVMOE | 0.014 | 8.1414 | 0 |
| RELD_MOEL | 0.405 | 6.3388 | 393 |
| RELD_MTL | 0.561 | 6.2963 | 607 |

### OVRPL  (selector top1=0.486, mean_cost=7.5979)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.085 | 8.2996 | 0 |
| MVMOE | 0.036 | 8.9026 | 0 |
| RELD_MOEL | 0.385 | 7.6341 | 376 |
| RELD_MTL | 0.494 | 7.5929 | 624 |

### VRPBL  (selector top1=0.553, mean_cost=9.2486)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.013 | 10.6000 | 0 |
| MVMOE | 0.048 | 11.2127 | 0 |
| RELD_MOEL | 0.402 | 9.3095 | 489 |
| RELD_MTL | 0.537 | 9.2492 | 511 |

### VRPBTW  (selector top1=0.497, mean_cost=19.5273)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 24.6857 | 0 |
| MVMOE | 0.118 | 21.3102 | 0 |
| RELD_MOEL | 0.394 | 19.6309 | 203 |
| RELD_MTL | 0.488 | 19.5263 | 797 |

### VRPLTW  (selector top1=0.475, mean_cost=18.4298)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 23.1636 | 0 |
| MVMOE | 0.088 | 20.0123 | 0 |
| RELD_MOEL | 0.410 | 18.4951 | 350 |
| RELD_MTL | 0.502 | 18.4269 | 650 |

### OVRPBL  (selector top1=0.551, mean_cost=6.3059)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.010 | 7.2184 | 0 |
| MVMOE | 0.016 | 7.8678 | 0 |
| RELD_MOEL | 0.428 | 6.3443 | 363 |
| RELD_MTL | 0.546 | 6.3068 | 637 |

### OVRPBTW  (selector top1=0.505, mean_cost=10.6048)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 15.2537 | 0 |
| MVMOE | 0.095 | 11.8587 | 0 |
| RELD_MOEL | 0.444 | 10.6443 | 547 |
| RELD_MTL | 0.461 | 10.6090 | 453 |

### OVRPLTW  (selector top1=0.489, mean_cost=10.1019)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 14.4558 | 0 |
| MVMOE | 0.078 | 11.3122 | 0 |
| RELD_MOEL | 0.430 | 10.1273 | 566 |
| RELD_MTL | 0.492 | 10.1036 | 434 |

### VRPBLTW  (selector top1=0.512, mean_cost=19.6767)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.001 | 24.7936 | 0 |
| MVMOE | 0.107 | 21.5061 | 0 |
| RELD_MOEL | 0.382 | 19.7998 | 135 |
| RELD_MTL | 0.510 | 19.6770 | 865 |

### OVRPBLTW  (selector top1=0.487, mean_cost=10.5919)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 15.2835 | 0 |
| MVMOE | 0.075 | 12.0405 | 0 |
| RELD_MOEL | 0.447 | 10.6280 | 343 |
| RELD_MTL | 0.478 | 10.5928 | 657 |

## D. Arm distribution (how often selector picks each method)

- **TSP**: DACT:0.0% | DIFUSCO:4.1% | ELG:0.0% | GLOP:1.0% | INVIT:0.0% | LEHD:74.4% | LIH:0.0% | OMNI:20.5% | T2T:0.0% | UDC:0.0%
- **CVRP**: DACT:0.0% | ELG:0.0% | GLOP:0.0% | ICAM:16.0% | INVIT:0.0% | LEHD:5.6% | OMNI:26.9% | RELD_CVRP:51.5% | UDC:0.0%
- **ATSP**: GLOP:0.0% | MATNET:25.0% | MATPOENET:75.0%
- **OVRP**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:11.3% | RELD_MTL:88.7%
- **VRPB**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:43.3% | RELD_MTL:56.7%
- **VRPL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:30.0% | RELD_MTL:70.0%
- **VRPTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:28.5% | RELD_MTL:71.5%
- **OVRPTW**: MTPOMO:0.0% | MVMOE:0.1% | RELD_MOEL:51.6% | RELD_MTL:48.3%
- **OVRPB**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:39.3% | RELD_MTL:60.7%
- **OVRPL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:37.6% | RELD_MTL:62.4%
- **VRPBL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:48.9% | RELD_MTL:51.1%
- **VRPBTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:20.3% | RELD_MTL:79.7%
- **VRPLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:35.0% | RELD_MTL:65.0%
- **OVRPBL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:36.3% | RELD_MTL:63.7%
- **OVRPBTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:54.7% | RELD_MTL:45.3%
- **OVRPLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:56.6% | RELD_MTL:43.4%
- **VRPBLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:13.5% | RELD_MTL:86.5%
- **OVRPBLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:34.3% | RELD_MTL:65.7%

### 0selection2 图和补充结果入口

- top1 图: [top1_vs_single_methods_test.png](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R25_multigen_partition_support_seed2/analysis_test_20260420/top1_vs_single_methods_test.png)
- mean cost 图: [mean_cost_vs_single_methods_test.png](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R25_multigen_partition_support_seed2/analysis_test_20260420/mean_cost_vs_single_methods_test.png)
- arm 分布图: [arm_distribution_test.png](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R25_multigen_partition_support_seed2/analysis_test_20260420/arm_distribution_test.png)
- loss curve: [loss_curve.png](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R25_multigen_partition_support_seed2/analysis_test_20260420/loss_curve.png)
- 4 月 21 日诊断总整理: [2026-04-21-post-review-results.md](/public/home/zhoucl/shiys/0selection2/review-stage/2026-04-21-post-review-results.md)
- smoke-only 结果入口: [_smoke_r30a_pairwin_switch/analysis_test.json](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/_smoke_r30a_pairwin_switch/analysis_test.json)

## 当前一句话判断

- `0selection` 已经从旧的 diagnostic base 前进到 `R40D` 这条 first-significant-positive 主线，但还需要 multi-seed 才能真正坐稳。
- `0selection2` 目前仍由 `R25` 扛正式 test 主结果，后续 `R26-R30` 更多是在解释为什么模型虽然 near-SBS-safe，却始终没把 hidden winners 打开。
