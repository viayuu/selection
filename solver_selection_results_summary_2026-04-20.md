# Solver Selection 双线结果汇总

日期: 2026-04-20

## 口径说明

- 这版按 [观测指标.md](/public/home/zhoucl/shiys/0selection/观测指标/观测指标.md) 的风格重写，尽量保持和 [solver_selection_results_summary_2026-04-19.md](/public/home/zhoucl/shiys/solver_selection_results_summary_2026-04-19.md) 一样的结构。
- `0selection` 4 月 20 日最新的 `R25-R39` 多数只有 aggregate test 表，所以 detailed 长表使用我刚补出的 [R18_alltail test analysis](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R18_alltail/analysis_test_20260420/analysis_test.json)。
- `0selection2` detailed 长表使用最新且已完成完整 test 分析的 [R25_multigen_partition_support_seed2](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R25_multigen_partition_support_seed2/analysis_test_20260420/analysis_test.json)。
- `0selection` 的 `analyze.py` 当前目录里没有本地 `audit.json`，所以这次 detailed test analysis 使用共享的 [0selection2 audit.json](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/audit.json) 重新跑出报表。两边共用同一套 18-problem 数据和 solver pool，口径在此说明。

## 一页总览

| Line | Setting | top1 | top2 | top3 | macro vs SBS | 备注 |
|---|---|---:|---:|---:|---:|---|
| 0selection | R18_alltail test-detailed | 0.5164 | 0.8736 | 0.9840 | -0.0119% | 当前 main base 的完整 per-problem 报表 |
| 0selection | R26b gated blend best exploratory | 0.5249 | — | — | -0.0267% | 当前 best point estimate，但 CI 跨 0 |
| 0selection | R35 dual-retrieval agreement | 0.5225 | — | — | Δcost=-0.0022% | pre-registered, not significant |
| 0selection | R38 fusion head | 0.5222 | — | — | Δcost=-0.0044% | pre-registered, not significant |
| 0selection2 | R25 latest test-confirmed | 0.5353 | 0.8848 | 0.9871 | -0.1247% | 当前最新且最强的 test-confirmed 单 seed |
| 0selection2 | R25 latest val snapshot | 0.5376 | — | — | -0.1396% | `eval_epoch0.json`，已被 test 大体确认 |
| 0selection2 | R6 earlier 3-seed mean test | 0.5221 | 0.8734 | 0.9833 | -0.0812% | 先前 multi-seed confirm |

## 0selection: 当前结果解读

- 这条线现在已经明显收敛成 **diagnostic paper** 路线，而不是继续冲 overall effect 的 improvement paper。
- 当前 detailed test base 结果：
  - `macro_top1 = 0.5164`
  - `macro_top2 = 0.8736`
  - `macro_top3 = 0.9840`
  - `macro_vs_sbs = -0.0119%`
  - `n_problems_beat_sbs = 10`
  - `n_problems_match_sbs = 7`
- 当前 point estimate 最好的 aggregate intervention 仍然是 [R26b_fine_gate](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R26b_fine_gate/blend_test.md)：
  - best config `a=0.40, d=0.08`
  - `macro_top1 = 0.5249`
  - `macro_vs_sbs = -0.0267%`
  - 但 bootstrap CI 仍跨 0，没有成为 reliable improvement story
- 4 月 20 日 fresh loop 最终结论见 [CLAIMS_FROM_RESULTS.md](/public/home/zhoucl/shiys/0selection/CLAIMS_FROM_RESULTS.md)：
  - 存在明显 `support collapse`
  - retrieval 确实能做 subset-level rescue
  - 但所有全局修正都没有显著提升 overall top1
  - oracle-pro 的终判是“**Stop. Write diagnostic paper.**”

### 0selection 详细观测指标报表

详细 JSON 见 [analysis_test.json](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R18_alltail/analysis_test_20260420/analysis_test.json)。下面直接保留核心长表。

# 观测指标报表

Split: **test**  ·  18 problems

## A. Selector top-k vs oracle, and per-method top1 winrate

| Problem | selector top1 | top2 | top3 | best top1 (oracle) | beat_top1 | lose_top1 |
|---|---:|---:|---:|---:|---|---|
| TSP | 0.715 | 0.866 | 0.948 | 0.706 | DACT, DIFUSCO, ELG, GLOP, INVIT, LEHD, LIH, OMNI, T2T, UDC | — |
| CVRP | 0.412 | 0.737 | 0.930 | 0.413 | DACT, ELG, GLOP, ICAM, INVIT, LEHD, OMNI, UDC | RELD_CVRP |
| ATSP | 0.700 | 0.995 | 1.000 | 0.701 | GLOP, MATNET | MATPOENET |
| OVRP | 0.488 | 0.828 | 0.969 | 0.481 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPB | 0.484 | 0.872 | 0.989 | 0.483 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPL | 0.456 | 0.757 | 0.933 | 0.454 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPTW | 0.485 | 0.887 | 1.000 | 0.473 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPTW | 0.430 | 0.881 | 1.000 | 0.480 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPB | 0.531 | 0.937 | 0.992 | 0.561 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPL | 0.498 | 0.836 | 0.964 | 0.494 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPBL | 0.536 | 0.893 | 0.993 | 0.537 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPBTW | 0.492 | 0.857 | 1.000 | 0.488 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPLTW | 0.500 | 0.883 | 1.000 | 0.502 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPBL | 0.568 | 0.934 | 0.995 | 0.546 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPBTW | 0.494 | 0.891 | 1.000 | 0.461 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPLTW | 0.502 | 0.907 | 1.000 | 0.492 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPBLTW | 0.510 | 0.869 | 0.999 | 0.510 | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPBLTW | 0.494 | 0.895 | 1.000 | 0.478 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| **MACRO** | **0.5164** | **0.8736** | **0.9840** |  |  |  |

## B. Mean cost — selector vs SBS, and per-method mean cost

| Problem | selector mean_cost | SBS mean_cost (best) | vs_sbs (%) | beat_mean_cost | lose_mean_cost |
|---|---:|---:|---:|---|---|
| TSP | 8.1525 | 8.1579 | -0.066% | DACT, DIFUSCO, ELG, GLOP, INVIT, LEHD, LIH, OMNI, T2T, UDC | — |
| CVRP | 18.3337 | 18.3323 | +0.008% | DACT, ELG, GLOP, ICAM, INVIT, LEHD, OMNI, UDC | RELD_CVRP |
| ATSP | 1.7171 | 1.7159 | +0.066% | GLOP, MATNET | MATPOENET |
| OVRP | 7.6057 | 7.6094 | -0.049% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPB | 9.2930 | 9.2951 | -0.022% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPL | 12.4562 | 12.4568 | -0.005% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPTW | 18.5667 | 18.5741 | -0.040% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPTW | 10.1299 | 10.1233 | +0.065% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPB | 6.3008 | 6.2963 | +0.071% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPL | 7.5930 | 7.5929 | +0.002% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPBL | 9.2495 | 9.2492 | +0.004% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPBTW | 19.5261 | 19.5263 | -0.001% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPLTW | 18.4273 | 18.4269 | +0.003% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPBL | 6.3024 | 6.3068 | -0.069% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPBTW | 10.6029 | 10.6090 | -0.058% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPLTW | 10.0997 | 10.1036 | -0.039% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPBLTW | 19.6770 | 19.6770 | +0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPBLTW | 10.5840 | 10.5928 | -0.083% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| **MACRO vs_sbs** |  |  | **-0.0119%** |  |  |

## C. Per-method comparison (mean cost on val/test)

### TSP  (selector top1=0.715, mean_cost=8.1525)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| DACT | 0.000 | 10.1287 | 0 |
| DIFUSCO | 0.056 | 9.2355 | 0 |
| ELG | 0.007 | 16.9025 | 0 |
| GLOP | 0.014 | 8.6876 | 0 |
| INVIT | 0.007 | 8.7046 | 0 |
| LEHD | 0.706 | 8.1579 | 982 |
| LIH | 0.000 | 116.1080 | 0 |
| OMNI | 0.210 | 8.3188 | 18 |
| T2T | 0.000 | 9.4649 | 0 |
| UDC | 0.000 | 13.0844 | 0 |

### CVRP  (selector top1=0.412, mean_cost=18.3337)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| DACT | 0.000 | 22.1375 | 0 |
| ELG | 0.001 | 61.4521 | 0 |
| GLOP | 0.000 | 22.0717 | 0 |
| ICAM | 0.154 | 18.5471 | 56 |
| INVIT | 0.000 | 20.8204 | 0 |
| LEHD | 0.261 | 18.5653 | 38 |
| OMNI | 0.171 | 18.6303 | 1 |
| RELD_CVRP | 0.413 | 18.3323 | 905 |
| UDC | 0.000 | 31.3299 | 0 |

### ATSP  (selector top1=0.700, mean_cost=1.7171)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| GLOP | 0.024 | 1.9203 | 0 |
| MATNET | 0.275 | 2.3139 | 1 |
| MATPOENET | 0.701 | 1.7159 | 999 |

### OVRP  (selector top1=0.488, mean_cost=7.6057)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.090 | 8.4301 | 0 |
| MVMOE | 0.047 | 9.1483 | 0 |
| RELD_MOEL | 0.382 | 7.6451 | 408 |
| RELD_MTL | 0.481 | 7.6094 | 592 |

### VRPB  (selector top1=0.484, mean_cost=9.2930)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.022 | 10.7014 | 0 |
| MVMOE | 0.059 | 11.5846 | 0 |
| RELD_MOEL | 0.436 | 9.3303 | 262 |
| RELD_MTL | 0.483 | 9.2951 | 738 |

### VRPL  (selector top1=0.456, mean_cost=12.4562)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.127 | 13.3434 | 0 |
| MVMOE | 0.063 | 14.1639 | 0 |
| RELD_MOEL | 0.356 | 12.4897 | 3 |
| RELD_MTL | 0.454 | 12.4568 | 997 |

### VRPTW  (selector top1=0.485, mean_cost=18.5667)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 23.2402 | 0 |
| MVMOE | 0.099 | 20.2432 | 0 |
| RELD_MOEL | 0.428 | 18.6273 | 302 |
| RELD_MTL | 0.473 | 18.5741 | 698 |

### OVRPTW  (selector top1=0.430, mean_cost=10.1299)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 14.5072 | 0 |
| MVMOE | 0.091 | 11.4039 | 0 |
| RELD_MOEL | 0.429 | 10.1432 | 442 |
| RELD_MTL | 0.480 | 10.1233 | 558 |

### OVRPB  (selector top1=0.531, mean_cost=6.3008)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.020 | 7.3061 | 0 |
| MVMOE | 0.014 | 8.1414 | 0 |
| RELD_MOEL | 0.405 | 6.3388 | 451 |
| RELD_MTL | 0.561 | 6.2963 | 549 |

### OVRPL  (selector top1=0.498, mean_cost=7.5930)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.085 | 8.2996 | 0 |
| MVMOE | 0.036 | 8.9026 | 0 |
| RELD_MOEL | 0.385 | 7.6341 | 33 |
| RELD_MTL | 0.494 | 7.5929 | 967 |

### VRPBL  (selector top1=0.536, mean_cost=9.2495)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.013 | 10.6000 | 0 |
| MVMOE | 0.048 | 11.2127 | 0 |
| RELD_MOEL | 0.402 | 9.3095 | 1 |
| RELD_MTL | 0.537 | 9.2492 | 999 |

### VRPBTW  (selector top1=0.492, mean_cost=19.5261)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 24.6857 | 0 |
| MVMOE | 0.118 | 21.3102 | 0 |
| RELD_MOEL | 0.394 | 19.6309 | 9 |
| RELD_MTL | 0.488 | 19.5263 | 991 |

### VRPLTW  (selector top1=0.500, mean_cost=18.4273)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 23.1636 | 0 |
| MVMOE | 0.088 | 20.0123 | 0 |
| RELD_MOEL | 0.410 | 18.4951 | 5 |
| RELD_MTL | 0.502 | 18.4269 | 995 |

### OVRPBL  (selector top1=0.568, mean_cost=6.3024)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.010 | 7.2184 | 0 |
| MVMOE | 0.016 | 7.8678 | 0 |
| RELD_MOEL | 0.428 | 6.3443 | 299 |
| RELD_MTL | 0.546 | 6.3068 | 701 |

### OVRPBTW  (selector top1=0.494, mean_cost=10.6029)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 15.2537 | 0 |
| MVMOE | 0.095 | 11.8587 | 0 |
| RELD_MOEL | 0.444 | 10.6443 | 287 |
| RELD_MTL | 0.461 | 10.6090 | 713 |

### OVRPLTW  (selector top1=0.502, mean_cost=10.0997)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 14.4558 | 0 |
| MVMOE | 0.078 | 11.3122 | 0 |
| RELD_MOEL | 0.430 | 10.1273 | 274 |
| RELD_MTL | 0.492 | 10.1036 | 726 |

### VRPBLTW  (selector top1=0.510, mean_cost=19.6770)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.001 | 24.7936 | 0 |
| MVMOE | 0.107 | 21.5061 | 0 |
| RELD_MOEL | 0.382 | 19.7998 | 0 |
| RELD_MTL | 0.510 | 19.6770 | 1000 |

### OVRPBLTW  (selector top1=0.494, mean_cost=10.5840)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 15.2835 | 0 |
| MVMOE | 0.075 | 12.0405 | 0 |
| RELD_MOEL | 0.447 | 10.6280 | 351 |
| RELD_MTL | 0.478 | 10.5928 | 649 |

## D. Arm distribution (how often selector picks each method)

- **TSP**: DACT:0.0% | DIFUSCO:0.0% | ELG:0.0% | GLOP:0.0% | INVIT:0.0% | LEHD:98.2% | LIH:0.0% | OMNI:1.8% | T2T:0.0% | UDC:0.0%
- **CVRP**: DACT:0.0% | ELG:0.0% | GLOP:0.0% | ICAM:5.6% | INVIT:0.0% | LEHD:3.8% | OMNI:0.1% | RELD_CVRP:90.5% | UDC:0.0%
- **ATSP**: GLOP:0.0% | MATNET:0.1% | MATPOENET:99.9%
- **OVRP**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:40.8% | RELD_MTL:59.2%
- **VRPB**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:26.2% | RELD_MTL:73.8%
- **VRPL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.3% | RELD_MTL:99.7%
- **VRPTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:30.2% | RELD_MTL:69.8%
- **OVRPTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:44.2% | RELD_MTL:55.8%
- **OVRPB**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:45.1% | RELD_MTL:54.9%
- **OVRPL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:3.3% | RELD_MTL:96.7%
- **VRPBL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.1% | RELD_MTL:99.9%
- **VRPBTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.9% | RELD_MTL:99.1%
- **VRPLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.5% | RELD_MTL:99.5%
- **OVRPBL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:29.9% | RELD_MTL:70.1%
- **OVRPBTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:28.7% | RELD_MTL:71.3%
- **OVRPLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:27.4% | RELD_MTL:72.6%
- **VRPBLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **OVRPBLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:35.1% | RELD_MTL:64.9%

### 0selection 图和补充结果入口

- top1 图: [top1_vs_single_methods_test.png](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R18_alltail/analysis_test_20260420/top1_vs_single_methods_test.png)
- mean cost 图: [mean_cost_vs_single_methods_test.png](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R18_alltail/analysis_test_20260420/mean_cost_vs_single_methods_test.png)
- arm 分布图: [arm_distribution_test.png](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R18_alltail/analysis_test_20260420/arm_distribution_test.png)
- 最新 aggregate intervention:
  - [R31 final test](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R31_final_test/blend_test.md)
  - [R35 dual](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R35_dual/dual.md)
  - [R38 fusion](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R38_fusion/fusion.md)
  - [R39 gross decomposition](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/R39_gross/gross_decomp.md)

## 0selection2: 当前结果解读

- 这条线仍然是 **效果提升** 主线，不是 diagnostic pivot。
- 当前最新且已完成的完整 test 报表是 [R25_multigen_partition_support_seed2](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R25_multigen_partition_support_seed2/analysis_test_20260420/analysis_test.json)：
  - `macro_top1 = 0.5353`
  - `macro_top2 = 0.8848`
  - `macro_top3 = 0.9871`
  - `macro_vs_sbs = -0.1247%`
  - `macro_vbs_gap_closed = +8.32%`
  - `n_problems_beat_sbs = 13`
  - `n_problems_match_sbs = 8`
- 这已经明显强于上一版里引用的 R6 3-seed mean test：
  - R6 3-seed mean: `0.5221 / 0.8734 / 0.9833 / -0.0812%`
  - 当前 R25 test-confirmed: `0.5353 / 0.8848 / 0.9871 / -0.1247%`
- 从已完成 test run 排名看，`R25` 已经成为当前最新且最强的 test-confirmed 结果。
- `0selection2` 这条线的臂分布也更活跃：和 `0selection` 相比，TSP/CVRP/ATSP 以及不少 MVRP 问题都不再那么极端地塌到单一 arm 上。

### 0selection2 详细观测指标报表

详细 JSON 见 [analysis_test.json](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R25_multigen_partition_support_seed2/analysis_test_20260420/analysis_test.json)。下面直接保留核心长表。

# 观测指标报表

Split: **test**  ·  18 problems

## A. Selector top-k vs oracle, and per-method top1 winrate

| Problem | selector top1 | top2 | top3 | best top1 (oracle) | beat_top1 | lose_top1 |
|---|---:|---:|---:|---:|---|---|
| TSP | 0.747 | 0.926 | 0.973 | 0.706 | DACT, DIFUSCO, ELG, GLOP, INVIT, LEHD, LIH, OMNI, T2T, UDC | — |
| CVRP | 0.529 | 0.846 | 0.960 | 0.413 | DACT, ELG, GLOP, ICAM, INVIT, LEHD, OMNI, RELD_CVRP, UDC | — |
| ATSP | 0.879 | 0.988 | 1.000 | 0.701 | GLOP, MATNET, MATPOENET | — |
| OVRP | 0.495 | 0.834 | 0.954 | 0.481 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPB | 0.500 | 0.878 | 0.991 | 0.483 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPL | 0.452 | 0.766 | 0.948 | 0.454 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPTW | 0.481 | 0.891 | 1.000 | 0.473 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPTW | 0.458 | 0.885 | 1.000 | 0.480 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPB | 0.540 | 0.937 | 0.991 | 0.561 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPL | 0.486 | 0.833 | 0.960 | 0.494 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPBL | 0.553 | 0.888 | 0.994 | 0.537 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPBTW | 0.497 | 0.870 | 1.000 | 0.488 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPLTW | 0.475 | 0.877 | 1.000 | 0.502 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPBL | 0.551 | 0.931 | 0.997 | 0.546 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPBTW | 0.505 | 0.885 | 1.000 | 0.461 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPLTW | 0.489 | 0.898 | 1.000 | 0.492 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPBLTW | 0.512 | 0.888 | 0.999 | 0.510 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPBLTW | 0.487 | 0.906 | 1.000 | 0.478 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| **MACRO** | **0.5353** | **0.8848** | **0.9871** |  |  |  |

## B. Mean cost — selector vs SBS, and per-method mean cost

| Problem | selector mean_cost | SBS mean_cost (best) | vs_sbs (%) | beat_mean_cost | lose_mean_cost |
|---|---:|---:|---:|---|---|
| TSP | 8.1287 | 8.1579 | -0.357% | DACT, DIFUSCO, ELG, GLOP, INVIT, LEHD, LIH, OMNI, T2T, UDC | — |
| CVRP | 18.2802 | 18.3323 | -0.284% | DACT, ELG, GLOP, ICAM, INVIT, LEHD, OMNI, RELD_CVRP, UDC | — |
| ATSP | 1.6882 | 1.7159 | -1.614% | GLOP, MATNET, MATPOENET | — |
| OVRP | 7.6076 | 7.6094 | -0.024% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPB | 9.2908 | 9.2951 | -0.046% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPL | 12.4543 | 12.4568 | -0.020% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPTW | 18.5728 | 18.5741 | -0.007% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPTW | 10.1282 | 10.1233 | +0.048% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPB | 6.3000 | 6.2963 | +0.059% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPL | 7.5979 | 7.5929 | +0.066% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPBL | 9.2486 | 9.2492 | -0.006% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPBTW | 19.5273 | 19.5263 | +0.005% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPLTW | 18.4298 | 18.4269 | +0.016% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPBL | 6.3059 | 6.3068 | -0.014% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPBTW | 10.6048 | 10.6090 | -0.039% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPLTW | 10.1019 | 10.1036 | -0.016% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPBLTW | 19.6767 | 19.6770 | -0.001% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPBLTW | 10.5919 | 10.5928 | -0.008% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| **MACRO vs_sbs** |  |  | **-0.1247%** |  |  |

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
- loss 曲线: [loss_curve.png](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R25_multigen_partition_support_seed2/analysis_test_20260420/loss_curve.png)
- 最新 val snapshot: [eval_epoch0.json](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R25_multigen_partition_support_seed2/eval_epoch0.json)

## 当前最关键的结论

- `0selection`：现在最可信、最成熟的是 **diagnostic paper** 路线。overall effect 最高能到 `0.5249` 的 nominal top1，但没有统计显著性；完整 detailed test 的 main base 结果是 `0.5164 / -0.0119%`。
- `0selection2`：现在仍在 **效果提升** 路线上，而且最新 test-confirmed `R25` 已经到 `0.5353 / -0.1247%`，明显强于 `0selection` 当前 main base，也强于 `0selection2` 之前的 R6 multi-seed line。
