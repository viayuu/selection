# Solver Selection 双线结果汇总

日期: 2026-04-19


## 口径说明

- `0selection` 的 in-distribution 详细观测指标直接采用现成报表 [eval_R4_ensemble3_test/metrics_report.md](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/eval_R4_ensemble3_test/metrics_report.md)。
- `0selection` 的 zero-shot / gated / specialist 对比数字来自 [AUTO_REVIEW.md](/public/home/zhoucl/shiys/0selection/review-stage/AUTO_REVIEW.md) 的 Round 4-9 汇总。
- `0selection2` 的“当前最新 raw selector”采用 R6 两阶段训练线的 3 个 seed test 聚合：
  - [analysis_test.json](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R6_init_softstats_soft_risk_bal_seed0/analysis_test/analysis_test.json)
  - [analysis_test.json](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R6_init_softstats_soft_risk_bal_seed1/analysis_test/analysis_test.json)
  - [analysis_test.json](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R6_init_softstats_soft_risk_bal_seed2/analysis_test/analysis_test.json)
- `0selection2` 的早期保守 gated 结论来自 [CLAIMS_FROM_RESULTS.md](/public/home/zhoucl/shiys/0selection2/CLAIMS_FROM_RESULTS.md)。

## 一页总览

| Line | Setting | top1 | top2 | top3 | macro vs SBS | 备注 |
|---|---|---:|---:|---:|---:|---|
| 0selection | raw unified, like-for-like 3-seed mean test | 0.5131 | — | — | +0.0453% | Round 9 reviewer re-eval |
| 0selection | raw unified, 3-seed logit ensemble test | 0.5157 | 0.8734 | 0.9827 | +0.0188% | 当前现成的最强 in-domain 观测报表 |
| 0selection | specialists, 3-seed mean test | 0.5197 | — | — | -0.0100% | in-domain 仍略强于 unified |
| 0selection | zero-shot Split-A gated | — | — | — | -0.002% | held-out 5 problems, 5 seeds |
| 0selection | zero-shot Split-B gated | — | — | — | +0.007% | held-out 5 problems, 3 seeds |
| 0selection | structured Leave-TW gated | — | — | — | -0.000% | held-out 8 TW variants, 3 seeds |
| 0selection2 | early conservative strict gate | 0.5143* | 0.8731* | 0.9827* | +0.001% | *best-seed gated diagnostic; 3-seed strict gate macro from early line |
| 0selection2 | current raw R6 3-seed mean test | 0.5221 | 0.8734 | 0.9833 | -0.0812% | 当前最强 raw line |
| 0selection2 | current raw R6 best seed (seed2) test | 0.5240 | 0.8742 | 0.9837 | -0.0973% | 当前单 seed 最强 |

## 0selection: 当前结果解读

- 这条线的强项已经不只是 in-domain raw selector，而是 `zero-shot + safety gate + structured holdout` 的论文叙事。
- 单看 in-domain raw unified，当前仍略输 SBS / specialists：Round 9 复核给出的 like-for-like test 是 `0.5131 / +0.0453%`，3-seed ensemble 可改善到 `0.5157 / +0.0188%`，但 specialists 仍是 `0.5197 / -0.0100%`。
- 这条线最强的证据是 zero-shot gating：
  - Random Split-A held-out gated: `-0.002% [-0.005, +0.000]`
  - Random Split-B held-out gated: `+0.007% [-0.006, +0.027]`
  - Structured Leave-TW held-out gated: `-0.000% [-0.004, +0.006]`
- 所以 `0selection` 目前更像“统一 selector 在 unseen composite / unseen family 上可 zero-shot transfer，并在 val-calibrated gating 下稳定 match SBS”的故事。

### 0selection 详细观测指标报表

详细现成报表见 [metrics_report.md](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/eval_R4_ensemble3_test/metrics_report.md)。下面直接保留其核心内容。

# 观测指标报表

Split: **test**  ·  18 problems

## A. Selector top-k vs oracle, and per-method top1 winrate

| Problem | selector top1 | top2 | top3 | best top1 (oracle) | best top2 | best top3 | beat_top1 | lose_top1 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| TSP | 0.734 | 0.885 | 0.961 | 0.706 | 1.000 | 1.000 | DACT, DIFUSCO, ELG, GLOP, INVIT, LEHD, LIH, OMNI, T2T, UDC | — |
| CVRP | 0.407 | 0.723 | 0.912 | 0.413 | 1.000 | 1.000 | DACT, ELG, GLOP, ICAM, INVIT, LEHD, OMNI, UDC | RELD_CVRP |
| ATSP | 0.701 | 0.996 | 1.000 | 0.701 | 1.000 | 1.000 | GLOP, MATNET | — |
| OVRP | 0.481 | 0.828 | 0.956 | 0.481 | 1.000 | 1.000 | MTPOMO, MVMOE, RELD_MOEL | — |
| VRPB | 0.483 | 0.879 | 0.990 | 0.483 | 1.000 | 1.000 | MTPOMO, MVMOE, RELD_MOEL | — |
| VRPL | 0.454 | 0.754 | 0.932 | 0.454 | 1.000 | 1.000 | MTPOMO, MVMOE, RELD_MOEL | — |
| VRPTW | 0.473 | 0.879 | 1.000 | 0.473 | 1.000 | 1.000 | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPTW | 0.480 | 0.897 | 1.000 | 0.480 | 1.000 | 1.000 | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPB | 0.561 | 0.929 | 0.990 | 0.561 | 1.000 | 1.000 | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPL | 0.494 | 0.837 | 0.962 | 0.494 | 1.000 | 1.000 | MTPOMO, MVMOE, RELD_MOEL | — |
| VRPBL | 0.537 | 0.893 | 0.993 | 0.537 | 1.000 | 1.000 | MTPOMO, MVMOE, RELD_MOEL | — |
| VRPBTW | 0.488 | 0.856 | 1.000 | 0.488 | 1.000 | 1.000 | MTPOMO, MVMOE, RELD_MOEL | — |
| VRPLTW | 0.502 | 0.882 | 1.000 | 0.502 | 1.000 | 1.000 | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPBL | 0.546 | 0.928 | 0.994 | 0.546 | 1.000 | 1.000 | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPBTW | 0.461 | 0.887 | 1.000 | 0.461 | 1.000 | 1.000 | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPLTW | 0.492 | 0.900 | 1.000 | 0.492 | 1.000 | 1.000 | MTPOMO, MVMOE, RELD_MOEL | — |
| VRPBLTW | 0.510 | 0.869 | 0.999 | 0.510 | 1.000 | 1.000 | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPBLTW | 0.478 | 0.900 | 1.000 | 0.478 | 1.000 | 1.000 | MTPOMO, MVMOE, RELD_MOEL | — |
| **MACRO** | **0.5157** | **0.8734** | **0.9827** | | | | | |

## B. Mean cost — selector vs SBS, and per-method mean cost

| Problem | selector mean_cost | SBS mean_cost (best) | vs_sbs (%) | beat_mean_cost | lose_mean_cost |
|---|---:|---:|---:|---|---|
| TSP | 8.1794 | 8.1579 | +0.263% | DACT, DIFUSCO, ELG, GLOP, INVIT, LIH, OMNI, T2T, UDC | LEHD |
| CVRP | 18.3462 | 18.3323 | +0.076% | DACT, ELG, GLOP, ICAM, INVIT, LEHD, OMNI, UDC | RELD_CVRP |
| ATSP | 1.7159 | 1.7159 | -0.000% | GLOP, MATNET | — |
| OVRP | 7.6094 | 7.6094 | -0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| VRPB | 9.2951 | 9.2951 | -0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| VRPL | 12.4568 | 12.4568 | -0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| VRPTW | 18.5741 | 18.5741 | +0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPTW | 10.1233 | 10.1233 | -0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPB | 6.2963 | 6.2963 | -0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPL | 7.5929 | 7.5929 | -0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| VRPBL | 9.2492 | 9.2492 | +0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| VRPBTW | 19.5263 | 19.5263 | -0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| VRPLTW | 18.4269 | 18.4269 | +0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPBL | 6.3068 | 6.3068 | -0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPBTW | 10.6090 | 10.6090 | -0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPLTW | 10.1036 | 10.1036 | -0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| VRPBLTW | 19.6770 | 19.6770 | -0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| OVRPBLTW | 10.5928 | 10.5928 | +0.000% | MTPOMO, MVMOE, RELD_MOEL | — |
| **MACRO vs_sbs** | | | **+0.0188%** | | |

## C. Per-method comparison (mean cost on val/test)

### TSP  (selector top1=0.734, mean_cost=8.1794)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| DACT | 0.000 | 10.1287 | 0 |
| DIFUSCO | 0.056 | 9.2355 | 0 |
| ELG | 0.007 | 16.9025 | 0 |
| GLOP | 0.014 | 8.6876 | 0 |
| INVIT | 0.007 | 8.7046 | 0 |
| LEHD | 0.706 | 8.1579 | 914 |
| LIH | 0.000 | 116.1080 | 1 |
| OMNI | 0.210 | 8.3188 | 85 |
| T2T | 0.000 | 9.4649 | 0 |
| UDC | 0.000 | 13.0844 | 0 |

### CVRP  (selector top1=0.407, mean_cost=18.3462)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| DACT | 0.000 | 22.1375 | 0 |
| ELG | 0.001 | 61.4521 | 0 |
| GLOP | 0.000 | 22.0717 | 0 |
| ICAM | 0.154 | 18.5471 | 75 |
| INVIT | 0.000 | 20.8204 | 0 |
| LEHD | 0.261 | 18.5653 | 146 |
| OMNI | 0.171 | 18.6303 | 0 |
| RELD_CVRP | 0.413 | 18.3323 | 779 |
| UDC | 0.000 | 31.3299 | 0 |

### ATSP  (selector top1=0.701, mean_cost=1.7159)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| GLOP | 0.024 | 1.9203 | 0 |
| MATNET | 0.275 | 2.3139 | 0 |
| MATPOENET | 0.701 | 1.7159 | 1000 |

### OVRP  (selector top1=0.481, mean_cost=7.6094)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.090 | 8.4301 | 0 |
| MVMOE | 0.047 | 9.1483 | 0 |
| RELD_MOEL | 0.382 | 7.6451 | 0 |
| RELD_MTL | 0.481 | 7.6094 | 1000 |

### VRPB  (selector top1=0.483, mean_cost=9.2951)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.022 | 10.7014 | 0 |
| MVMOE | 0.059 | 11.5846 | 0 |
| RELD_MOEL | 0.436 | 9.3303 | 0 |
| RELD_MTL | 0.483 | 9.2951 | 1000 |

### VRPL  (selector top1=0.454, mean_cost=12.4568)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.127 | 13.3434 | 0 |
| MVMOE | 0.063 | 14.1639 | 0 |
| RELD_MOEL | 0.356 | 12.4897 | 0 |
| RELD_MTL | 0.454 | 12.4568 | 1000 |

### VRPTW  (selector top1=0.473, mean_cost=18.5741)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 23.2402 | 0 |
| MVMOE | 0.099 | 20.2432 | 0 |
| RELD_MOEL | 0.428 | 18.6273 | 0 |
| RELD_MTL | 0.473 | 18.5741 | 1000 |

### OVRPTW  (selector top1=0.480, mean_cost=10.1233)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 14.5072 | 0 |
| MVMOE | 0.091 | 11.4039 | 0 |
| RELD_MOEL | 0.429 | 10.1432 | 0 |
| RELD_MTL | 0.480 | 10.1233 | 1000 |

### OVRPB  (selector top1=0.561, mean_cost=6.2963)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.020 | 7.3061 | 0 |
| MVMOE | 0.014 | 8.1414 | 0 |
| RELD_MOEL | 0.405 | 6.3388 | 0 |
| RELD_MTL | 0.561 | 6.2963 | 1000 |

### OVRPL  (selector top1=0.494, mean_cost=7.5929)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.085 | 8.2996 | 0 |
| MVMOE | 0.036 | 8.9026 | 0 |
| RELD_MOEL | 0.385 | 7.6341 | 0 |
| RELD_MTL | 0.494 | 7.5929 | 1000 |

### VRPBL  (selector top1=0.537, mean_cost=9.2492)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.013 | 10.6000 | 0 |
| MVMOE | 0.048 | 11.2127 | 0 |
| RELD_MOEL | 0.402 | 9.3095 | 0 |
| RELD_MTL | 0.537 | 9.2492 | 1000 |

### VRPBTW  (selector top1=0.488, mean_cost=19.5263)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 24.6857 | 0 |
| MVMOE | 0.118 | 21.3102 | 0 |
| RELD_MOEL | 0.394 | 19.6309 | 0 |
| RELD_MTL | 0.488 | 19.5263 | 1000 |

### VRPLTW  (selector top1=0.502, mean_cost=18.4269)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 23.1636 | 0 |
| MVMOE | 0.088 | 20.0123 | 0 |
| RELD_MOEL | 0.410 | 18.4951 | 0 |
| RELD_MTL | 0.502 | 18.4269 | 1000 |

### OVRPBL  (selector top1=0.546, mean_cost=6.3068)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.010 | 7.2184 | 0 |
| MVMOE | 0.016 | 7.8678 | 0 |
| RELD_MOEL | 0.428 | 6.3443 | 0 |
| RELD_MTL | 0.546 | 6.3068 | 1000 |

### OVRPBTW  (selector top1=0.461, mean_cost=10.6090)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 15.2537 | 0 |
| MVMOE | 0.095 | 11.8587 | 0 |
| RELD_MOEL | 0.444 | 10.6443 | 0 |
| RELD_MTL | 0.461 | 10.6090 | 1000 |

### OVRPLTW  (selector top1=0.492, mean_cost=10.1036)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 14.4558 | 0 |
| MVMOE | 0.078 | 11.3122 | 0 |
| RELD_MOEL | 0.430 | 10.1273 | 0 |
| RELD_MTL | 0.492 | 10.1036 | 1000 |

### VRPBLTW  (selector top1=0.510, mean_cost=19.6770)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.001 | 24.7936 | 0 |
| MVMOE | 0.107 | 21.5061 | 0 |
| RELD_MOEL | 0.382 | 19.7998 | 0 |
| RELD_MTL | 0.510 | 19.6770 | 1000 |

### OVRPBLTW  (selector top1=0.478, mean_cost=10.5928)
| Method | oracle winrate | mean_cost | pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 15.2835 | 0 |
| MVMOE | 0.075 | 12.0405 | 0 |
| RELD_MOEL | 0.447 | 10.6280 | 0 |
| RELD_MTL | 0.478 | 10.5928 | 1000 |

## D. Arm distribution (how often selector picks each method)

- **TSP**: DACT:0.0% | DIFUSCO:0.0% | ELG:0.0% | GLOP:0.0% | INVIT:0.0% | LEHD:91.4% | LIH:0.1% | OMNI:8.5% | T2T:0.0% | UDC:0.0%
- **CVRP**: DACT:0.0% | ELG:0.0% | GLOP:0.0% | ICAM:7.5% | INVIT:0.0% | LEHD:14.6% | OMNI:0.0% | RELD_CVRP:77.9% | UDC:0.0%
- **ATSP**: GLOP:0.0% | MATNET:0.0% | MATPOENET:100.0%
- **OVRP**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **VRPB**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **VRPL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **VRPTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **OVRPTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **OVRPB**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **OVRPL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **VRPBL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **VRPBTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **VRPLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **OVRPBL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **OVRPBTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **OVRPLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **VRPBLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **OVRPBLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%

## 0selection2: 当前结果解读

- 这条线先前的保守结论是：raw unified 略微不安全，但 strict gated SBS-fallback 可以把 macro degradation 压到 `+0.001%`。
- 2026-04-19 的新推进已经转向 **raw selector 本体增强**：`soft + stats` 预训练，再接 `balanced soft_risk` 微调。
- 这条新线 3-seed mean test 已经到 `top1=0.5221`、`top2=0.8734`、`top3=0.9833`、`macro_vs_sbs=-0.0812%`。
- 平均来看，18 个问题里大约 `3` 个问题明显 beat SBS，`15-16` 个问题基本 match SBS；最亮眼的收益来自 `ATSP`，其次是 `TSP`。
- 所以 `0selection2` 目前已经不只是“靠 gate 保命”，而是在往“raw unified selector 本身就略优于 SBS”推进。

### A. Selector top-k vs SBS-top1

| Problem | selector top1 | top2 | top3 | best top1 (SBS) | beat_top1 | lose_top1 |
|---|---:|---:|---:|---:|---|---|
| TSP | 0.704 | 0.864 | 0.949 | 0.706 | DACT, DIFUSCO, ELG, GLOP, INVIT, LIH, OMNI, T2T, UDC | LEHD |
| CVRP | 0.413 | 0.748 | 0.932 | 0.413 | DACT, ELG, GLOP, ICAM, INVIT, LEHD, OMNI, RELD_CVRP, UDC | — |
| ATSP | 0.843 | 0.994 | 1.000 | 0.701 | GLOP, MATNET, MATPOENET | — |
| OVRP | 0.481 | 0.828 | 0.956 | 0.481 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPB | 0.483 | 0.879 | 0.990 | 0.483 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPL | 0.449 | 0.754 | 0.933 | 0.454 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPTW | 0.473 | 0.879 | 1.000 | 0.473 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPTW | 0.475 | 0.894 | 1.000 | 0.480 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPB | 0.559 | 0.930 | 0.991 | 0.561 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPL | 0.494 | 0.837 | 0.962 | 0.494 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPBL | 0.537 | 0.893 | 0.993 | 0.537 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPBTW | 0.488 | 0.856 | 1.000 | 0.488 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPLTW | 0.502 | 0.882 | 1.000 | 0.502 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPBL | 0.546 | 0.928 | 0.994 | 0.546 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPBTW | 0.468 | 0.886 | 1.000 | 0.461 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPLTW | 0.492 | 0.899 | 1.000 | 0.492 | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPBLTW | 0.510 | 0.869 | 0.999 | 0.510 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPBLTW | 0.480 | 0.901 | 1.000 | 0.478 | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| **MACRO** | **0.5221** | **0.8734** | **0.9833** |  |  |  |

### B. Mean cost — selector vs SBS, and per-method mean cost

| Problem | selector mean_cost | SBS mean_cost (best) | vs_sbs (%) | beat_mean_cost | lose_mean_cost |
|---|---:|---:|---:|---|---|
| TSP | 8.1513 | 8.1579 | -0.081% | DACT, DIFUSCO, ELG, GLOP, INVIT, LEHD, LIH, OMNI, T2T, UDC | — |
| CVRP | 18.3323 | 18.3323 | +0.000% | DACT, ELG, GLOP, ICAM, INVIT, LEHD, OMNI, RELD_CVRP, UDC | — |
| ATSP | 1.6914 | 1.7159 | -1.430% | GLOP, MATNET, MATPOENET | — |
| OVRP | 7.6094 | 7.6094 | +0.000% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPB | 9.2951 | 9.2951 | +0.000% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPL | 12.4599 | 12.4568 | +0.025% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPTW | 18.5741 | 18.5741 | +0.000% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPTW | 10.1263 | 10.1233 | +0.030% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPB | 6.2963 | 6.2963 | +0.000% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| OVRPL | 7.5929 | 7.5929 | +0.000% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPBL | 9.2492 | 9.2492 | +0.000% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPBTW | 19.5263 | 19.5263 | +0.000% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| VRPLTW | 18.4269 | 18.4269 | +0.000% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPBL | 6.3068 | 6.3068 | +0.000% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPBTW | 10.6086 | 10.6090 | -0.004% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPLTW | 10.1039 | 10.1036 | +0.003% | MTPOMO, MVMOE, RELD_MOEL | RELD_MTL |
| VRPBLTW | 19.6770 | 19.6770 | +0.000% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| OVRPBLTW | 10.5922 | 10.5928 | -0.006% | MTPOMO, MVMOE, RELD_MOEL, RELD_MTL | — |
| **MACRO vs_sbs** |  |  | **-0.0812%** |  |  |

### C. Per-method comparison (3-seed mean pick count, test split)

#### TSP  (selector top1=0.704, mean_cost=8.1513)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| DACT | 0.000 | 10.1287 | 0.0 |
| DIFUSCO | 0.056 | 9.2355 | 25.3 |
| ELG | 0.007 | 16.9025 | 0.0 |
| GLOP | 0.014 | 8.6876 | 0.0 |
| INVIT | 0.007 | 8.7046 | 0.0 |
| LEHD | 0.706 | 8.1579 | 872.7 |
| LIH | 0.000 | 116.1080 | 0.0 |
| OMNI | 0.210 | 8.3188 | 102.0 |
| T2T | 0.000 | 9.4649 | 0.0 |
| UDC | 0.000 | 13.0844 | 0.0 |

#### CVRP  (selector top1=0.413, mean_cost=18.3323)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| DACT | 0.000 | 22.1375 | 0.0 |
| ELG | 0.001 | 61.4521 | 0.0 |
| GLOP | 0.000 | 22.0717 | 0.0 |
| ICAM | 0.154 | 18.5471 | 0.0 |
| INVIT | 0.000 | 20.8204 | 0.0 |
| LEHD | 0.261 | 18.5653 | 0.0 |
| OMNI | 0.171 | 18.6303 | 0.0 |
| RELD_CVRP | 0.413 | 18.3323 | 1000.0 |
| UDC | 0.000 | 31.3299 | 0.0 |

#### ATSP  (selector top1=0.843, mean_cost=1.6914)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| GLOP | 0.024 | 1.9203 | 0.0 |
| MATNET | 0.275 | 2.3139 | 173.0 |
| MATPOENET | 0.701 | 1.7159 | 827.0 |

#### OVRP  (selector top1=0.481, mean_cost=7.6094)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| MTPOMO | 0.090 | 8.4301 | 0.0 |
| MVMOE | 0.047 | 9.1483 | 0.0 |
| RELD_MOEL | 0.382 | 7.6451 | 0.0 |
| RELD_MTL | 0.481 | 7.6094 | 1000.0 |

#### VRPB  (selector top1=0.483, mean_cost=9.2951)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| MTPOMO | 0.022 | 10.7014 | 0.0 |
| MVMOE | 0.059 | 11.5846 | 0.0 |
| RELD_MOEL | 0.436 | 9.3303 | 2.3 |
| RELD_MTL | 0.483 | 9.2951 | 997.7 |

#### VRPL  (selector top1=0.449, mean_cost=12.4599)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| MTPOMO | 0.127 | 13.3434 | 0.0 |
| MVMOE | 0.063 | 14.1639 | 0.0 |
| RELD_MOEL | 0.356 | 12.4897 | 83.0 |
| RELD_MTL | 0.454 | 12.4568 | 917.0 |

#### VRPTW  (selector top1=0.473, mean_cost=18.5741)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 23.2402 | 0.0 |
| MVMOE | 0.099 | 20.2432 | 0.0 |
| RELD_MOEL | 0.428 | 18.6273 | 0.0 |
| RELD_MTL | 0.473 | 18.5741 | 1000.0 |

#### OVRPTW  (selector top1=0.475, mean_cost=10.1263)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 14.5072 | 0.0 |
| MVMOE | 0.091 | 11.4039 | 0.0 |
| RELD_MOEL | 0.429 | 10.1432 | 209.3 |
| RELD_MTL | 0.480 | 10.1233 | 790.7 |

#### OVRPB  (selector top1=0.559, mean_cost=6.2963)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| MTPOMO | 0.020 | 7.3061 | 0.0 |
| MVMOE | 0.014 | 8.1414 | 0.0 |
| RELD_MOEL | 0.405 | 6.3388 | 6.0 |
| RELD_MTL | 0.561 | 6.2963 | 994.0 |

#### OVRPL  (selector top1=0.494, mean_cost=7.5929)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| MTPOMO | 0.085 | 8.2996 | 0.0 |
| MVMOE | 0.036 | 8.9026 | 0.0 |
| RELD_MOEL | 0.385 | 7.6341 | 0.0 |
| RELD_MTL | 0.494 | 7.5929 | 1000.0 |

#### VRPBL  (selector top1=0.537, mean_cost=9.2492)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| MTPOMO | 0.013 | 10.6000 | 0.0 |
| MVMOE | 0.048 | 11.2127 | 0.0 |
| RELD_MOEL | 0.402 | 9.3095 | 0.0 |
| RELD_MTL | 0.537 | 9.2492 | 1000.0 |

#### VRPBTW  (selector top1=0.488, mean_cost=19.5263)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 24.6857 | 0.0 |
| MVMOE | 0.118 | 21.3102 | 0.0 |
| RELD_MOEL | 0.394 | 19.6309 | 0.0 |
| RELD_MTL | 0.488 | 19.5263 | 1000.0 |

#### VRPLTW  (selector top1=0.502, mean_cost=18.4269)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 23.1636 | 0.0 |
| MVMOE | 0.088 | 20.0123 | 0.0 |
| RELD_MOEL | 0.410 | 18.4951 | 0.0 |
| RELD_MTL | 0.502 | 18.4269 | 1000.0 |

#### OVRPBL  (selector top1=0.546, mean_cost=6.3068)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| MTPOMO | 0.010 | 7.2184 | 0.0 |
| MVMOE | 0.016 | 7.8678 | 0.0 |
| RELD_MOEL | 0.428 | 6.3443 | 0.0 |
| RELD_MTL | 0.546 | 6.3068 | 1000.0 |

#### OVRPBTW  (selector top1=0.468, mean_cost=10.6086)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 15.2537 | 0.0 |
| MVMOE | 0.095 | 11.8587 | 0.0 |
| RELD_MOEL | 0.444 | 10.6443 | 31.7 |
| RELD_MTL | 0.461 | 10.6090 | 968.3 |

#### OVRPLTW  (selector top1=0.492, mean_cost=10.1039)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 14.4558 | 0.0 |
| MVMOE | 0.078 | 11.3122 | 0.0 |
| RELD_MOEL | 0.430 | 10.1273 | 3.3 |
| RELD_MTL | 0.492 | 10.1036 | 996.7 |

#### VRPBLTW  (selector top1=0.510, mean_cost=19.6770)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| MTPOMO | 0.001 | 24.7936 | 0.0 |
| MVMOE | 0.107 | 21.5061 | 0.0 |
| RELD_MOEL | 0.382 | 19.7998 | 0.0 |
| RELD_MTL | 0.510 | 19.6770 | 1000.0 |

#### OVRPBLTW  (selector top1=0.480, mean_cost=10.5922)
| Method | oracle winrate | mean_cost | mean pick count |
|---|---:|---:|---:|
| MTPOMO | 0.000 | 15.2835 | 0.0 |
| MVMOE | 0.075 | 12.0405 | 0.0 |
| RELD_MOEL | 0.447 | 10.6280 | 9.3 |
| RELD_MTL | 0.478 | 10.5928 | 990.7 |

### D. Arm distribution (3-seed mean)

- **TSP**: DACT:0.0% | DIFUSCO:2.5% | ELG:0.0% | GLOP:0.0% | INVIT:0.0% | LEHD:87.3% | LIH:0.0% | OMNI:10.2% | T2T:0.0% | UDC:0.0%
- **CVRP**: DACT:0.0% | ELG:0.0% | GLOP:0.0% | ICAM:0.0% | INVIT:0.0% | LEHD:0.0% | OMNI:0.0% | RELD_CVRP:100.0% | UDC:0.0%
- **ATSP**: GLOP:0.0% | MATNET:17.3% | MATPOENET:82.7%
- **OVRP**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **VRPB**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.2% | RELD_MTL:99.8%
- **VRPL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:8.3% | RELD_MTL:91.7%
- **VRPTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **OVRPTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:20.9% | RELD_MTL:79.1%
- **OVRPB**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.6% | RELD_MTL:99.4%
- **OVRPL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **VRPBL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **VRPBTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **VRPLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **OVRPBL**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **OVRPBTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:3.2% | RELD_MTL:96.8%
- **OVRPLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.3% | RELD_MTL:99.7%
- **VRPBLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.0% | RELD_MTL:100.0%
- **OVRPBLTW**: MTPOMO:0.0% | MVMOE:0.0% | RELD_MOEL:0.9% | RELD_MTL:99.1%

### 图和曲线

- `0selection` 现成 in-domain 图表入口: [metrics_report.md](/public/home/zhoucl/shiys/0selection/code/unified_selector/runs/eval_R4_ensemble3_test/metrics_report.md)
- `0selection2` 当前 R6 line 图表:
  [top1_vs_single_methods_test.png](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R6_init_softstats_soft_risk_bal_seed0/analysis_test/top1_vs_single_methods_test.png)
  [mean_cost_vs_single_methods_test.png](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R6_init_softstats_soft_risk_bal_seed0/analysis_test/mean_cost_vs_single_methods_test.png)
  [arm_distribution_test.png](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R6_init_softstats_soft_risk_bal_seed0/analysis_test/arm_distribution_test.png)
  [loss_curve.png](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R6_init_softstats_soft_risk_bal_seed0/analysis_test/loss_curve.png)
- `0selection2` 训练日志: [seed0 train.log](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R6_init_softstats_soft_risk_bal_seed0/train.log), [seed1 train.log](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R6_init_softstats_soft_risk_bal_seed1/train.log), [seed2 train.log](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R6_init_softstats_soft_risk_bal_seed2/train.log)
