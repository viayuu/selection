# R32e Solver Features: Full Training and Test

30 epochs; seed 2; batch 640; RTX 4090; solver feature weight 0.3.
Primary checkpoint: `best.pt`, selected only by validation top1-safe score.
Selected epoch: 24 (zero-based), completed epoch 25.
18 problems: 10,000 training, 1,000 validation and 1,000 test instances each. No LIB evaluation.
Historical comparisons are descriptive, not a controlled feature ablation: batch/runtime changed.
ALL is the unweighted mean across problems. Percentage columns average per-problem percentages,
not ratios of the ALL mean costs. SBS is the best fixed solver on the reported split (benchmark reference).

## 结论

- 已完整训练 30 个 epoch，并完成 18 个问题、共 18,000 个 test 实例的评估。主 checkpoint 为验证集选出的第 25 轮 `best.pt`（日志 epoch=24），没有按 test 结果换模型。
- 相比旧 R32d 的同一选模规则，test top1 从 45.73% 升至 46.02%，仅增加 0.28 个百分点；top2/top3 也小幅提高。
- mean cost 从 11.1821 升至 11.1869，略有退步；宏平均 vs_sbs 从 -0.922% 变为 -0.901%，宏平均 vs_Oracle 从 +1.028% 变为 +1.050%。
- 坍缩没有改善：hidden mass 从 5.96% 升至 7.83%，宏平均最终臂覆盖率从 55.36% 降至 51.31%。hidden mass 指“从未被选中、但实际能够获胜的方法”对应的 oracle 胜率总量，再对问题取平均。
- TSP、ATSP 的 top1 和 cost 同时改善；CVRP 的 top1 下降 1.3 个百分点、cost 增加约 0.0299，是需要重点检查的退步项。VRPLTW 的 top1 提高 2.0 个百分点，但 cost 也变差，说明 top1 增加不必然降低代价。
- 固定特征矩阵为 `[22, 34]`；特征 MLP 第一层权重相对初始化的 L2 变化为 0.9809，确认新分支实际参与训练，并非未接入。
- 这是单 seed、单次运行。batch 从 448 改为 640、环境和 attention 内核也变化，因此不是严格的特征消融，不能把微小增减全部归因于新特征。当前不宜替换旧的主模型。

## 运行记录

- 环境：`easynco`，PyTorch 2.5.1+cu124，RTX 4090；FP16 AMP、SDPA attention、GPU 数据缓存。
- 起止：2026-09-29 23:51:13 至 2026-09-30 00:04:28（Asia/Shanghai）；含缓存、训练和测试共 13 分 15 秒，训练约 11.4 分钟。
- batch=640，30 个 epoch，共 8,100 个优化步。沿用原训练器 `drop_last`，每个问题每轮随机使用 9,600/10,000 个训练样本。
- GPU 每 10 秒采样；从首次 GPU-Util 达到 50% 后至结束，利用率均值 74.6%、中位数 72%、峰值 100%，包含验证、保存等间隙。显存峰值 18,820 MiB；训练计算阶段常见吞吐约 7,800–8,400 实例/秒。
- 训练和测试退出码均为正常结束（launcher exit_code=0）；GPU 已释放，没有遗留训练进程。
- W&B 已离线记录到 `wandb/offline-run-20260929_235251-v4ayhoft`；当前环境无登录凭据，未上传云端。
- `source.tar.gz`、`args.json`、`pip_before.txt` / `pip_after.txt`、`label_manifest.json` 保存了代码、参数、环境及标签顺序和哈希。

图表：`loss_validation_curve.png`，以及 `val_` / `test_` 开头的 `top1_vs_single_methods.png`、`mean_cost_vs_single_methods.png`、`arm_distribution.png`。

## Historical Comparison

| Model | Epoch (0-based) | top1 | top2 | top3 | mean cost | vs SBS | hidden mass | arm coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R32d best (val top1-safe) | 29 | 0.4573 | 0.7981 | 0.9135 | 11.1821 | -0.922% | 0.0596 | 0.5536 |
| R32d historical best_top3_safe | 26 | 0.4595 | 0.7987 | 0.9120 | 11.1819 | -0.935% | 0.0476 | 0.5504 |
| R32e solver features (val top1-safe) | 24 | 0.4602 | 0.8009 | 0.9147 | 11.1869 | -0.901% | 0.0783 | 0.5131 |

## VAL Results

| Problem | top1 | top2 | top3 | mean_cost | Gap | SBS cost (vs_sbs) | Oracle (vs_Oracle) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ALL** | **0.4683** | **0.8047** | **0.9134** | **11.1875** | **—** | **11.3413 (-0.938%)** | **11.0589 (+1.029%)** |
| TSP | 0.348 | 0.579 | 0.733 | 7.9196 | — | 7.9578 (-0.480%) | 7.8695 (+0.637%) |
| CVRP | 0.353 | 0.571 | 0.726 | 18.2985 | — | 18.2943 (+0.023%) | 18.1100 (+1.041%) |
| ATSP | 0.770 | 0.972 | 0.995 | 1.6393 | — | 1.6735 (-2.045%) | 1.6296 (+0.597%) |
| OVRP | 0.338 | 0.657 | 0.839 | 7.4588 | — | 7.4617 (-0.039%) | 7.4035 (+0.747%) |
| VRPB | 0.561 | 0.917 | 0.994 | 9.3340 | — | 9.3403 (-0.067%) | 9.2683 (+0.709%) |
| VRPL | 0.321 | 0.562 | 0.717 | 12.4026 | — | 12.4072 (-0.037%) | 12.3440 (+0.475%) |
| VRPTW | 0.449 | 0.754 | 0.904 | 17.9242 | — | 18.6571 (-3.928%) | 17.6603 (+1.495%) |
| OVRPTW | 0.494 | 0.933 | 1.000 | 10.0464 | — | 10.0484 (-0.020%) | 9.9585 (+0.882%) |
| OVRPB | 0.556 | 0.968 | 0.985 | 6.3218 | — | 6.3220 (-0.002%) | 6.2607 (+0.977%) |
| OVRPL | 0.384 | 0.700 | 0.846 | 7.4340 | — | 7.4376 (-0.047%) | 7.3796 (+0.738%) |
| VRPBL | 0.533 | 0.909 | 0.990 | 9.3564 | — | 9.3593 (-0.031%) | 9.2846 (+0.774%) |
| VRPBTW | 0.442 | 0.755 | 0.904 | 19.1016 | — | 19.6697 (-2.888%) | 18.7444 (+1.906%) |
| VRPLTW | 0.448 | 0.767 | 0.915 | 17.6492 | — | 18.5091 (-4.646%) | 17.4194 (+1.319%) |
| OVRPBL | 0.531 | 0.955 | 0.987 | 6.3090 | — | 6.3103 (-0.020%) | 6.2459 (+1.010%) |
| OVRPBTW | 0.501 | 0.900 | 0.998 | 10.5089 | — | 10.5068 (+0.020%) | 10.3798 (+1.243%) |
| OVRPLTW | 0.497 | 0.923 | 1.000 | 10.1259 | — | 10.1272 (-0.013%) | 10.0398 (+0.857%) |
| VRPBLTW | 0.409 | 0.738 | 0.909 | 19.0093 | — | 19.5198 (-2.615%) | 18.6515 (+1.918%) |
| OVRPBLTW | 0.495 | 0.925 | 0.999 | 10.5363 | — | 10.5412 (-0.046%) | 10.4112 (+1.201%) |

### VAL Arm Distribution

| Problem | hidden mass | zero picks / pool size | picks (all methods) |
| --- | ---: | ---: | --- |
| TSP | 0.2470 | 3/8 | BQ: 47.5%; DIFUSCO: 0.0%; DIFUSCO500: 5.5%; ELG: 20.7%; LEHD: 0.0%; OMNI: 0.0%; T2T: 0.7%; T2T500: 25.6% |
| CVRP | 0.0590 | 3/10 | BQ: 0.2%; ELG: 0.0%; ICAM: 18.2%; LEHD: 2.8%; MVMOE: 0.0%; MoSES_CaDA: 9.8%; MoSES_RF: 7.9%; OMNI: 48.6%; RELD_CVRP: 12.5%; RouteFinder: 0.0% |
| ATSP | 0.0790 | 2/5 | GLOP: 0.0%; ICAM_ATSP: 25.3%; MATNET: 20.4%; MATPOENET: 0.0%; UNICO_MatPOENet: 54.3% |
| OVRP | 0.1590 | 4/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 44.3%; MoSES_RF: 35.7%; RELD_MOEL: 0.0%; RELD_MTL: 0.0%; RouteFinder: 20.0% |
| VRPB | 0.0830 | 5/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 53.3%; RELD_MTL: 46.7%; RouteFinder: 0.0% |
| VRPL | 0.1910 | 2/7 | MTPOMO: 0.4%; MVMOE: 0.0%; MoSES_CaDA: 47.3%; MoSES_RF: 39.1%; RELD_MOEL: 12.7%; RELD_MTL: 0.5%; RouteFinder: 0.0% |
| VRPTW | 0.0710 | 2/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 30.6%; MoSES_RF: 19.1%; RELD_MOEL: 24.1%; RELD_MTL: 19.8%; RouteFinder: 6.4% |
| OVRPTW | 0.0670 | 5/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 72.3%; RELD_MTL: 27.7%; RouteFinder: 0.0% |
| OVRPB | 0.0320 | 5/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 53.1%; RELD_MTL: 46.9%; RouteFinder: 0.0% |
| OVRPL | 0.1390 | 4/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 42.2%; MoSES_RF: 35.0%; RELD_MOEL: 0.0%; RELD_MTL: 0.0%; RouteFinder: 22.8% |
| VRPBL | 0.0860 | 5/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 56.9%; RELD_MTL: 43.1%; RouteFinder: 0.0% |
| VRPBTW | 0.0000 | 1/7 | MTPOMO: 0.0%; MVMOE: 5.3%; MoSES_CaDA: 9.9%; MoSES_RF: 5.5%; RELD_MOEL: 33.1%; RELD_MTL: 32.5%; RouteFinder: 13.7% |
| VRPLTW | 0.0000 | 1/7 | MTPOMO: 0.0%; MVMOE: 0.8%; MoSES_CaDA: 29.1%; MoSES_RF: 11.4%; RELD_MOEL: 28.4%; RELD_MTL: 18.6%; RouteFinder: 11.7% |
| OVRPBL | 0.0450 | 5/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 54.0%; RELD_MTL: 46.0%; RouteFinder: 0.0% |
| OVRPBTW | 0.0010 | 4/7 | MTPOMO: 0.0%; MVMOE: 4.6%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 56.7%; RELD_MTL: 38.7%; RouteFinder: 0.0% |
| OVRPLTW | 0.0730 | 5/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 69.0%; RELD_MTL: 31.0%; RouteFinder: 0.0% |
| VRPBLTW | 0.0920 | 2/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 10.9%; MoSES_RF: 0.5%; RELD_MOEL: 33.4%; RELD_MTL: 37.6%; RouteFinder: 17.6% |
| OVRPBLTW | 0.0800 | 5/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 61.4%; RELD_MTL: 38.6%; RouteFinder: 0.0% |

Detailed per-method comparisons, SBS top1/2/3, pick/oracle ratios: `analysis_val.json`.

## TEST Results

| Problem | top1 | top2 | top3 | mean_cost | Gap | SBS cost (vs_sbs) | Oracle (vs_Oracle) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ALL** | **0.4602** | **0.8009** | **0.9147** | **11.1869** | **—** | **11.3339 (-0.901%)** | **11.0567 (+1.050%)** |
| TSP | 0.358 | 0.581 | 0.732 | 8.0454 | — | 8.0829 (-0.464%) | 7.9946 (+0.634%) |
| CVRP | 0.316 | 0.562 | 0.728 | 18.3724 | — | 18.3323 (+0.219%) | 18.1586 (+1.177%) |
| ATSP | 0.765 | 0.962 | 0.993 | 1.6363 | — | 1.6697 (-1.998%) | 1.6246 (+0.724%) |
| OVRP | 0.355 | 0.649 | 0.828 | 7.4669 | — | 7.4695 (-0.035%) | 7.4087 (+0.785%) |
| VRPB | 0.507 | 0.914 | 0.985 | 9.2893 | — | 9.2951 (-0.061%) | 9.2178 (+0.776%) |
| VRPL | 0.309 | 0.528 | 0.718 | 12.3545 | — | 12.3607 (-0.050%) | 12.2950 (+0.483%) |
| VRPTW | 0.426 | 0.741 | 0.907 | 17.6787 | — | 18.4357 (-4.106%) | 17.4240 (+1.462%) |
| OVRPTW | 0.459 | 0.908 | 0.999 | 10.1192 | — | 10.1233 (-0.040%) | 10.0336 (+0.853%) |
| OVRPB | 0.530 | 0.964 | 0.988 | 6.3015 | — | 6.2963 (+0.083%) | 6.2329 (+1.101%) |
| OVRPL | 0.384 | 0.674 | 0.849 | 7.4467 | — | 7.4537 (-0.094%) | 7.3937 (+0.717%) |
| VRPBL | 0.550 | 0.931 | 0.993 | 9.2502 | — | 9.2492 (+0.010%) | 9.1773 (+0.794%) |
| VRPBTW | 0.416 | 0.767 | 0.916 | 19.0393 | — | 19.5263 (-2.494%) | 18.6905 (+1.866%) |
| VRPLTW | 0.430 | 0.752 | 0.915 | 17.6398 | — | 18.4269 (-4.271%) | 17.3725 (+1.539%) |
| OVRPBL | 0.567 | 0.971 | 0.991 | 6.3030 | — | 6.3068 (-0.061%) | 6.2418 (+0.981%) |
| OVRPBTW | 0.496 | 0.903 | 0.998 | 10.6111 | — | 10.6090 (+0.020%) | 10.4782 (+1.268%) |
| OVRPLTW | 0.478 | 0.916 | 0.999 | 10.1071 | — | 10.1036 (+0.034%) | 10.0172 (+0.897%) |
| VRPBLTW | 0.427 | 0.774 | 0.927 | 19.1163 | — | 19.6770 (-2.849%) | 18.7886 (+1.744%) |
| OVRPBLTW | 0.510 | 0.919 | 0.999 | 10.5862 | — | 10.5928 (-0.062%) | 10.4710 (+1.100%) |

### TEST Arm Distribution

| Problem | hidden mass | zero picks / pool size | picks (all methods) |
| --- | ---: | ---: | --- |
| TSP | 0.1880 | 2/8 | BQ: 52.1%; DIFUSCO: 0.1%; DIFUSCO500: 5.7%; ELG: 16.5%; LEHD: 0.0%; OMNI: 0.0%; T2T: 0.4%; T2T500: 25.2% |
| CVRP | 0.1770 | 4/10 | BQ: 0.0%; ELG: 0.0%; ICAM: 18.6%; LEHD: 3.0%; MVMOE: 0.0%; MoSES_CaDA: 7.7%; MoSES_RF: 8.3%; OMNI: 48.0%; RELD_CVRP: 14.4%; RouteFinder: 0.0% |
| ATSP | 0.0930 | 2/5 | GLOP: 0.0%; ICAM_ATSP: 24.8%; MATNET: 20.4%; MATPOENET: 0.0%; UNICO_MatPOENet: 54.8% |
| OVRP | 0.1700 | 4/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 43.3%; MoSES_RF: 36.9%; RELD_MOEL: 0.0%; RELD_MTL: 0.0%; RouteFinder: 19.8% |
| VRPB | 0.0820 | 5/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 54.4%; RELD_MTL: 45.6%; RouteFinder: 0.0% |
| VRPL | 0.0260 | 1/7 | MTPOMO: 0.4%; MVMOE: 0.0%; MoSES_CaDA: 46.1%; MoSES_RF: 40.5%; RELD_MOEL: 11.6%; RELD_MTL: 1.1%; RouteFinder: 0.3% |
| VRPTW | 0.0660 | 2/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 31.9%; MoSES_RF: 19.2%; RELD_MOEL: 25.1%; RELD_MTL: 18.9%; RouteFinder: 4.9% |
| OVRPTW | 0.0920 | 5/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 70.6%; RELD_MTL: 29.4%; RouteFinder: 0.0% |
| OVRPB | 0.0360 | 5/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 54.0%; RELD_MTL: 46.0%; RouteFinder: 0.0% |
| OVRPL | 0.1420 | 4/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 43.2%; MoSES_RF: 35.5%; RELD_MOEL: 0.0%; RELD_MTL: 0.0%; RouteFinder: 21.3% |
| VRPBL | 0.0610 | 5/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 55.5%; RELD_MTL: 44.5%; RouteFinder: 0.0% |
| VRPBTW | 0.0000 | 1/7 | MTPOMO: 0.0%; MVMOE: 4.6%; MoSES_CaDA: 8.0%; MoSES_RF: 6.4%; RELD_MOEL: 34.6%; RELD_MTL: 32.4%; RouteFinder: 14.0% |
| VRPLTW | 0.0000 | 1/7 | MTPOMO: 0.0%; MVMOE: 0.5%; MoSES_CaDA: 28.9%; MoSES_RF: 9.8%; RELD_MOEL: 27.2%; RELD_MTL: 22.2%; RouteFinder: 11.4% |
| OVRPBL | 0.0290 | 5/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 53.5%; RELD_MTL: 46.5%; RouteFinder: 0.0% |
| OVRPBTW | 0.0010 | 4/7 | MTPOMO: 0.0%; MVMOE: 4.9%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 55.2%; RELD_MTL: 39.9%; RouteFinder: 0.0% |
| OVRPLTW | 0.0790 | 5/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 66.3%; RELD_MTL: 33.7%; RouteFinder: 0.0% |
| VRPBLTW | 0.0930 | 2/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 9.8%; MoSES_RF: 0.5%; RELD_MOEL: 32.4%; RELD_MTL: 39.2%; RouteFinder: 18.1% |
| OVRPBLTW | 0.0750 | 5/7 | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 60.9%; RELD_MTL: 39.1%; RouteFinder: 0.0% |

Detailed per-method comparisons, SBS top1/2/3, pick/oracle ratios: `analysis_test.json`.
