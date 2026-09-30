# R33a Dual-Stream Selector

## 结果结论

已完成新结构的 30 轮训练和 18,000 个 test 实例评估。主 checkpoint 由验证集选出，为第 23 轮（epoch=22）的 `best.pt`，没有按 test 更换模型。

- 验证集：top1=0.4685，vs_sbs=-0.961%。
- 测试集：top1=0.4568，top2=0.7956，top3=0.9103，mean cost=11.1881，vs_sbs=-0.898%，vs_Oracle=+1.053%。
- 相比上一版 R32e，test top1 下降 0.33 个百分点，mean cost 增加约 0.0013。本次完整实验没有体现出整体效果提升，因此保留为独立实验分支，不替换旧 checkpoint。
- CVRP 的 top1 提高 0.9 个百分点、cost 降低约 0.0206；VRPB 的 top1 和 cost 也同时改善。但 TSP、OVRP、OVRPL 的 top1 分别下降 1.5、3.4、1.8 个百分点。
- ATSP 的 cost 降低约 0.0027，但 top1 下降 1.4 个百分点，再次说明两个目标不能互相替代。
- 新方法按要求移除了旧的辅助头，并将方法特征权重设为 1.0；因此这是整套新路径与旧方法的比较，不是只改变双向 attention 的严格单因素消融。单 seed 的结果不能说明该结构没有进一步优化空间。

## 实现与验证

- 主实现：`code/V4/dual_stream.py`，通过 `--architecture dual_stream` 启用。旧模式和旧 checkpoint 仍可使用。
- 复用已有 4 层节点 self-attention；联合编码只接收实际节点序列，不接收交互前的全局池化向量。
- 两层联合编码均采用同步双向更新，两个方向及不同层参数独立；Decoder 的 attention 也独立。
- 新路径仅输出共享打分头的 logits，没有旧 pre/gap/support 分数的叠加，也没有伪造旧辅助输出。
- 17 项自动测试通过，包括 mask、候选重排、batch 样本隔离、同步更新、梯度连接、无标签推理、新旧 checkpoint 和评估接口。
- 54 份标签文件哈希与 R32e 相同，见 `label_manifest.json`。
- 独立 CLI 评估与训练内评估已逐问题核对，见 `evaluation_verification.json` 和 `standalone_test_aligned/`。独立入口已统一采用训练器的 TF32 设置；首次默认 FP32 的诊断结果保留在 `standalone_test_fp32_default/`，不是主报告口径。

## 运行记录

- 启动脚本：`code/V4/run_v4_dualstream_seed2.sh`；环境 `easynco`，RTX 4090。
- 起止：2026-09-30 01:02:11 至 01:15:18（Asia/Shanghai），含缓存、训练和训练脚本内测试共 13 分 07 秒；另完成了独立评估复核。
- batch=640，30 轮，共 8,100 个优化步。GPU 数据缓存、SDPA 和 FP16 AMP 均启用。
- GPU 每 10 秒采样，从首次利用率达到 50% 后至训练脚本结束，利用率均值 87.7%、中位数 92%、峰值 100%，显存峰值 18,650 MiB。
- W&B 离线记录位于 `wandb/offline-run-20260930_010348-jew459xa`，未上传云端。
- 修改前代码备份为 `source_before.tar.gz`；训练源码快照为 `source.tar.gz`；参数和环境分别见 `args.json`、`pip_freeze.txt`。
- 曲线：`training_curves.png`；逐问题方法对比及选择频率图为 `val_`、`test_` 开头的 PNG。

## Architecture and Protocol

Existing 4-layer node self-attention -> 2 synchronous bidirectional cross-attention layers.
Solver initialization: LayerNorm(ID + feature MLP); the fixed 34-feature table is unchanged.
Updated nodes -> masked mean/max pooling -> instance query over updated solvers.
Shared scorer [h, context, solver] -> logits. No mixed-score external D and no label input.
No legacy pre/gap/support scores or auxiliary heads in this mode.
Training: from scratch, seed 2, 30 epochs, batch 640, AdamW lr=2e-4, wd=1e-4, FP16, dropout=0.1.
Loss: 0.35 winner-balanced CE + 0.30 top-focused pairwise + 0.08 sequential top-k CE + 0.02 expected cost risk.
18 problems, each with 10,000 train / 1,000 val / 1,000 test. No LIB or unseen-problem zero-shot evaluation.
Each training epoch uses 9,600 randomly shuffled samples per problem (existing drop_last behavior).
Primary checkpoint: best.pt, epoch 22 (zero-based), selected exclusively on validation top1-safe score.
Test never selects the checkpoint. ALL percentages are macro means of per-problem percentages, not ratios of aggregate costs.
SBS is the best fixed method on the reported split, used as a benchmark reference only.
This compares complete methods, not an isolated architectural ablation: the old auxiliary heads were removed and feature weight changed to 1.0.

## Test Comparison

| Model | Epoch (0-based) | top1 | top2 | top3 | mean cost | vs SBS | vs Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R32d | 29 | 0.4573 | 0.7981 | 0.9135 | 11.1821 | -0.922% | +1.028% |
| R32e solver features | 24 | 0.4602 | 0.8009 | 0.9147 | 11.1869 | -0.901% | +1.050% |
| R33a dual-stream | 22 | 0.4568 | 0.7956 | 0.9103 | 11.1881 | -0.898% | +1.053% |

## VAL Results

| Problem | top1 | top2 | top3 | mean_cost | Gap | SBS cost (vs_sbs) | Oracle (vs_Oracle) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ALL** | **0.4685** | **0.8018** | **0.9124** | **11.1851** | **—** | **11.3413 (-0.961%)** | **11.0589 (+1.006%)** |
| TSP | 0.345 | 0.573 | 0.721 | 7.9242 | — | 7.9578 (-0.422%) | 7.8695 (+0.696%) |
| CVRP | 0.355 | 0.561 | 0.714 | 18.2843 | — | 18.2943 (-0.055%) | 18.1100 (+0.963%) |
| ATSP | 0.776 | 0.965 | 0.992 | 1.6359 | — | 1.6735 (-2.249%) | 1.6296 (+0.386%) |
| OVRP | 0.355 | 0.656 | 0.834 | 7.4581 | — | 7.4617 (-0.048%) | 7.4035 (+0.738%) |
| VRPB | 0.543 | 0.916 | 0.995 | 9.3384 | — | 9.3403 (-0.020%) | 9.2683 (+0.757%) |
| VRPL | 0.317 | 0.561 | 0.712 | 12.4041 | — | 12.4072 (-0.025%) | 12.3440 (+0.487%) |
| VRPTW | 0.449 | 0.766 | 0.913 | 17.9024 | — | 18.6571 (-4.045%) | 17.6603 (+1.371%) |
| OVRPTW | 0.503 | 0.933 | 0.999 | 10.0443 | — | 10.0484 (-0.041%) | 9.9585 (+0.862%) |
| OVRPB | 0.553 | 0.968 | 0.986 | 6.3221 | — | 6.3220 (+0.002%) | 6.2607 (+0.981%) |
| OVRPL | 0.381 | 0.691 | 0.863 | 7.4354 | — | 7.4376 (-0.029%) | 7.3796 (+0.756%) |
| VRPBL | 0.541 | 0.914 | 0.987 | 9.3535 | — | 9.3593 (-0.062%) | 9.2846 (+0.742%) |
| VRPBTW | 0.420 | 0.744 | 0.906 | 19.0920 | — | 19.6697 (-2.937%) | 18.7444 (+1.855%) |
| VRPLTW | 0.446 | 0.764 | 0.911 | 17.6706 | — | 18.5091 (-4.530%) | 17.4194 (+1.442%) |
| OVRPBL | 0.538 | 0.954 | 0.986 | 6.3090 | — | 6.3103 (-0.020%) | 6.2459 (+1.010%) |
| OVRPBTW | 0.505 | 0.903 | 0.999 | 10.5024 | — | 10.5068 (-0.042%) | 10.3798 (+1.181%) |
| OVRPLTW | 0.495 | 0.918 | 1.000 | 10.1261 | — | 10.1272 (-0.010%) | 10.0398 (+0.860%) |
| VRPBLTW | 0.405 | 0.742 | 0.907 | 18.9951 | — | 19.5198 (-2.688%) | 18.6515 (+1.842%) |
| OVRPBLTW | 0.506 | 0.903 | 0.999 | 10.5335 | — | 10.5412 (-0.073%) | 10.4112 (+1.175%) |

### VAL Arm Distribution

| Problem | Selection frequency |
| --- | --- |
| TSP | BQ: 56.1%; DIFUSCO: 1.6%; DIFUSCO500: 4.0%; ELG: 9.0%; LEHD: 0.0%; OMNI: 0.0%; T2T: 1.4%; T2T500: 27.9% |
| CVRP | BQ: 2.2%; ELG: 0.0%; ICAM: 19.7%; LEHD: 9.1%; MVMOE: 0.4%; MoSES_CaDA: 11.8%; MoSES_RF: 8.6%; OMNI: 30.9%; RELD_CVRP: 17.3%; RouteFinder: 0.0% |
| ATSP | GLOP: 0.0%; ICAM_ATSP: 27.9%; MATNET: 14.0%; MATPOENET: 0.0%; UNICO_MatPOENet: 58.1% |
| OVRP | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 46.3%; MoSES_RF: 31.5%; RELD_MOEL: 0.0%; RELD_MTL: 0.3%; RouteFinder: 21.9% |
| VRPB | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 59.7%; RELD_MTL: 40.3%; RouteFinder: 0.0% |
| VRPL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 45.3%; MoSES_RF: 39.4%; RELD_MOEL: 10.4%; RELD_MTL: 2.6%; RouteFinder: 2.3% |
| VRPTW | MTPOMO: 0.0%; MVMOE: 0.4%; MoSES_CaDA: 25.6%; MoSES_RF: 17.7%; RELD_MOEL: 21.7%; RELD_MTL: 27.9%; RouteFinder: 6.7% |
| OVRPTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 72.5%; RELD_MTL: 27.5%; RouteFinder: 0.0% |
| OVRPB | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 54.5%; RELD_MTL: 45.5%; RouteFinder: 0.0% |
| OVRPL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 47.0%; MoSES_RF: 31.5%; RELD_MOEL: 0.0%; RELD_MTL: 0.5%; RouteFinder: 21.0% |
| VRPBL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 59.8%; RELD_MTL: 40.2%; RouteFinder: 0.0% |
| VRPBTW | MTPOMO: 0.0%; MVMOE: 3.2%; MoSES_CaDA: 9.3%; MoSES_RF: 10.1%; RELD_MOEL: 19.3%; RELD_MTL: 46.8%; RouteFinder: 11.3% |
| VRPLTW | MTPOMO: 0.0%; MVMOE: 1.6%; MoSES_CaDA: 29.3%; MoSES_RF: 16.7%; RELD_MOEL: 22.4%; RELD_MTL: 19.9%; RouteFinder: 10.1% |
| OVRPBL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 54.9%; RELD_MTL: 45.1%; RouteFinder: 0.0% |
| OVRPBTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 74.5%; RELD_MTL: 25.5%; RouteFinder: 0.0% |
| OVRPLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 71.4%; RELD_MTL: 28.6%; RouteFinder: 0.0% |
| VRPBLTW | MTPOMO: 0.0%; MVMOE: 4.3%; MoSES_CaDA: 11.4%; MoSES_RF: 2.7%; RELD_MOEL: 19.7%; RELD_MTL: 45.7%; RouteFinder: 16.2% |
| OVRPBLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 68.5%; RELD_MTL: 31.5%; RouteFinder: 0.0% |

All single-method comparisons and SBS top1/2/3 are in `analysis_val.json`.

## TEST Results

| Problem | top1 | top2 | top3 | mean_cost | Gap | SBS cost (vs_sbs) | Oracle (vs_Oracle) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ALL** | **0.4568** | **0.7956** | **0.9103** | **11.1881** | **—** | **11.3339 (-0.898%)** | **11.0567 (+1.053%)** |
| TSP | 0.343 | 0.574 | 0.722 | 8.0480 | — | 8.0829 (-0.432%) | 7.9946 (+0.667%) |
| CVRP | 0.325 | 0.558 | 0.705 | 18.3518 | — | 18.3323 (+0.106%) | 18.1586 (+1.064%) |
| ATSP | 0.751 | 0.952 | 0.993 | 1.6336 | — | 1.6697 (-2.160%) | 1.6246 (+0.557%) |
| OVRP | 0.321 | 0.644 | 0.823 | 7.4717 | — | 7.4695 (+0.029%) | 7.4087 (+0.850%) |
| VRPB | 0.515 | 0.916 | 0.986 | 9.2875 | — | 9.2951 (-0.082%) | 9.2178 (+0.756%) |
| VRPL | 0.299 | 0.503 | 0.702 | 12.3554 | — | 12.3607 (-0.043%) | 12.2950 (+0.491%) |
| VRPTW | 0.420 | 0.736 | 0.909 | 17.6772 | — | 18.4357 (-4.114%) | 17.4240 (+1.454%) |
| OVRPTW | 0.461 | 0.908 | 1.000 | 10.1223 | — | 10.1233 (-0.010%) | 10.0336 (+0.884%) |
| OVRPB | 0.537 | 0.964 | 0.989 | 6.2998 | — | 6.2963 (+0.055%) | 6.2329 (+1.074%) |
| OVRPL | 0.366 | 0.675 | 0.849 | 7.4478 | — | 7.4537 (-0.078%) | 7.3937 (+0.732%) |
| VRPBL | 0.553 | 0.937 | 0.992 | 9.2468 | — | 9.2492 (-0.026%) | 9.1773 (+0.758%) |
| VRPBTW | 0.421 | 0.758 | 0.916 | 19.0633 | — | 19.5263 (-2.371%) | 18.6905 (+1.994%) |
| VRPLTW | 0.432 | 0.743 | 0.895 | 17.6437 | — | 18.4269 (-4.250%) | 17.3725 (+1.561%) |
| OVRPBL | 0.564 | 0.971 | 0.990 | 6.3039 | — | 6.3068 (-0.047%) | 6.2418 (+0.995%) |
| OVRPBTW | 0.500 | 0.905 | 0.998 | 10.6175 | — | 10.6090 (+0.080%) | 10.4782 (+1.330%) |
| OVRPLTW | 0.484 | 0.918 | 1.000 | 10.1067 | — | 10.1036 (+0.030%) | 10.0172 (+0.893%) |
| VRPBLTW | 0.428 | 0.748 | 0.917 | 19.1193 | — | 19.6770 (-2.834%) | 18.7886 (+1.760%) |
| OVRPBLTW | 0.503 | 0.911 | 0.999 | 10.5903 | — | 10.5928 (-0.023%) | 10.4710 (+1.140%) |

### TEST Arm Distribution

| Problem | Selection frequency |
| --- | --- |
| TSP | BQ: 55.9%; DIFUSCO: 1.1%; DIFUSCO500: 4.5%; ELG: 8.4%; LEHD: 0.0%; OMNI: 0.0%; T2T: 1.2%; T2T500: 28.9% |
| CVRP | BQ: 2.0%; ELG: 0.0%; ICAM: 19.5%; LEHD: 9.1%; MVMOE: 0.1%; MoSES_CaDA: 11.1%; MoSES_RF: 8.7%; OMNI: 29.7%; RELD_CVRP: 19.7%; RouteFinder: 0.1% |
| ATSP | GLOP: 0.0%; ICAM_ATSP: 26.7%; MATNET: 14.2%; MATPOENET: 0.0%; UNICO_MatPOENet: 59.1% |
| OVRP | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 46.7%; MoSES_RF: 30.2%; RELD_MOEL: 0.0%; RELD_MTL: 0.6%; RouteFinder: 22.5% |
| VRPB | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 60.4%; RELD_MTL: 39.6%; RouteFinder: 0.0% |
| VRPL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 43.9%; MoSES_RF: 42.4%; RELD_MOEL: 9.0%; RELD_MTL: 2.3%; RouteFinder: 2.4% |
| VRPTW | MTPOMO: 0.0%; MVMOE: 1.2%; MoSES_CaDA: 25.0%; MoSES_RF: 18.3%; RELD_MOEL: 22.4%; RELD_MTL: 27.4%; RouteFinder: 5.7% |
| OVRPTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 75.3%; RELD_MTL: 24.7%; RouteFinder: 0.0% |
| OVRPB | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 56.7%; RELD_MTL: 43.3%; RouteFinder: 0.0% |
| OVRPL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 47.7%; MoSES_RF: 32.0%; RELD_MOEL: 0.0%; RELD_MTL: 0.2%; RouteFinder: 20.1% |
| VRPBL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 59.4%; RELD_MTL: 40.6%; RouteFinder: 0.0% |
| VRPBTW | MTPOMO: 0.0%; MVMOE: 3.6%; MoSES_CaDA: 8.8%; MoSES_RF: 10.5%; RELD_MOEL: 20.1%; RELD_MTL: 45.6%; RouteFinder: 11.4% |
| VRPLTW | MTPOMO: 0.0%; MVMOE: 1.4%; MoSES_CaDA: 29.6%; MoSES_RF: 16.2%; RELD_MOEL: 22.3%; RELD_MTL: 20.7%; RouteFinder: 9.8% |
| OVRPBL | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 53.5%; RELD_MTL: 46.5%; RouteFinder: 0.0% |
| OVRPBTW | MTPOMO: 0.0%; MVMOE: 0.1%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 73.0%; RELD_MTL: 26.9%; RouteFinder: 0.0% |
| OVRPLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 70.6%; RELD_MTL: 29.4%; RouteFinder: 0.0% |
| VRPBLTW | MTPOMO: 0.0%; MVMOE: 4.2%; MoSES_CaDA: 10.2%; MoSES_RF: 3.0%; RELD_MOEL: 20.1%; RELD_MTL: 44.5%; RouteFinder: 18.0% |
| OVRPBLTW | MTPOMO: 0.0%; MVMOE: 0.0%; MoSES_CaDA: 0.0%; MoSES_RF: 0.0%; RELD_MOEL: 68.0%; RELD_MTL: 32.0%; RouteFinder: 0.0% |

All single-method comparisons and SBS top1/2/3 are in `analysis_test.json`.
