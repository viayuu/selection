# NSS TRAIN/VAL 与 R77B：已完成任务比较

执行与结果检查：Sol；仅本地 CPU 读取已有缓存，不交 Astra 复核。TSP/CVRP均已有完成记录；正式 freeze 及 locked_train_val_handoff.json 由 Popper 收尾，本文不替代正式交接，也不重新检查其状态。

TRAIN 边界：共享 70,114 原始实例/题，方法真实 winner 计数复用 R77B 已有 native 表，并核对 NSS TRAIN cache stamp 的数据目录、方法名和行数。cpu_cache/*train.pt 是成本/native winner/输入缓存，不是模型 TRAIN 预测；NSS TRAIN 选择分布不可用，CSV 对应字段留空，不填0。

## TSP：完成，原统计保留

NSS：50轮/27,400 updates，最佳第48轮（JSON零基 47），模型 SHA `c677a8cd4ee473ee47bdedff871050827f75717c321ae9d551a3edc7bc78994b`。R77B：ep28/update34,776。同一 3,895 VAL 行已映射到原始快照 source_index，候选名称、FP64成本、native winner逐行一致。

| VAL指标 | R77B | NSS | NSS−B |
| --- | ---: | ---: | ---: |
| Top1 native (%) | 45.750963 | 43.825417 | -1.925546 |
| Top1 FP32标签 (%) | 45.417202 | 43.363286 | -2.053915 |
| Mean cost | 7.674409672 | 7.672828214 | -0.001581458 |
| Actual regret (%) | 0.410812799 | 0.389351501 | -0.021461298 |
| 零选择且真实获胜的方法对 | 4 | 4 | +0 |
| 上述方法覆盖native winner数 | 1036 | 1036 | +0 |

FP32 argmin 与 native winner 有 44 行不同；主比较使用 native Top1，不混用两种标签。Actual regret = mean((所选FP64成本 / 同池FP64最小成本 − 1)×100)。

| 方法 | TRAIN真实winner | B TRAIN选择 | VAL真实winner | B VAL选择/命中 | NSS VAL选择/命中 |
| --- | ---: | ---: | ---: | --- | --- |
| BQ | 20,669 | 24,095 | 1,159 | 1,347/662 | 1,175/585 |
| DIFUSCO | 6,084 | 0 | 330 | 0/0 | 0/0 |
| DIFUSCO500 | 865 | 0 | 44 | 0/0 | 0/0 |
| ELG | 5,184 | 7,916 | 296 | 443/166 | 359/131 |
| LEHD | 11,881 | 4 | 641 | 0/0 | 0/0 |
| OMNI | 404 | 0 | 21 | 0/0 | 0/0 |
| T2T | 22,538 | 37,763 | 1,267 | 2,083/948 | 2,353/987 |
| T2T500 | 2,489 | 336 | 137 | 22/6 | 8/4 |

B 的 VAL 零选择真实获胜方法：DIFUSCO, DIFUSCO500, LEHD, OMNI；其中 NSS 仍零选：DIFUSCO, DIFUSCO500, LEHD, OMNI。这组 native winner 共 1,036 行，B/NSS 命中分别 0/0。覆盖改善只能以这些行的实际命中与成本判断，非零选择本身不构成修复。

结果判断：NSS−B 的 native Top1 为 -1.925546 个百分点（正确实例数差 -75）；平均成本差 -0.001581458，actual regret 差 -0.021461298 个百分点。Top1 与成本的方向分别报告，不能互相替代。

在 B 零类真实winner这一相同子集，B/NSS actual regret 为 0.764720% / 0.703734%（均未命中的类别不因此变成覆盖修复）。

来源：[NSS现成VAL预测](/public/home/shiys/0selection2/code/V4/runs/R79_NSS_new_data_batch1024/TSP/best_val_predictions.npz)、[完成记录](/public/home/shiys/0selection2/code/V4/runs/R79_NSS_new_data_batch1024/TSP/training_complete.json)、[R77B锁定VAL](/public/home/shiys/0selection2/code/V4/runs/R77_sample_size_batch1024/selected_train_val_predictions/B/val.npz)。

## CVRP：完成，本次补充

NSS：50轮/27,400 updates，最佳第40轮（JSON零基39），模型 SHA `b1e7595eb16a89e3bdf526c3c4d8b05ce3d7d4dc39ab1241667657924e3fc574`。R77B：ep28/update34,776。同一3,895个VAL实例已按原始快照source_index和候选名称对齐，FP64成本及native winner逐行一致。本次未重算TSP结果。

| VAL指标 | R77B | NSS | NSS−B |
| --- | ---: | ---: | ---: |
| Top1 native (%) | 46.315789 | 44.749679 | -1.566110 |
| Top1 FP32标签 (%) | 46.315789 | 44.749679 | -1.566110 |
| Mean cost | 18.127197079 | 18.121459035 | -0.005738044 |
| Actual regret (%) | 0.689987131 | 0.620847226 | -0.069139904 |
| 零选择且真实获胜的方法对 | 1 | 0 | -1 |
| 上述零选择方法覆盖native winner数 | 84 | 0 | -84 |

CVRP的FP32 argmin与native winner没有分歧。零选择方法数减少不等于覆盖修复：RouteFinder由B选0次变为NSS选2次（0.05135%，仍为near-zero），两次均未命中。其84个真实winner在B/NSS下均命中0次。

| 方法 | TRAIN真实winner | B TRAIN选择 | VAL真实winner | B VAL选择/命中 | NSS VAL选择/命中 |
| --- | ---: | ---: | ---: | --- | --- |
| BQ | 8,999 | 5,375 | 486 | 281/94 | 26/10 |
| ELG | 1,428 | 764 | 71 | 35/6 | 57/8 |
| ICAM | 13,158 | 14,526 | 711 | 831/439 | 674/379 |
| LEHD | 10,116 | 2,146 | 558 | 100/46 | 27/17 |
| MVMOE | 1,506 | 2,732 | 100 | 150/29 | 194/39 |
| MoSES_CaDA | 1,832 | 703 | 95 | 51/5 | 42/7 |
| MoSES_RF | 2,163 | 2,363 | 120 | 136/26 | 152/30 |
| OMNI | 17,073 | 24,741 | 944 | 1,353/794 | 1,465/823 |
| RELD_CVRP | 12,676 | 16,764 | 726 | 958/365 | 1,256/430 |
| RouteFinder | 1,163 | 0 | 84 | 0/0 | 2/0 |

结果判断：NSS native Top1较B下降1.566110个百分点（1,743对1,804，少61次命中）；平均成本降低0.005738044，actual regret降低0.069139904个百分点。NSS在RouteFinder真实winner的相同84行子集上actual regret为0.918607%，低于B的1.269559%，但两者类别命中均为0，不能称为覆盖修复。CVRP目录未发现TRAIN模型预测；NSS TRAIN模型选择分布继续不可用，CSV字段留空，不将成本/native winner缓存冒充预测。

来源：[NSS现成CVRP VAL预测](/public/home/shiys/0selection2/code/V4/runs/R79_NSS_new_data_batch1024/CVRP/best_val_predictions.npz)、[CVRP完成记录](/public/home/shiys/0selection2/code/V4/runs/R79_NSS_new_data_batch1024/CVRP/training_complete.json)、[R77B锁定VAL](/public/home/shiys/0selection2/code/V4/runs/R77_sample_size_batch1024/selected_train_val_predictions/B/val.npz)。仅补充本节及CSV的CVRP行，TSP数值保留。

## 结论边界

这是各自按VAL选定模型在同一VAL上的描述比较，不是独立泛化验证或单变量因果结论；不能将方法非零选择频数等同修复，也不计算标签后择优联合成绩。没有读取 TEST/LIB、没有新推理/GPU/训练/tests/profile/模型修改/来源追踪。本文只分别列出TSP/CVRP，不扩展为18题整体成绩。
