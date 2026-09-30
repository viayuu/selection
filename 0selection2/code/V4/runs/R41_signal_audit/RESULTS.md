# R41 阶段性结果

## 1. 冻结18任务 test

四份权重先锁定路径、SHA256和验证选型指标，再运行完整 test。没有训练新 selector，没有改旧权重、模型结构、候选池或标签，也没有根据 test 重选 checkpoint。主决策全部使用分类 logits；R39A 未监督的性能头不参与选择。

| 模型 | 固定轮次 | Test Top1 | Top2 | Top3 | Mean cost | macro vs SBS | macro vs Oracle | actual regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R34 | 50 | 0.4734 | 0.8128 | 0.9220 | 11.172881 | -0.9982% | +0.9508% | 0.9819% |
| R39A | 53 | 0.4712 | 0.8141 | 0.9245 | 11.173855 | -0.9936% | +0.9555% | 0.9866% |
| R40A | 13 | 0.4684 | 0.8082 | 0.9208 | 11.172058 | -0.9967% | +0.9517% | 0.9808% |
| R40B | 24 | 0.4729 | 0.8080 | 0.9197 | 11.175296 | -0.9813% | +0.9677% | 0.9970% |

**没有发现 R40 的明显 test 提升。** R34 的严格 Top1 略高；R40A 的跨问题原始 mean cost 略低，但 macro vs SBS 没有胜过 R34。两种聚合不能混用。小差异不能宣称可靠的结构优势；仍按原验证选型保留 R39A 作为既有参考，不根据这次 test 反向调参。

每个模型的 ALL+18问题表、完整方法选择比例、逐方法对比见 `test_comparison.md`、`test_comparison.csv`、`test_arm_distribution.csv`。问题族汇总见 `test_family_comparison.csv`，图在 `test_method_plots/`。

## 2. 评估完整性

- Top1 按 native ind；原始 raw_label.pkl 成本直接使用 FP64 汇总，不把缓存 FP32 重新转 FP64 冒充原精度。
- 无增强，eval模式，保留尾批；局部几何标准化从 checkpoint buffer 恢复，没有在 test 重新拟合。
- 逐实例结果重算核验通过，共72000次完整 test 选择；四份权重和原模型源码均未改变。
- 验证重放通过。R34/R39A 使用历史评估batch640，R40A/B使用256。初次统一256时，R34少数近分数样本发生浮点选择变化；该次在读取test前被拦下，日志与预测保留在 `frozen_eval_initial_batch256.log` 和 `validation_batch256_initial/`。
- R39A 的核验样本预测从完整train/val前向结果提取，保留原索引、cost、logits，避免抽小子集后不同padding/batch形状引入比较漂移。
- W&B 已记录为offline，不声称云端已同步。本轮没有训练，因此不新增虚构的训练loss曲线。
- 40项测试通过；512个正式solver成本及4次预检通过更严格的参数、预算、独立进程和完整输入字段校验。早期R41运行源码没有完整快照，保留这一溯源限制；原哈希不覆盖，后续复核阶段保存不可变源码快照。见 `SOURCE_REVISIONS.md`。

## 3. 标签核验实际完成范围

抽样在查看R41 test前锁定：TSP、OVRPTW各64 train+64 val，seed2，按真实规模比例抽样；不按winner、模型错例或gap挑选。原始输入及哈希保存在 `audit_inputs/`、`audit_indices.json`。

**完整候选池核验未完成，不能报告完整池winner稳定率。** TSP的旧NSS标签缺少原solver命令、预算和权重绑定；OVRPTW的MTPOMO/MVMOE有有效decoder歧义，另外三个方法有原环境/行为绑定缺口。具体证据见 `original_solver_provenance.md`、`solver_manifest.json`、`blocked_dependencies.md`。

能够确认原入口的 RELD_MOEL/RELD_MTL 已完成8例预检，再完成每方法每split64例、两次独立进程复跑，共512个实际重复成本记录。配置固定为argmax、N个起点、aug=1、sample=1，没有为了制造随机性改变求解方式。原标签没有被覆盖，也没有把多次最小值作为新标签。

| 已实测的 OVRPTW 双方法对比 | train | val |
| --- | ---: | ---: |
| 样本数 | 64 | 64 |
| 两次成本最大差异 | 0 | 0 |
| RELD_MOEL−RELD_MTL 胜负符号稳定率 | 100% | 100% |
| 与旧标签胜负符号一致率 | 100% | 100% |
| 与旧成本最大差异 | 4.96e-11 | 4.94e-11 |
| R39A 在这两个方法中的正确率 | 65.62% | 53.12% |
| R39A 相对这两个方法最小成本的regret | 0.7984% | 0.6766% |

这里后两行是**双方法口径，不是7方法Oracle口径**。详见 `head_pair_stability.md/csv`。原成本差异与文本小数截断一致。该双方法的稳定监督仍有相当多实例被R39A选错，不能把这些错例主要解释成重复求解噪声。

`repeated_costs.npz` 保留所有已执行结果，缺失/未执行位置为NaN；`label_stability.md/csv`、`label_stability_by_gap.csv`明确保留完整池未测状态，不用两方法结果填满全池。

## 4. 下一步判断

首先补齐原标签生成recipe，尤其是TSP的权重/预算绑定和MTPOMO/MVMOE当次有效decoder，而不是立即训练R42或全量重标。顶层greedy不一定是实际greedy，源码支持sampling但缺少run-to-revision绑定，这个缺口不能靠调seed或换当前默认值解决。

在已核验的 OVRPTW RELD头部对比中，没有发现标签运行波动；更值得优先检验实例/约束关系的可泛化表示。这个结论只覆盖固定128例和两个方法，**不证明所有标签稳定，也不能计算整个项目的最高可达Top1**。

R41没有加入Adapter、改变Q/K/V、solver特征、loss或几何增强。所有可执行工作已经结束，不留任务让用户自行监控；缺失原配置的部分作为明确阻塞保留，而不是伪装成已完成的全池核验。
