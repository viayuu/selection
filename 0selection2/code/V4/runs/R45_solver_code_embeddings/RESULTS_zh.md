# R45：方法源码 Embedding 实验结果

## 实现与训练

- 使用已经生成并锁定的真实 `voyage-code-4` Embedding，没有再次调用 API。
- 每个 solver 保留 Encoder、Decision、Inference、Config 四个语义视图。
  固定向量经过可训练的角色投影和融合，与可训练 ID embedding 相加后归一化，
  替换 R43A 的 34 维手工特征分支；实例编码、双流交互、评分和 maximin 决策不变。
- 三组均从头训练完整 18 任务，seed=2、batch=128，复用 R43A 比较监督和 R-Drop。
  A 为手工特征对照；B 为正确源码映射；C 为固定错配映射，permutation_seed=4502。
- A/B/C 的共同参数初值一致；B/C 的全部可训练参数初值一致，仅固定语义归属不同。
  A 参数量 1,842,953，B/C 各 1,986,697，A/B 不是严格的等容量对照。
- 两张 RTX 3090 并行运行 B/C，再运行 A。训练峰值显存约 8.9 GiB，无 OOM。
  W&B 使用 offline 模式，本地指标和曲线完整保留。

| 组别 | 训练结束轮次 | 验证成本最佳轮次 | 成功更新次数 |
|---|---:|---:|---:|
| A：手工特征 | 40 | 38 | 56,880 |
| B：正确源码 Embedding | 38，预设早停 | 30 | 54,036 |
| C：错配源码 Embedding | 36，预设早停 | 28 | 51,192 |

每轮每个问题完整覆盖 10,000 个训练实例，不丢尾批。三组完成验证选型后，
先锁定各自的一份 best.pt，再分别评估一次完整 test；没有按 test 更换 checkpoint。

## 完整 18 问题 Test

每组 18,000 个实例。Top1 使用原生 ind；百分比指标按问题等权宏平均，
不是用 ALL 的两个平均成本直接相除。actual regret 是逐实例相对 Oracle 损失的宏平均。

| 模型 | Top1 | Top2 | Top3 | mean_cost | vs_SBS | vs_Oracle | actual regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| 历史 R43A，固定参考 | 47.944% | 81.567% | 92.472% | 11.168108 | -1.0324% | +0.9150% | 0.9433% |
| R45A：手工特征 | 47.911% | 81.744% | 92.433% | 11.168366 | -1.0320% | +0.9157% | 0.9432% |
| R45B：正确源码 Embedding | 47.400% | 81.389% | 92.311% | 11.170249 | -1.0149% | +0.9330% | 0.9634% |
| R45C：错配源码 Embedding | 47.817% | 81.633% | 92.322% | 11.170659 | -1.0080% | +0.9404% | 0.9682% |

## 结论

**本轮没有获得正确源码语义带来明确收益的证据，不建议用 R45B 替换 R43A 基线。**

- B 相对本轮匹配对照 A：Top1 下降 0.511 个百分点，mean_cost 增加 0.001883，
  actual regret 增加 0.02015 个百分点。纠正 863 例，同时把 955 例原本正确的选择改错。
- B 相对同容量错配对照 C：Top1 下降 0.417 个百分点，mean_cost 仅减少 0.000410，
  actual regret 减少 0.00480 个百分点。这是指标取舍，不构成整体胜出。
- B 在 TSP 上比 A 多选对 9 例，但选择成本仍更高；CVRP、ATSP 和 MVRP 宏平均
  的 Top1、mean_cost、actual regret 均未优于 A，收益没有在问题族之间形成一致方向。

这是单个训练 seed、单个固定错配映射下的方向筛选，不能据此断言所有源码表征均无效。
部分历史求解配置使用显式记录的本机默认值，仍保留 26 项未完全核实的部署来源；
本轮结论针对这版近似源码表征，不代表完整还原了所有历史标签的求解配置。

## 结果入口

- `comparison.md` / `comparison.csv`：ALL、18 个问题、问题族及验证最佳/最终/末五轮结果。
- `decision_changes.csv`：纠正、伤害、双方均错但成本变好/变坏的逐问题归因。
- `observations.csv`：验证最佳点和 test 的 108 行观测指标与单方法对比。
- `curves/comparison.png`：完整训练/验证对照；`curves/convergence_zoom.png`：后半程放大。
- 各训练目录：`args.json`、`train.log`、`history.json`、`best.pt`、`last.pt` 和分项 loss 曲线。
- `curves/A/`、`curves/B/`、`curves/C/`：验证最佳点和 test 的逐问题方法性能、选择分布图。
- `locked_checkpoints.json`、`test_predictions/`、`prediction_replay.json`：锁定权重、逐实例预测和重算核对。

54 份 test 预测均为每问题 1,000 个不同实例，成本保持 FP64；保存预测重算指标误差为 0。
固定 Embedding 已随 checkpoint 保存，加载模型和推理不需要 API key 或联网。
