# R42 实验结论

## 结论

R42 已完成两组完整18任务训练和冻结权重 test。关系图＋solver-query 路径已经实现并通过验证，但本轮没有达到性能突破目标，也没有显示出相对双流对照的明确优势。保留 R34／R39A 作为主参考，不因这次微小差异替换基线。

| 模型 | Test Top1 | Mean cost | Macro vs SBS | Macro vs Oracle | Actual regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| R34 | 47.344% | 11.172881 | -0.9982% | +0.9508% | 0.9819% |
| R39A | 47.122% | 11.173855 | -0.9936% | +0.9555% | 0.9866% |
| R40A | 46.844% | 11.172058 | -0.9967% | +0.9517% | 0.9808% |
| R40B | 47.289% | 11.175296 | -0.9813% | +0.9677% | 0.9970% |
| R42A：双流＋R-Drop | 47.350% | 11.172455 | -0.9999% | +0.9484% | 0.9715% |
| R42B：关系图＋solver-query＋R-Drop | 47.222% | 11.171769 | -1.0002% | +0.9481% | 0.9726% |

R42A 相对 R39A 的 Test Top1 增加 **0.228 个百分点**，actual regret 相对下降约 **1.53%**。R42B 对应为 **+0.100 个百分点、下降约1.41%**。B 的原始平均成本略低，但 Top1 与 actual regret 略差于 A，不能据此宣布 B 全面胜出。

## 验证与分族

两组从头训练，seed=2，使用相同保存的任务／实例序列、batch128、完整自然分布、winner-cost＋双向KL。无增强、重标或候选池修改。A/B 停止规则和预算上限相同，实际停止轮次与计算量不同。

| 模型 | 停止轮次 | 验证成本最佳轮次 | 最佳 Val Top1 | 最佳 Val regret | 最后5轮 Val Top1 | 最后5轮 Val regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R42A | 30 | 22 | 48.472% | 0.9290% | 48.449% | 0.9371% |
| R42B | 38 | 30 | 48.628% | 0.9336% | 48.404% | 0.9441% |

相对 R39A 验证集，A/B Top1 分别增加 **0.044／0.200 个百分点**，actual regret 相对下降 **1.70%／1.22%**。两者均未达到预设的 **Top1 +2个百分点且 regret -10%** 目标。终点训练／验证 Top1 为 A **49.11%／48.49%**、B **48.81%／48.31%**，未出现 R40 那种严重分离；但训练协议也不同，不能单独归因于 R-Drop。

| Test 问题族 | R39A Top1 | R42A Top1 | R42B Top1 | R39A regret | R42A regret | R42B regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSP | 37.20% | 37.20% | 36.10% | 0.6470% | 0.6142% | 0.6435% |
| CVRP | 46.20% | 43.60% | 45.90% | 0.6289% | 0.6570% | 0.6462% |
| ATSP | 76.30% | 76.70% | 76.00% | 0.4125% | 0.3485% | 0.4045% |
| MVRP（15任务宏平均） | 45.90% | 46.32% | 46.13% | 1.0713% | 1.0578% | 1.0542% |

MVRP 有小幅收益，但 B 的 MVRP regret 仅相对下降约1.60%，不是预期的大幅改善；CVRP 的成本表现反而有所退化。关系信息真实进入消息传递，不是只添加 attention bias，solver-query 也确实读取第2／4层节点；本次结果不能归结为“新分支没接上”，也不能据此断言所有关系建模无效或标签有噪声。

按问题内配对实例重采样2000次，A−R39A 的 Test Top1 差值95%区间为 **[-0.328,+0.772]个百分点**，B−R39A 为 **[-0.461,+0.717]个百分点**；两者 mean cost 与 actual regret 差值区间同样包含0。完整结果在 [paired_test_bootstrap.csv](paired_test_bootstrap.csv)。这是固定 checkpoint 的样本重采样，不是多训练seed的稳健性证明。

## 实现与核验

- B：4层32维边消息＋128维全局attention；保留第2／4层节点 memory；每 solver 四个 query、两层独立 cross-attention、共享评分头。没有旧的节点回写或公共 context decoder。
- 旧模型逻辑和历史权重未修改。34维 solver 特征、局部几何、native ind、候选顺序沿用原实现；成本与 winner 不进入前向。
- 每轮每任务79次成功更新，保留16例尾批。A 共42,660次，B 共54,036次；两次 dropout 前向合为一次更新。
- 44项原／新路径测试通过，另有1项真实 CUDA 强制梯度溢出测试通过：相同批次／dropout流重试，仅成功更新一次。
- 两组各18,000个验证与18,000个 test 预测均按原始 FP64 成本独立重算，指标完全吻合；test 数据／标签哈希与 R41 一致。最佳验证 checkpoint 重放为0选择翻转、0 logits 差异。
- test 只使用事先锁定的 A第22轮／B第30轮，未根据 test 改选轮次或拼接问题最优模型。同一 test 在 R41 已查看过，本次是阶段性固定版本比较，不是新的从未查看过的留出集。
- 启动快照保留原样。训练与 test 完成后仅修正汇总脚本的表格布局，记录见 [post_training_reporting_patch.json](post_training_reporting_patch.json)，没有修改模型、损失、评估或重跑 test。
- W&B 已保存两组离线记录；未配置在线凭据。所有训练任务均已正常结束。

完整 ALL＋18问题 Top1/2/3、cost、SBS／Oracle、臂分布和历史比较：[comparison.md](comparison.md)。CSV：[test_comparison.csv](test_comparison.csv)、[test_family_comparison.csv](test_family_comparison.csv)、[test_arm_distribution.csv](test_arm_distribution.csv)。

曲线：[Top1](comparison_top1.png)、[CE](comparison_ce.png)、[mean cost](comparison_mean_cost.png)、[actual regret](comparison_actual_regret_pct.png)。各 run 内另有分族、逐问题和训练KL曲线。

权重：[R42A best.pt](dual_stream_rdrop_seed2/best.pt)、[R42B best.pt](relation_query_rdrop_seed2/best.pt)。配置、日志、history、last.pt、逐轮验证预测和 test 预测均保留。
