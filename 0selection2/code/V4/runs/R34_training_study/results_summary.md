# R34：训练协议与监督目标改进

日期：2026-09-30。18个问题，seed=2。三组完整配方对照，不是三个随机种子。

## 结论

本轮有可测量的提升，但不是大幅突破，也尚未达到80% top1。
按验证集选择的模型是 **R34 winner-cost，第50轮的 best.pt**。
固定这份 checkpoint 后，在18000个 test 实例上评估，top1 为 **47.3444%**。
相对同口径重测的 R33a 双流模型，提升 **1.7222个百分点**，mean cost 从 **11.1881 降至11.1729**。
相对 R32e 方法特征模型，提升 **1.3944个百分点**。

本轮保留双流 encoder/decoder、34维方法特征、d=128、4个注意力头、4层初始编码和2层联合编码。
没有通过增大模型或换 Q/K 方向取得这次结果。

## 实际修改

- 停用未经数据验证、由索引猜测的 `coord_dist` embedding；真实规模、节点和约束信息仍保留。
- CE 和主评估采用离线标签原有的 `ind`，避免 FP32 成本并列及排序规则改变 winner。
- 新增 `ce`、`winner_cost` 两种监督模式，旧目标继续作为对照；不改动数据或 split。
- 新增3轮 warmup、验证平台期降学习率、最低12000次成功更新后才允许早停，上限60轮。
- 保存 optimizer、scheduler、AMP scaler、RNG、更新计数和选择状态；修复恢复旧 checkpoint 时的统计策略兼容性。
- 每轮保存验证指标，每5轮及结束时完整评估训练集；保存曲线、梯度范数、学习率与成功更新数。

winner-cost 使用 `0.35 * CE + 0.10 * winner_pair + 0.02 * risk`。
其中 CE 不做类别重加权；winner_pair 比较 winner 与所有成本更高的候选，包括 runner-up。
相对成本尺度为0.01，pair 权重为 `clip(relative_gain / 0.01, 0.1, 5)`，risk 同样按0.01缩放。
模型推理仍直接输出 logits，真实 solver cost 只用于监督和评估。

## 验证集选型

| 配方 | 实际训练轮数 | 总成功更新 | 最佳轮次 | 最佳权重成功更新 | val top1 | val vs_SBS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原目标对照 | 50 | 13495 | 40 | 10795 | 47.5500% | -0.9965% |
| 纯 CE | 60 | 16194 | 60 | 16194 | 48.1833% | -1.0324% |
| **winner-cost** | **60** | **16194** | **50** | **13495** | **48.4667%** | **-1.0544%** |

轮次为1-based。原目标对照触发早停；另两组达到60轮上限。
三组采用同一调度和停止规则，但实际学习率轨迹、总预算并不完全相同。
选型规则预先固定为 `val_top1 - 0.25 * max(0, val_vs_sbs_pct)`；三组最佳点的 vs_SBS 均为负。
先验证任务完成和更新预算，再复制并校验 checkpoint 哈希，保存不可随意改写的选型记录，之后才评估 test。
另外两组没有读取 test 结果进行再选型。记录见 [validation_selection.json](validation_selection.json)。

## 同口径 Test 对比

| 模型 | top1 | top2 | top3 | mean_cost | macro vs_SBS | macro vs_Oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R32e：加入方法特征 | 0.4595 | 0.8010 | 0.9149 | 11.1869 | -0.9011% | +1.0502% |
| R33a：双流联合编码 | 0.4562 | 0.7958 | 0.9104 | 11.1881 | -0.8982% | +1.0533% |
| **R34 winner-cost** | **0.4734** | **0.8128** | **0.9220** | **11.1729** | **-0.9982%** | **+0.9508%** |

旧模型使用原有 `best.pt`，在相同 native-winner 指标和数值设置下重新评估，因此 top1 可能与旧报告的 FP32 并列排序口径略有不同。
旧 R32e/R33a 的54个标签文件哈希与当前数据全部一致。
SBS 是当前评估 split 上平均成本最低的固定方法，是 hindsight comparator；Oracle 是当前候选池内 VBS，不是路线问题的理论最优解。
ALL 的 top1/top2/top3、cost 和百分比均为逐问题宏平均；百分比不是把 ALL 的两个 cost 直接相除。

完整的18问题 top1/top2/top3、cost、SBS/Oracle 和臂分布表：
[test_summary.md](../R34_winner_cost_seed2_4090/test_summary.md)。本轮未测 LIB 或 zero-shot。

### 改善与退步

- 相对 R33a，16/18个问题的 top1 和 mean cost 同时改善。
- CVRP：top1 从32.5%到44.9%，增加12.4个百分点；mean cost 减少约0.0981。
- ATSP、OVRP、OVRPTW、VRPLTW 的 top1 分别增加2.2、2.5、2.6、2.5个百分点。
- VRPL：top1 下降0.9个百分点，mean cost 增加约0.00212。
- VRPBL：top1 下降0.3个百分点，mean cost 增加约0.00033。
- 逐实例对比：相对 R33a，修正1996个原错误，同时把1686个原正确改错，净增加310个正确选择。

完整差异见 [paired_comparison.json](paired_comparison.json)。这些是同一 test 集上的单次观测，不是跨 seed 的显著性结论。

## 曲线说明

[三配方对比曲线](recipe_curves.png) 按成功 optimizer 更新次数对齐，包含同口径 train/val CE、top1 和 vs_SBS。
[18问题验证曲线](per_problem_val_curves.png) 展示各问题的差异。
[最终配方训练曲线](../R34_winner_cost_seed2_4090/training_curves.png) 包含 loss、学习率和梯度范数。
完整原始记录见各 run 的 `history.json`、`train_eval_epoch*.json`、`eval_epoch*.json`。

训练总 loss 与分项是不同量，分项没有乘最终加权系数；不同配方的总 loss 不应直接比较。
`train_eval` 使用完整训练集、eval 模式、无增强，才适合与 val CE/top1 对比。
winner-cost 第60轮完整训练集 top1 为49.72%，同期 val 为48.40%；不是训练准确率很高但验证完全失效。
30轮之后仍有收益，但后段改善变慢；达到60轮上限不等于证明完全收敛。
仅凭这次实验不能认定继续增加轮数就能达到80%，也不能认定方法元信息或双流结构本身已经被独立验证有效。

## 复现与范围

- 最终权重：[best.pt](../R34_winner_cost_seed2_4090/best.pt)；冻结副本：[R34_winner_cost.pt](checkpoints/R34_winner_cost.pt)。
- 参数：[args.json](../R34_winner_cost_seed2_4090/args.json)；日志：[train.log](../R34_winner_cost_seed2_4090/train.log)。
- 结果及哈希：[comparison.json](comparison.json)、[label_manifest.json](label_manifest.json)。
- 使用 easynco、RTX4090、batch=640、FP16、SDPA 和 GPU 数据缓存；训练阶段433次采样的平均 GPU-Util 为84.1%，中位数90%，最大显存约18.54 GiB。
- W&B 为离线记录，未上传云端。25项代码测试和5项评估入口测试通过；保存的逐实例 logits 能重现全部18问题的 test top1。
- 原目标对照保留历史 FP32-argmin 类别频率，其 CE 标签仍使用 native winner。该统计与 native 计数涉及77/180000个训练记录，后两组不使用类别权重；不把后续权重统计修正算作已实验验证的收益。
- `source.tar.gz` 为启动时源码；`source_after_audit.tar.gz` 为 winner-cost 启动前版本。后续修正仅涉及恢复策略记录，不改变本轮已运行目标。细节见 [plan.md](plan.md)。

建议保留 R34 winner-cost 作为这次实验的新基线。接下来优先做等预算的重复 seed 和有针对性的表示层对照，而不是根据本次 test 逐问题调参。
