# R76 LIB / TRAIN 分布对比

同一 epoch 18 锁定模型；仅复用已有预测，按候选方法名称对齐。LIB 真 winner 使用 native 标签，绝不作为模型选择频数。TRAIN 的模型选择来自 logits argmax，不是 TRAIN 标签。
| 数据集 | LIB数 | TRAIN数 | LIB Top1 | LIB 选过方法数 | TRAIN选过方法数 | 选法 TV 距离 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TSPLIB | 49 | 70,114 | 28.57% | 3 | 3 | 12.84% |
| CVRPLIB | 100 | 70,114 | 33.00% | 5 | 7 | 18.71% |

1. CVRPLIB 的 RELD_CVRP 真实获胜 45/100 次，但模型选择 0 次；同 checkpoint 的 TRAIN 模型选择比例为 4.63%。这描述固定模型的分布，不用于调参。
2. LIB 样本数较小（TSPLIB 49、CVRPLIB 100）；这是固定模型的分布观察，不是新规则或训练方案的验证。
3. 选法分布变化与真实 winner 分布变化是不同量，宽表分别保留；TV=半个各方法选择比例绝对差之和，0%完全一致，100%完全分离。
主要图表见 `by_problem.html`；原始选择/标签频数见 `selection_distribution.csv`，TRAIN 全18题模型选法另存 `train_selection_distribution.csv`。
