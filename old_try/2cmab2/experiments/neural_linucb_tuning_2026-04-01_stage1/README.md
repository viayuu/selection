# Neural-LinUCB 第一批调参实验

这批实验的目标不是直接出最终结论，而是先回答下面几个更关键的问题：

1. 现在的 `loss 下降但 val 不明显变好`，是不是因为表示层训练过勤、过强。
2. 现在的训练轨迹里，是否存在探索不足，导致后续表示层只在少数 arm 上反复拟合。
3. `reward` 口径、线性头历史窗口、图编码器是否训练，这几件事里哪一件对验证集更敏感。

## 设计思路

- 先做小规模筛选：
  - `--max-samples-per-problem 1000`
  - 也就是先用 `TSP 1000 + CVRP 1000`
- 每组只跑 `seed=0`
- 每隔 `500` step 做一次 val
- 每隔 `500` step 自动保存 checkpoint
- 采用串行执行：
  - 避免多个实验同时抢同一张 GPU
  - 也更方便看日志和比较结果

如果这批里有 2 到 3 组明显更好，再把它们升到：

- 多个 `seed`
- 全量数据

## 实验列表

| 实验名 | 主要假设 | 关键改动 |
| --- | --- | --- |
| `s1_00_baseline_default` | 作为当前默认配置基线 | 保持当前默认参数 |
| `s1_01_less_rep_fit` | 表示层训练太频繁、太久，导致 recent buffer 被过拟合 | `train_every=200`, `representation_steps=2`, `representation_buffer_size=2000`, `initial_pulls=5` |
| `s1_02_more_explore` | 训练数据覆盖不够，需要更强探索 | 在 `s1_01` 基础上 `alpha=2.0` |
| `s1_03_less_explore` | 当前 bonus 可能过强，导致训练样本太噪 | 在 `s1_01` 基础上 `alpha=0.5` |
| `s1_04_full_linear_history` | 线性头只看有限窗口，可能丢失太多历史 bandit 信息 | 在 `s1_02` 基础上 `linear_head_buffer_size=0` |
| `s1_05_autosaea_reward` | 现在的 reward 口径和最终 `mean_cost` 不够一致 | 在 `s1_02` 基础上 `reward_mode=autosaea` |
| `s1_06_freeze_graph_encoder` | bandit 监督太稀疏，训练图编码器反而会带偏底层表示 | 在 `s1_02` 基础上 `--freeze-encoder` |
| `s1_07_low_lr` | 表示层更新步子太大，训练集 loss 虽降但泛化不稳 | 在 `s1_02` 基础上 `lr=3e-4` |

## 启动方式

主脚本：

- [launch_stage1.sh](/public/home/zhoucl/shiys/2cmab2/experiments/neural_linucb_tuning_2026-04-01_stage1/launch_stage1.sh)

推荐后台启动方式：

在这台服务器上，`setsid` 比 `nohup` 更稳，推荐优先用：

```bash
/bin/bash /public/home/zhoucl/shiys/2cmab2/experiments/neural_linucb_tuning_2026-04-01_stage1/start_stage1_detached.sh
```

如果你更习惯手写命令，也可以直接：

```bash
setsid /bin/bash /public/home/zhoucl/shiys/2cmab2/experiments/neural_linucb_tuning_2026-04-01_stage1/launch_stage1.sh \
  < /dev/null \
  > /public/home/zhoucl/shiys/2cmab2/experiments/neural_linucb_tuning_2026-04-01_stage1/setsid_stage1.log 2>&1 &
```

`nohup` 版本仍然保留，作为备选：

```bash
nohup /bin/bash /public/home/zhoucl/shiys/2cmab2/experiments/neural_linucb_tuning_2026-04-01_stage1/launch_stage1.sh \
  > /public/home/zhoucl/shiys/2cmab2/experiments/neural_linucb_tuning_2026-04-01_stage1/nohup_stage1.log 2>&1 &
```

## 输出位置

这一批实验统一输出到：

- `/public/home/zhoucl/shiys/2cmab2/outputs/tuning_2026-04-01_stage1/`

每个实验一个子目录，目录名和上面的实验名一致。

## 优先关注哪些指标

优先看：

1. `periodic_val.jsonl`
2. `summary.json`
3. `neural_linucb.log`

建议最先比较：

- `val_greedy.overall.mean_cost`
- `val_greedy.overall.top1_accuracy`
- `val_greedy.by_problem.tsp.mean_cost`
- `val_greedy.by_problem.cvrp.mean_cost`

不要只盯着：

- `loss`

因为当前 `loss` 优化的是：

- 用历史 `theta_at_pull` 和当前 `phi(x,a)` 的内积去拟合 reward

它只能说明表示层越来越会拟合训练窗口里的 bandit 监督，不代表最终 argmax 选臂一定更好。
