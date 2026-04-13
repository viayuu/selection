# Neural-LinUCB 第二批调参实验（5k / problem）

这一批实验的目标是解决上一批 `1k / problem` 实验的两个主要问题：

1. 训练集太小，最终 `val / test` 排名不稳定；
2. 周期性验证点太少，难以判断训练过程中谁真的更稳。

## 本批设置

- 每个问题使用 `5000` 个样本：
  - `TSP 5000`
  - `CVRP 5000`
- 总样本数约 `10000`
- 按默认比例切分后：
  - `train ≈ 7000`
  - `val ≈ 1500`
  - `test ≈ 1500`
- `epochs = 1`
- 周期性验证：
  - `--eval-every-steps 500`
  - 这样在 `7000` 个 train step 中，大约会有 `14` 次周期性 val
- checkpoint：
  - `--checkpoint-every-steps 1000`

## 为什么这次仍然保留 8 组

上一批虽然规模偏小，但已经能看出一些弱趋势：

- `low_lr`
- `less_rep_fit`
- `more_explore`
- `autosaea_reward`

更值得继续观察。

但由于上一批 `val` 和 `test` 排名并不一致，如果这次一上来就只保留前 2~4 组，容易过早排除掉真正有潜力的设定。

所以这一批仍然保留和 stage1 完全一致的 8 组，用更大的数据和更多的 val 点重新比较。

## 实验列表

| 实验名 | 说明 |
| --- | --- |
| `s2_00_baseline_default` | 当前默认配置基线 |
| `s2_01_less_rep_fit` | 降低表示层训练频率与强度 |
| `s2_02_more_explore` | 增强探索 |
| `s2_03_less_explore` | 更弱探索 |
| `s2_04_full_linear_history` | 线性头使用全历史 |
| `s2_05_autosaea_reward` | 改 reward 口径 |
| `s2_06_freeze_graph_encoder` | 冻结底层图编码器 |
| `s2_07_low_lr` | 更小学习率 |

## 启动方式

推荐用 `setsid` 脱离运行：

```bash
/bin/bash /public/home/zhoucl/shiys/2cmab2/experiments/neural_linucb_tuning_2026-04-01_stage2_5k/start_stage2_detached.sh
```

主脚本：

- [launch_stage2.sh](/public/home/zhoucl/shiys/2cmab2/experiments/neural_linucb_tuning_2026-04-01_stage2_5k/launch_stage2.sh)

脱离启动脚本：

- [start_stage2_detached.sh](/public/home/zhoucl/shiys/2cmab2/experiments/neural_linucb_tuning_2026-04-01_stage2_5k/start_stage2_detached.sh)

## 监控方式

总日志：

- `/public/home/zhoucl/shiys/2cmab2/experiments/neural_linucb_tuning_2026-04-01_stage2_5k/setsid_stage2.log`

实时看总日志：

```bash
tail -f /public/home/zhoucl/shiys/2cmab2/experiments/neural_linucb_tuning_2026-04-01_stage2_5k/setsid_stage2.log
```

看某个实验自己的训练日志，例如 baseline：

```bash
tail -f /public/home/zhoucl/shiys/2cmab2/outputs/tuning_2026-04-01_stage2_5k/s2_00_baseline_default/neural_linucb.log
```

查后台进程：

```bash
pgrep -af "launch_stage2.sh|2cmab2/run_offline.py"
```
