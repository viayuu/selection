# Neural-LinUCB 5k 队列实验

这个目录用于在**不打断当前实验**的前提下，排队启动下一批 `Neural-LinUCB` 5k 调参实验。

## 这批实验的目标

当前重点验证的是：

- 最新 `problem-specific heads` 版本下
- 如果把探索强度从旧配置继续调低
- 是否能缓解之前看到的 test 泛化不稳、`pointerformer / omni` 被过度选中的问题

本批默认会串行执行 3 组：

1. `ph5k_a05_e005`
2. `ph5k_a03_e005`
3. `ph5k_a03_e002`

它们的共同点是：

- `max_samples_per_problem = 5000`
- `problem_specific_heads = true`
- `representation_balance_mode = problem_arm`
- `representation_loss_reweight = true`

主要区别是：

- `alpha`
- `train_epsilon`

## 如何启动

在仓库根目录执行：

```bash
/bin/bash 2cmab2/experiments/neural_linucb_tuning_2026-04-02_problem_heads_5k_queue/start_in_tmux.sh
```

默认 tmux session 名：

```bash
nlin_ph5k_queue_0402
```

也可以自己传一个名字：

```bash
/bin/bash 2cmab2/experiments/neural_linucb_tuning_2026-04-02_problem_heads_5k_queue/start_in_tmux.sh my_session_name
```

## 如何查看

看 tmux：

```bash
tmux attach -t nlin_ph5k_queue_0402
```

看日志：

```bash
tail -f 2cmab2/experiments/neural_linucb_tuning_2026-04-02_problem_heads_5k_queue/nlin_ph5k_queue_0402.log
```

## 输出目录

实验结果会写到：

```bash
2cmab2/outputs/tuning_2026-04-02_problem_heads_5k_queue/
```

对应子目录分别是：

- `ph5k_a05_e005`
- `ph5k_a03_e005`
- `ph5k_a03_e002`

## 等待策略

真正开跑前，这个脚本会先等 GPU 满足下面条件：

- 利用率 `<= 35%`
- 空闲显存 `>= 12000MB`
- 连续满足 `3` 次检查

默认每 `120s` 检查一次。

如果你想改得更激进或更保守，可以在启动前覆盖环境变量，例如：

```bash
WAIT_UTIL_THRESHOLD=20 WAIT_FREE_MEM_MB=16000 WAIT_INTERVAL_SEC=180 \
/bin/bash 2cmab2/experiments/neural_linucb_tuning_2026-04-02_problem_heads_5k_queue/start_in_tmux.sh
```
