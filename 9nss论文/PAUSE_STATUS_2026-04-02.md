# NSS 多任务实验暂停记录

记录时间：2026-04-02 10:09 左右

## 当前状态

- 原始 NSS 未修改
- 正式训练已启动，但当前已暂停
- 当前暂停在 `schemeA`
- `schemeB` 尚未开始

## 当前落盘进度

- `schemeA` 指标日志：
  `/public/home/zhoucl/shiys/9nss论文/neural-solver-selection_schemeA/train_logs/config_multitask_rank_2025/metrics.jsonl`
- 当前指标日志已落盘到 `epoch 36`
- 但当前可直接恢复训练的 checkpoint 不是 `epoch 36`，而是 best checkpoint：
  `/public/home/zhoucl/shiys/9nss论文/neural-solver-selection_schemeA/train_logs/config_multitask_rank_2025/checkpoint_best.pt`
- 该 best checkpoint 对应：
  - `epoch = 13`
  - `best_metric = 3.4287885427474976`
- 总控脚本：
  `/public/home/zhoucl/shiys/9nss论文/launch_multitask_schemes_ab.sh`

## 暂停相关 PID

- `1000826`：外层顺序执行脚本
- `505619`：conda run 外层进程
- `505831`：内层 bash
- `506020`：实际训练 python

其中当前真正被暂停的是：

- `505831`
- `506020`

## 继续实验

如果仍保持 SIGSTOP 暂停方式，继续可执行：

```bash
kill -CONT 505831 506020
```

如果之后改为“释放显存但保留已保存进度”的方式，那么不能再从内存里的 `epoch 36` 原地恢复，
只能从上面的 `checkpoint_best.pt` 继续。

如果要再确认状态，可看：

```bash
grep -E '^State:|^Pid:|^PPid:' /proc/506020/status
tail -f /public/home/zhoucl/shiys/9nss论文/neural-solver-selection_schemeA/train_logs/config_multitask_rank_2025/metrics.jsonl
```
