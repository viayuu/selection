# 1online

`1online/` 是单实例、在线、on-policy 的 TSP 算子选择实验目录。

## 方案概览

- 训练单位：`一个 TSPLIB 实例 = 一个 checkpoint`
- 训练方式：在线 REINFORCE
- 决策结构：
  - 先选一次初始化方法
  - 再固定迭代步数，每一步选一个 operator
- 训练样本：对同一个基实例在线生成几何增强 batch
- 奖励：`-final_length`（即最终最优解长度的相反数）
- baseline：当前 augmentation batch 的 mean reward
- 训练动作：按策略分布采样
- 评估动作：`argmax`

## 目录结构

- `train_online_instance_policy.py`：训练入口
- `evaluate_online_instance_policy.py`：评测入口
- `instance_data.py`：单个 TSPLIB 实例读取与归一化
- `augmentations.py`：`identity / rotate / reflect / mix` 在线增强
- `policy.py`：初始化头 + operator 头
- `state_encoder.py`：状态编码器
- `env.py`：初始化 + 迭代式搜索环境
- `solver_zoo.py`：初始化器与 operator 适配层
- `trainer.py`：REINFORCE、trace、聚合指标输出
- `easynco_bootstrap.py`：EasyNCO 本地导入与运行期兼容补丁

## 默认动作空间

- `init_zoo = lehd elg difusco`
- `operator_zoo = two_opt lehd_rrc_step dact_2opt_step`

可以通过 CLI 覆盖。

补充说明：日志里的 `reward_mean` 现在表示 `episode_reward` 的均值，也就是 `-final_length` 的均值。

## 训练示例

```bash
PYTHONPATH=. python 1online/train_online_instance_policy.py \
  --device cuda:0 \
  --easynco_solver_device cuda:0 \
  --dact_solver_device cuda:0 \
  --tsplib_root EasyNCO/data/datasets/tsplib \
  --instance_name kroA100 \
  --train_steps 500 \
  --rollout_steps 100 \
  --aug_batch_size 16 \
  --eval_aug_size 128 \
  --eval_batch_size 32 \
  --eval_every 50 \
  --save_every 20 \
  --output_dir 1online/outputs/kroA100_run1
```

也可以直接指定文件：

```bash
PYTHONPATH=. python 1online/train_online_instance_policy.py \
  --instance_path EasyNCO/data/datasets/tsplib/tsplib/kroA100.tsp
```

## 评测示例

```bash
PYTHONPATH=. python 1online/evaluate_online_instance_policy.py \
  --checkpoint 1online/outputs/kroA100_run1/selector_best.pt \
  --tsplib_root EasyNCO/data/datasets/tsplib \
  --instance_name kroA100 \
  --device cuda:0 \
  --easynco_solver_device cuda:0 \
  --dact_solver_device cuda:0 \
  --eval_aug_size 128 \
  --eval_batch_size 32
```

## 输出文件

训练输出目录下至少包含：

- `config.json`
- `run.log`
- `train_metrics.jsonl`
- `eval_metrics.jsonl`
- `latest_metrics.json`
- `selector_last.pt`
- `selector_best.pt`
- `train_traces/`
- `eval_traces/`

独立评测输出目录下至少包含：

- `eval.log`
- `eval_metrics.json`
- `REPORT.md`
- `original_instance_actions.jsonl`
- `original_instance_actions.meta.json`
- `original_instance_actions_step_summary.json`
- `augmented_instance_actions.jsonl`
- `augmented_instance_actions.meta.json`
- `augmented_instance_actions_step_summary.json`

## Trace 格式

每条 `jsonl` 记录对应一个样本，包含：

- `train_step`
- `instance_id`
- `augmentation_type`
- `init_action_id`
- `init_action_name`
- `init_probs`
- `operator_action_ids`
- `operator_action_names`
- `operator_probs`
- `init_length`
- `current_length_by_step`
- `best_length_by_step`
- `reward_by_step`
- `final_length`
- `improvement`
- `normalized_improvement`
- `episode_reward`

`*_step_summary.json` 会统计每一个 rollout step 上各 operator 的：

- `counts`
- `distribution`
- `named_distribution`

`reward_by_step` 是单步 best-length 改善量；真正用于训练的 episode reward 是 `episode_reward = -final_length`。

这两个文件配合起来可以直接分析：

- 是否坍缩到单一初始化器
- 是否坍缩到单一 operator
- 是所有 step 都坍缩，还是只在某些 step 坍缩
- 不同 augmentation 上动作是否发生切换

## 说明

- `1online` 不修改 `EasyNCO` 源码。
- augmentation、初始化器和迭代器都遵循“**优先调用 EasyNCO 原生实现**，兼容性受限时再 fallback 到本地实现”的原则。
- `two_opt` 会优先尝试 EasyNCO 原生 2-opt 路径；只有在当前环境依赖不兼容时才退回本地实现。
- `easynco_bootstrap.py` 里加入了轻量兼容 stub，目的是让在线推理/训练入口尽量少依赖 Lightning 训练栈。
