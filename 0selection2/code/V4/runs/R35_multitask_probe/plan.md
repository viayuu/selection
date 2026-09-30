# R35 多任务共享训练对照

执行日期：2026-09-30。只完成联合续训 / TSP 单任务续训及 train/val 整理，不读取 test。

- 起点：R34 winner-cost `best.pt`，第50轮。两组严格恢复全部模型、AdamW moments/step 和 AMP scaler，然后把 LR 设置为固定 `2e-5`。
- 模型代码、solver ID / 34维特征、全部候选池、loss 均沿用 R34。无 Adapter，无 Q/K/V 改动。
- loss：`0.35 CE + 0.10 winner_pair + 0.02 scaled_cost_risk`；cost_scale=0.01，native ind，无类别权重。
- batch=640，dropout=0.1，weight_decay=1e-4，无数据增强。关闭 warmup、scheduler、early stop；保留 R34 的梯度裁剪1.0和 FP16/SDPA。
- A：每轮按现有全局问题顺序，18个问题各执行一次成功更新；B：仅执行一次 TSP 更新。两组都结束于300次 TSP 成功更新。
- 三个续训 seed：2/3/4；每个 seed 的 A/B 从同一个源 optimizer 独立复制，防止状态污染。
- 每个任务有独立 CPU Generator 生成索引表，独立 CPU/CUDA dropout 随机状态；TSP batch/RNG 的 SHA256 逐次记录并跨 A/B 校验。
- 采样沿用随机打乱、每次完整 pass 丢弃不足640的训练尾批；下一 pass 重新打乱。AMP 跳步降低 scaler 后重试相同 batch 和 dropout，成功前不推进 batch 序列。
- 评估点：0、30、60、…、300次 TSP 成功更新。A 在完成对应整轮18个任务之后评估。
- 每次完整评估 TSP train/val，eval 模式、FP32、无增强、不丢尾批，保存 logits/indices/native winner/cost/pool IDs。
- 主比较：300次更新 endpoint；同时汇报180/210/240/270/300五点评估均值。不会按最高点选模型。
- 统一横轴：TSP 成功更新。A 每个 seed 总5400次成功更新，B 总300次。
- 输出：六个实验目录、comparison.csv/md、top1/CE/mean_cost 曲线、逐实例配对差异和 verification.json。
- W&B：保留离线记录，沿用项目 selector。所有长任务在 tmux，日志与 GPU 采样落盘。

判断采用 endpoint 与 last5 的共同趋势：B 在多个 seed 中同时改善 train/val top1 且成本没有明显恶化，才支持继续测试问题专属 Adapter；只改善 train 不足以支持泛化瓶颈结论。A/B 都基本不变则转向实例表征与标签可预测性。
本实验无法区分其他任务对共享权重和 Adam moments 的各自影响，也不能代表其它17个任务。
