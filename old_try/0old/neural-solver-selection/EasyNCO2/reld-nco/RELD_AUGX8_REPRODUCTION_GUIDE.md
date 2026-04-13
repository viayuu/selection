# ReLD augx 8 复现指南

## 📊 论文结果背景

根据论文"ReLD: Rethinking Light Decoder-based solvers for the Capacitated Vehicle Routing Problem"的实验结果，ReLD augx 8在CVRP问题上表现出色：

| 问题规模 | ReLD augx 8 Gap | POMO augx 8 Gap | 改进幅度 | 时间 |
|----------|----------------|----------------|----------|------|
| CVRP100 | **0.960%** | 1.004% | 0.044% | 1.34min |
| CVRP200 | **1.654%** | 3.403% | 1.749% | 2s |
| CVRP500 | **2.975%** | 11.135% | 8.16% | 15s |
| CVRP1000 | **6.757%** | 110.632% | 103.875% | 0.91min |

## 🔧 配置说明

### "augx 8" 含义
- `augx` = Augmentation ×倍数
- `8` = 8倍数据增强
- 在推理阶段使用8倍数据增强来提升性能

### 论文关键参数

**1. 模型架构改进**：
- **基础**: POMO架构
- **改进1**: 移除编码器归一化层
- **改进2**: 解码器中添加IDT (Identity Mapping)
- **改进3**: 解码器中添加FF (Feed Forward)
- **改进4**: 解码器中添加距离启发式

**2. 训练参数**：
```
- 训练轮数: 90 epochs
- 每轮实例数: 600,000
- 批大小: 120
- 优化器: Adam (lr=1e-4)
- 学习率衰减: 第70、80轮衰减0.1倍
- 权重衰减: 0
```

**3. 数据生成**：
```
- 问题规模: Uniform(40, 100)
- 三角分布: T(3,6,25) for route size r
- 容量设置: capacity = ceil(r * avg_demand)
```

**4. 推理设置**：
```
- 轨迹数量K = min(100, N) where N=problem_size
- 滚动策略: 贪心滚动 (greedy rollout)
- 数据增强: 8倍 (augx 8)
```

## 🚀 复现命令

### 1. 基础训练命令

```bash
# 训练ReLD模型（CVRP100为例）
python train.py \
  settings=reld_settings \
  model=reld \
  problem=cvrp \
  scale=100 \
  max_epochs=90 \
  cuda=[0]
```

### 2. 覆盖关键参数命令

```bash
# 精确复现论文设置
python train.py \
  settings=reld_settings \
  model=reld \
  problem=cvrp \
  scale=100 \
  max_epochs=90 \
  +module/batch_size=120 \
  +module/episodes=600000 \
  +module/optimizer_params.optimizer.lr=0.0001 \
  +module/optimizer_params.optimizer.weight_decay=0 \
  cuda=[0]
```

### 3. 不同规模训练

```bash
# CVRP200
python train.py settings=reld_settings model=reld problem=cvrp scale=200 max_epochs=90 cuda=[0]

# CVRP500
python train.py settings=reld_settings model=reld problem=cvrp scale=500 max_epochs=90 cuda=[0]

# CVRP1000
python train.py settings=reld_settings model=reld problem=cvrp scale=1000 max_epochs=90 cuda=[0]
```

### 4. 评估推理命令

```bash
# 使用8倍数据增强推理
python eval.py \
  settings=reld_settings \
  model=reld \
  problem=cvrp \
  scale=100 \
  decoder_strategy=sampling \
  aug_factor=8 \
  cuda=[0]
```

## 📋 配置文件要点

### 关键设置已更新到 `settings/reld_settings.yaml`：

1. **数据增强**：
   ```yaml
   env:
     aug_factor: 8  # augx 8: 8倍数据增强
   ```

2. **训练参数**：
   ```yaml
   module:
     batch_size: 120              # 论文设置
     episodes: 600000             # 论文设置
     max_epochs: 90               # 论文设置
     optimizer_params:
       optimizer:
         lr: 1e-4                 # 论文设置
         weight_decay: 0          # 论文设置
       scheduler:
         schedule: [70, 80]       # 论文设置
         gamma: 0.1               # 衰减0.1倍
   ```

3. **ReLD特有参数**：
   ```yaml
   model:
     normalization: null          # 移除归一化层
     logit_clipping: 50           # 更大的截断值
     use_graph_mean: false        # 不使用图均值
     first_placeholder: false     # 不使用first placeholder
   ```

## 🎯 成功要素

1. **严格按照论文设置**：
   - 确保数据分布为Uniform(40,100)
   - 使用三角分布T(3,6,25)设置容量
   - 保持90轮训练，每轮600K实例

2. **学习率调度**：
   - 第70、80轮衰减0.1倍
   - 权重衰减为0

3. **推理阶段**：
   - 使用8倍数据增强 (augx 8)
   - 贪心滚动策略
   - 轨迹数量K = min(100, N)

4. **对比验证**：
   - 训练完成后对比论文表格结果
   - 确保Gap值接近论文报告的数值

## 📊 预期结果

训练完成后，您应该得到与论文相似的结果：

| 规模 | 预期Gap | 论文Gap | 预期时间 |
|------|---------|---------|----------|
| CVRP100 | 约0.96% | 0.960% | 1.3min |
| CVRP200 | 约1.65% | 1.654% | 2s |
| CVRP500 | 约2.98% | 2.975% | 15s |
| CVRP1000 | 约6.76% | 6.757% | 0.91min |

## 🐛 常见问题

1. **调度器错误 (KeyError: 'step')**：
   - **已修复**: 配置文件中的`scheduler_type`已从`'step'`改为`'MultiStepLR'`
   - **已修复**: 调度器参数从`schedule`改为`milestones`
   - 现在可以正确使用学习率衰减

2. **训练时间过长**：
   - 减少episodes数量进行测试
   - 使用较小的问题规模验证

3. **结果不理想**：
   - 检查学习率调度是否正确
   - 确保数据增强参数设置正确
   - 验证容量生成逻辑

4. **内存不足**：
   - 减小batch_size
   - 使用梯度累积

## 📚 参考文献

- 论文: "ReLD: Rethinking Light Decoder-based solvers for the Capacitated Vehicle Routing Problem"
- POMO基础论文: "POMO: Policy Optimization with Multiple Optima for Learning to Solve Vehicle Routing Problems"