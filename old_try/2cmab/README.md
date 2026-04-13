# 2cmab — Online Neural-LinUCB 求解器选择

## 任务

给定一个组合优化问题实例（TSP 或 CVRP），在线学习自动选择最佳的神经求解器初始化方法。

## 算法

| 方法 | 特征 | 探索策略 | 论文 |
|------|------|---------|------|
| **Online Neural-LinUCB** (主方案) | DualEncoder 256维 → 64维 φ | last-layer UCB + 周期性网络更新 | Xu et al., 2020 |
| **LinUCB** | 34维手工+交互 | UCB 置信上界 | Li et al., 2010 |
| Random | - | 随机 | baseline |
| SingleBest Global | - | 全局固定最优 | baseline |
| SingleBest Per Problem | - | 按问题类型固定最优 | baseline |
| Oracle | - | 事后最优 (上界) | - |

### Online Neural-LinUCB 核心流程

对齐论文 Algorithm 1 (Xu et al., 2020 "Neural Contextual Bandits with UCB-based Exploration"):

```
for t = 1, 2, ..., T:
    1. 收到实例 x_t, 用当前网络提取特征 φ(x_t) (64维)
    2. 对每个可行臂 k, 计算 UCB_k = θ_kᵀφ + α√(φᵀ Z_k⁻¹ φ)
    3. 选择 UCB 最大的臂, 执行并观察 reward
    4. 更新 LinUCB 统计量: Z_k += φφᵀ, b_k += r·φ
    5. 每隔 q 步:
       a. 用历史数据重训 encoder + context_proj + reward_heads (J 步梯度下降, 梯度累积)
       b. 用新的 φ 重建所有臂的 Z_k 和 b_k (关键: φ 变了旧统计量失效)
```

## 文件说明

```
2cmab/
├── config.py              # ★ 所有可调参数集中配置 (修改此文件调参)
├── arm_config.py          # 12个统一臂 (TSP 12个, CVRP 8个) + action mask
├── build_cost_matrix.py   # 从 EasyNCO/results/test/ 构建 cost 矩阵
├── features.py            # NSS 手工特征 (9维几何 + scale + demand)
├── nss_encoder.py         # NSS Encoder_h (层次注意力编码器)
├── encoder.py             # DualEncoder (TSP/CVRP 各一个 Encoder_h)
├── reward.py              # 排名奖励 r=(K+1-rank)/K (AutoSAEA 公式24)
├── linucb.py              # LinUCB (34维交互特征)
├── neural_linucb.py       # Online Neural-LinUCB (主方案)
├── baselines.py           # Random / SingleBest / Oracle
├── run_offline.py         # 主实验入口
├── data/                  # cost矩阵和结果 (自动生成)
│   ├── tsp100_costs.npz
│   ├── cvrp100_costs.npz
│   └── results.json
├── logs/                  # 实验详细日志 (自动生成)
└── checkpoints/           # 模型 checkpoint (自动生成)
```

## 快速开始

### 前置条件

```bash
pip install numpy scipy torch hdbscan
```

### 步骤 1: 构建 cost 矩阵（只需运行一次）

```bash
cd neural-solver-selection
python 2cmab/build_cost_matrix.py
```

扫描 `EasyNCO/results/test/` 并生成 `2cmab/data/tsp100_costs.npz` 和 `cvrp100_costs.npz`。

### 步骤 2: 配置参数

打开 `2cmab/config.py`，所有参数都集中在此文件，带有详细中文注释：

```python
# 快速调试: 每种类型只取 1000 条
N_INSTANCES = 1000
DEVICE = "cuda"

# UCB 探索系数
ALPHA = 1.0

# 每隔 50 步重训网络, 每次训练 10 个 epoch
TRAIN_EVERY = 50
TRAIN_STEPS = 10
BATCH_SIZE = 64

# 每 500 步保存一次 checkpoint
CHECKPOINT_EVERY = 500
```

也可以不改文件，直接用命令行参数覆盖（见下方）。

### 步骤 3: 运行实验

```bash
# 方式 1: 使用 config.py 中的默认参数
python 2cmab/run_offline.py --method neural_linucb

# 方式 2: 命令行覆盖参数 (快速调试)
python 2cmab/run_offline.py --method neural_linucb \
    --device cuda \
    --n-instances 1000 \
    --verbose

# 方式 3: 自定义超参数
python 2cmab/run_offline.py --method neural_linucb \
    --alpha 0.5 \
    --train-every 100 \
    --train-steps 20 \
    --batch-size 128 \
    --device cuda

# LinUCB
python 2cmab/run_offline.py --method linucb --alpha 1.0

# 全部方法
python 2cmab/run_offline.py --method all --device cuda
```

### 步骤 4: 查看结果

#### 终端输出

```
============================================================
结果汇总
============================================================
  random                        : reward=0.5531
  single_best_global            : reward=0.8547, top1_hit=0.2107, regret=0.1363
  single_best_per_problem       : reward=0.8662, top1_hit=0.3617, regret=0.1248
  oracle                        : reward=0.9910
  neural_linucb                 : reward=0.8012, top1_hit=0.4567, regret=0.0890
```

#### 输出文件

| 路径 | 内容 |
|------|------|
| `data/results.json` | 所有方法的指标 (mean_reward, top1_hit_rate, regret, method_distribution) |
| `logs/neural_linucb_alpha1.0_q50_J10_seed42.log` | 详细日志 (每次选臂分数、reward、网络训练 loss) |
| `checkpoints/step500.pt` | 训练中间 checkpoint |
| `checkpoints/step1000.pt` | 训练中间 checkpoint |
| `checkpoints/final_alpha1.0_q50_J10_bs64_seed42.pt` | 训练完成的最终模型 |

## 参数说明

所有参数在 `config.py` 中集中管理，命令行同名参数可覆盖。

### 数据参数

| 参数 | config.py | 命令行 | 默认值 | 含义 |
|------|-----------|--------|--------|------|
| 实例数 | `N_INSTANCES` | `--n-instances` | 0 (全部) | 每种问题类型取多少条，0=全部10000条，快速调试设1000 |
| 随机种子 | `SEED` | `--seed` | 42 | 影响数据划分和训练顺序 |
| 设备 | `DEVICE` | `--device` | cpu | `cpu` 或 `cuda` |

### UCB 参数

| 参数 | config.py | 命令行 | 默认值 | 含义 |
|------|-----------|--------|--------|------|
| 探索系数 | `ALPHA` | `--alpha` | 1.0 | UCB 中的 α，越大越倾向探索，典型范围 0.1~2.0 |

### 网络参数 (仅 config.py 可调)

| 参数 | config.py | 默认值 | 含义 |
|------|-----------|--------|------|
| 特征维度 | `PHI_DIM` | 64 | context_proj 输出维度，也是 LinUCB 的工作维度 |
| 正则化 | `REG` | 1.0 | Z 矩阵初始化为 λI 中的 λ |
| Encoder 学习率 | `LR_ENCODER` | 1e-4 | 较小，防止大 encoder 过拟合 |
| Head 学习率 | `LR_HEAD` | 1e-3 | context_proj + reward_heads 的学习率 |

### 训练调度参数

| 参数 | config.py | 命令行 | 默认值 | 含义 |
|------|-----------|--------|--------|------|
| 重训间隔 | `TRAIN_EVERY` | `--train-every` | 50 | 每 q 步触发一次网络重训 (论文中的 q) |
| 训练步数 | `TRAIN_STEPS` | `--train-steps` | 10 | 每次重训的 epoch 数 (论文中的 J) |
| Batch 大小 | `BATCH_SIZE` | `--batch-size` | 64 | 梯度累积 mini-batch，越大训练越快越稳定 |
| 历史上限 | `MAX_HISTORY` | `--max-history` | 5000 | 训练只用最近 H 条历史，0=不限制 |

### Checkpoint 参数

| 参数 | config.py | 命令行 | 默认值 | 含义 |
|------|-----------|--------|--------|------|
| 是否保存 | `SAVE_CHECKPOINT` | `--save-checkpoint` | True | 是否保存模型 |
| 保存目录 | `CHECKPOINT_DIR` | `--checkpoint-dir` | checkpoints | checkpoint 存放目录 |
| 保存间隔 | `CHECKPOINT_EVERY` | `--checkpoint-every` | 500 | 每 N 步保存一次，0=只保存最终模型 |

### 日志参数

| 参数 | config.py | 命令行 | 默认值 | 含义 |
|------|-----------|--------|--------|------|
| 日志目录 | `LOG_DIR` | `--log-dir` | logs | 日志文件存放目录 |
| 终端详细输出 | `VERBOSE` | `--verbose` | False | True 则终端也显示每次选臂的详细分数 |

## Checkpoint 说明

### 保存内容

每个 `.pt` 文件包含完整的训练状态，可以恢复训练或直接推理：

- 网络权重: encoder, context_proj, reward_heads
- 优化器状态: Adam 的动量 (m, v)
- LinUCB 统计量: Z 矩阵, b 向量
- 训练进度: 交互轮次 t, 每个臂的拉取次数和累计奖励
- 超参数快照: 方便核对

### 保存时机

- **周期保存**: 每 `CHECKPOINT_EVERY` 步保存一次 → `checkpoints/step500.pt`
- **最终保存**: 训练结束后保存 → `checkpoints/final_alpha1.0_q50_J10_bs64_seed42.pt`

### 加载 checkpoint (代码示例)

```python
from encoder import DualEncoder
from neural_linucb import NeuralLinUCB
from arm_config import N_ARMS, ALL_METHODS

encoder = DualEncoder()
nl = NeuralLinUCB(n_arms=N_ARMS, encoder=encoder, arm_names=ALL_METHODS, device="cuda")
nl.load_checkpoint("2cmab/checkpoints/final_alpha1.0_q50_J10_bs64_seed42.pt")

# 直接推理: 给定实例坐标, 选择最佳求解器
arm = nl.select(coords, None, "tsp", 100, action_mask)
print(f"推荐求解器: {ALL_METHODS[arm]}")
```

## 日志详解

日志自动保存到 `logs/` 目录，文件名格式: `{method}_alpha{α}_q{q}_J{J}_seed{seed}.log`

```
# 选臂: 显示每个可行臂的 mu(exploitation), bonus(exploration), ucb(总分)
09:15:23 | t=42 | TSP | select arm=1(elg)
09:15:23 |      0(lehd            ): mu=+0.3214  bonus=0.8901  ucb=+1.2115
09:15:23 |      1(elg             ): mu=+0.5432  bonus=0.7654  ucb=+1.3086 <--
09:15:23 |      2(invit           ): mu=+0.1234  bonus=0.9012  ucb=+1.0246

# 更新: reward, 累计拉取次数, 平均 reward, 特征范数
09:15:23 | t=43 | update arm=1(elg) | reward=0.7500 | pulls=8 | avg_reward=0.6875 | phi_norm=3.45

# 网络重训: 臂统计汇总 + 训练 loss + 统计量重建
09:15:30 | t=50 | === 触发网络重训 (train_every=50) ===
09:15:30 |   臂统计汇总:
09:15:30 |      0(lehd            ): pulls=   12  avg_reward=0.6543
09:15:31 |   [fit_repr] step 5/10, loss=0.041088, n_train=50, n_updates=1
09:15:32 |   [rebuild] 用新 φ 重建 LinUCB 统计量 (n=50)

# 训练 vs Oracle 逐实例对比
09:15:23 | t=0 | TRAIN | TSP | chosen=0(lehd) cost=7.77 reward=0.92 | oracle=3(icam) cost=7.77 reward=1.00 | match=NO

# 测试结果
09:25:00 | === TEST 结果: mean_reward=0.8012, top1_hit=0.4567, regret=0.0890 ===
```

## 12 个臂 (求解器初始化方法)

| idx | 方法名 | TSP | CVRP |
|-----|--------|-----|------|
| 0 | lehd | o | o |
| 1 | elg | o | o |
| 2 | invit | o | o |
| 3 | icam | o | o |
| 4 | dact | o | o |
| 5 | lih | o | o |
| 6 | udc | o | o |
| 7 | omni | o | o |
| 8 | pointerformer | o | x |
| 9 | difusco | o | x |
| 10 | t2t | o | x |
| 11 | glop | o | x |

## 关键设计

- **在线学习**: 逐个实例交互, encoder 每 q 步持续更新, 不需要预先收集全部数据
- **持久网络**: encoder, context_proj, reward_heads 都是持久组件, 每次重训在上一轮参数基础上继续优化 (对齐原论文)
- **梯度累积**: TSP/CVRP 输入维度不同无法拼 batch, 用梯度累积等价实现 mini-batch
- **滑动窗口训练**: 网络训练只用最近 H 条历史, 防止后期重训变慢; rebuild 统计量仍用全部历史
- **统计量重建**: 网络更新后 φ 变了, 必须重建 Z/b (论文核心设计)
- **周期 checkpoint**: 每 N 步自动保存, 训练中断可从最近的 checkpoint 恢复
- **12 个统一臂**: TSP 和 CVRP 共有方法合并, TSP-only 方法通过 mask 排除
- **排名奖励**: r = (K+1-rank)/K, 有界 [1/K, 1], 非稀疏, 不受量级影响
