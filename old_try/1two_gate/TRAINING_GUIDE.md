# 双层强化学习求解器选择训练指南

> 说明：本文档中的源码目录已统一写为当前命名，即 `1two_gate/` 与 `EasyNCO/`。  
> 但如果直接运行当前代码，少量默认输出字符串仍可能落到 `my/outputs`，这是代码里的历史残留，不是本文档继续沿用旧命名。

## 一、预训练模型配置 (EasyNCO/pretrained/)

### 1.1 是否需要预训练模型？

**答案：不是必须的，但强烈建议使用！**

原因分析：

#### ✅ 使用预训练模型的优势
1. **框架可运行性**：代码能正常运行，但初始化策略随机
2. **训练收敛性**：预训练模型提供良好的初始化，加速收敛
3. **最终性能**：预训练模型的初始化质量直接影响最终求解质量
4. **研究完整性**：验证框架在有良好初始化下的表现

#### ❌ 不使用预训练模型的劣势
1. **随机初始化**：POMO/AM/LEHD从随机权重开始，初始化质量差
2. **训练困难**：Gate需要学习从差初始化中恢复，增加训练难度
3. **性能上限**：即使策略学好了，求解器本身也无法产生好解
4. **研究价值低**：无法评估框架在良好条件下的真实表现

### 1.2 预训练模型获取方式

#### 方法1：使用提供的Google Drive链接（推荐）

```bash
# 1. 访问Google Drive链接
# https://drive.google.com/drive/folders/1ElpVg7-uPrJf-gqEjDev4rTvOaJ91HoX?usp=sharing

# 2. 下载所有预训练模型到 EasyNCO/pretrained/ 目录
# 下载完成后应该有：
# EasyNCO/pretrained/model_epoch_90.ckpt  # Lightning格式
# EasyNCO/pretrained/model_epoch_90.pt     # PyTorch格式
```

#### 方法2：从原始仓库下载并转换

```bash
# 1. 从原始仓库下载模型（如果有权限）
# POMO: https://github.com/yd-kwon/POMO
# AM: https://github.com/wouterkool/attention-please-to-tour
# LEHD: https://github.com/yd-kwon/LEHD

# 2. 转换为EasyNCO格式（需要参考EasyNCO/pretrained/README.md的转换脚本）
```

### 1.3 预训练模型的使用配置

训练脚本中通过命令行参数指定预训练模型路径：

```bash
python 1two_gate/train_two_gate_tsp.py \
    --problem_size 50 \
    --device cuda:0 \
    --pomo_ckpt EasyNCO/pretrained/pomo_tsp50.ckpt \
    --am_ckpt EasyNCO/pretrained/am_tsp50.ckpt \
    --lehd_ckpt EasyNCO/pretrained/lehd_tsp50.ckpt
```

---

## 二、完整训练流程详解

### 2.1 环境准备

```bash
# 1. 确保项目路径正确
cd /mnt/d/Study/3/neural-solver-selection

# 2. 检查Python环境（需要PyTorch + TensorDict + TorchRL）
python -c "import torch; import tensordict; import torchrl; print('环境OK')"

# 3. 如果缺少依赖，安装：
# pip install torch tensordict torchrl
```

### 2.2 最简单的训练命令（从零开始）

```bash
# 基础训练：不使用预训练模型，仅验证框架可运行
python 1two_gate/train_two_gate_tsp.py \
    --problem_size 50 \
    --batch_size 32 \
    --train_steps 200 \
    --lr 1e-3 \
    --device cuda:0
```

**这个命令会做什么？**
1. **不加载预训练模型**：POMO/AM/LEHD使用随机初始化的权重
2. **训练200步**：每步训练Gate 1和Gate 2
3. **每20步打印日志**：显示loss、路径长度、熵、gate分布等
4. **每100步评估**：在评估集上测试平均路径长度
5. **保存模型**：训练结束后保存到 `1two_gate/outputs/two_gate_tsp_seed2024_n50.pt`

**预期结果**：
- 框架能正常运行
- 但由于求解器未训练，性能会很差
- 适合验证代码逻辑正确性

### 2.3 使用预训练模型的完整训练命令

```bash
# 推荐配置：使用预训练模型 + 完整zoo
python 1two_gate/train_two_gate_tsp.py \
    --problem_size 50 \
    --batch_size 32 \
    --train_steps 200 \
    --lr 1e-3 \
    --device cuda:0 \
    --init_zoo pomo am lehd \
    --iter_zoo none rrc_lehd \
    --pomo_ckpt EasyNCO/pretrained/pomo_tsp50.ckpt \
    --am_ckpt EasyNCO/pretrained/am_tsp50.ckpt \
    --lehd_ckpt EasyNCO/pretrained/lehd_tsp50.ckpt \
    --baseline critic_batch_mean \
    --critic_coef 0.5
```

**参数解释**：
- `--init_zoo pomo am lehd`：初始化器列表（3个）
- `--iter_zoo none rrc_lehd`：迭代器列表（2个）
- `--pomo_ckpt`：POMO预训练模型路径
- `--am_ckpt`：AM预训练模型路径
- `--lehd_ckpt`：LEHD预训练模型路径（用于初始化+RRC迭代）
- `--baseline critic_batch_mean`：使用Critic + Batch Mean的混合baseline
- `--critic_coef 0.5`：Critic损失权重系数

### 2.4 仅评估已训练模型

```bash
# 评估模式：加载已有checkpoint并运行评估
python 1two_gate/train_two_gate_tsp.py \
    --problem_size 50 \
    --device cuda:0 \
    --load 1two_gate/outputs/two_gate_tsp_seed2024_n50.pt \
    --only_eval \
    --eval_batches 10
```

---

## 三、代码调用EasyNCO方法的详细流程

### 3.1 核心调用链路图

```
train_two_gate_tsp.py (训练主流程)
    │
    ├─► ensure_local_easynco()         [easynco_bootstrap.py]
    │   └─► 创建 EasyNCO 包别名指向 EasyNCO/
    │   └─► 轻量级导入：只导入需要的模块
    │
    ├─► _build_initializer_zoo()          [构建初始化器zoo]
    │   ├─► 导入 EasyNCO.neural_solvers.envs.TSPEnv
    │   ├─► 导入 EasyNCO.neural_solvers.methods.pomo.policy.POMOPolicy
    │   ├─► 导入 EasyNCO.neural_solvers.methods.am.policy.AttentionModelPolicy
    │   ├─► 导入 EasyNCO.neural_solvers.methods.lehd.policy.LEHDPolicy
    │   ├─► 导入 EasyNCO.neural_solvers.pipeline.initialization.POMOInitialization
    │   ├─► 导入 EasyNCO.neural_solvers.pipeline.initialization.ARInitialization
    │   └─► 包装为 NeuralARInitializer [solver_zoo.py]
    │
    ├─► _build_iterator_zoo()             [构建迭代器zoo]
    │   ├─► 导入 EasyNCO.neural_solvers.envs.TSPEnv
    │   ├─► 导入 EasyNCO.neural_solvers.methods.lehd.policy.LEHDPolicy
    │   └─► 包装为 RRCLIHStyleLEHDIterator [solver_zoo.py]
    │
    └─► 训练循环
        ├─► 生成TSP实例: _make_tsp_batch()
        ├─► Gate 1选择: policy.gate1(feat1) → 采样动作a1
        ├─► 运行初始化: run_initializers_by_action(coords, a1, init_zoo)
        │   └─► 对每个batch索引，调用对应的初始化器.solve()
        │       └─► NeuralARInitializer.solve()
        │           └─► initialization.run(env, coords, strategy, phase)
        │               └─► EasyNCO的POMOInitialization/ARInitialization
        │                   └─► policy.forward(state_td)  [实际求解！]
        │
        ├─► Gate 2选择: policy.gate2(feat2) → 采样动作a2
        ├─► 运行迭代: run_iterators_by_action(coords, a2, sol0, iter_zoo)
        │   └─► 对每个batch索引，调用对应的迭代器.improve()
        │       └─► RRCLIHStyleLEHDIterator.improve()
        │           └─► 循环执行RRC（破坏-修复）
        │               └─► policy.pre_forward() + policy(state_td)  [实际求解！]
        │
        └─► 计算奖励并更新策略
```

### 3.2 详细代码执行流程

#### 步骤1：训练初始化

```python
# train_two_gate_tsp.py:240
def train(cfg: TrainConfig):
    # 1. 设置EasyNCO环境别名
    ensure_local_easynco()  # [easynco_bootstrap.py]
    #   → 创建 EasyNCO 包指向 EasyNCO/
    #   → 轻量级导入，避免加载整个EasyNCO/

    # 2. 配置PyTorch默认设备
    _configure_torch_defaults(cfg.device)
    #   → 如果用CUDA，设置默认tensor类型为torch.cuda.FloatTensor

    # 3. 设置随机种子
    _set_seed(cfg.seed)

    # 4. 构建初始化器zoo
    init_zoo = _build_initializer_zoo(cfg)
    #   见下文详细解释

    # 5. 构建迭代器zoo
    iter_zoo = _build_iterator_zoo(cfg)
    #   见下文详细解释

    # 6. 构建双门控网络
    policy, optimizer, feat_cfg = _build_gates(cfg, ...)
    #   policy.encoder: 论文风格图编码器
    #   policy.gate1: 选择初始化器的MLP
    #   policy.gate2: 选择迭代器的MLP
    #   policy.value: Critic网络（如果baseline=critic）
```

#### 步骤2：构建初始化器zoo

```python
# train_two_gate_tsp.py:93
def _build_initializer_zoo(cfg: TrainConfig):
    zoo = []

    # 初始化器1: POMO
    if "pomo" in cfg.init_zoo:
        # 1. 创建TSP环境（EasyNCO的env）
        env = TSPEnv(
            problem_size=cfg.problem_size,
            pomo_size=cfg.pomo_size or cfg.problem_size,
            device=cfg.device,
            aug_type=None,
            aug_factor=1
        )

        # 2. 创建POMO策略（EasyNCO的policy）
        policy = POMOPolicy(env_name="tsp")
        policy.to(cfg.device)

        # 3. 加载预训练权重（如果提供）
        if cfg.pomo_ckpt is not None:
            rep = load_policy_checkpoint(policy, cfg.pomo_ckpt, device=cfg.device)
            #   → 加载checkpoint文件
            #   → 提取state_dict
            #   → policy.load_state_dict(state_dict, strict=False)

        # 4. 冻结策略参数（不训练求解器）
        policy.requires_grad_(False)

        # 5. 创建POMO初始化器（EasyNCO的pipeline）
        init = POMOInitialization(policy=policy)

        # 6. 包装为我们的NeuralARInitializer
        zoo.append(NeuralARInitializer(
            name="pomo",
            env=env,
            initialization=init,
            decoder_strategy=cfg.decoder_strategy  # "greedy" or "sampling"
        ))

    # 初始化器2: AM（类似流程）
    if "am" in cfg.init_zoo:
        env = TSPEnv(problem_size=cfg.problem_size, pomo_size=1, ...)
        policy = AttentionModelPolicy(env_name="tsp")
        policy.to(cfg.device)
        if cfg.am_ckpt is not None:
            rep = load_policy_checkpoint(policy, cfg.am_ckpt, device=cfg.device)
        policy.requires_grad_(False)
        init = ARInitialization(policy=policy)  # 注意：AM用ARInitialization
        zoo.append(NeuralARInitializer(name="am", env=env, initialization=init, ...))

    # 初始化器3: LEHD（类似流程）
    if "lehd" in cfg.init_zoo:
        env = TSPEnv(..., method_name="lehd", ...)
        policy = LEHDPolicy(phase="test", env_name="tsp")
        policy.to(cfg.device)
        if cfg.lehd_ckpt is not None:
            rep = load_policy_checkpoint(policy, cfg.lehd_ckpt, device=cfg.device)
        policy.requires_grad_(False)
        init = LEHDInitialization(policy=policy)
        zoo.append(NeuralARInitializer(name="lehd", env=env, initialization=init, ...))

    return zoo
```

**关键点**：
- `TSPEnv`：EasyNCO提供的环境类，管理问题实例和状态
- `Policy`：神经网络的策略，包含Transformer编码器-解码器
- `Initialization`：EasyNCO的pipeline组件，调用policy进行求解
- `NeuralARInitializer`：我们的包装类，统一接口

#### 步骤3：训练循环 - 单步详解

```python
# train_two_gate_tsp.py:283
for step in range(1, cfg.train_steps + 1):
    # === 步骤3.1: 生成TSP实例 ===
    coords = _make_tsp_batch(cfg.batch_size, cfg.problem_size, cfg.device)
    # coords: [batch_size, problem_size, 2]
    #   例如：[32, 50, 2] 表示32个实例，每个50个城市，每个城市2维坐标

    # === 步骤3.2: Gate 1 - 选择初始化器 ===
    # 3.2.1 提取实例特征
    feat1 = tsp_instance_features(coords, cfg=feat_cfg, encoder=policy.encoder)
    # feat1: [batch_size, feature_dim]
    #   encoder是论文风格的图编码器（层次化或简单版）
    #   提取图嵌入 + 实例规模特征

    # 3.2.2 Gate 1前向传播，输出初始化器的对数概率
    dist1 = Categorical(logits=policy.gate1(feat1))
    # policy.gate1: MLP，输出 [batch_size, num_init] 的logits
    # num_init = len(init_zoo)，例如 3（pomo, am, lehd）

    # 3.2.3 采样动作（选择初始化器）
    a1 = dist1.sample()  # [batch_size]，每个值是 {0,1,...,num_init-1}
    logp1 = dist1.log_prob(a1)  # [batch_size]，采样概率的对数

    # 3.2.4 运行选中的初始化器
    sol0 = run_initializers_by_action(coords, a1, init_zoo)
    # 内部流程：
    #   for i in range(num_init):
    #       idx = (a1 == i).nonzero()  # 找到选择初始化器i的batch索引
    #       if idx.numel() > 0:
    #           sub = init_zoo[i].solve(coords[idx])  # 调用EasyNCO求解！
    #           out_tour[idx] = sub.tour
    #           out_len[idx] = sub.length
    # sol0.tour: [batch_size, problem_size]
    # sol0.length: [batch_size]

    # === 步骤3.3: Gate 2 - 选择迭代器 ===
    # 3.3.1 提取Gate 2特征（实例特征 + 初始解特征）
    feat2 = tsp_gate2_features(
        coords, sol0.tour, sol0.length,
        cfg=feat_cfg, encoder=policy.encoder
    )
    # feat2: [batch_size, feature_dim + tour_features_dim]
    #   包含：实例特征 + 初始解长度 + 初始解统计特征

    # 3.3.2 Gate 2前向传播，输出迭代器的对数概率
    dist2 = Categorical(logits=policy.gate2(feat2))
    # policy.gate2: MLP，输出 [batch_size, num_iter] 的logits
    # num_iter = len(iter_zoo)，例如 2（none, rrc_lehd）

    # 3.3.3 采样动作（选择迭代器）
    a2 = dist2.sample()
    logp2 = dist2.log_prob(a2)

    # 3.3.4 运行选中的迭代器
    sol1 = run_iterators_by_action(coords, a2, sol0, iter_zoo)
    # 内部流程：
    #   for i in range(num_iter):
    #       idx = (a2 == i).nonzero()
    #       if idx.numel() > 0:
    #           sub1 = SolutionBatch(tour=sol0.tour[idx], length=sol0.length[idx])
    #           sub2 = iter_zoo[i].improve(coords[idx], sub1)
    #               └─► 调用EasyNCO的RRC迭代优化！
    #           out_tour[idx] = sub2.tour
    #           out_len[idx] = sub2.length
    # sol1.tour: [batch_size, problem_size]
    # sol1.length: [batch_size]

    # === 步骤3.4: 计算奖励和baseline ===
    length = sol1.length  # [batch_size]
    reward = -length      # [batch_size]，越短越好，所以是负长度

    # baseline计算（三种模式）
    if cfg.baseline == "batch_mean":
        baseline_length = length.mean()
        baseline = -baseline_length
    elif cfg.baseline == "critic_batch_mean":
        baseline_length = length.mean()
        baseline_reward = -baseline_length
        value_pred = policy.value(feat1).squeeze(-1)
        baseline = baseline_reward + value_pred
        critic_loss = ((reward - baseline_reward - value_pred) ** 2).mean()
    elif cfg.baseline == "critic":
        baseline = policy.value(feat1).squeeze(-1)
        critic_loss = ((reward - baseline) ** 2).mean()

    # 计算优势函数
    adv = reward - baseline  # [batch_size]

    # === 步骤3.5: 计算损失并更新策略 ===
    # Actor损失：鼓励采样高奖励的动作
    actor_loss = -((logp1 + logp2) * adv.detach()).mean()
    #   adv.detach(): 停止梯度，避免影响baseline计算

    # 总损失
    loss = actor_loss + cfg.critic_coef * critic_loss

    # 反向传播
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

### 3.3 EasyNCO实际求解调用详解

#### 初始化阶段：POMO求解

```python
# solver_zoo.py:118
def solve(self, coords) -> SolutionBatch:
    with TorchNoGrad():
        # 调用EasyNCO的POMOInitialization.run()
        state_td, _out = self.initialization.run(
            self.env,           # TSPEnv
            coords,           # [batch_size, problem_size, 2]
            self.decoder_strategy,  # "greedy" or "sampling"
            "eval"            # phase
        )

    # state_td是TensorDict，包含：
    #   "reward": [batch_size, pomo_size]  # 负路径长度
    #   "selected_node_list": [batch_size, pomo_size, problem_size]  # 路径

    # 提取每个实例的最佳解（pomo_size个解中最好的）
    reward = state_td.get("reward")  # [B, pomo]
    best_idx = reward.max(dim=1).indices  # [B]
    best_tour = selected_node_list[torch.arange(B), best_idx]  # [B,N]
    best_length = -reward[torch.arange(B), best_idx]  # [B]

    return SolutionBatch(tour=best_tour, length=best_length)
```

**内部调用链**：
```
POMOInitialization.run()
    └─► TSPEnv.reset(problems=coords)
    └─► policy.pre_forward(reset_td)
    └─► for step in range(problem_size):
            state_td = policy(state_td)  # 自回归解码
            state_td = env.step(state_td)
    └─► 返回 state_td (TensorDict)
```

#### 迭代阶段：RRC优化

```python
# solver_zoo.py:227
def improve(self, coords, sol0: SolutionBatch) -> SolutionBatch:
    with TorchNoGrad():
        self.policy.eval()
        best_selected_node_list = sol0.tour[:, None, :]  # [B,1,N]

        # RRC循环：多次破坏-修复
        for _ in range(self.max_steps):
            # 1. 破坏当前解（随机选择一条子路径）
            partial_len, first_node_index, subpath_length, solution_copy = \
                self._destroy_solution(coords, best_selected_node_list)

            # 2. 修复被破坏的子路径
            # 2.1 加载被破坏的问题到环境
            self.env.load_problems(problems, batch_size)
            self.env.solution = destroyed_solution
            self.env.problems = destroyed_problem

            # 2.2 通过LEHD策略重新构建子路径
            reset_td = self.env.reset()
            self.policy.pre_forward(reset_td)
            state_td = self.env.pre_step()

            done = False
            current_step = 0
            while not done:
                if current_step == 0:
                    selected = self.env.solution[:, :, -1]
                elif current_step == 1:
                    selected = self.env.solution[:, :, 0]
                else:
                    selected = self.policy(state_td)  # 调用LEHD policy！

                next_td["action"] = selected
                current_step += 1
                state_td = self.env.step(next_td)
                done = state_td["done"].all()

            # 2.3 修复后的子路径
            repaired_sub_solution = torch.roll(
                self.env.selected_node_list, shifts=-1, dims=2
            )
            repaired_length = -state_td["reward"]

            # 3. 接受或拒绝修复结果
            best_selected_node_list = self._accept_repaired_solution(
                repaired_sub_solution,
                prev_length,
                repaired_length,
                first_node_index,
                subpath_length,
                solution_copy
            )

        best_tour = best_selected_node_list.squeeze(1)  # [B,N]
        best_length = self.env._get_travel_distance(
            problems=coords,
            selected_node_list=best_selected_node_list,
            batch_size=batch_size,
            pomo_size=1,
        ).squeeze(1)

        return SolutionBatch(tour=best_tour, length=best_length)
```

**关键点**：
- RRC通过随机破坏子路径，然后使用LEHD策略重新构建
- 破坏-修复过程重复`max_steps`次（默认3次）
- 每次只接受改进的解（如果修复后更差则保持原解）

---

## 四、实际训练命令示例

### 4.1 快速验证（10秒内完成）

```bash
# 最小化配置：不使用预训练，快速验证框架
python 1two_gate/train_two_gate_tsp.py \
    --problem_size 20 \
    --batch_size 4 \
    --train_steps 10 \
    --device cpu \
    --init_zoo pomo \
    --iter_zoo none \
    --log_every 2 \
    --eval_every 10
```

### 4.2 标准训练（几分钟完成）

```bash
# 标准配置：使用预训练模型，完整zoo
python 1two_gate/train_two_gate_tsp.py \
    --problem_size 50 \
    --batch_size 32 \
    --train_steps 500 \
    --lr 1e-3 \
    --device cuda:0 \
    --init_zoo pomo am \
    --iter_zoo none rrc_lehd \
    --pomo_ckpt EasyNCO/pretrained/model_epoch_90.pt \
    --am_ckpt EasyNCO/pretrained/model_epoch_90.pt \
    --lehd_ckpt EasyNCO/pretrained/model_epoch_90.pt \
    --baseline critic_batch_mean \
    --critic_coef 0.5 \
    --log_every 20 \
    --eval_every 100
```

### 4.3 大规模训练（研究级）

```bash
# 大规模配置：更长的训练，更多的评估
python 1two_gate/train_two_gate_tsp.py \
    --problem_size 100 \
    --batch_size 64 \
    --train_steps 5000 \
    --lr 1e-4 \
    --device cuda:0 \
    --init_zoo pomo am lehd \
    --iter_zoo none rrc_lehd two_opt \
    --pomo_ckpt EasyNCO/pretrained/pomo_tsp100.ckpt \
    --am_ckpt EasyNCO/pretrained/am_tsp100.ckpt \
    --lehd_ckpt EasyNCO/pretrained/lehd_tsp100.ckpt \
    --pomo_size 100 \
    --rrc_steps 5 \
    --two_opt_iters 1000 \
    --baseline critic_batch_mean \
    --critic_coef 0.5 \
    --log_every 50 \
    --eval_every 200 \
    --eval_batches 10
```

---

## 五、预期输出和结果解读

### 5.1 训练日志示例

```bash
[config] {
  "seed": 2024,
  "problem_size": 50,
  "batch_size": 32,
  "train_steps": 200,
  "lr": 0.001,
  "device": "cuda:0",
  "init_zoo": ["pomo", "am"],
  "iter_zoo": ["none", "rrc_lehd"],
  ...
}

[train step     1] loss=3.8154 len0=8.2345 len1=7.9823 adv=-0.2341±0.5678 critic_mse=0.1234 entropy=1.2345 gate1=(pomo:0.52,am:0.48) gate2=(none:0.65,rrc_lehd_3:0.35)

[eval step   100] mean_len=7.6543 best=7.6543 gate1=(pomo:0.60,am:0.40) gate2=(none:0.70,rrc_lehd_3:0.30)

...
```

**字段解释**：
- `loss`: 总损失（actor_loss + critic_coef * critic_loss）
- `len0`: 初始化后的平均路径长度
- `len1`: 迭代后的平均路径长度
- `adv`: 优势函数的均值和标准差
- `critic_mse`: Critic的MSE损失（如果使用critic）
- `entropy`: 策略熵（探索程度）
- `gate1`: Gate 1的动作分布（pomo占比、am占比等）
- `gate2`: Gate 2的动作分布（none占比、rrc_lehd_3占比等）

### 5.2 收敛标志

**正常的训练收敛**：
1. `len1` 逐渐下降
2. `gate1` 和 `gate2` 的分布趋于稳定
3. `entropy` 不会过大（>2）或过小（<0.1）
4. `eval mean_len` 逐步改善

**需要调整的信号**：
1. `gate1` 或 `gate2` 的分布完全坍缩到单个动作（如pomo:1.0）→ 探索不足
2. `loss` 不下降或震荡剧烈 → 学习率问题
3. `len1` 不下降 → 求解器质量差或训练步数不足

---

## 六、常见问题排查

### 6.1 EasyNCO导入错误

```python
ModuleNotFoundError: No module named 'EasyNCO.neural_solvers.envs.TSPEnv'
```

**原因**：`ensure_local_easynco()` 未被调用

**解决**：
```python
# 确保在train()函数开始时调用
def train(cfg: TrainConfig):
    ensure_local_easynco()  # 这行必须存在！
    ...
```

### 6.2 设备不匹配错误

```python
RuntimeError: Expected all tensors to be on the same device
```

**原因**：没有配置PyTorch默认设备

**解决**：
```python
# 在train()开始时调用
_configure_torch_defaults(cfg.device)  # 这行必须存在！
```

### 6.3 预训练模型加载失败

```bash
[solver_load] pomo: EasyNCO/pretrained/pomo_tsp50.ckpt missing=123 unexpected=45
```

**解决**：
- `missing`：模型权重不匹配，可能需要重新转换
- `unexpected`：checkpoint包含额外字段，正常现象，可以忽略

### 6.4 CUDA内存不足

```bash
RuntimeError: CUDA out of memory
```

**解决**：
- 减小 `batch_size`
- 减小 `problem_size`
- 减小 `pomo_size`

---

## 七、完整训练脚本模板

```bash
#!/bin/bash
# training_template.sh

# 配置
PROBLEM_SIZE=50
BATCH_SIZE=32
TRAIN_STEPS=1000
DEVICE="cuda:0"
LR=1e-3

# 预训练模型路径
POMO_CKPT="EasyNCO/pretrained/pomo_tsp50.pt"
AM_CKPT="EasyNCO/pretrained/am_tsp50.pt"
LEHD_CKPT="EasyNCO/pretrained/lehd_tsp50.pt"

# 运行训练
python 1two_gate/train_two_gate_tsp.py \
    --problem_size $PROBLEM_SIZE \
    --batch_size $BATCH_SIZE \
    --train_steps $TRAIN_STEPS \
    --lr $LR \
    --device $DEVICE \
    --init_zoo pomo am lehd \
    --iter_zoo none rrc_lehd \
    --pomo_ckpt $POMO_CKPT \
    --am_ckpt $AM_CKPT \
    --lehd_ckpt $LEHD_CKPT \
    --baseline critic_batch_mean \
    --critic_coef 0.5 \
    --pomo_size 50 \
    --rrc_steps 3 \
    --log_every 50 \
    --eval_every 200 \
    --eval_batches 5

# 训练完成后，模型保存在：
# 1two_gate/outputs/two_gate_tsp_seed{SEED}_n{PROBLEM_SIZE}.pt
```

---

## 八、总结

### 核心要点
1. **EasyNCO/pretrained/** 不是必须的，但强烈建议使用预训练模型
2. **EasyNCO调用链**：`env → policy → initialization.run()` 实际求解
3. **训练框架**：Gate 1选择初始化器 → Gate 2选择迭代器 → 计算奖励 → 更新策略
4. **baseline机制**：batch_mean（简单）、critic（学习值函数）、critic_batch_mean（混合）

### 训练建议
- **初期验证**：使用小配置、少步数、不加载预训练模型
- **正式训练**：使用预训练模型、完整zoo、合理的训练步数
- **超参数**：lr=1e-3（默认）、critic_coef=0.5、baseline=critic_batch_mean

### 下一步
1. 运行快速验证命令，确保代码能跑通
2. 从Google Drive下载预训练模型
3. 运行标准训练命令，观察收敛情况
4. 根据训练日志调整超参数
