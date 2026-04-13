# 原论文实现 vs `1two_gate/` 两层 Gate 强化学习实现：超详细对照（含代码定位）

> 目标：用“**按真实代码执行顺序**”的方式，把两套实现讲清楚，并明确回答：  
> 1) 原论文训练的是什么（是不是 MLP）？`1two_gate/` 训练的是什么（是不是 MLP）？  
> 2) 原论文的 padding/mask 在哪里、怎么做、怎么用？`1two_gate/` 是否也有？  
>
> 讨论范围：  
> - 论文代码：`neural-solver-selection/`（监督学习 selector + 离线标签）  
> - 你的实现：`1two_gate/`（两层 gate + Actor-Critic RL，在线真实调用 `EasyNCO/` 求解器得到 reward）  
> - `EasyNCO/` 仅作为 `1two_gate/` 调用的“求解器库/环境”；与论文无关，且排除 `9Reld/`。

---

## 目录

1. [三套目录定位](#1-三套目录定位)
2. [原论文实现全流程（`neural-solver-selection/`）](#2-原论文实现全流程-neural-solver-selection)
3. 你的实现全流程（`1two_gate/`）
4. [问题回答：训练的是什么？是不是 MLP？](#4-问题回答训练的是什么是不是-mlp)
5. [问题回答：padding/mask 具体怎么做？两边是否都有？](#5-问题回答paddingmask-具体怎么做两边是否都有)
6. [代码索引速查（建议收藏）](#6-代码索引速查建议收藏)

---

## 1. 三套目录定位

- `paper.md`：论文正文（Markdown）。
- `neural-solver-selection/`：原作者论文实现代码（**单 gate + 监督学习**）。  
  - 训练/评估依赖离线标签 `datasets/*/raw_label.pkl`，训练时不跑 solver。
- `EasyNCO/`：EasyNCO 平台（与你论文无关；`1two_gate/` 会调用；排除 `9Reld/`）。
- `1two_gate/`：你实现的两层 gate + Actor-Critic 强化学习原型（当前先做 TSP）。

---

## 2. 原论文实现全流程（`neural-solver-selection/`）

### 2.1 论文框架的三组件如何落到代码

论文说的三组件：

1) Feature extraction（实例特征）  
2) Selection model（输出 solver logits/score）  
3) Selection strategies（top-k/rejection/top-p 等策略评估）

代码映射（按真实文件）：

- Feature extraction：
  - 手工特征：`neural-solver-selection/dataset.py:54`（`manual_features`）
  - 学习式特征（注意力 encoder）：`neural-solver-selection/model.py:116`（`Naive_Encoder`）、`neural-solver-selection/model.py:147`（`Encoder_h`）
- Selection model：`neural-solver-selection/model.py:7`（`Selection_model`）
- Selection strategies：`neural-solver-selection/trainer.py:156`（`test`，包含 top-k/rejection/top-p）

### 2.2 入口脚本 `run.py`：从配置到训练器（按执行顺序）

入口：`neural-solver-selection/run.py:22`

核心执行链：

1) 读 config：`run.py:33-40`
2) 设置随机种子：`run.py:45`（调用 `utils.seed_everything`）
3) 读数据与离线标签：`run.py:90-91`（调用 `utils.prepare_dataset`）
4) 构建 `SelectionDataset`：`run.py:93-94`
5) 初始化模型：`run.py:105-109`
6) 初始化 trainer 并 `trainer.run(...)`：`run.py:111-119`

你可以把它理解为：

```
config -> dataset + raw_label -> SelectionDataset -> Selection_model -> trainer.run
```

### 2.3 数据与离线标签：`prepare_dataset` 到底返回了什么？

实现：`neural-solver-selection/utils.py:48`

关键逻辑：

- 读训练实例：`utils.py:54-56`（`datasets/{problem_type}train/dataset.pkl`）
- 读测试/验证/LIB 实例：`utils.py:57-65`（根据 `name` 选择 `val/test/LIB`）
- 读训练离线标签：`utils.py:66-68`（`datasets/{problem_type}train/raw_label.pkl`）
- 读测试离线标签：`utils.py:69-78`
- 对每个实例拼 `label=[ind,cost,time,gap]`：`utils.py:79-86`

这一步非常关键：**训练 selector 的监督信号来自离线标签**（不是在线跑 solver）。

### 2.4 `SelectionDataset`：手工特征与数据增强

实现：`neural-solver-selection/dataset.py:10`

- 如果 `manual_feature=True`，会在初始化时为每个实例预计算手工特征：`dataset.py:16-21`（调用 `manual_features`）。
- 如果 `data_aug=True`，会对坐标做 8-fold augmentation：`dataset.py:23-42`（调用 `utils.augment_xy_by_8_fold`）。

### 2.5 padding + mask：论文里“具体怎么做”的（你问的重点之一）

实现：`neural-solver-selection/dataset.py:105`（`collate_fn`）

核心几行（建议你对照看一眼）：

```python
# neural-solver-selection/dataset.py:105-128
# 重点：把变长样本 padding 到 batch 内 max_length，并构造 ninf_mask (0 / -inf)

lengths = np.array([data.shape[0] for data in batch_x])   # :116
max_length = np.max(lengths)                              # :117
batch_x = [F.pad(..., max_length - lengths[i]) ...]       # :118
ninf_mask = torch.zeros(len(batch_x), max_length)         # :119
for i in range(len(batch_x)):
    ninf_mask[i, lengths[i]: max_length] = float('-inf')  # :120-121
lengths = torch.tensor(lengths, dtype=torch.float32)      # :122
```

这里同时构造了两类“变长信息”：

- `lengths`（后续叫 `scales`）：每个样本真实 N（padding 前），用于当作额外特征输入。  
- `ninf_mask`：用于 attention/pooling 屏蔽 padding token。

### 2.6 模型：`Selection_model` 的结构（encoder + head）

实现：`neural-solver-selection/model.py:7`

#### 2.6.1 encoder 的选择（Naive vs Hierarchical）

```python
# neural-solver-selection/model.py:13-19
if model_params['pooling']:
    self.encoder = Encoder_h(**model_params)
    feature_dim = 2 * model_params['embedding_dim'] + 1
else:
    self.encoder = Naive_Encoder(**model_params)
    feature_dim = model_params['embedding_dim'] + 1
```

这说明：

- `pooling=True`（默认配置 `config_TSP.yml`）：使用 `Encoder_h`（层次化 pooling），输出是 `2*embedding_dim`，再拼 `scale` 得到 `2*embedding_dim+1`。
- `pooling=False`：使用 `Naive_Encoder`，输出是 `embedding_dim`，再拼 `scale` 得到 `embedding_dim+1`。

#### 2.6.2 head：MLP 输出 solver logits

默认 `ns_feature=False` 时：

```python
# neural-solver-selection/model.py:34-37
self.classifier = nn.Sequential(
    nn.Linear(feature_dim, model_params['embedding_dim']),
    nn.GELU(),
    nn.Linear(model_params['embedding_dim'], model_params['output_dim']))
```

即一个两层 MLP（中间 GELU），输出维度 `output_dim=num_solvers`。

#### 2.6.3 手工特征路线：`Naive_classifier`（纯 MLP）

`manual_feature=True` 时走：`neural-solver-selection/model.py:96`（`Naive_classifier`）。

它直接对 `(manual_features + scales)` 做 MLP 分类，不走注意力 encoder。

### 2.7 论文 selector 的深度学习“框架类型”：Transformer 还是 GNN？

看 `EncoderLayer` 就很清楚（`neural-solver-selection/model.py:271`）：

- `Wq/Wk/Wv` 线性投影：`model.py:279-282`
- `multi_head_attention(..., rank2_ninf_mask=mask)`：`model.py:298`
- FFN + 残差/归一化：`model.py:304-306`

而 mask 如何进入 attention 的 logits：

```python
# neural-solver-selection/model.py:324-359（节选）
score_scaled = score / sqrt(key_dim)                       # :340
score_scaled = score_scaled + rank2_ninf_mask[...]         # :342-343
weights = Softmax(dim=3)(score_scaled)                     # :347
out = weights @ v                                          # :351
```

结论：**论文 selector encoder 是 Transformer-style 的全连接自注意力（Graph Transformer / Attention Model 风格），不是传统 message-passing GNN。**

### 2.8 训练循环：`trainer.train_one_epoch` 怎么更新参数？

实现：`neural-solver-selection/trainer.py:106`

关键逻辑（按代码顺序）：

1) `self.model.train()`：`trainer.py:113`
2) batch 解包：`trainer.py:115-123`（拿到 `x/y/cost/scales/mask/...`）
3) forward：`trainer.py:127`
4) loss：
   - CE：`trainer.py:131-133`
   - rank：`trainer.py:133-135`
5) `l.backward()` + `self.optimizer.step()`：`trainer.py:136-139`

优化器是在 `__init__` 中用 `self.model.parameters()` 构建的：

```python
# neural-solver-selection/trainer.py:15-44（节选）
self.model = model.to(self.device)                          # :28
self.optimizer = torch.optim.Adam(self.model.parameters(),  # :43
                                  lr=..., weight_decay=...)
```

所以：**被训练的参数集合就是 selector 模型的全部参数**（encoder + MLP head；如果是 manual_feature 路线则只有 MLP）。

### 2.9 RankingLoss：到底在做什么？

实现：`neural-solver-selection/loss.py:7`

它不是简单的 “argmin(cost)” 分类，而是用 cost 的排序信息构造多轮 NLL：

```python
# neural-solver-selection/loss.py:13-20
for i in range(self.top_k):
    cur_cost, ind = costs.topk(self.num_solvers - i, largest=True)
    cur_label = cur_cost.min(dim=1)[1]
    cur_logits = torch.take_along_dim(logits, ind, 1)
    loss += F.nll_loss(F.log_softmax(cur_logits, 1), cur_label)
```

直观解释（对照代码行为）：

- 第 0 轮：在全体 solver 里选 cost 最小的当 label
- 第 1 轮：把“最差的一个 solver”剔掉后再选最小（通过 `topk(self.num_solvers - i, largest=True)` 达到“逐步缩小集合”的效果）
- …累计多轮 NLL  

这种 loss 会比纯 top-1 分类更“利用 cost 信息”，也更鲁棒（论文里也强调这一点）。

### 2.10 选择策略评估：`trainer.test` 怎么离线模拟 top-k/rejection/top-p？

实现：`neural-solver-selection/trainer.py:156`

#### 2.10.1 推理输出收集

关键行：

- `score_mat.append(softmax(y_pred))`：`trainer.py:195`
- `gap_mat.append(gap)`、`time_mat.append(time_cost)`：`trainer.py:196-197`
- selector 推理耗时：`select_time = (time.time()-start)/num_instances`：`trainer.py:200-201`

#### 2.10.2 top-k

```python
# neural-solver-selection/trainer.py:225-235（节选）
_, topk_ind = score_mat.topk(k, 1, largest=True)          # :226
topk_gap = gap_mat.gather(1, topk_ind).min(dim=1)[0]      # :227
topk_time = time_mat.gather(1, topk_ind).sum(dim=1)       # :228
```

含义：

- gap：对 top-k 候选 solver 集合取 `min(gap)`（模拟“跑 k 个 solver 取最好”）
- time：对 top-k 候选 solver 集合取 `sum(time)`（模拟“都跑了的总耗时”）再加 `select_time`

#### 2.10.3 rejection-based（置信度拒绝）

```python
# neural-solver-selection/trainer.py:238（节选）
sort_ind = score_mat.max(dim=1)[0].sort(descending=True)[1].cpu().numpy()
```

先按 `max softmax` 从高到低排序，再把低置信度部分用 top-k，高置信度部分用 top-1（见 `trainer.py:242-255`）。

#### 2.10.4 top-p

实现：`neural-solver-selection/trainer.py:257-287`

对每个样本从 top-1 开始逐步扩大集合，直到累计概率 ≥ p。

---

## 3. 你的实现全流程（`1two_gate/`）

> 你的实现和原论文最根本的区别：  
> - 原论文 selector 用监督学习，**训练/评估不需要真实跑 solver**（离线标签）。  
> - `1two_gate/` 用 RL 训练两个 gate，每 step 必须 **真实调用** initializer/iterator 得到最终长度，再构造 `reward=-length`。

### 3.1 让 `EasyNCO/` 可被导入：`ensure_local_easynco`

原因：虽然仓库里已经有 `EasyNCO/` 目录，但这条原型线仍需要统一处理本地导入、轻量 stub 和部分兼容补丁。  
解决：通过 `ensure_local_easynco()` 在运行时补齐这些兼容层，而不是直接修改 `EasyNCO/`。

实现：`1two_gate/easynco_bootstrap.py:9`

关键说明：

- 创建/修改 `sys.modules["EasyNCO"]` 的 `__path__` 指向 `EasyNCO/`：`easynco_bootstrap.py:28-42`
- 提供轻量 stub 避免导入整个 data/methods/backbones：`easynco_bootstrap.py:43+`
- 给 DIFUSCO 的 Cython 扩展提供 NumPy fallback：`easynco_bootstrap.py:183-188` 与 `_try_install_difusco_cython_merge_fallback`

### 3.2 数据：训练无限流 + 评估固定集（分布对齐论文 gaussian mixture）

实现：`1two_gate/tsp_data.py`

- 训练：`TSPBatchGenerator.sample(...)` 每 step 生成新 batch（`tsp_data.py:126-164`）
- gaussian mixture 逻辑：`tsp_data.py:48-86`（`num_modes=0` 时退化 uniform；否则混合并 MinMax 到 `[0,1]^2`）

训练脚本在 `train(...)` 中创建 generator：`1two_gate/train_two_gate_tsp.py:523-528`

### 3.3 特征提取：默认使用“论文同款 encoder”的 `1two_gate/` 本地重写版

实现分两层：

1) encoder（Transformer-style）：`1two_gate/paper_encoder.py`
2) 特征拼接逻辑：`1two_gate/features.py`

#### 3.3.1 Gate1 的实例特征 `feat1`

实现：`1two_gate/features.py:tsp_instance_features`（`features.py:77`）

关键逻辑：

- 默认 mask：`features.py:71-75`（全 0 的 `(B,N)` mask）
- `graph_emb = encoder(coords, mask)`：`features.py:94`
- 拼接 `scale=N`：`features.py:96-99`

#### 3.3.2 Gate2 的条件特征 `feat2 = g(data, sol0)`

实现：`1two_gate/features.py:tsp_gate2_features`（`features.py:154`）

会在 `feat1` 上拼接：

- `length0`：`features.py:164-170`
- `tour_stats`（mean/std/min/max）：`features.py:171-173`（细节在 `tsp_tour_features`，`features.py:129-151`）

#### 3.3.3 encoder 的结构类型（同样是 Transformer-style）

`1two_gate/paper_encoder.py:163`（`EncoderLayer`）与论文版本对应，关键结构一致：

- Q/K/V + multi-head attention：`paper_encoder.py:171-189`
- mask 加到 attention logits：`paper_encoder.py:206-227`（`score_scaled += rank2_ninf_mask[...]`）
- 层次化 pooling：`paper_encoder.py:48`（`Encoder_h`）和 `paper_encoder.py:94`（`Encoder_block_h`）

### 3.4 Gate/critic 的网络：MLP（但你训练的不止 MLP）

实现：`1two_gate/gates.py`

- gate MLP：`gates.py:14-26`
- critic MLP：`gates.py:29-45`
- 组合对象：`TwoGateAC`（`gates.py:47-107`，内部持有 `encoder/gate1/gate2/value1/value2`）

### 3.5 统一求解接口与按 action 分流：`1two_gate/solver_zoo.py`

你必须把多种 initializer/iterator 统一为一个 RL 可用的接口：

- `SolutionBatch`：`solver_zoo.py:7-15`（reward 定义为 `-length`）
- 批内分流执行：
  - initializer：`solver_zoo.py:42-60`
  - iterator：`solver_zoo.py:63-82`
- solver 调用统一包裹在 `TorchNoGrad`：`solver_zoo.py:85-98`

例如 `NeuralARInitializer.solve(...)` 从 EasyNCO 的 rollout 里取 best tour：

- 运行 initialization：`solver_zoo.py:121-128`
- 取 reward / selected_node_list：`solver_zoo.py:129-138`
- 选 best（max reward = min length）：`solver_zoo.py:139-148`

### 3.6 构建 initializer zoo / iterator zoo：显式冻结 solver（不训练它们）

构建 initializer zoo：`1two_gate/train_two_gate_tsp.py:_build_initializer_zoo`（`train_two_gate_tsp.py:227`）

以 POMO 为例（你能看到明确冻结）：

```python
# 1two_gate/train_two_gate_tsp.py:244-256（节选）
policy = POMOPolicy(env_name="tsp")          # :246
policy.to(cfg.device)                        # :247
... load_policy_checkpoint ...               # :248-251
policy.requires_grad_(False)                 # :252  <- 冻结 solver
init = POMOInitialization(policy=policy)     # :253
zoo.append(NeuralARInitializer(...))         # :254-256
```

iterator zoo 同理：`1two_gate/train_two_gate_tsp.py:_build_iterator_zoo`（在同文件后续），例如 LEHD RRC：

- 加载/冻结 LEHDPolicy：`train_two_gate_tsp.py` 中 `policy.requires_grad_(False)`（同上风格）
- wrapper 是 `RRCLIHStyleLEHDIterator`：`1two_gate/solver_zoo.py:346`

### 3.7 构建 gates/critic（真正要训练的模型）

实现：`1two_gate/train_two_gate_tsp.py:_build_gates`（`train_two_gate_tsp.py:463`）

关键逻辑：

- 先构造 encoder：`train_two_gate_tsp.py:470-474`（`build_paper_tsp_encoder`）
- 用 dummy forward 推断 `feat1/feat2` 维度：`train_two_gate_tsp.py:475-485`
- 构造 gate1/gate2：`train_two_gate_tsp.py:487-488`
- 构造 critic（baseline=critic 或 critic_batch_mean 时）：`train_two_gate_tsp.py:491-493`
- 关键：优化器是 `policy.parameters()`（会包含 encoder+gates+critic）：`train_two_gate_tsp.py:497`

```python
# 1two_gate/train_two_gate_tsp.py:495-498
policy = TwoGateAC(...).to(cfg.device)                 # :495
optimizer = optim.Adam(list(policy.parameters()), ...) # :497
```

### 3.8 AC 训练循环：两次决策、reward、baseline、更新（逐行对齐你的设定）

主循环：`1two_gate/train_two_gate_tsp.py:594`

关键片段（非常建议你对照读这段，基本就是你的伪代码）：

```python
# 1two_gate/train_two_gate_tsp.py:594-657（节选）
feat1 = tsp_instance_features(coords, ...)             # :598
dist1 = Categorical(logits=policy.gate1(feat1))        # :599
a1 = dist1.sample()                                    # :600
logp1 = dist1.log_prob(a1)                             # :601
sol0 = run_initializers_by_action(coords, a1, init_zoo)# :603

feat2 = tsp_gate2_features(coords, sol0.tour, sol0.length, ...) # :606
dist2 = Categorical(logits=policy.gate2(feat2))        # :607
a2 = dist2.sample()                                    # :608
logp2 = dist2.log_prob(a2)                             # :609
sol1 = run_iterators_by_action(coords, a2, sol0, iter_zoo)      # :611

reward = -sol1.length                                  # :614
... baseline 计算 ...                                   # :617-647
adv1 = reward - baseline1                              # :649
adv2 = reward - baseline2                              # :650
actor_loss = -(logp1 * adv1.detach() + logp2 * adv2.detach()).mean() # :652
loss = actor_loss + cfg.critic_coef * critic_loss      # :653
loss.backward(); optimizer.step()                      # :655-657
```

你要求的 5 个点在这段代码里的对应关系：

1) **AC 架构**：有 actor（gate1/gate2）与 critic（value1/value2，可选），见 `train_two_gate_tsp.py:491-493` 与训练循环 `:617-645`。  
2) **baseline = batch mean length**：`baseline="batch_mean"` 分支，`train_two_gate_tsp.py:617-622`。  
3) **reward 用 score（长度）**：明确写 `reward = -length`：`train_two_gate_tsp.py:613-614`。  
4) **loss 同时包含两次决策**：`actor_loss` 同时包含 `logp1` 与 `logp2`：`train_two_gate_tsp.py:652`。  
5) **固定迭代次数**：iterator 的步数由 config 决定（如 `rrc_steps/dact_steps/lih_steps`），构造 iterator 时就固定了（见 `train_two_gate_tsp.py:73-75`、以及 iterator wrapper 的 `max_steps` 参数）。

---

## 4. 问题回答：训练的是什么？是不是 MLP？

### 4.1 原论文训练的是什么？

**结论（按代码真实参数更新集合）**：原论文训练的是 selector 模型本体（encoder + head），不是训练 solver zoo。

证据链：

1) 模型是 `Selection_model` 或 `Naive_classifier`：`neural-solver-selection/run.py:104-109`  
2) 优化器绑定 `self.model.parameters()`：`neural-solver-selection/trainer.py:43`  
3) `train_one_epoch` 对 `y_pred = self.model(...)` 反向传播：`neural-solver-selection/trainer.py:127-139`

**是不是 MLP？**

- 如果 `manual_feature=True`：模型是 `Naive_classifier`（基本就是 MLP），可以说“训练的是 MLP”。  
  - 代码：`neural-solver-selection/model.py:96-114`
- 默认（`manual_feature=False`）：模型是 `Selection_model = Transformer-style encoder + MLP head`：
  - encoder：`Naive_Encoder/Encoder_h`（注意力网络）  
    - 代码：`neural-solver-selection/model.py:116`、`neural-solver-selection/model.py:147`、`neural-solver-selection/model.py:271`
  - head：MLP 输出 solver logits  
    - 代码：`neural-solver-selection/model.py:34-37`

所以严格说：**论文默认不是只训练 MLP，而是训练 “注意力 encoder + MLP head”。**

### 4.2 你的实现训练的是什么？

**结论（按代码真实参数更新集合）**：你训练的是 `TwoGateAC`（encoder + gate1/gate2 + critic），solver 组件来自 `EasyNCO/` 并被冻结。

证据链：

1) solver 冻结：`1two_gate/train_two_gate_tsp.py:252`（`policy.requires_grad_(False)`，各 initializer/iterator 里同理）  
2) solver 调用在 no-grad 模式：`1two_gate/solver_zoo.py:85-98`（`TorchNoGrad = torch.inference_mode()`）  
3) optimizer 绑定 `policy.parameters()`：`1two_gate/train_two_gate_tsp.py:497`  
4) 训练循环对 `loss.backward()`：`1two_gate/train_two_gate_tsp.py:655-657`

**是不是 MLP？**

- gate1/gate2 是 MLP：`1two_gate/gates.py:14-26`
- critic 是 MLP：`1two_gate/gates.py:29-45`
- 但你同样在训练 encoder（Transformer-style 注意力网络）：`1two_gate/paper_encoder.py`

所以严格说：**你也不是“只训练 MLP”，而是训练 “注意力 encoder + MLP gates (+ MLP critic)”。**

---

## 5. 问题回答：padding/mask 具体怎么做？两边是否都有？

### 5.1 原论文：为什么必须 padding/mask？

原因：训练集规模混合（例如 `data_config.yml` 里 50~500），同 batch 样本 N 不一致，所以要 padding。

实现证据：

- padding + mask 构造：`neural-solver-selection/dataset.py:115-123`
- mask 用于 attention：`neural-solver-selection/model.py:324-359`（`score_scaled += rank2_ninf_mask[...]`）
- mask 用于 pooling：`neural-solver-selection/model.py:139-145`（`emb_mask = where(mask==-inf,0,1)`）
- mask 用于层次化下采样：`neural-solver-selection/model.py:249-268`（`scores = scores + mask` + topk）

### 5.2 你的实现：是否也有 padding/mask？

**mask：有。padding：目前默认没有。**

证据：

- mask 的默认生成：`1two_gate/features.py:71-75`（`_tsp_default_mask` 返回全 0 `(B,N)`）
- mask 被传入 encoder：`1two_gate/features.py:94`（`graph_emb = encoder(coords, mask)`）
- encoder 内部用 mask 做 attention：`1two_gate/paper_encoder.py:206-227`（与论文同样把 mask 加到 attention logits）

为什么“默认没有 padding”？

- 你的训练脚本默认 `problem_size` 固定：`1two_gate/train_two_gate_tsp.py:41-48`（`TrainConfig.problem_size`）  
  每 step 生成的 `coords` 形状固定 `(B, N, 2)`：`1two_gate/train_two_gate_tsp.py:595`。
- 所以同一个 batch 内 N 不变，不需要 padding。

如果未来你要像论文一样“同 batch 混合不同 N”，需要额外实现：

- 一个 `collate_fn` 风格的 padding 管线（可以参考论文 `neural-solver-selection/dataset.py:105-128` 的思路）
- 并把 `scales=lengths` 作为显式输入特征（你目前 `scale=N` 是固定的）

---

## 6. 代码索引速查（建议收藏）

### 6.1 原论文（`neural-solver-selection/`）

- 入口与整体流程：`neural-solver-selection/run.py:22`
- 数据+离线标签读取：`neural-solver-selection/utils.py:48`
- Dataset 与 padding/mask：`neural-solver-selection/dataset.py:105`
- 选择器模型：`neural-solver-selection/model.py:7`
- 注意力 encoder：
  - `Naive_Encoder`：`neural-solver-selection/model.py:116`
  - `Encoder_h`：`neural-solver-selection/model.py:147`
  - `EncoderLayer`：`neural-solver-selection/model.py:271`
  - `multi_head_attention`（mask 关键逻辑）：`neural-solver-selection/model.py:324`
- 训练循环：`neural-solver-selection/trainer.py:106`
- 离线策略评估（top-k/rejection/top-p）：`neural-solver-selection/trainer.py:156`
- RankingLoss：`neural-solver-selection/loss.py:7`

### 6.2 你的实现（`1two_gate/`）

- 入口与训练主循环：`1two_gate/train_two_gate_tsp.py:501`（`train`）、`1two_gate/train_two_gate_tsp.py:594`（主循环）
- baseline/adv/actor-critic 更新：`1two_gate/train_two_gate_tsp.py:617-657`
- 构建 gates/critic：`1two_gate/train_two_gate_tsp.py:463`
- solver 冻结：`1two_gate/train_two_gate_tsp.py:252`（各 solver 分支类似）
- EasyNCO alias/stub：`1two_gate/easynco_bootstrap.py:9`
- TSP 数据分布生成：`1two_gate/tsp_data.py:126`
- 特征提取（含 mask 与 gate2 条件特征）：`1two_gate/features.py:71`、`1two_gate/features.py:77`、`1two_gate/features.py:154`
- 论文同款 encoder（本地重写）：`1two_gate/paper_encoder.py:10`、`1two_gate/paper_encoder.py:48`、`1two_gate/paper_encoder.py:163`
- gate/critic MLP：`1two_gate/gates.py:14`、`1two_gate/gates.py:29`
- 分流调用 initializer/iterator：`1two_gate/solver_zoo.py:42`、`1two_gate/solver_zoo.py:63`
