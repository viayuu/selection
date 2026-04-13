# AGENTS.md

本文件用于给后续自动化编码助手提供“项目记忆”和协作约定；适用范围为仓库根目录及其子目录（除非有更深层的 AGENTS.md 覆盖）。

## 0. 目录与文件定位（避免混淆）
- 作者论文对应的实现代码位于 `neural-solver-selection/` 子目录；下文提到的 `run.py`/`dataset.py`/`trainer.py`/`datasets/*` 等若未写前缀，默认指该子目录中的同名路径。
- `paper.md` 是论文正文；`CLAUDE.md` 提供对论文与代码的解释线索。
- `final/` 是你自己的独立 NCO 平台项目，**与作者论文无关**；并且明确排除 `final/reld-nco/` 不考虑。
- `my/` 是你在做的双 gate + AC 强化学习原型（尽量只调用 `final/`，不改 `final/`）；`my.md`/`implementation_verification_report.md`/`ans.md` 分别记录思路、复盘与问答。

## 1. 项目目标（一句话）

这是 ICML 2025 论文《Neural Solver Selection for Combinatorial Optimization》的代码：训练一个“选择器”模型，在实例级别从一个固定的神经求解器集合（solver zoo）中挑选最合适的求解器/组合策略，以获得更好的平均最优性差距（gap）与较低额外开销。

## 2. 关键认知（不要误解项目在做什么）

- 该仓库默认**不包含/不训练** Omni、DIFUSCO、LEHD 等具体神经求解器；它们的运行结果被离线固化为监督标签（`datasets/*/raw_label.pkl`）。
- 训练/测试时主要学习：`instance -> solver scores`；评估时用这些 score 离线“模拟” top-k / rejection / top-p 策略带来的 gap/time。
- `paper.md` 是论文正文（已转 Markdown），`CLAUDE.md` 也提供了对论文与代码的解释线索。
- 用户的后续研究设想在 `my.md`：将“选择对象”从“完整求解器”改为“求解 pipeline 的两阶段组件”（初始化/迭代），并考虑用强化学习替代纯监督标注的组合爆炸。

## 3. 代码结构速览（paper -> code）

**入口与配置**
- `run.py`：解析参数/加载配置；训练与测试共用。
- `config_TSP.yml` / `config_CVRP.yml`：核心超参数（loss、是否手工特征、是否层次化 pooling 等）。

**数据与标签（监督来源）**
- `datasets/*/dataset.pkl`：实例集合（TSP: 坐标；CVRP: depot+customer+需求）。
- `datasets/*/raw_label.pkl`：每实例离线标签：`cost/time/gap` 向量与 `ind=argmin(cost)`。
- `utils.py:prepare_dataset()`：统一读入 train/val/test/LIB，拼装成 `SelectionDataset` 使用的结构。
- `datasets/process_raw_label.py`：将各求解器的 `result_*.txt` 汇总成 `raw_label.pkl`（并可计算 gap）。
- `datasets/generate_data.py` + `datasets/data_utils.py`：合成数据生成（高斯混合等；CVRP capacity/demand 模式）。

**特征提取（Feature extraction）**
- 手工特征：`dataset.py:manual_features()`（论文对比的 manual features 路线）。
- 图注意力编码器：`model.py:Naive_Encoder`（全连接自注意力 + mask mean pooling）。
- 层次化图编码器：`model.py:Encoder_h/Encoder_block_h`（downsampling pooling，提升 OOD 泛化）。

**选择模型（Selection model）**
- `model.py:Selection_model`：编码器输出 + 规模特征 `scales` 拼接 -> MLP 输出每 solver 的分数。
- `train_params.manual_feature=True` 时走 `model.py:Naive_classifier`（仅用手工特征+规模）。

**训练损失（Selection loss）**
- 分类（CE/NLL）：`trainer.py` 中 `torch.nn.NLLLoss()`。
- 排序（Ranking）：`loss.py:RankingLoss`（利用 cost 向量学习 solver 的相对排序；通常更鲁棒）。

**选择策略评估（Selection strategies）**
- `trainer.py:test()`：基于预测 score 离线计算：
  - top-k：取 top-k solver 子集，gap 取子集中最小，time 求和。
  - rejection-based：按 `max softmax` 置信度分位拒绝一部分实例，对低置信度用 top-k，其余用 top-1。
  - top-p：按分数累积达到阈值 p 的最小 solver 集合。

## 4. 重要运行方式（常用命令）

- 训练（示例，TSP，ranking loss）：`python run.py --config_name config_TSP.yml --loss rank --seed 2024 --gpu_id 0`
- 测试预训练：`python run.py --gpu_id 0 --load config_TSP.yml_rank_2024 --test_file TSPLIB --exp_name config_TSP.yml_rank_TSPLIB`
- 批量脚本：`run_experiment.sh`、`run_test.sh`（由 `experiments/*.py` 生成）。

注意：仓库里 `train_logs/` 提供部分预训练 checkpoint；在缺少完整依赖（如 torch）时无法直接运行，需要先确保环境就绪。

## 5. EasyNCO 平台（`final/`，排除 `final/reld-nco/`）

该目录是你自己的一个 NCO 平台项目（与作者论文无关）：集成多种组合优化问题（TSP/CVRP/ATSP/…）与多种求解方法（AM/POMO/LEHD/DIFUSCO/DeepACO/…），并提供统一的训练/评估入口与配置系统。
66

**入口与配置**
- `final/train.py`：训练入口（Hydra + PyTorch Lightning），根据配置实例化 `env + model(policy) + initialization + iteration + module + trainer` 并调用 `trainer.fit()`。
- `final/eval.py`：评估入口，加载 `test_loader` 指定的 checkpoint，并调用 `trainer.test()`。
- `final/configs.yaml`：全局 Hydra 默认配置（`mode/model/problem/scale/...`、数据路径、log/ckpt 输出目录等）。
- `final/settings/*.yaml`：方法级配置模板（定义 env、model、initialization、iteration、module、trainer、test_loader）。典型如：
  - `final/settings/pomo_settings.yaml`
  - `final/settings/reld_settings.yaml`
  - `final/settings/flexible_composition.yaml`（支持把不同 solver 的 initialization/iteration 进行自由组合）。

**核心抽象与代码组织**
- 数据（`final/data/`）
  - `final/data/main.py:generate_data()`：按 `problem_name` 返回对应的 DataLoader 生成器。
  - 典型数据生成器：`final/data/TSPGenerator.py`、`final/data/CVRPGenerator.py`，既支持随机分布生成，也支持从 `.pt/.pkl/.txt` 与 TSPLIB/CVRPLIB 文件加载。
  - 数据增强：`final/data/data_utils.py`（如 `augment_pomo/rotate/reflect/...`）。
- 环境（`final/neural_solvers/envs/`）
  - 每个问题一个 `EnvBase` 子类（torchrl）：如 `final/neural_solvers/envs/TSPEnv.py`、`final/neural_solvers/envs/CVRPEnv.py`。
  - 通过 `TensorDict` 传递状态（mask、已选节点、容量/载重等），终止时返回 `reward = -cost`。
  - 支持 augmentation：在 `load_problems()` 中按 `aug_type/aug_factor` 扩展 batch。
- 求解方法（`final/neural_solvers/methods/`）
  - 每个方法通常包含：
    - `Policy`（`nn.Module`）：给定 `TensorDict` 输出 `action/prob`（如 `POMOPolicy`、`AttentionModelPolicy`、`ReLDPolicy` 等）。
    - `Initialization`：生成初始解（多为自回归 rollout；如 `POMOInitialization`）。
    - `Iteration`：可选的迭代改进步骤（一些方法有，如 LEHD/DeepACO/DACT/…）。
  - `final/neural_solvers/methods/__init__.py` 汇总导出各方法组件，便于 Hydra 通过 `_target_` 实例化。
- Pipeline 分离（`final/neural_solvers/pipeline/`）
  - `final/neural_solvers/pipeline/initialization.py`：定义 `Initialization/ARInitialization` 接口（rollout 并记录 `reward/likelihood`）。
  - `final/neural_solvers/pipeline/iteration.py`：定义 `Iteration` 接口与 `NoIteration` 默认实现。
- 训练阶段（`final/phases/`）
  - `final/phases/rl/ar_reinforce.py:ARREINFORCELightning`：自回归方法的 REINFORCE 训练/评估主模块：
    - `training_step()`：`initialization.run(...) -> iteration.run(...) -> calculate_loss()`。
    - `calculate_loss()`：`-(log_prob * (reward - baseline)).mean()`（baseline 来自 `final/phases/rl/baselines.py`）。
  - `final/phases/rl/baselines.py`：Shared/Rollout/Warmup/Critic 等 baseline 实现（POMO 常用 SharedBaseline）。
- 工具（`final/utils/utils.py`）
  - 设备/DDP 配置、日志打印、checkpoint 命名。
  - `load_model()`：适配不同来源 checkpoint 的字段/命名差异，加载到平台 policy。

**其它组件**
- `final/exact_solvers/`：统一封装 LKH/EAX/HGS/OR-Tools/Gurobi/CPLEX/PyVRP 等，用于求解或产生参考解（见 `final/exact_solvers/README.md`、`final/exact_solvers/use_main.py`）。
- `final/pretrained/`：存放从原作者仓库下载并转换格式后的预训练模型（见 `final/pretrained/README.md`）。
- `final/GUI/`：Tkinter + paramiko 的远程运行/可视化界面（方法与任务列表、参数面板等）。
- `final/co_bench/`：CO-Bench 相关的评测封装（大量问题的 template/evaluation）。

## 6. 用户拟开展的新论文工作（按用户说明整理为“可实现”的方案）

### 6.1 整体前提与原论文的局限
- 整体前提：一个模型不可能适配所有的实例，模型肯定具有偏向性。
- 原论文做法：只有一个 gate；把代表性模型组成 solver zoo，用 gate 将实例分配给某个模型（把“一对多”变成“一对一”）。
- 原论文训练方式：监督学习；对每个实例先用每个 model 求解得到结果并排序，作为训练信号。
- 原论文关键假设：一个模型的 pipeline 适用于所有数据；但该假设不成立，因为求解 pipeline 可分为“初始化”和“迭代”两阶段，两个阶段对实例的偏好可能不同。

### 6.2 新方案：双 gate 解耦 pipeline（多分裂问题）
- 目标：细化整个 pipeline，用两个 gate 分别选择两个阶段的组件，并学习/训练这两个 gate。
- Gate 1（初始化选择）：
  - 输入：原始问题实例 data。
  - 动作：从初始化器集合中选一个（POMO/RELD/LEHD/指针网络 等）。
  - 输出：初始化解 `sol0`。
- Gate 2（迭代选择）：
  - 输入：`data + sol0`（Gate 1 的结果与 data 共同作为 Gate 2 的条件）。
  - 动作：从迭代器集合中选一个（传统 2-opt/大规模搜索/RRC 等）。
  - 执行约束：先不考虑时间；固定迭代次数，只选择迭代方式。
- 输出：最终解 `sol1`。
- 为什么不用监督学习：初始化器×迭代器存在 `n*m` 种组合可能性，监督标注（全组合求解并打标签）代价太大。

### 6.3 强化学习训练（AC + batch mean baseline；reward 用解的长度）
- 强化学习架构：Actor-Critic（AC）。
- reward（以最小化长度为目标的写法）：用最终解的 score（求解出来的长度）构造奖励信号（实现时通常用 `reward = -length`）。
- baseline：用 batch 内实例路径长度的均值（或等价的 batch mean reward）作为 baseline。
- loss：计算时同时把上层（初始化 Gate 1）与下层（迭代 Gate 2）都考虑进 loss function（同一个最终 reward 归因到两次决策）。

**训练批次的一次前向与损失（伪代码表达意图）**
```text
for batch in instances:
  s0 = state_from(data)
  a1, logp1 = gate1_actor.sample(s0)         # 选初始化器
  sol0 = initializer[a1](data)

  s1 = state_from(data, sol0)
  a2, logp2 = gate2_actor.sample(s1)         # 选迭代器（固定迭代次数）
  sol1 = improver[a2](data, sol0)

  length = score(sol1)                        # 路径长度
  reward = -length

baseline = mean(reward over batch)            # 或 mean(length)
adv = reward - baseline
loss = -mean((logp1 + logp2) * adv)           # 同时更新两层 gate
```

### 6.4 当前落地代码（原型，先跑通框架）

- 代码放在 `my/`：两层 gate + AC（baseline= batch mean），先支持 TSP。
- 入口：`my/train_two_gate_tsp.py`（随机生成 TSP 坐标；训练 gate，不训练 solver）。
- 调用 `final/`（不改 `final/`）：
  - Gate1 initializer：`POMOPolicy` / `AttentionModelPolicy` / `LEHDPolicy`（通过 rollout 得到 `sol0`）。
  - Gate2 iteration：`LEHDPolicy` 驱动的 RRC 风格 destroy/repair（固定 `rrc_steps`）。
- 关键约定：
  - `reward = -length`（length 由 `TSPEnv._get_travel_distance` 计算）。
  - loss 同时包含两次决策：`(logp1 + logp2) * (reward - baseline)`。
  - 为了让 `final/` 的 `EasyNCO.*` 绝对导入可用且避免导入整个 solver zoo：`my/easynco_bootstrap.py` 在运行时把 `EasyNCO`（以及兼容 `neural_solvers.*`）映射到 `final/`，并对 `EasyNCO.data/methods/envs/backbones` 做轻量 stub（不改 `final/`）。
  - Gate 特征提取默认使用 `my/paper_encoder.py` 中内置的论文同款实例编码器（`Encoder_h`/`Naive_Encoder`），并额外把 Gate2 条件信息做成 `data + sol0`（`sol0` 的长度与简单 tour 统计量）。
  - baseline 支持两种模式：
    - `batch_mean`：严格按用户说明，用 batch 内 `mean(length)` 作为 baseline（实现里换算到 reward 空间）。
    - `critic`：可选 actor-critic（value 网络预测 baseline）。
    - `critic_batch_mean`：同时使用 `mean(length)` 常数 baseline + 可学习 `V(s)` 预测残差（兼顾两者）。

## 7. 工作约定

- `final/` 是一个独立的 NCO 平台目录；用户明确要求阅读其内容时再进入。
- 默认排除 `final/reld-nco/`（用户明确要求不阅读该子目录）。
- 改动尽量聚焦：先保证复现实验与数据流清晰，再做大规模重构。
- 所有新增实验/配置/脚本要写清楚输入输出与数据格式，避免隐式依赖（尤其是离线标签与真实求解的边界）。
