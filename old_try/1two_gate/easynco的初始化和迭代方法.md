# EasyNCO 平台：初始化方法 & 迭代方法（全面清单）

> 统计来源：`EasyNCO/` 目录下的实现代码（主要扫描 `EasyNCO/neural_solvers/pipeline/` 与 `EasyNCO/neural_solvers/methods/`）。  
> 排除范围：`9Reld/`（按你的约定不考虑）。

---

## 0. 平台的“Initialization / Iteration”抽象（先统一概念）

EasyNCO 把“求解 pipeline”抽象成两段：

1. **Initialization（初始化）**：给定实例 `batch`，产生一个初始解（以及训练需要的 likelihood / reward 等信息）。  
2. **Iteration（迭代改进）**：在已有解（通常在 `td` / `initialization_out` 里）基础上做改进；也可以是 No-Op（什么都不做）。

对应的接口定义在：
- `EasyNCO/neural_solvers/pipeline/initialization.py`
  - `Initialization`：初始化抽象基类
  - `ARInitialization`：通用的“自回归 rollout 初始化”（适用于很多只有 `policy.py` 的方法）
- `EasyNCO/neural_solvers/pipeline/iteration.py`
  - `Iteration`：迭代抽象基类
  - `NoIteration`：通用的“无迭代”（直接返回初始化结果）

---

## 1. 方法目录总览（methods 文件夹里到底有什么）

`EasyNCO/neural_solvers/methods/` 下共有 **28 个方法目录**（按目录名计）：

> 下面的 `policy/init/iter` 表示该目录下是否存在 `policy.py / initialization.py / iteration.py` 文件。

| method 目录 | policy | init | iter |
|---|---:|---:|---:|
| `am` | ✅ | ❌ | ❌ |
| `dact` | ✅ | ✅ | ✅ |
| `deepaco` | ❌ | ✅ | ✅ |
| `difusco` | ✅ | ✅ | ✅ |
| `dpn` | ✅ | ✅ | ❌ |
| `drl_hgnn` | ✅ | ✅ | ❌ |
| `elg` | ✅ | ✅ | ❌ |
| `glop` | ✅ | ✅ | ✅ |
| `htsp` | ✅ | ✅ | ❌ |
| `icam` | ✅ | ✅ | ❌ |
| `insertion` | ❌ | ✅ | ❌ |
| `invit` | ✅ | ❌ | ❌ |
| `l2s` | ✅ | ✅ | ✅ |
| `lehd` | ✅ | ✅ | ✅ |
| `lih` | ✅ | ✅ | ✅ |
| `matnet` | ✅ | ✅ | ❌ |
| `matpoenet` | ✅ | ✅ | ❌ |
| `mtpomo` | ✅ | ❌ | ❌ |
| `mvmoe` | ✅ | ✅ | ❌ |
| `nlns` | ✅ | ✅ | ✅ |
| `omni` | ✅ | ✅ | ❌ |
| `pointerformer` | ✅ | ❌ | ❌ |
| `pomo` | ✅ | ✅ | ❌ |
| `psl` | ✅ | ✅ | ❌ |
| `ptr_nets` | ✅ | ❌ | ❌ |
| `reld` | ✅ | ✅ | ❌ |
| `t2t` | ✅ | ✅ | ✅ |
| `udc` | ✅ | ✅ | ✅ |

补充说明（重要）：
- **没有 `initialization.py` 并不代表不能做初始化**：很多方法用的是通用的 `ARInitialization`（见上面的 pipeline 定义），只要 `policy` 实现了平台约定的接口即可。
- `deepaco` 的“model”不在 `methods/deepaco/policy.py` 里实现：它的 settings 默认用的是 `EasyNCO/neural_solvers/backbones/GNN/net.py` 里的 GNN 模型（见 `EasyNCO/settings/deepaco_settings.yaml`）。

---

## 2. 初始化方法（Initializer）清单

### 2.1 通用初始化器（不绑定具体 method）

| 初始化器 | 文件 | 说明 |
|---|---|---|
| `ARInitialization` | `EasyNCO/neural_solvers/pipeline/initialization.py` | 通用自回归 rollout 初始化：`env.load_problems(...)` + 反复 `policy(state_td) → env.step(...)`，得到 `reward/likelihood` |

### 2.2 各 method 自带的初始化器（`methods/*/initialization.py`）

> 下面是“类名级别”的清单：哪些目录提供了独立的 Initialization 类、它们继承自什么基类。

| method | 初始化类（类名） | 基类 | 文件 |
|---|---|---|---|
| `pomo` | `POMOInitialization` | `ARInitialization` | `EasyNCO/neural_solvers/methods/pomo/initialization.py` |
| `elg` | `ELGInitialization` | `ARInitialization` | `EasyNCO/neural_solvers/methods/elg/initialization.py` |
| `dpn` | `DPNInitialization` | `ARInitialization` | `EasyNCO/neural_solvers/methods/dpn/initialization.py` |
| `psl` | `PSLInitialization` | `ARInitialization` | `EasyNCO/neural_solvers/methods/psl/initialization.py` |
| `reld` | `ReLDInitialization` | `ARInitialization` | `EasyNCO/neural_solvers/methods/reld/initialization.py` |
| `matnet` | `MatNetInitialization` | `ARInitialization` | `EasyNCO/neural_solvers/methods/matnet/initialization.py` |
| `matpoenet` | `MatPOENetInitialization` | `ARInitialization` | `EasyNCO/neural_solvers/methods/matpoenet/initialization.py` |
| `icam` | `ICAMInitialization` | `ARInitialization` | `EasyNCO/neural_solvers/methods/icam/initialization.py` |
| `lehd` | `LEHDInitialization` | `Initialization` | `EasyNCO/neural_solvers/methods/lehd/initialization.py` |
| `lih` | `LIHInitialization` | `Initialization` | `EasyNCO/neural_solvers/methods/lih/initialization.py` |
| `dact` | `DACTInitialization` | `Initialization` | `EasyNCO/neural_solvers/methods/dact/initialization.py` |
| `nlns` | `NLNSInitialization` | `Initialization` | `EasyNCO/neural_solvers/methods/nlns/initialization.py` |
| `difusco` | `DIFUSCOInitialization` | `Initialization` | `EasyNCO/neural_solvers/methods/difusco/initialization.py` |
| `t2t` | `T2TInitialization` | `Initialization` | `EasyNCO/neural_solvers/methods/t2t/initialization.py` |
| `deepaco` | `DeepACOInitialization` | `Initialization` | `EasyNCO/neural_solvers/methods/deepaco/initialization.py` |
| `glop` | `GLOPInitialization` | `Initialization` | `EasyNCO/neural_solvers/methods/glop/initialization.py` |
| `udc` | `UDCInitialization` | `Initialization` | `EasyNCO/neural_solvers/methods/udc/initialization.py` |
| `omni` | `OMNIInitialization` | `Initialization` | `EasyNCO/neural_solvers/methods/omni/initialization.py` |
| `htsp` | `HTSPInitialization` | `Initialization` | `EasyNCO/neural_solvers/methods/htsp/initialization.py` |
| `drl_hgnn` | `DRL_HGNNInitialization` | `Initialization` | `EasyNCO/neural_solvers/methods/drl_hgnn/initialization.py` |
| `insertion` | `INSERTIONInitialization` | `Initialization` | `EasyNCO/neural_solvers/methods/insertion/initialization.py` |
| `mvmoe` | `MVRPInitialization` | `Initialization` | `EasyNCO/neural_solvers/methods/mvmoe/initialization.py` |
| `l2s` | `L2SInitialization` | `Initialization` | `EasyNCO/neural_solvers/methods/l2s/initialization.py` |

补充说明（权重/训练相关，按代码可观察到的事实）：
- **绝大多数初始化器**会调用某个神经 `policy`（或背后的 `model`），因此通常需要加载权重（预训练或训练得到）。  
  - 典型例子：`POMOInitialization/ELGInitialization/ReLDInitialization/...` 都需要 `policy` 能正常 forward。
- **`INSERTIONInitialization`** 是启发式插入法（在 `initialization.py` 注释里明确写了多种插入策略），本身不依赖神经网络权重。
- 一些方法在代码里对 train phase 还没实现（例如 `DRL_HGNNInitialization` 直接 `NotImplementedError`），但这不影响“平台里存在该初始化器”这一事实。

### 2.3 仅有 `policy.py`、默认用 `ARInitialization` 的方法（无独立 `initialization.py`）

这些方法目录下没有 `initialization.py`，平台通常用 `ARInitialization` 做初始化（可在对应 settings 中看到）：

| method | policy 文件 | settings 默认 initialization |
|---|---|---|
| `am` | `EasyNCO/neural_solvers/methods/am/policy.py` | 通常为 `EasyNCO.neural_solvers.pipeline.ARInitialization` |
| `invit` | `EasyNCO/neural_solvers/methods/invit/policy.py` | `EasyNCO/settings/invit_settings.yaml` 使用 `ARInitialization` |
| `mtpomo` | `EasyNCO/neural_solvers/methods/mtpomo/policy.py` | `EasyNCO/settings/mtpomo_settings.yaml`（可查看其 initialization 配置） |
| `pointerformer` | `EasyNCO/neural_solvers/methods/pointerformer/policy.py` | `EasyNCO/settings/pointerformer_settings.yaml`（可查看其 initialization 配置） |
| `ptr_nets` | `EasyNCO/neural_solvers/methods/ptr_nets/policy.py` |（如需使用，通常也可用 `ARInitialization`，取决于 policy 是否满足接口） |

---

## 3. 迭代方法（Iterator / Improvement）清单

### 3.1 通用迭代器（不绑定具体 method）

| 迭代器 | 文件 | 说明 |
|---|---|---|
| `NoIteration` | `EasyNCO/neural_solvers/pipeline/iteration.py` | No-Op：直接返回初始化结果，不做改进 |

### 3.2 各 method 自带的迭代器（`methods/*/iteration.py`）

> 下表列的是“真正作为 pipeline Iteration 使用的类”（继承自 `Iteration`）。一些文件里还有工具类（`Iteration_tool` 等）这里不作为“可选迭代器动作”单独列出。

| method | 迭代类（类名） | 文件 | 代码层面的“迭代本质”（尽量按源码表述） |
|---|---|---|---|
| `lehd` | `LEHDIteration` | `EasyNCO/neural_solvers/methods/lehd/iteration.py` | 调用 `TSPLEHDIteration/CVRPLEHDIteration`；其中 TSP 是 **RRC（Random Re-Construction）**：随机破坏子路径→用 policy 重建→按改进接受 |
| `dact` | `DACTIteration` | `EasyNCO/neural_solvers/methods/dact/iteration.py` | 基于 `Iteration_tool` 做 **2-opt 类交换**；每步用 policy 产生交换对，再执行 2-opt 更新 |
| `lih` | `LIHIteration` | `EasyNCO/neural_solvers/methods/lih/iteration.py` | 同样是 **2-opt 类局部搜索**，区别在于实现细节与约束处理（尤其 CVRP 里会检查可行性） |
| `difusco` | `DIFUSCOIteration` | `EasyNCO/neural_solvers/methods/difusco/iteration.py` | 直接调用扩散模型 `model.iteration(...)`；注释写明可用 **2-opt/MCTS** 做改进 |
| `t2t` | `T2TIteration` | `EasyNCO/neural_solvers/methods/t2t/iteration.py` | 与 DIFUSCO 类似，调用 `model.iteration(...)`；注释也写明 **2-opt/MCTS** |
| `nlns` | `NLNSIteration` | `EasyNCO/neural_solvers/methods/nlns/iteration.py` | **NLNS（Neural Large Neighborhood Search）**：destroy & repair 的大邻域搜索流程（含时间/重启等控制逻辑） |
| `glop` | `GLOPIteration` | `EasyNCO/neural_solvers/methods/glop/iteration.py` | 分解与子问题求解（`*_decompose_and_solve`）的迭代流程（面向子问题组合/合并） |
| `udc` | `UDCIteration` | `EasyNCO/neural_solvers/methods/udc/iteration.py` | 训练/评估都有实现；包含“子问题采样/滚动/合并”的迭代（内部还有反复 rollout / merge） |
| `deepaco` | `DeepACOIteration` | `EasyNCO/neural_solvers/methods/deepaco/iteration.py` | 基于 `load_policy(...)` 构造 ACO，在 `max_steps` 内反复 search，输出平均结果 |
| `l2s` | `L2SIteration` | `EasyNCO/neural_solvers/methods/l2s/iteration.py` | 针对调度问题的局部搜索迭代（图结构 + tabu 等；见 `Iteration_tool`） |

---

## 4. “能不能用作我的两阶段 gate 动作空间？”（简短结论）

如果你要把平台组件接入你现在的两阶段 gate（Gate1 选初始化 / Gate2 选迭代）：

- **Gate1（初始化动作空间）**：  
  - 基本上“所有能产生解”的方法都可以当 initializer（包括用 `ARInitialization` 的纯 policy 方法、以及有独立 `*Initialization` 的方法）。
  - 特例：`INSERTIONInitialization` 不需要权重，适合当“传统启发式初始化”基线。

- **Gate2（迭代动作空间）**：  
  - 平台里“显式作为 Iteration 组件”的主要就 10 个（见上表 3.2）+ 一个 `NoIteration`。
  - 其中有的强依赖特定 policy/数据结构（例如 `LIHIteration` 在代码里对 CVRP 有大量约束处理；`DIFUSCOIteration/T2TIteration` 依赖扩散模型的 `model.iteration`；`NLNSIteration` 依赖其工具类流程与 policy_dict 结构）。

---

## 5. 你后续如果要自动化抽取（可选）

如果你希望以后每次更新平台后自动生成这份清单，可以基于以下“稳健的判据”：

1. 以 `EasyNCO/neural_solvers/methods/<method>/` 为单位扫描：
   - 是否存在 `initialization.py`（有则解析 `*Initialization` 类）
   - 是否存在 `iteration.py`（有则解析继承自 `Iteration` 的类）
2. 再补上 pipeline 的通用 `ARInitialization / NoIteration`。

---

## 6. 仅整理：能解单目标 TSP 的初始化/迭代方法子集

> 这里的“能解 TSP”指：在 `env_name='tsp'`（或对应的 `TSPEnv` 数据格式）下，代码路径明确支持并能产生 TSP tour（或等价表示），最终可得到 tour length。  
> 注意：有些方法的 pipeline 是“初始化只产中间结构，真正的解在迭代阶段产出”（例如 DIFUSCO/T2T/DeepACO），因此它们在两阶段 gate 框架里**不一定**能直接当作“可交换的 initializer/iterator”，但它们本身属于“平台能解 TSP”的方法。

### 6.1 TSP 可用初始化（Initializer / Initialization）

#### 6.1.1 通用初始化器（跨 method）

| 初始化器 | 适用 TSP 吗 | 常见用法（TSP） | 文件 |
|---|---|---|---|
| `ARInitialization` | ✅ | 适用于 **AM / INVIT / Pointerformer / (以及其它满足接口的 policy)**：自回归 rollout 直接生成 tour | `EasyNCO/neural_solvers/pipeline/initialization.py` |

#### 6.1.2 method 自带初始化器（明确支持 TSP）

| method | 初始化类 | 说明（面向 TSP） | 文件 |
|---|---|---|---|
| `pomo` | `POMOInitialization` | 自回归 rollout（POMO 变体）直接生成 tour | `EasyNCO/neural_solvers/methods/pomo/initialization.py` |
| `lehd` | `LEHDInitialization` | rollout 生成 tour（LEHD policy） | `EasyNCO/neural_solvers/methods/lehd/initialization.py` |
| `elg` | `ELGInitialization` | rollout 生成 tour；TSP/CVRP 都有分支（policy 里显式区分） | `EasyNCO/neural_solvers/methods/elg/initialization.py` |
| `icam` | `ICAMInitialization` | 本质是 `ARInitialization` 的封装（`policy.model` 走 TSP 分支） | `EasyNCO/neural_solvers/methods/icam/initialization.py` |
| `omni` | `OMNIInitialization` | meta-learning 初始化/训练入口；代码里显式支持 `env.env_name=='tsp'` | `EasyNCO/neural_solvers/methods/omni/initialization.py` |
| `htsp` | `HTSPInitialization` | HTSP 的测试入口（`HTSP_test_step` 输出 TSP length） | `EasyNCO/neural_solvers/methods/htsp/initialization.py` |
| `insertion` | `INSERTIONInitialization` | 传统插入启发式（支持 TSP/CVRP；不依赖神经权重） | `EasyNCO/neural_solvers/methods/insertion/initialization.py` |
| `dact` | `DACTInitialization` | 为 2-opt 类迭代准备初始解结构；实现里对 `env.env_name=='tsp'` 有分支 | `EasyNCO/neural_solvers/methods/dact/initialization.py` |
| `lih` | `LIHInitialization` | 为 LIH 的 2-opt 迭代准备初始解结构；对 `problems_type=='tsp'` 有分支 | `EasyNCO/neural_solvers/methods/lih/initialization.py` |
| `udc` | `UDCInitialization` | `_run_eval_tsp` 会显式生成 permutation `solution`（TSP） | `EasyNCO/neural_solvers/methods/udc/initialization.py` |
| `glop` | `GLOPInitialization` | `problem=='tsp'` 分支用随机插入/多宽度方案生成候选解 | `EasyNCO/neural_solvers/methods/glop/initialization.py` |
| `difusco` | `DIFUSCOInitialization` | 初始化阶段产出 DIFUSCO 的中间结构（非通用 tour）；后续由 `DIFUSCOIteration` 产最终指标 | `EasyNCO/neural_solvers/methods/difusco/initialization.py` |
| `t2t` | `T2TInitialization` | 同 DIFUSCO：初始化产中间结构，迭代阶段产结果 | `EasyNCO/neural_solvers/methods/t2t/initialization.py` |
| `deepaco` | `DeepACOInitialization` | 初始化阶段仅传递 batch；真正 search 在 `DeepACOIteration` | `EasyNCO/neural_solvers/methods/deepaco/initialization.py` |

#### 6.1.3 “只有 policy.py、但能解 TSP”的方法（通常配 `ARInitialization`）

| method | policy（支持 TSP 的证据） | 文件 |
|---|---|---|
| `am` | `AttentionModelPolicy(env_name='tsp' 默认)` | `EasyNCO/neural_solvers/methods/am/policy.py` |
| `invit` | `INVITPolicy` 里 `node_dim = 2 if env_name == 'tsp' else 3` | `EasyNCO/neural_solvers/methods/invit/policy.py` |
| `pointerformer` | `PointerformerPolicy(env_name='tsp' 默认)` | `EasyNCO/neural_solvers/methods/pointerformer/policy.py` |

### 6.2 TSP 可用迭代（Iterator / Iteration）

#### 6.2.1 通用迭代器

| 迭代器 | 适用 TSP 吗 | 文件 |
|---|---|---|
| `NoIteration` | ✅ | `EasyNCO/neural_solvers/pipeline/iteration.py` |

#### 6.2.2 method 自带迭代器（明确支持 TSP）

| method | 迭代类 | 说明（面向 TSP） | 文件 |
|---|---|---|---|
| `lehd` | `LEHDIteration` | TSP 分支为 **RRC**（`TSPLEHDIteration`） | `EasyNCO/neural_solvers/methods/lehd/iteration.py` |
| `dact` | `DACTIteration` | 2-opt 类交换迭代；实现里对 `env.env_name=='tsp'` 有显式分支 | `EasyNCO/neural_solvers/methods/dact/iteration.py` |
| `lih` | `LIHIteration` | LIH 的 2-opt 类迭代；实现里对 `env.env_name=='tsp'` 有显式分支 | `EasyNCO/neural_solvers/methods/lih/iteration.py` |
| `difusco` | `DIFUSCOIteration` | 调用扩散模型 `model.iteration(...)`（注释写明 2-opt/MCTS） | `EasyNCO/neural_solvers/methods/difusco/iteration.py` |
| `t2t` | `T2TIteration` | 调用 `model.iteration(...)`（注释写明 2-opt/MCTS） | `EasyNCO/neural_solvers/methods/t2t/iteration.py` |
| `udc` | `UDCIteration` | 包含 `_run_*_tsp` 分支（train/eval 都有） | `EasyNCO/neural_solvers/methods/udc/iteration.py` |
| `glop` | `GLOPIteration` | 代码里包含 tsp 分解/求解流程（`tsp_decompose_and_solve`） | `EasyNCO/neural_solvers/methods/glop/iteration.py` |
| `deepaco` | `DeepACOIteration` | `problem_type=='tsp'` 分支会构造 `ACO_TSP` 并在 `max_steps` 内 search | `EasyNCO/neural_solvers/methods/deepaco/iteration.py` |

### 6.3 反向校验：明确“不是（单目标）TSP”的 method（因此本节不列）

以下 method 的默认 env / 数据结构与 TSP 不一致（因此不作为“能解单目标 TSP”的整理对象）：
- `nlns`：CVRP 大邻域搜索（初始化数据格式是 `(batch, 1+N, 3)` 含 demand）  
- `l2s`、`drl_hgnn`：调度（JSSP/FJSP）  
- `matnet`、`matpoenet`：ATSP/FFSP 相关  
- `mtpomo`、`mvmoe`：MVRP 相关  
- `dpn`：MTSP 相关  
- `psl`：`MOTSPEnv`（多目标 TSP，不是单目标 TSP）







## 7. 每种方法能解决的问题（problem/env）

> 判据说明（避免“想当然”）：  
> 1) 优先看 `EasyNCO/settings/*_settings.yaml` 的 `env._target_`（平台默认跑哪个 env）。  
> 2) 再看 `EasyNCO/neural_solvers/methods/<method>/` 中对 `env_name / problem` 的显式分支（方法代码“明确支持”哪些问题）。  
> 3) 若代码里有分支、但平台没有对应 env 或 settings，我会标注为“代码分支存在/平台未显式接入”。

### 7.1 缩写对照（便于读表）

| 缩写 | 中文含义（常用译法） | 对应 env |
|---|---|---|
| TSP | 旅行商问题 | `TSPEnv` |
| CVRP | 容量约束车辆路径问题 | `CVRPEnv` |
| ATSP | 非对称旅行商问题 | `ATSPEnv` |
| FFSP | 流水车间调度（Flow Shop） | `FFSPEnv` |
| JSSP | 作业车间调度（Job Shop） | `JSSPEnv` |
| FJSP | 柔性作业车间调度（Flexible Job Shop） | `FJSPEnv` |
| MOTSP | 多目标 TSP | `MOTSPEnv` |
| MOCVRP | 多目标 CVRP | `MOCVRPEnv` |
| MOKP | 多目标背包 | `MOKPEnv` |
| MTSP | 多旅行商 | `MTSPEnv` |
| MPDP | 多取送（Pickup & Delivery） | `MPDPEnv` |
| MDVRP | 多仓库 VRP | `MDVRPEnv` |
| OP | 定向越野（Orienteering Problem） | `OPEnv` |
| PCTSP | 奖励收集 TSP（Prize-Collecting TSP） | `PCTSPEnv` |
| SOP | 序列排序问题（Sequential Ordering Problem） | `SOPEnv` |
| SMTWTP | 单机总加权拖期（Single Machine Total Weighted Tardiness） | `SMTWTPEnv` |
| RCPSP | 资源受限项目调度（Resource-Constrained Project Scheduling） | `RCPSPEnv` |
| MKP | 多维背包（Multi-dimensional Knapsack） | `MKPEnv` |
| BPP | 装箱（Bin Packing） | `BPPEnv` |
| HCP | 哈密顿回路（Hamiltonian Cycle） | `HCPEnv` |
| MVRP/MTVRP | 多变体 VRP（平台里聚合为 `mtvrp`） | `MVRPEnv` |

### 7.2 方法 → 可解问题（按代码/配置确认）

| method | 可解问题（明确） | settings 默认 env | 证据（最小集合） | 备注 |
|---|---|---|---|---|
| `am` | TSP, CVRP | （无独立 settings） | `EasyNCO/neural_solvers/methods/am/am_encoder.py`、`EasyNCO/neural_solvers/methods/am/policy.py` | Encoder 注释写明“目前只支持 tsp/cvrp” |
| `pomo` | TSP, CVRP | `TSPEnv` | `EasyNCO/settings/pomo_settings.yaml`、`EasyNCO/neural_solvers/methods/pomo/policy.py` | 同样依赖 AM 的 encoder/decoder（tsp/cvrp） |
| `omni` | TSP, CVRP | `CVRPEnv` | `EasyNCO/settings/omni_settings.yaml`、`EasyNCO/neural_solvers/methods/omni/policy.py` | 代码结构是“POMO/AM 体系 + meta-learning”，问题支持同 encoder 约束 |
| `elg` | TSP, CVRP | `TSPEnv` | `EasyNCO/settings/elg_settings.yaml`、`EasyNCO/neural_solvers/methods/elg/policy.py` | `if env_name == 'tsp' / 'cvrp'` |
| `lehd` | TSP, CVRP | `TSPEnv` | `EasyNCO/settings/lehd_settings.yaml`、`EasyNCO/neural_solvers/methods/lehd/policy.py` | `if env_name == 'tsp' / 'cvrp'` |
| `dact` | TSP, CVRP | `CVRPEnv`（注释写可切 TSP） | `EasyNCO/settings/dact_settings.yaml`、`EasyNCO/neural_solvers/methods/dact/policy.py` | `if env_name == 'tsp' / 'cvrp'` |
| `lih` | TSP, CVRP | `CVRPEnv`（注释写可切 TSP） | `EasyNCO/settings/lih_settings.yaml`、`EasyNCO/neural_solvers/methods/lih/policy.py` | `env_name: tsp/cvrp`，迭代是神经 2-opt 风格 |
| `icam` | TSP, CVRP | `TSPEnv` | `EasyNCO/settings/icam_settings.yaml`、`EasyNCO/neural_solvers/methods/icam/policy.py` | `if env_name == 'tsp' / 'cvrp'` |
| `invit` | TSP, CVRP | `TSPEnv` | `EasyNCO/settings/invit_settings.yaml`、`EasyNCO/neural_solvers/methods/invit/policy.py` | `node_dim = 2 if tsp else 3` |
| `insertion` | TSP, CVRP | （无独立 settings） | `EasyNCO/neural_solvers/methods/insertion/initialization.py` | 传统启发式插入初始化（不依赖神经权重） |
| `pointerformer` | TSP | `TSPEnv` | `EasyNCO/settings/pointerformer_settings.yaml`、`EasyNCO/neural_solvers/methods/pointerformer/policy.py` | `pre_forward` 直接处理 `td['locs']`（坐标） |
| `htsp` | TSP | `TSPEnv` | `EasyNCO/settings/htsp_settings.yaml`、`EasyNCO/neural_solvers/methods/htsp/policy.py` | HTSP 自带 PPO/IMPALA 体系，低层可选 POMO/LKH |
| `difusco` | TSP；（代码分支：MIS） | `TSPEnv` | `EasyNCO/settings/difusco_settings.yaml`、`EasyNCO/neural_solvers/methods/difusco/policy.py` | `env_name in {'tsp','mis'}`；平台 env 列表里未看到 MIS 对应 Env |
| `t2t` | TSP；（代码分支：MIS） | `TSPEnv` | `EasyNCO/settings/t2t_settings.yaml`、`EasyNCO/neural_solvers/methods/t2t/policy.py` | `env_name in {'tsp','mis'}`；同上 |
| `udc` | TSP, CVRP | `TSPEnv` | `EasyNCO/settings/udc_settings.yaml`、`EasyNCO/neural_solvers/methods/udc/policy.py` | `if env_name == 'tsp' / 'cvrp'` |
| `glop` | TSP, ATSP, CVRP, PCTSP | `DUMMYEnv`（外包一层） | `EasyNCO/settings/glop_settings.yaml`、`EasyNCO/neural_solvers/methods/glop/policy.py` | `problem in ['tsp','atsp','cvrp','pctsp']` |
| `matnet` | ATSP, FFSP | `ATSPEnv` | `EasyNCO/settings/matnet_settings.yaml`、`EasyNCO/neural_solvers/methods/matnet/policy.py` | `if env_name == 'atsp' / 'ffsp'` |
| `matpoenet` | ATSP；（代码分支：HCP） | `ATSPEnv` | `EasyNCO/settings/matpoenet_settings.yaml`、`EasyNCO/neural_solvers/methods/matpoenet/initialization.py` | 初始化里有 `if problems == 'hcp'` 特判；要跑 HCP 需配 `HCPEnv` |
| `nlns` | CVRP | `CVRPEnv` | `EasyNCO/settings/nlns_settings.yaml`、`EasyNCO/neural_solvers/methods/nlns/iteration.py` | 大邻域搜索（destroy/repair），数据结构含 depot/demand |
| `reld` | CVRP | `CVRPEnv` | `EasyNCO/settings/reld_settings.yaml`、`EasyNCO/neural_solvers/methods/reld/policy.py` | 解码显式依赖 `td['load']` 等 CVRP 状态 |
| `dpn` | MTSP, MPDP, MDVRP, FMDVRP | `MTSPEnv` | `EasyNCO/settings/dpn_settings.yaml`、`EasyNCO/neural_solvers/methods/dpn/dpn_decoder.py` | `env_name` 分支覆盖 `mtsp/mpdp/mdvrp/fmdvrp` |
| `psl` | MOTSP；（代码分支：MOCVRP, MOKP） | `MOTSPEnv` | `EasyNCO/settings/psl_settings.yaml`、`EasyNCO/neural_solvers/methods/psl/policy.py` | policy 里显式分支 `motsp/mocvrp/mokp` |
| `deepaco` | TSP, CVRP, OP, PCTSP, SOP, SMTWTP, RCPSP, MKP, BPP | `TSPEnv`（通过 `problem` 切换） | `EasyNCO/settings/deepaco_settings.yaml`、`EasyNCO/neural_solvers/methods/deepaco/initialization.py` | `load_policy(..., env_name)` 覆盖上述 9 类问题 |
| `drl_hgnn` | FJSP | `FJSPEnv` | `EasyNCO/settings/drl_hgnn_settings.yaml`、`EasyNCO/neural_solvers/methods/drl_hgnn/policy.py` | 调度问题（图注意力 + actor/critic） |
| `l2s` | JSSP | `JSSPEnv` | `EasyNCO/settings/l2s_settings.yaml`、`EasyNCO/neural_solvers/methods/l2s/policy.py` | 调度问题（GIN/DGHAN 等 GNN） |
| `mtpomo` | MVRP/MTVRP（多变体：VRPB/CVRP/OVRP/VRPTW/VRPL/OVRPL…） | `MVRPEnv` | `EasyNCO/settings/mtpomo_settings.yaml`、`EasyNCO/neural_solvers/envs/MVRPEnv.py` | settings 里 `train_problems` 给出了平台支持的多变体列表 |
| `mvmoe` | MVRP/MTVRP（多变体：VRPB/CVRP/OVRP/VRPTW/OVRPTW/VRPL/OVRPL…） | `MVRPEnv` | `EasyNCO/settings/mvmoe_settings.yaml`、`EasyNCO/neural_solvers/envs/MVRPEnv.py` | 同上；MoE 版本还包含 routing 的 aux loss |
| `ptr_nets` | （未完整接入，需补 settings/initializer 才能跑） | （无） | `EasyNCO/neural_solvers/methods/ptr_nets/policy.py` | 当前只看到 `Attention/Pointer` 组件定义，没有标准 pipeline 配置 |



