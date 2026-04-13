# AGENTS.md

本文件用于给后续自动化编码助手提供“项目记忆”和协作约定；适用范围为 `final/` 目录及其子目录（但本仓库约定默认不进入/不修改 `final/reld-nco/`，除非用户明确要求）。

## 0. 目录定位（避免混淆）

- `final/` 是 EasyNCO 平台源码：集成多种组合优化问题（TSP/CVRP/ATSP/…）与多种神经/混合求解方法，并提供统一训练/评估入口与配置系统。
- 代码内部普遍使用 `EasyNCO.*` 作为绝对导入前缀；但在本仓库里目录名是 `final/`，不是 `EasyNCO/`，因此直接运行 `final/train.py`/`final/eval.py` 前通常需要先处理包名映射：
  - 方案 A：在仓库根目录创建 `EasyNCO -> final` 的软链接（让 `import EasyNCO` 解析到 `final/`）。
  - 方案 B：在运行脚本最开始调用 `my/easynco_bootstrap.ensure_local_easynco()` 做运行时 alias（本仓库原型代码常用）。
- 默认排除：`final/reld-nco/`（该子目录不纳入平台维护/阅读范围，除非用户明确要求）。

## 1. 入口与配置（Hydra）

- 训练入口：`final/train.py`（Hydra + Lightning）
- 测试/评估入口：`final/eval.py`
- 全局默认配置：`final/configs.yaml`
  - `defaults: - settings: reld_settings`（可通过 CLI 覆盖）
  - `mode: train|test`，`model`，`problem`，`scale` 等运行关键参数
  - Hydra 输出目录：`dir: ./results/${mode}/${model}_${problem}/...`
- 方法级配置：`final/settings/*_settings.yaml`
  - 常见 keys：`env/model/initialization/iteration/module/trainer/test_loader/model_checkpoint`
  - `env._target_` 在 `train.py`/`eval.py` 中会被动态覆盖为 `EasyNCO.neural_solvers.envs.{cfg.problem.upper()}Env`

### 常用命令（示例）

- `python final/train.py settings=pomo_settings mode=train model=pomo problem=tsp cuda=[0]`
- `python final/eval.py settings=pomo_settings mode=test model=pomo problem=tsp cuda=[0] test_data_path=...`
  - 注意：`test_data_path/val_data_path` 通常是相对 `final/data/datasets/` 的路径（脚本内部会拼接 `dataset_path`）。

## 2. 平台执行流水线（核心抽象）

训练/测试基本遵循相同的组件拼装顺序：

1. `env`：`EasyNCO.neural_solvers.envs/*Env.py`（torchrl `EnvBase`），维护 `TensorDict` 状态与 `reward`
2. `policy`：`EasyNCO.neural_solvers.methods.<method>.policy`
3. `initialization`：`EasyNCO.neural_solvers.pipeline.Initialization`（常见 `ARInitialization` 子类）负责 rollout 得到初始解/轨迹
4. `iteration`：`EasyNCO.neural_solvers.pipeline.Iteration`（或 `NoIteration`）负责可选的解改进
5. `module`：LightningModule，默认常用 `EasyNCO.phases.ARREINFORCELightning`（REINFORCE/RL 训练与评测）

### 关键约定

- `reward` 统一使用“越大越好”的方向：routing 问题通常 `reward = -cost`（见如 `final/neural_solvers/envs/TSPEnv.py`）。
- POMO/augmentation：
  - `pomo_size` 表示并行轨迹数（multi-start）
  - `aug_type/aug_factor` 控制数据增强；env 内部会把 `env_batch_size` 乘上 `aug_factor`
- 数据 batch 可能是 `Tensor` 或 `dict`（见 `final/neural_solvers/pipeline/initialization.py` 对 dict 的兼容处理）；新增 dataloader/Env 时需保持兼容。

## 3. 数据与数据集

- 生成器入口：`final/data/main.py:generate_data(problem_name)` 返回对应 `*Generator`
- 内置数据目录：`final/data/datasets/`（包含 uniform 生成数据、LIB 数据、OOD 数据等）
- 常见路径组织：`final/data/datasets/<dataset_name>/...`

## 4. Checkpoint 加载

- `final/utils/utils.py:load_model()` 兼容多种 checkpoint 格式：
  - `{'state_dict': ...}`（平台格式）
  - `{'model_state_dict': ...}`（部分作者仓库格式，如 ReLD）
  - 直接 `state_dict` dict
- 若新增/移植方法，优先在此集中补充“参数名对齐/兼容”策略，而不是在各个 policy 内散落零碎 hack。

## 5. 依赖与导入注意事项（避免踩坑）

- `final/neural_solvers/methods/__init__.py` 会一次性导入“solver zoo”，包含不少可选依赖；在只用某个方法时，优先 `import EasyNCO.neural_solvers.methods.<method>...` 避免触发全量导入。
- `final/neural_solvers/backbones/__init__.py` 会导入 `GNN/partition_net.py`（依赖 `torch_geometric`）。若环境缺少该依赖，避免 import 该 `__init__`，或参考 `my/easynco_bootstrap.py` 的轻量 stub 做隔离。
- 不要在 `final/` 内部为“临时可运行”而改成相对导入：保持 `EasyNCO.*` 路径一致性，便于与原平台对齐与复用 checkpoint。

## 6. 变更约定（给未来助手）

- 改动尽量局部：优先通过新增 `settings/*.yaml`、新增 method 子目录、或扩展 `load_model`/`generate_data` 映射实现功能。
- 不在 `final/reld-nco/` 做改动（除非用户明确要求）。
- 若需要跑训练/评测，优先用小 scale、小 batch 做 smoke test，再逐步扩大配置。

