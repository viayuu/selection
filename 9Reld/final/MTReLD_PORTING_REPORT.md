# MTReLD 移植报告（ReLD Multi-Task → EasyNCO 平台 `final/`）

> 本报告记录：将作者 multi-task 版本的 ReLD（两种模型：`MTL` 与 `MOE_LIGHT`）移植进 EasyNCO 平台（本仓库的 `final/`）的全过程与关键实现细节，便于后续复现实验、排查差异、继续扩展训练/评测流程。

## 1. 背景与目标

### 1.1 目标

- 在 EasyNCO 平台里新增 **multi-task 版本 ReLD** 的推理能力，能够像作者代码一样在 **16 种 VRP 变体**上评测：
  - `CVRP`, `OVRP`, `VRPB`, `VRPL`, `VRPTW`, `OVRPTW`,
  - `OVRPB`, `OVRPL`, `VRPBL`, `VRPBTW`, `VRPLTW`,
  - `OVRPBL`, `OVRPBTW`, `OVRPLTW`, `VRPBLTW`, `OVRPBLTW`
- 平台侧提供两个等价于作者实现的模型变体：
  - **ReLD-MTL**：作者 `--model_type=MTL`  → 平台 `model=mtreld_mtl`
  - **ReLD-MoEL(light)**：作者 `--model_type=MOE_LIGHT` → 平台 `model=mtreld`
- 推理（inference）对齐作者结果：同一 checkpoint、同一测试集、同一 POMO/增强/解码策略下，指标应能精确对齐或仅存在极小数值误差。

### 1.2 关键对齐点（为什么会“差一点点”）

multi-task 评测容易出现“看似只差一点点”的误差，常见根因包括：

- 随机性/确定性设置不同（seed、cudnn deterministic、torch/numpy/random）
- `matmul_precision` 不一致（`medium` vs `high` vs `highest`）
- augmentation/POMO 的维度组织不同（aug_factor、batch_size 的对应关系）
- `norm` / `ffidt` 等模型结构开关不同（会导致权重无法语义对齐）
- env 中特征 dtype 不一致（例如 `open` bool vs float，导致 decoder 输入不同）

本次移植的核心工作之一就是把上述“隐性差异源”消掉。

## 2. 源实现与对照材料

### 2.1 作者 multi-task 评测入口（参考）

- `final/reld-nco/Multi-Task/test.py`
- `final/reld-nco/Multi-Task/Tester.py`

作者侧推理参数要点（摘自 `test.py`）：

- `--problem_size=100`, `--pomo_size=100`
- `--embedding_dim=128`, `--encoder_layer_num=6`, `--head_num=8`, `--qkv_dim=16`, `--ff_hidden_dim=512`
- `--logit_clipping=10`
- 推荐 ReLD 变体：`--norm=none --ffidt`
- 推理：`--test_episodes=1000 --aug_factor=8 --aug_batch_size=100 --eval_type=argmax`
  - 当 `aug_factor!=1` 时，作者会把 `test_batch_size` 自动置为 `aug_batch_size`（默认 100）

### 2.2 代码级差异分析（ReLD multi-task 相对 MVMoE）

- 对照实现目录：`final/compare/mvmoe/` 与 `final/compare/reld/`
- 总结文档：`final/compare/RELD_changes_from_MVMoE.md`

结论（与移植直接相关）：

- 运行配置：**encoder 去 normalization**（`norm=none`）
- 结构改动：decoder 增加 **IDT + FF**（`ffidt=True`）
- MoE-light 变体：当启用 `ffidt` 时，**禁用 decoder 的 MoE combine / hierarchical gating**，始终走 dense combine（对齐论文/代码设定）

## 3. 平台侧新增/修改文件清单（与 MTReLD 相关）

### 3.1 新增：方法实现（methods）

- `final/neural_solvers/methods/mtreld/policy.py`
- `final/neural_solvers/methods/mtreld/mtreld_decoder.py`
- `final/neural_solvers/methods/mtreld/__init__.py`
- `final/neural_solvers/methods/mtreld_mtl/policy.py`
- `final/neural_solvers/methods/mtreld_mtl/mtl_model.py`
- `final/neural_solvers/methods/mtreld_mtl/__init__.py`

对应关系：

- `mtreld`：实现作者 `MOE_LIGHT`（但移植时沿用平台目录命名 `mtreld`）
- `mtreld_mtl`：实现作者 `MTL`

### 3.2 新增：Hydra 配置（settings）

- `final/settings/mtreld_settings.yaml`
- `final/settings/mtreld_mtl_settings.yaml`

### 3.3 平台适配修改（让 multi-task inference 正确跑通并对齐）

- `final/eval.py`
  - 识别 `model in ["mtpomo", "mvmoe", "mtreld", "mtreld_mtl"]` 时，强制使用 `MVRPEnv`，并注入 `env.test_problem=cfg.problem.upper()`（避免 `NoneType` 错误）
  - 用平台 `seed_everything()` 对齐作者 seed 行为
  - `matmul_precision` 由配置控制（不再写死 `medium`）
  - 在 `final/` 内运行时，显式把 `EasyNCO` 包名映射到当前目录，避免导入到另一个 `EasyNCO/` 目录
- `final/phases/rl/ar_reinforce.py`
  - `train_dataloader()` 支持 `mtreld/mtreld_mtl` 的 multi-task 数据生成分支
  - `test_dataloader()` 对 `problem_name == "MVRP"` 走专用逻辑并强制要求 `env.test_problem` 存在，避免静默读错数据
- `final/neural_solvers/envs/MVRPEnv.py`
  - 将 `open` 特征统一转成 `float32`（解决 decoder `attr` 拼接时的 dtype 不一致）
- `final/neural_solvers/methods/__init__.py`
  - 暴露 `MTReLDPolicy`（便于 `EasyNCO.neural_solvers.methods.MTReLDPolicy` 风格导入；同时保持 `mtreld_settings.yaml` 里用全路径更稳）

### 3.4 推理脚本（可直接跑 16 任务并保存日志）

- `final/test.bash`
  - 统一运行参数
  - 自动保存日志到 `final/logs/mtreld_runs/run_*.txt`
  - 同时把每个问题的输出 tee 到 `final/logs/mtreld_moe_light/*.txt` / `final/logs/mtreld_mtl/*.txt`

## 4. 实现细节

### 4.1 平台 pipeline 如何承载 multi-task ReLD

平台侧推理调用链（简化）：

- `final/eval.py`
  1) Hydra 实例化 `env / policy / initialization / iteration / module`
  2) `load_model(policy, ckpt_path, device, model_name)`
  3) Lightning `trainer.test(module)`
- `final/phases/rl/ar_reinforce.py:ARREINFORCELightning.test_step()`
  - `initialization.run(env, batch, decoder_strategy, phase="eval")`
  - `iteration.run(...)`（这里是 `NoIteration`，直接透传 initialization_out）
  - 记录/打印 `no_aug_score` 与 `aug_score`

multi-task 版本采用：

- `initialization = POMOInitialization`（本质是 ARInitialization + POMO rollout）
- `iteration = NoIteration`
- `module = ARREINFORCELightning`

这样可与作者 `Tester._test_one_batch()` 的评测逻辑一一对应：

- env 内部做 8-fold augmentation（`aug_factor=8`）
- reward reshape 成 `(aug, batch, pomo)` 后：
  - `no_aug_score` = augmentation=0 的最优 POMO
  - `aug_score` = augmentation 维度上再取最优

### 4.2 数据与环境：MVRPEnv / MVRPGenerator

#### 4.2.1 数据文件组织（平台约定）

multi-task 数据放在：

- `final/data/datasets/mt/<PROB>/<prob_lower><scale>_uniform.pkl`

例如：

- `final/data/datasets/mt/CVRP/cvrp100_uniform.pkl`
- `final/data/datasets/mt/VRPTW/vrptw100_uniform.pkl`

对应作者默认：

- `./data/<PROB>/<prob_lower><problem_size>_uniform.pkl`

#### 4.2.2 MVRPGenerator 的关键约束

`final/data/MVRPGenerator.py` 在加载 `.pkl` 时会根据 `test_problem` 决定需要解析的字段：

- 是否包含 `TW`（时间窗）
- 是否包含 `B`（backhaul）
- 是否包含 `L`（route length limit）

因此 **必须**在评测时设置 `test_problem`（例如 `VRPTW`），否则会触发类似错误：

- `TypeError: argument of type 'NoneType' is not iterable`（`if "TW" in test_problem`）

平台修复方式：

- `final/eval.py` 对 multi-task 模型强制注入 `env.test_problem = cfg.problem.upper()`
- `final/phases/rl/ar_reinforce.py:test_dataloader()` 在 `problem_name == "MVRP"` 时显式传入 `test_problem=self.env.test_problem`

#### 4.2.3 `open` 特征的 dtype 对齐

ReLD decoder 的动态属性向量：

- `attr = [load, current_time, length, open]`

其中 `open` 在 `MVRPEnv` 内由 `o_flag`（是否是 OVRP 类问题）派生，原始是 bool。为保证 `torch.cat` 与线性层输入一致，平台将其统一转换为 `float32`：

- 修改位置：`final/neural_solvers/envs/MVRPEnv.py`
- 表现：`open` 在 reset / pre_step / step 输出里都是 float（0.0/1.0）

### 4.3 `mtreld`（对应作者 `MOE_LIGHT`）实现

**Encoder（复用 `mvmoe` 的 `MTMoEEncoder`）**

- 调用（接口不变，仍返回 `encoded_nodes, moe_loss`）：`final/neural_solvers/methods/mtreld/policy.py`

```python
self.encoded_nodes, moe_loss = self.encoder(depot_xy, node_xy_demand_tw)
```

- 关键对齐：关闭 normalization（不改 encoder 代码，只改配置）：`final/settings/mtreld_settings.yaml`

```yaml
normalization: 'none'
```

**Decoder（在 `MTMoEDecoder` 框架上做 ReLD 改动）**

- `ffidt=True`：加入 IDT + FF：`final/neural_solvers/methods/mtreld/mtreld_decoder.py`

```python
mh_atten_out = mh_atten_out + encoded_last_node + self.attr_mapping(attr)
mh_atten_out = self.feed_forward(mh_atten_out) + mh_atten_out
```

- `ffidt=True`：禁用 decoder 的 MoE combine（强制 dense combine）：`final/neural_solvers/methods/mtreld/mtreld_decoder.py`

```python
use_decoder_moe = (MoE_param.get("num_experts", 1) > 1) and ("Dec" in MoE_param.get("expert_loc", [])) and (not self.ffidt)
mh_atten_out = self.multi_head_combine_dense(out_concat)
```

### 4.4 `mtreld_mtl`（对应作者 `MTL`）实现

**Encoder（替换 `MTMoEEncoder` → `MTL_Encoder`）**

- 替换点：`final/neural_solvers/methods/mtreld_mtl/policy.py`

```python
self.encoder = MTL_Encoder(**model_params)
```

- 不能直接复用 `MTMoEEncoder` 的原因：`mvmoe` 的 encoder layer 写死需要 `moe_loss`，MTL 不产生 `moe_loss`

```python
out2, moe_loss = self.feedForward(out1)  # mvmoe
out2 = self.feedForward(out1)            # mtreld_mtl
```

**Decoder（替换 `MTMoEDecoder` → `MTL_Decoder`，保留 `ffidt`）**

- 替换点：`final/neural_solvers/methods/mtreld_mtl/policy.py`

```python
self.decoder = MTL_Decoder(**model_params)
```

- dense combine（无 MoE / gating）：`final/neural_solvers/methods/mtreld_mtl/mtl_model.py`

```python
self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
```

- `ffidt=True`：IDT + FF：`final/neural_solvers/methods/mtreld_mtl/mtl_model.py`

```python
mh_atten_out = mh_atten_out + encoded_last_node + self.attr_mapping(attr.clone())
mh_atten_out = self.feed_forward(mh_atten_out) + mh_atten_out
```

## 5. 推理配置与运行方法（只用 `final/`）

### 5.1 关键文件路径（数据与模型）

- multi-task 数据：`final/data/datasets/mt/`
- multi-task 预训练模型：
  - ReLD-MoEL(light)：`final/pretrained/pretrained/reld_moe_light/epoch-5000.pt`
  - ReLD-MTL：`final/pretrained/pretrained/reld_mtl/epoch-5000.pt`

### 5.2 直接用 `eval.py` 跑单个问题（示例）

在 `final/` 目录下：

**(A) 跑 `mtreld`（MOE_LIGHT）**

```bash
python eval.py \
  settings=mtreld_settings mode=test model=mtreld problem=cvrp scale=100 cuda=[0] \
  seed=2024 matmul_precision=highest decoder_strategy=greedy \
  test_data_path="mt/CVRP/cvrp100_uniform.pkl" \
  settings.test_loader.model_dirpath="pretrained/pretrained/reld_moe_light" \
  settings.test_loader.model_filename="epoch-5000.pt" \
  episodes=1000 batch_size=100 \
  ++settings.env.aug_factor=8 \
  ++settings.module.test_data_params.mode=1
```

**(B) 跑 `mtreld_mtl`（MTL）**

```bash
python eval.py \
  settings=mtreld_mtl_settings mode=test model=mtreld_mtl problem=cvrp scale=100 cuda=[0] \
  seed=2024 matmul_precision=highest decoder_strategy=greedy \
  test_data_path="mt/CVRP/cvrp100_uniform.pkl" \
  settings.test_loader.model_dirpath="pretrained/pretrained/reld_mtl" \
  settings.test_loader.model_filename="epoch-5000.pt" \
  episodes=1000 batch_size=100 \
  ++settings.env.aug_factor=8 \
  ++settings.module.test_data_params.mode=1
```

说明：

- `problem=` 传小写（Hydra config 里常用），平台会自动注入 `env.test_problem=PROBLEM.upper()` 供 MVRPGenerator 使用
- `episodes=1000` 对齐作者 `--test_episodes=1000`
- `batch_size=100` 对齐作者 `--aug_batch_size=100`（当 `aug_factor=8` 时）
- `++settings.env.aug_factor=8` 对齐作者 `--aug_factor=8`
- `++settings.module.test_data_params.mode=1` 对齐多任务数据生成/时间窗缩放方式（见 MVRPGenerator）

### 5.3 用脚本一键跑 16 个问题并保存日志

推荐直接用：

- `final/test.bash`

当前脚本默认只测一个问题用于 smoke test，如需全量评测，删掉脚本中的：

- `PROBS="OVRP"`

然后在 `final/` 下运行：

```bash
bash test.bash
```

输出：

- 总日志：`final/logs/mtreld_runs/run_*.txt`
- 分问题日志：
  - `final/logs/mtreld_moe_light/<prob>_100_aug8.txt`
  - `final/logs/mtreld_mtl/<prob>_100_aug8.txt`

## 6. Checkpoint 加载与“参数转换”策略

### 6.1 作者 checkpoint 格式

作者 multi-task checkpoint（`epoch-5000.pt`）结构是典型的：

- `{'model_state_dict': ..., 'epoch': ..., 'problem': ..., ...}`

作者在 `final/reld-nco/Multi-Task/Tester.py` 中以严格模式加载：

- `model.load_state_dict(checkpoint['model_state_dict'], strict=True)`

### 6.2 平台加载入口与兼容逻辑

平台加载在 `final/eval.py` 中完成：

- `policy = load_model(policy, full_model_path, device=cuda_device, model_name=cfg.model)`

核心兼容在：

- `final/utils/utils.py:load_model()`

支持三种常见结构：

1) 作者格式：`model_state_dict`
2) Lightning 常见平台格式：`state_dict`
3) 直接是 state_dict 字典

并提供了参数名匹配策略（对齐平台/作者 key 命名差异）：

- 自动尝试 `policy.` 前缀
- 自动尝试移除 `encoder.` / `decoder.` 等前缀
- 对关键结构差异会做 shape 校验，发现不一致直接报错，避免 silent wrong load

> 实践建议：对于新方法移植，优先让平台实现的 module 命名尽量贴近作者（`encoder/decoder/layers/...`），这样可以做到“零转换直接加载”；只有在结构或命名不可避免不一致时，再扩展 `load_model()` 的映射逻辑或写离线转换脚本。

## 7. 作者参数 vs 平台参数：一致性检查清单（强烈建议逐项核对）

对齐作者推理（`final/reld-nco/Multi-Task/test.py`）时，平台侧需要保证：

- **模型结构**：
  - `embedding_dim=128`, `encoder_layer_num=6`, `head_num=8`, `qkv_dim=16`, `ff_hidden_dim=512`
  - `logit_clipping=10`
  - `norm=none`（平台：`normalization: 'none'`）
  - `ffidt=True`
  - `MOE_LIGHT`：`num_experts=4`, `topk=2`, `routing_level=node`, `routing_method=input_choice`, `expert_loc` 对齐
- **解码策略**：作者 `eval_type=argmax` 等价于平台 `decoder_strategy=greedy`
- **POMO/增强**：
  - `problem_size=100`, `pomo_size=100`（平台：`scale=100` 且 env.pomo_size 跟随 problem_size）
  - `aug_factor=8`
  - 当 `aug_factor=8` 时，**batch_size=100**（否则显存/吞吐与作者不同，也可能引入数值差异）
- **seed 与数值实现**：
  - `seed=2024`
  - `matmul_precision=highest`（建议与作者跑 log 时一致）
  - 环境 `mode=1`（影响时间窗/缩放逻辑）

## 8. 结果验证资产

为了对齐检查，保留了作者与平台的输出日志，并用 notebook 自动对齐比较：

- 作者日志：`final/logs/tsplib_70/lehd/author.txt`
- 平台日志：`final/logs/tsplib_70/lehd/my.txt`
- 对齐分析 notebook：`final/logs/tsplib_70/lehd/mtreld_author_vs_my_analysis.ipynb`

该 notebook 会把 `mtreld` 与 `mtreld_mtl` 的 `no_aug_score/aug_score` 与作者对应版本逐问题对齐，并输出差值统计与可视化。

## 9. 常见报错与排障

### 9.1 `TypeError: argument of type 'NoneType' is not iterable`（MVRPGenerator）

原因：`test_problem` 没有传入，导致 `if "TW" in test_problem` 报错。

解决：

- 使用 `final/eval.py`（已自动注入 `env.test_problem`）
- 确保 `model` 是 `mtreld` / `mtreld_mtl`（触发 multi-task 分支）

### 9.2 `torch.cat` / `Linear` 输入 dtype 报错（bool 与 float）

原因：`open` 作为 bool 与其他 float 特征拼接。

解决：

- `final/neural_solvers/envs/MVRPEnv.py` 已把 `open` 转成 `float32`。

### 9.3 结果“差一点点”

优先核对（按影响从大到小）：

- seed 与 deterministic 是否一致（作者：`seed_everything()`）
- `matmul_precision` 是否一致（建议 `highest`）
- 是否确实启用了 `norm=none` 与 `ffidt=True`
- `aug_factor`、`batch_size` 是否对齐作者（`aug_factor=8` 时 batch_size=100）

## 10. 已知限制与后续可改进点（可选）

- 目前主要验证的是 **inference 对齐**；如需在平台中复现作者训练（Train_ALL），建议新增专用训练入口或扩展 `train.py` 在 `problem=MVRP` 下的 workflow（包括 train_problems 采样逻辑与 checkpoint 保存策略对齐）。
- 本次移植实现了 compare 版本强调的 `IDT + FF`；未额外实现 distance heuristic（`-log(dist)`）等论文中可能出现的增强项（与 `final/compare/RELD_changes_from_MVMoE.md` 的说明一致）。
- `final/neural_solvers/methods/__init__.py` 全量导入 solver zoo，可能触发可选依赖（如 `torch_geometric`）导入问题；日常使用建议在 Hydra `settings.model._target_` 里写 **明确的全路径**（已按此实现）。
