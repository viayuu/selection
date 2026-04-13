# my/（双 Gate 强化学习原型）

本目录实现你的新论文设想：把 solver 选择从“整求解器”细化为两阶段 pipeline，并用强化学习训练两个 gate：

- Gate1：从初始化器集合中选 1 个（例如 `final/` 里的 POMO / AM 等），生成初始解 `sol0`
- Gate2：在 `data + sol0` 条件下，从迭代器集合中选 1 个（例如 `final/` 里的 RRC 等），固定迭代步数得到 `sol1`
- 奖励：`reward = -length(sol1)`
- baseline：按你的要求用 batch 内 `mean(length)`（实现中再换算到 reward 空间）
- loss：`loss = -mean((logp1 + logp2) * (reward - baseline))`

实现目标以“跑通框架”为主，不追求性能。

## 依赖

本原型会直接调用 `final/` 下的 EasyNCO 组件，因此至少需要：

- `torch`
- `tensordict`
- `torchrl`

（如果你要跑 `final/` 的更多方法，可能还需要额外依赖；本目录不负责安装依赖）

说明：`final/` 里部分 `__init__.py` 会导入大量方法/环境（可能依赖 `torch_geometric`、`sklearn`、`scipy` 等）。
为了让本原型在“只用 POMO/AM/LEHD + TSP”时尽量少依赖，`my/easynco_bootstrap.py` 会在运行时对
`EasyNCO.data` / `EasyNCO.neural_solvers.methods` / `EasyNCO.neural_solvers.envs` / `EasyNCO.neural_solvers.backbones`
做轻量 stub（不改 `final/` 代码），避免不必要的全量导入。

如果你在 `--init_zoo` 里启用更“重”的方法：
- `elg`：可能需要 `absl-py` 等依赖
- `difusco`：通常需要 `scipy`、`torch_sparse`，以及编译 `final/neural_solvers/methods/difusco/merge` 下的 cython 扩展

这些依赖由你的运行环境负责准备，`my/` 目录不做自动安装。

## Feature extraction（与论文一致的编码器）

默认特征提取不再用最初的简单手工统计，而是在 `my/` 中内置了一份与论文一致的编码器实现：

- `my/paper_encoder.py`：从论文代码的 `Naive_Encoder/Encoder_h` 拆出来并重写到 `my/`（避免直接依赖 repo 根目录的 `model.py`）
- `my/features.py` 默认 `extractor="paper"`，内部用 `my/paper_encoder.py:Encoder_h`（与 `config_TSP.yml:model_params` 默认一致）
- `tsp_instance_features(...)` 输出 `graph_emb + scale`，对应论文 `Selection_model.acquire_feature(...)`

## 运行

在仓库根目录执行：

```bash
python my/train_two_gate_tsp.py --problem_size 50 --batch_size 32 --train_steps 200
```

默认 zoo：
- Gate1 initializer：`pomo`, `am`（也支持 `lehd` / `elg` / `difusco`）
- Gate2 iteration：`none`, `rrc_lehd`（也支持 `two_opt` / `dact`；`lih` 仅支持 CUDA）

Gate2 迭代器说明（TSP）：
- `none`：不做改进，直接返回 `sol0`
- `rrc_lehd`：LEHD 的 RRC 风格 destroy/repair（固定 `--rrc_steps`）
- `two_opt`：传统 2-opt（来自 DIFUSCO 工具；固定 `--two_opt_iters`）
- `dact`：DACT 的神经 2-opt（调用平台 `DACTIteration`；固定 `--dact_steps`；当前为稳妥默认在 CPU 上运行并回传结果）
- `lih`：LIH 的神经 2-opt（调用平台 `LIHIteration`；固定 `--lih_steps`；平台代码硬编码 `.cuda()`，因此只支持 CUDA）

你也可以通过参数改 `pomo_size`、`rrc_steps` 等。
也可以显式指定 zoo：

```bash
python my/train_two_gate_tsp.py --init_zoo pomo am lehd elg difusco --iter_zoo none rrc_lehd two_opt dact
```

评估默认会在一份固定的 eval set 上跑 `eval_batches * batch_size` 个实例（按 batch 切分求均值）。
从当前版本开始，评估不再“每次重新随机生成 batch”，而是：
- 启动时按 `--eval_seed` + `--eval_dist` **生成一份固定 eval set**（大小 = `eval_batches * batch_size`），之后每次评估都在这份固定数据上跑，便于可复现对比；
- 或者用 `--eval_data_path xxx.pt` 指定一个固定的 `.pt` 数据集作为 eval set（默认只取前 `eval_batches * batch_size` 个实例控制开销）。

训练数据仍然是 **实时生成的无限数据流**，可用 `--train_dists` 指定分布（与论文的合成数据一致的生成逻辑）：
- `--train_dists uniform`：均匀分布（`[0,1]^2`）
- `--train_dists gaussian`：高斯混合分布（含 `num_modes=0` 时的 uniform 退化）
- 也支持混合：`--train_dists gaussian uniform`（每个实例随机选一种分布）

### 只评估 / 断点加载

- 只评估（greedy）：
  - `python my/train_two_gate_tsp.py --only_eval --load my/outputs/two_gate_tsp_seed2024_n50.pt`
- 继续训练：
  - `python my/train_two_gate_tsp.py --load my/outputs/two_gate_tsp_seed2024_n50.pt --train_steps 200`

### 用训练好的 gate 求解一个 TSP 数据集（.pt）

示例：求解 `final/data/datasets/test_dataset_tsp_uniform/test_tsp100_nums10000_uniform.pt`（10000 个 TSP100 实例）：

```bash
python my/solve_dataset_tsp.py \
  --gate_ckpt my/outputs/two_gate_tsp_seed2024_n100.pt \
  --data_path final/data/datasets/test_dataset_tsp_uniform/test_tsp100_nums10000_uniform.pt \
  --device cuda:0 \
  --batch_size 128
```

注意：`--init_zoo/--iter_zoo` 的顺序必须与训练时一致（默认会从 `gate_ckpt` 里的 `cfg` 读取）。
输出默认保存为 `my/outputs/<dataset_stem>_solved_n{N}.txt`，也可以用 `--out_path xxx.txt` 指定；如不想在 txt 里写 tour（文件更小），加 `--no_tour`。
txt 每行格式：`idx len0 len1 init iter [tour]`（其中 `len0` 是初始化解长度，`len1` 是迭代后长度）。

### baseline / AC

- 默认 baseline：`critic_batch_mean`（同时使用 `mean(length)` 常数 baseline + 可学习 `V(s)` 预测残差）
- 可选 baseline：`critic`（加入一个可学习 value 网络，做 actor-critic）
  - `python my/train_two_gate_tsp.py --baseline critic --critic_coef 0.5`
- 推荐 baseline：`critic_batch_mean`（同时使用 `mean(length)` 常数 baseline + 可学习 `V(s)` 预测残差，更符合“兼顾两者”的 AC 直觉）
  - `python my/train_two_gate_tsp.py --baseline critic_batch_mean --critic_coef 0.5`
 - 仅用 batch mean baseline（严格按你的原始说明，不训练 critic）：
  - `python my/train_two_gate_tsp.py --baseline batch_mean`

### 加载 solver checkpoint（可选）

默认 solver policy 不加载权重（随机初始化），只用于把框架跑通。你可以传入自己的 `final/` 兼容 checkpoint：

```bash
python my/train_two_gate_tsp.py \
  --pomo_ckpt /path/to/pomo.ckpt \
  --am_ckpt /path/to/am.ckpt \
  --lehd_ckpt /path/to/lehd.ckpt \
  --elg_ckpt /path/to/elg.ckpt \
  --difusco_ckpt /path/to/difusco.ckpt \
  --dact_ckpt /path/to/dact.ckpt \
  --lih_ckpt /path/to/lih.ckpt
```

也支持把 ckpt 统一放在仓库根目录的 `model/` 下（例如 `model/lehd_tsp100.ckpt`、`model/elg_tsp100.ckpt`、`model/difusco_tsp100.ckpt`），
并通过 `--model_dir model` 让脚本按文件名关键字自动匹配加载（若你显式传了 `--*_ckpt`，则以显式路径为准）。

示例（用你放在 `model/` 的 3 个初始化器 + LEHD-RRC 迭代器，先跑通 TSP100）：

```bash
python my/train_two_gate_tsp.py \
  --problem_size 100 \
  --init_zoo difusco elg lehd \
  --iter_zoo none rrc_lehd \
  --model_dir model \
  --device cuda:0
```
