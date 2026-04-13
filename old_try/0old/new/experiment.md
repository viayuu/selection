# Experiments（EasyNCO 平台命令复现实验记录）

> 目标：只用 **EasyNCO 平台现有代码**（`python EasyNCO/eval.py ...`），不新增/不修改代码；给出可直接执行的命令，并解释每个实验为什么要这样配。

本文分两部分：
1) **TSPLIB（61 实例）**：你提出的“LEHD/GLOP 初始化×迭代”组合实验  
2) **CVRPLIB**：测试 `udc / glop / icam` 在 CVRPLIB 上的效果

---

## 0. 通用说明（非常关键）

### 0.1 从仓库根目录运行（避免 `import EasyNCO` 失败）

所有命令建议在仓库根目录 `/mnt/d/Study/3/neural-solver-selection` 执行，并显式加上：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python EasyNCO/eval.py ...
```

原因：`EasyNCO/eval.py` 会把 `EasyNCO/` 的上级目录插入 `sys.path`，但不同 shell/CWD 组合下你可能仍遇到路径问题；`PYTHONPATH=.` 是最稳的。

### 0.2 数据路径是相对 `EasyNCO/data/datasets/`

`EasyNCO/eval.py` 内部会把 `test_data_path` 拼成：

```
EasyNCO/data/datasets/<test_data_path>
```

所以：
- TSPLIB：直接用 `test_data_path=tsplib/tsplib`（这是现成目录）
- CVRPLIB：如果你的数据不在 `EasyNCO/data/datasets/` 下，需要先 **做软链接**（见第 2 部分）

### 0.3 为什么你的“跨方法 init+iter 组合”在 EasyNCO **无法纯靠配置做到**

你提出的组合（例如“LEHD 初始化 + GLOP 迭代”）从研究角度很合理，但在 **当前 EasyNCO 的架构**里，`eval.py` 的逻辑是：

1) 只加载 **一个** `policy = instantiate(settings.model)`  
2) 用同一个 `policy` 同时注入：
   - `initialization = instantiate(settings.initialization, policy=policy)`
   - `iteration = instantiate(settings.iteration, policy=policy)`

这带来一个硬约束：
- `LEHDInitialization` 必须配套 `LEHDPolicy`
- `GLOPIteration` 必须配套 `GLOPPolicy`（它会用到 `policy.lower_model[...]` 和 `policy.global_params`）

因此：
- **LEHD init + GLOP iter**：需要同时存在 `LEHDPolicy` 与 `GLOPPolicy` 两套权重，但 `eval.py` 只能加载一套 policy → **无法只靠命令行 override 实现**
- **GLOP init + LEHD iter**：同理也不行

在“不新增代码”的约束下，你目前能做的、严格等价的实验只有：
- `LEHD 初始化 + LEHD 迭代`（LEHD end-to-end）
- `GLOP 初始化 + GLOP 迭代`（GLOP end-to-end）

如果你后续允许“最小改动”，做跨方法组合的正确方式是：让 `eval.py`/module 支持 **同时加载两个 policy**（init_policy 与 iter_policy），或写一个“组合 policy wrapper”。但这已经超出本文“不要新增代码”的边界。

---

## 1) TSPLIB（61 个实例）上的实验：LEHD 与 GLOP

TSPLIB 61 实例目录在：
`EasyNCO/data/datasets/tsplib/tsplib`  
（你说的 61 个 `.tsp` 文件确实都在这里；`TSPGenerator` 会自动按 `.tsp` 遍历并用 batch_size=1 逐个评估。）

### 1.1 实验 A：LEHD 初始化 + LEHD 迭代（可直接跑）

**目的**：得到 LEHD 在 TSPLIB61 上的完整 end-to-end 表现（包含 iteration）。

**命令：**
```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python EasyNCO/eval.py \
  mode=test model=lehd problem=tsp decoder_strategy=greedy \
  settings=lehd_settings \
  test_data_path=tsplib/tsplib \
  settings.test_loader.model_dirpath=pretrained \
  settings.test_loader.model_filename=lehd_tsp100.ckpt \
  settings.module.iteration_params.max_steps=1000 \
  settings.env.aug_factor=1 \
  batch_size=1 \
  dir=./results/experiments/tsplib61/lehd_lehd
```

说明（你可以据此改超参）：
- `test_data_path=tsplib/tsplib`：触发 TSPLIB loader；每次一个实例（batch_size=1）
- `settings.module.iteration_params.max_steps=1000`：LEHD 迭代步数（预算）  

### 1.3 实验 C：GLOP 初始化 + GLOP 迭代（可直接跑，但要注意 `revision_lens`）

**关键风险**：`GLOPIteration` 会对 `global_params.revision_lens` 逐级分解求解子问题。  
如果 `revision_lens` 比 TSPLIB 中某些实例的 `N` 更大，可能导致分解过程出错或质量异常。

TSPLIB61 中最小 N 大约是 50 左右，因此建议把 `revision_lens` 控制在 `<=50` 的范围（一个稳妥的起步设置：`[50, 20]`）。

**命令（推荐起步版）：**
```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python EasyNCO/eval.py \
  mode=test model=glop problem=tsp scale=100 decoder_strategy=greedy \
  settings=glop_settings \
  test_data_path=tsplib/tsplib \
  settings.test_loader.model_dirpath=pretrained \
  settings.test_loader.model_filename=glop_policy_tsp.pt \
  settings.model.global_params.revision_lens=[50,20] \
  settings.model.global_params.iter=[10,5] \
  batch_size=1 \
  dir=./results/experiments/tsplib61/glop_glop
```

备注：
- 如果你遇到 `EasyNCO/neural_solvers/envs/TSPEnv.py` 里 `self.lib_data["edge_weight_type"]` 报 `TypeError: 'NoneType' object is not subscriptable`：这是因为 **GLOP 在 iteration 内部会创建并使用自己的 `lower_env (TSPEnv)`**，而 TSPLIB 的 `edge_weight_type/lib_data` 是挂在外层 env 上的；两者没有自动同步就会触发该错误。本仓库已在 `EasyNCO/neural_solvers/methods/glop/iteration.py` 开头补了一行把外层 `env.lib_data` 传给 `lower_env`，所以更新后应可直接跑通。

### 1.4 你提出的 4 个组合实验如何处理（在“不新增代码”的约束下）

你列的是：
1) lehd init + glop iter  
2) lehd init + lehd iter
3) glop init + glop iter  
4) glop init + lehd iter

在 EasyNCO 当前 `eval.py` 注入单一 `policy` 的机制下：
- ①/④：**无法只用命令行实现**
- ③：可做（见 1.3）


---

## 1.5（最小改动版）实现跨方法 init+iter：新增 settings + `eval.py` 支持“两套 policy”

你现在允许“最小改动”，那么**可以**用“新增 settings + 很小的 `EasyNCO/eval.py` 改动”来实现：
- `LEHD init + GLOP iter`
- `GLOP init + LEHD iter`

实现方式（已在本仓库完成）：
- `EasyNCO/eval.py` 支持一种 *two-policy composition* 模式：  
  如果 `settings` 里存在 `init_model/init_test_loader` 与 `iter_model/iter_test_loader`，则分别加载两套 ckpt，并把它们注入到 `initialization` 与 `iteration`。
- 新增了两个 settings 文件：  
  - `EasyNCO/settings/lehd_init_glop_iter_settings.yaml`  
  - `EasyNCO/settings/glop_init_lehd_iter_settings.yaml`

> 这仍然属于“用 EasyNCO 平台代码跑”，没有新写 pipeline 逻辑，只是让 `eval.py` 允许 init/iter 使用不同 policy。

### 1.5.1 实验 1：LEHD 初始化 + GLOP 迭代（TSPLIB61）

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python EasyNCO/eval.py \
  mode=test model=compose problem=tsp decoder_strategy=greedy \
  settings=lehd_init_glop_iter_settings \
  test_data_path=tsplib/tsplib \
  batch_size=1 \
  dir=./results/experiments/tsplib61/lehd_init_glop_iter
```

### 1.5.2 实验 4：GLOP 初始化 + LEHD 迭代（TSPLIB61）

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python EasyNCO/eval.py \
  mode=test model=compose problem=tsp  decoder_strategy=greedy \
  settings=glop_init_lehd_iter_settings \
  test_data_path=tsplib/tsplib \
  batch_size=1 \
  dir=./results/experiments/tsplib61/glop_lehd
```
---

## 2) CVRPLIB 上测试 `udc / glop / icam`

### 2.2 实验 D：ICAM on CVRPLIB（可直接跑）

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python EasyNCO/eval.py \
  mode=test model=icam problem=cvrp decoder_strategy=greedy \
  settings=icam_settings \
  test_data_path=cvrplib/Vrp-Set-X \
  varying_data_params.capacity_range=null \
  settings.test_loader.model_dirpath=pretrained \
  settings.test_loader.model_filename=icam_cvrp_best.ckpt \
  batch_size=1 \
  dir=./results/experiments/cvrplib/icam_vrp_set_x
```

说明：
- **一定要加** `varying_data_params.capacity_range=null`：否则会用 `EasyNCO/configs.yaml` 里的默认值 `capacity_range=[50,100]` 过滤数据
- 如果你只想测 `Vrp-Set-X` 这一组（而不是整个 XXL 目录），命令改成：

### 2.3 实验 E：UDC on CVRPLIB（可直接跑，但默认 `aug_factor=50` 很重）

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python EasyNCO/eval.py \
  mode=test model=udc problem=cvrp scale=1000 decoder_strategy=greedy \
  settings=udc_settings \
  test_data_path=cvrplib/Vrp-Set-X \
  varying_data_params.scale_range=null \
  varying_data_params.capacity_range=null \
  settings.model.feats=3 \
  settings.model.feats=3 \
  settings.test_loader.model_dirpath=pretrained \
  settings.test_loader.model_filename=udc_cvrp_best.ckpt \
  batch_size=1 \
  dir=./results/experiments/cvrplib/udc_vrp_set_x
```


**常见报错（你刚遇到的）**：
- `Shape mismatch for model_p.emb_net.v_lin0.weight: model [64,2], ckpt [64,3]`
  - 原因：UDC 在 CVRP 的图输入特征是 3 维（`norm_demand, r, theta`，见 `EasyNCO/neural_solvers/methods/glop/cvrp_heatmap.py`），对应 `feats=3`；但 `udc_settings.yaml` 默认写的是 `feats=2`（更像 TSP 用法），导致 ckpt 维度对不上。
  - 解决：命令里加 `settings.model.feats=3`（上面的命令已包含）。

### 2.4 实验 F：GLOP on CVRPLIB（当前仓库缺少对应 ckpt，无法直接跑）

**结论先说清楚**：  
`EasyNCO/utils/utils.py:load_model()` 是“严格加载”，如果 ckpt 缺关键参数会直接 `assert False`。  
而 `EasyNCO/pretrained/` 里目前只有 `glop_policy_tsp.pt`，没有 CVRP 版本（例如 `glop_policy_cvrp_1000.pt`）。

因此在不改代码的前提下：
- **你需要先拿到/放入一个 CVRP 对应的 GLOP ckpt**（文件名随意，但必须与 `GLOPPolicy(problem='cvrp')` 的 state_dict 完全匹配）

假设你已经有 `glop_policy_cvrp_1000.pt` 放到了 `EasyNCO/pretrained/`，则命令是：
```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python EasyNCO/eval.py \
  mode=test model=glop problem=cvrp scale=1000 decoder_strategy=greedy \
  settings=glop_settings \
  test_data_path=cvrplib-Vrp-Set-XXL \
  varying_data_params.scale_range=[0,1001] \
  settings.test_loader.model_dirpath=pretrained \
  settings.test_loader.model_filename=glop_policy_cvrp_1000.pt \
  batch_size=1 \
  dir=./results/experiments/cvrplib/glop
```

