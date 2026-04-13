# ReLD（Multi-Task）相对 MVMoE 的改动说明（基于 `final/compare/*`）

> 结论先行：`final/compare/reld/` 基本完全复用 `final/compare/mvmoe/` 的代码结构与训练/评测流程；**语义上的核心改动只有两类**：
>
> 1) 运行配置层面：训练/评测时建议 `--norm=none`（去掉 encoder 中的 normalization）；  
> 2) 代码实现层面：新增 `--ffidt` 开关，把 decoder 从 “POMO/MVMoE 的轻 decoder” 改成 “ReLD 风格的 IDT + FF 增强 decoder”，并在 MoE-light 版本中**禁用 decoder 的 MoE combine**以保持与论文设定一致。

本文只描述 **ReLD multi-task 相对 MVMoE 的差异**，不重复解释 MVMoE 的完整实现细节。

---

## 1. 对比对象与目录结构

- 基线：`final/compare/mvmoe/`（MVMoE 原始代码）
- 目标：`final/compare/reld/`（在 MVMoE 基础上做 ReLD multi-task 变体）

两者的 `envs/`、`Trainer.py`、`Tester.py`、`utils.py`、`generate_data.py` 等目录/文件在逻辑上保持一致；若你看到 “diff” 提示，主要是行尾/空白字符差异（`diff -w` 下大多无差异）。

---

## 2. 改动总览（真正有语义变化的文件）

### 2.1 新增 CLI/参数：`--ffidt`

新增文件改动：

- `final/compare/reld/train.py`
- `final/compare/reld/test.py`

变化点：

- 新增参数：`parser.add_argument('--ffidt', action='store_true')`
- 把 `ffidt` 写入 `model_params`：
  - `model_params["ffidt"] = args.ffidt`

目的：

- 让模型侧（decoder）能安全读取 `model_params['ffidt']`（默认 False）；
- 通过一个开关控制 “是否启用 ReLD 的 decoder 增强（IDT + FF）”。

### 2.2 模型侧：在 decoder 中加入 IDT + FF（ReLD 的核心结构改动）

涉及文件：

- `final/compare/reld/models/MTLModel.py`
- `final/compare/reld/models/MOEModel_Light.py`

#### A) `MTLModel.py`：对 MTL decoder 增强

改动位置：`class MTL_Decoder`

新增成员：

- `self.ffidt = self.model_params['ffidt']`
- 当 `ffidt=True` 时：
  - `self.attr_mapping = nn.Linear(4, embedding_dim, bias=False)`
  - `self.feed_forward = FeedForward(**model_params)`

新增前向逻辑（在 `mh_atten_out = multi_head_combine(out_concat)` 之后）：

- 当 `ffidt=True`：
  - **IDT（Identity Mapping）**：
    - `mh_atten_out = mh_atten_out + encoded_last_node + attr_mapping(attr)`
  - **FF（Feed-Forward + residual）**：
    - `mh_atten_out = feed_forward(mh_atten_out) + mh_atten_out`

其中 `attr` 是 decoder 的动态属性向量（维度 4），在上层模型里由环境状态拼出：

- `attr = [load, current_time, length, open]`

这对应 ReLD 论文中的：

- “Direct Influence of Context”（把 context 以残差方式直接注入 query 表示）
- “Powerful Query”（在聚合后加 FF 引入非线性）

> 注意：这里实现的是 **IDT + FF**；没有实现论文 Eq.(12) 的 `-log(dist)` distance heuristic。

#### B) `MOEModel_Light.py`：在 MoE-light 版本中同样加入 IDT + FF，并禁用 decoder 的 MoE combine

改动位置：`class MTL_Decoder`

新增成员：

- `self.ffidt = self.model_params['ffidt']`

关键行为 1：**当启用 `ffidt` 时，禁止 decoder 使用 MoE（hierarchical gating）**

原 MVMoE-light 的 decoder 有一段 “multi-head 输出 → dense / MoE 二选一（hierarchical gating）” 的逻辑。

ReLD multi-task 的改动是：

- 仅当 `not ffidt` 时才允许启用 decoder 的 hierarchical gating：
  - `if num_experts > 1 and 'Dec' in expert_loc and not ffidt: ...`
- forward 时也加了保护：
  - `if hierarchical_gating and not ffidt: ...`
- 启用 `ffidt` 时，始终走：
  - `mh_atten_out = self.multi_head_combine_dense(out_concat)`

理由（代码内也有注释）：**ReLD 系列模型在该设定下使用简单线性层 combine，多任务 MoE 仍主要体现在 encoder/其他位置**，以对齐论文中 “for simplicity and fair comparison” 的设定。

关键行为 2：**当启用 `ffidt` 时，加入 IDT + FF**

与 `MTLModel.py` 类似：

- `attr_mapping: Linear(4 → embedding)`
- `mh_atten_out = mh_atten_out + encoded_last_node + attr_mapping(attr)`
- `mh_atten_out = feed_forward(mh_atten_out) + mh_atten_out`

在该文件里 `feed_forward()` 的返回值是 `(out, moe_loss)` 风格，所以使用了 `[0]` 取出张量：

- `mh_atten_out = self.feed_forward(mh_atten_out)[0] + mh_atten_out`

---

## 3. 与 ReLD 论文设定的对应关系（你现在这份 multi-task 代码实现了什么/没实现什么）

你在 `final/compare/reld/` 中实现/强调的对应关系是：

- **去 normalization**：通过运行参数 `--norm=none` 实现（不是新代码逻辑，而是 MVMoE 原本就支持的 norm 选项）
- **decoder 增强**：通过 `--ffidt` 实现（新加开关 + decoder 内部结构变更）
  - IDT：`+ encoded_last_node + W(attr)`
  - FF：`+ FF(mh_atten_out)`
- **未包含**：distance heuristic（`-log(dist)`）与 “varying attribute / varying size” 训练策略的完整实现（这部分如果要对齐 ReLD-MoEL+，需要另行扩展）

---

## 4. 运行建议（基于 reld 目录的 README 约定）

- ReLD-MTL（多任务但非 MoE）：
  - `python train.py --problem=Train_ALL --model_type=MTL --norm=none --ffidt`
  - `python test.py  --problem=ALL       --model_type=MTL --norm=none --ffidt --checkpoint=...`

- ReLD-MoEL（MoE-light 体系）：
  - `python train.py --problem=Train_ALL --model_type=MOE_LIGHT --num_experts=4 --routing_level=node --routing_method=input_choice --norm=none --ffidt`
  - `python test.py  --problem=ALL       --model_type=MOE_LIGHT --num_experts=4 --routing_level=node --routing_method=input_choice --norm=none --ffidt --checkpoint=...`

> 注：`final/compare/reld/README.md` 里训练命令示例写的是 `--model_type=MOE`，但评测示例是 `MOE_LIGHT`；实际以你要用的模型类为准（在该代码结构里，MoE-light 对应 `models/MOEModel_Light.py`）。

