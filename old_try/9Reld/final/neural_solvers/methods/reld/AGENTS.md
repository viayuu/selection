# AGENTS.md (ReLD single-task port notes)

适用范围：`final/neural_solvers/methods/reld/` 及其子目录。

本目录是你已完成的 **ReLD（单任务 CVRP）移植到 EasyNCO 平台**的实现，目标对齐 `final/reld-nco/CVRP/CVRPModel.py`（以及论文 `final/reld-nco/paper.md` 的 ReLD：IDT + FF + distance heuristic，encoder 去 normalization）。

## 1. 文件与职责对照（author → platform）

- `final/neural_solvers/methods/reld/reld_encoder.py`
  - 对齐 author `CVRP_Encoder` / `EncoderLayer` / `FeedForward`
  - 关键点：无 normalization；FF 层参数命名为 `W1/W2` 以便对齐作者 checkpoint key。
- `final/neural_solvers/methods/reld/reld_decoder.py`
  - 对齐 author `CVRP_Decoder`（结构：`Wq_last/Wk/Wv/multi_head_combine + capacity_mapping + feed_forward + (-log(dist))`）
  - 关键点：IDT（残差注入 last node + capacity 映射）+ FF（残差前馈）+ 距离启发式 `-log(cur_dist)`。
- `final/neural_solvers/methods/reld/policy.py`
  - 平台 `Policy` 封装：`pre_forward()` 计算静态 embeddings 并缓存 K/V；`forward()` 按 POMO 风格处理前两步，然后调用 decoder 输出 `probs`，最终用 `get_post_search_strategy()` 做 sampling/greedy 选点。
- `final/neural_solvers/methods/reld/initialization.py`
  - `ReLDInitialization(ARInitialization)`：重写 `play_episode()`，每步从 `env._get_local_feature()` 取 `cur_dist` 并传给 policy/decoder，以支持 distance heuristic。

## 2. 平台侧数据流假设（必须满足）

- 环境：用于 ReLD 的 `CVRPEnv` 需要实现 `_get_local_feature()` 并在 `current_node != None` 时返回 dict，至少包含：
  - `cur_dist`: `(batch, pomo, problem+1)`，当前节点到所有节点的距离（含 depot），供 decoder 做 `-log(cur_dist)`。
- policy/decoder 依赖的关键字段（来自 `TensorDict`）：
  - `td["locs"]`: `(batch, problem+1, 3)`，首节点为 depot `(x,y,*)`，其余为 customer `(x,y,demand)`。
  - `td["load"]`: `(batch, pomo)`，remaining capacity（标量），用于 `Wq_last` 输入与 `capacity_mapping`。
  - `td["next"]["ninf_mask"]`: `(batch, pomo, problem+1)`，不可选节点 mask（`-inf`）。
  - `td["action"]`: 上一步动作（用于从静态 embeddings 里 gather `encoded_last_node`）。

## 3. ReLD 关键计算（与论文/作者代码的一致性）

- 解码第 t 步（从第三步开始）：
  1. `encoded_last_node` 与 `load` 拼接 → `Wq_last` 得到 query（按 head reshape）
  2. `MHA(query, K=static, V=static)` + `multi_head_combine`
  3. **IDT**：`+ encoded_last_node + capacity_mapping(load)`
  4. **FF**：`q_refined = feed_forward(mh_atten_out) + mh_atten_out`
  5. logits：`q_refined @ encoded_nodes^T / sqrt(d) - log(cur_dist)`，再 `tanh` clip，最后加 mask softmax。

## 4. POMO 初始化策略（平台实现）

`ReLDPolicy.forward()` 固定：

- `selected_count == 0`：选 depot（index 0）
- `selected_count == 1`：选 `1..pomo_size` 作为多轨迹起点（与 author 代码中“forcing_first_step / top-K 起点”策略不同）
- `selected_count >= 2`：走 ReLD decoder + `get_post_search_strategy()`

如果要严格复现论文里 “top-K first move”（或 author 代码的随机/强制起点策略），需要在这里改动（或在 env/module 提供 `start_node`）。

## 5. Checkpoint/参数名对齐注意事项

- 本仓库 `final/utils/utils.py:load_model()` 含针对 ReLD 的多策略 key 匹配（移除 `encoder.`/`decoder.` 前缀、`encoder.layers -> layers`、`decoder.feed_forward -> feed_forward` 等）。
- 因此：**不要随意重命名模块属性/层名（特别是 `embedding_depot/embedding_node/layers/W1/W2/capacity_mapping/Wq_last/Wk/Wv/multi_head_combine`）**，否则需要同步更新 `load_model()` 的匹配逻辑。

