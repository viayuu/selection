# Q&A：移植新模型到 EasyNCO（`final/`）时如何做“模型参数转换”

## 问题

如果我想移植一个新的模型到平台中，除了写方法代码，“模型参数转换”（checkpoint/权重转换）应该如何做？

## 回答

在 EasyNCO（`final/`）里，“模型参数转换”的核心是：**让外部 checkpoint 里的参数，能以正确的 key/shape/语义装载到你在平台里实现的 `Policy` 上**。平台评测路径是 `final/eval.py` 实例化 policy 后调用 `final/utils/utils.py:load_model()`，因此你的转换要么离线生成平台可直接加载的权重文件（推荐），要么在 `load_model()` 里做在线兼容。

---

## 0) 平台是怎么加载权重的（你要对齐的接口）

- `final/eval.py`：
  1. `policy = hydra.utils.instantiate(model_settings.model)`
  2. `policy = load_model(policy, full_model_path, device=cuda_device, model_name=cfg.model)`
- `final/utils/utils.py:load_model()` 当前支持的单模型 checkpoint 形态：
  - `{'state_dict': ...}`（平台 Lightning 常见格式）
  - `{'model_state_dict': ...}`（部分作者仓库格式；ReLD 已兼容）
  - 直接就是 `state_dict`（`{key: tensor, ...}`）
  - 特例：`model_name == 'nlns'` 时按目录加载多个 `.pt`

因此你“转换后的目标产物”建议是：**保存成 `{'state_dict': converted_state_dict}`，其中 `converted_state_dict` 的 key 尽量与 `policy.state_dict().keys()` 一致**（最省心）。

---

## 1) 先判断：你到底需不需要“转换”

### 场景 A：在平台内训练新模型（从零训练/继续训练）

如果你用 `final/train.py` 在平台内训练，Lightning 保存出来的权重本来就与平台实现一致，通常不需要额外转换；你只需要在 `final/settings/<your_model>_settings.yaml` 的 `test_loader` 指向训练产物即可。

### 场景 B：移植作者仓库预训练权重到平台（对齐复现/做 benchmark）

这是“模型参数转换”的主要场景：你要把**外部 checkpoint → 平台 `Policy` 可加载的 checkpoint**。

---

## 2) 推荐做法：离线转换（一次性生成平台权重）

离线转换的好处是：mapping 关系固定、可审计、可复现；`load_model()` 不需要写越来越多的兼容分支。

### Step 0：优先保证“结构一致”

转换不是万能胶：如果平台实现和作者实现的结构/超参数不一致，必然出现 shape mismatch 或效果严重偏离。

常见导致结构不一致的点：

- `embed_dim/num_heads/num_layers/ffn_hidden` 不一致
- attention 的 q/k/v 是“融合”还是“拆分”
- LayerNorm/BatchNorm 的位置、是否存在 normalization
- 输出 head/critic/value head 的结构差异

建议：先让你平台版模型在 forward 逻辑上尽量贴近原仓库（哪怕代码风格不同），再做权重转换。

### Step 1：拿到“目标端”参数清单（平台 policy 的 state_dict）

在平台里 instantiate 你的 policy（超参数与作者一致）：

- `tgt_sd = policy.state_dict()`
- 记录每个 key 的 shape：`{k: v.shape for k, v in tgt_sd.items()}`

这一步决定了“你需要填满哪些参数槽位”。

### Step 2：拿到“源端”参数清单（作者 checkpoint 的 state_dict）

读取 checkpoint，并抽取源参数字典：

- `raw = torch.load(src_path, map_location='cpu')`
- `src_sd = raw.get('state_dict') or raw.get('model_state_dict') or raw`

做第一轮通用清洗（常见前缀）：

- 去掉 DDP：`module.`
- 去掉外壳：`policy.` / `model.`（取决于原仓库保存方式）

### Step 3：建立 key 映射（mapping）

最稳妥的策略是：**显式维护一个 mapping：target_key → source_key**。

你也可以先写“规则生成候选 key”，再把最终确认过的匹配结果固化下来。平台里 ReLD 的在线兼容就是类似思路（`final/utils/utils.py:load_model()` 里对多个 `possible_names` 做尝试）。

强烈建议：转换脚本里对每个 target_key 都做“找到 source_key 且 shape 一致”的强校验，避免 silent wrong load。

### Step 4：处理非一一对应的参数（拼接/拆分/转置/reshape）

这是移植最关键的一步，常见情况如下：

#### 4.1 `QKV` 融合 ↔ 拆分

- 源端：`W_q/W_k/W_v` 三个线性层；目标端：一个 `W_qkv`
  - `W_qkv = torch.cat([W_q, W_k, W_v], dim=0)`（按输出维拼接）
  - `b_qkv = torch.cat([b_q, b_k, b_v], dim=0)`
- 源端融合、目标端拆分：反向 `chunk(3, dim=0)` 即可

#### 4.2 attention 的实现差异导致 reshape

有的实现会把权重按 head 拆维保存/计算（例如 `[n_heads, head_dim, ...]`），另一些是平铺矩阵。处理方式是：写清楚数学等价关系，然后用 `view/reshape/transpose` 做对齐；这类转换务必配合 Step 6 的“同输入对齐输出”验证。

#### 4.3 线性层转置

PyTorch `nn.Linear` 默认 weight 是 `[out_features, in_features]`。如果源权重是反的（较少见），需要 `W = W.t()`。

#### 4.4 位置编码/Embedding

- 可学习 PE：需要按 key 正常加载
- 函数生成 PE：通常不需要从 ckpt 加载
- 若你在平台里改了 PE 机制，优先回到 Step 0（结构一致）解决

### Step 5：组装 `converted_state_dict` 并保存平台格式

建议流程：

1. 遍历 `policy.state_dict()` 的每个 key：
   - 找 source key（mapping 或规则）
   - 取源 tensor 并做必要变换
   - 强校验 shape 一致
2. 保存：
   - `torch.save({'state_dict': converted_state_dict}, dst_path)`

把产物放在 `final/pretrained/`，并在你的 `final/settings/<your_model>_settings.yaml` 里配置：

```yaml
test_loader:
  model_dirpath: pretrained
  model_filename: <your_model>.pt
```

---

## 3) 验证转换正确性（非常重要）

验证建议从强到弱：

1. **同输入对齐输出（最强）**：同时跑原仓库模型与平台模型（加载转换后的权重），对同一个 batch 比较：
   - logits/probabilities 的 max/mean diff
   - greedy decoding 的 tour/成本是否一致或高度一致
2. **中间层对齐（次强）**：hook 若干关键层输出（embedding、encoder layer、decoder attention）比较误差
3. **端到端指标对齐（弱但实用）**：在一小批固定实例上，对齐论文/原仓库的 cost/gap（允许很小波动）

如果第 1 条能过，基本就说明“参数语义转换”正确；如果只能做到第 3 条，需要警惕：可能是环境/增强/解码策略差异，而不一定是权重错误。

---

## 4) 在线兼容 vs 离线转换：平台内怎么选？

### 离线转换（推荐）

- 一次性生成标准权重文件，后续评测路径最干净；
- 映射可版本化，出错更容易定位。

### 在线兼容（在 `load_model()` 里写兼容逻辑）

适用：

- 你要兼容很多不同来源/命名风格；
- 或者模型像 NLNS 一样需要加载多个文件并组装结构。

落地：

- 在 `final/utils/utils.py:load_model()` 中新增 `model_name == '<your_model>'` 的分支，或扩展更通用的 key 匹配策略；
- 仍建议保留强校验（missing key/shape mismatch 直接 fail）。

---

## 5)（可选）离线转换脚本骨架（示意）

> 下面是思路骨架，不是可直接运行的完整脚本；关键点是：抽取源 state_dict → 清洗 key → 显式映射/变换 → 强校验 → 保存平台格式。

```python
import torch

def load_src_state_dict(src_path: str) -> dict:
    raw = torch.load(src_path, map_location="cpu")
    sd = raw.get("state_dict") or raw.get("model_state_dict") or raw
    out = {}
    for k, v in sd.items():
        k = k.replace("module.", "")
        k = k.replace("policy.", "")
        k = k.replace("model.", "")
        out[k] = v
    return out

def convert(src_path: str, dst_path: str, policy) -> None:
    src_sd = load_src_state_dict(src_path)
    tgt_sd = policy.state_dict()

    mapping = {}  # target_key -> source_key（建议最终显式固化）
    converted = {}

    for tgt_key, tgt_tensor in tgt_sd.items():
        src_key = mapping.get(tgt_key, tgt_key)
        w = src_sd[src_key]
        # TODO: qkv merge/split/reshape/transpose if needed
        assert w.shape == tgt_tensor.shape, (tgt_key, w.shape, tgt_tensor.shape)
        converted[tgt_key] = w

    torch.save({"state_dict": converted}, dst_path)
```

如果你提供“原仓库 checkpoint 的 key 列表（前 20 个）”和“平台 policy.state_dict() 的 key 列表（前 20 个）”，就可以很快确定 mapping 规则，以及是否需要 qkv 拼接/拆分等结构性变换。
