# Neural Solver Selection for Combinatorial Optimization — 作者论文代码实现解读（`paper.md` ↔ `neural-solver-selection/`）

> 本文档只解释作者论文对应的实现：`neural-solver-selection/`。根目录下的 `final/` 与 `my/` 是你自己的独立项目/原型，与作者论文实现无关，不在本文档范围。

---

## 1. 论文要解决的问题：为什么需要“选择器”

论文核心观察是：**不同神经求解器在不同实例上表现互补**。也就是说，单一 solver（哪怕在均值指标上最强）也会在某些实例上明显落后于其他 solver；如果我们能在**实例级别**决定“把这个实例交给哪个 solver（或一组 solver）”，就有机会把整体平均 gap 拉下来，同时只付出少量额外开销（选择器推理时间 + 可能多跑几个 solver 的时间）。

作者把这个问题形式化为一个“Neural Solver Selection”问题：给定实例 `x`，输出一个对 solver 集合的打分/排序 `s(x)`，然后用一套**选择策略**（top-1/top-k/rejection/top-p）把 `s(x)` 转换成最终使用哪些 solver，并在离线的 `gap/time` 上评估整体性能。

对应到代码，作者实现的关键点是：**训练监督来自离线标签**，并且**测试时也主要做离线模拟评估**，不需要在线真正调用求解器求解（求解器结果已经固化在 `raw_label.pkl`）。

---

## 2. 总体框架：三部分（Feature extraction / Selection model / Selection strategy）

论文提出的通用协调框架包含三块（在代码中都有可对应的实现）：

1) **Feature extraction（特征提取）**  
把 COP 实例（图/点集）转换成一个向量表示，可选手工特征或学习型 encoder。

2) **Selection model（选择模型）**  
输入特征，输出一个长度为 `|S|` 的分数向量（每个维度对应一个 solver），分数越高表示越适合把该实例交给这个 solver。

3) **Selection strategy（选择策略）**  
把“分数向量”变成“真正的选择动作”：只选 top-1？选 top-k 并取最优？对低置信度实例回退到 top-k？用 top-p 自适应集合大小？等等。

代码对应关系（先给总览，后面展开）：

- Feature extraction：`neural-solver-selection/dataset.py:manual_features()`、`neural-solver-selection/model.py:Naive_Encoder`、`neural-solver-selection/model.py:Encoder_h`
- Selection model：`neural-solver-selection/model.py:Selection_model`（以及仅手工特征版本 `Naive_classifier`）
- Selection strategy：`neural-solver-selection/trainer.py:test()` 中的 top-k / rejection / top-p 离线计算逻辑

---

## 3. 数据与监督：离线标签如何进入训练与测试

### 3.1 实例数据：`dataset.pkl`

作者训练/验证主要用合成数据，文件形式是 `dataset.pkl`：

- 路径：`neural-solver-selection/datasets/*/dataset.pkl`  
  例如：`neural-solver-selection/datasets/TSPtrain/dataset.pkl`、`neural-solver-selection/datasets/CVRPval/dataset.pkl` 等。

生成方式见：

- `neural-solver-selection/datasets/generate_data.py`：读取 `data_config.yml`，按分布/容量类型采样问题规模，调用 `data_utils.py` 生成样本并保存。
- `neural-solver-selection/datasets/data_utils.py`：
  - `generate_tsp_data()`：支持 `Gaussian/Explosion/Rotation` 等分布，最终归一化到 `[0,1]^2`（通过 `MinMaxScaler`）。
  - `generate_vrp_data()`：先生成 customer 坐标 + depot 坐标，再生成 demand 与 capacity（`scale/triangular/random_triangular` 等规则），最后保存成字典结构：`{loc, depot, demand}`。

### 3.2 监督标签：`raw_label.pkl`

作者的监督信号来自“离线跑 solver zoo 得到的结果汇总”，保存在：

- 路径：`neural-solver-selection/datasets/*/raw_label.pkl`
- 每个实例（按字符串 key，如 `"0"`, `"1"`）对应一个字典，至少包含：
  - `cost`: `List[float]`，长度为 solver 数（该实例每个 solver 的目标值）
  - `time`: `List[float]`，长度同上（该实例每个 solver 的时间）
  - `gap`: `List[float]`，长度同上（若提供最优值则可计算；否则为 0）
  - `ind`: `int`，`argmin(cost)`，即最优 solver 的索引（用作分类标签）

汇总脚本见：

- `neural-solver-selection/datasets/process_raw_label.py`：读取 `results/result_*.txt`，把各方法的 `cost/time` 归档到 `raw_label.pkl`；可选读取 `result_opt.txt` 计算 gap；最后计算 `ind = argmin(cost)`。

> 这一步解释了为什么作者代码里“没有真正的 neural solver 训练”：训练时只需要这些离线的 `cost/time/gap` 向量作为监督与评估依据。

### 3.3 统一加载：`utils.py:prepare_dataset()`

训练/测试都通过 `neural-solver-selection/utils.py:prepare_dataset(problem_type, name)` 来读数据与标签：

- 永远会读训练集：
  - `datasets/{problem_type}train/dataset.pkl`
  - `datasets/{problem_type}train/raw_label.pkl`
- 测试集（实际上是 val/test/LIB 之一）取决于 `name`：
  - 如果 `name` 包含 `"LIB"`：读 `datasets/{problem_type}LIB/...`
  - 否则如果包含 `"test"`：读 `datasets/{problem_type}test/...`
  - 否则默认读 `datasets/{problem_type}val/...`

该函数返回四个对象：

- `train_instance_set`: `List[Tensor/Dict]`
- `train_label_set`: `List[[ind, cost_vec, time_vec, gap_vec]]`
- `test_instance_set`
- `test_label_set`

其中 CVRP 会在加载后做一次结构统一：

- `utils.py:process_instance_CVRP()`：把 `{depot, loc, demand}` 拼成形如 `(N+1, 3)` 的张量（含 depot 节点与 demand 通道），以便与 TSP 一样走同一套 encoder。

---

## 4. Dataset 与 DataLoader：模型实际看到的张量长什么样

### 4.1 `SelectionDataset`

`neural-solver-selection/dataset.py:SelectionDataset` 的 `__getitem__` 返回：

- 无手工特征：`[instance_tensor, label_tuple, index]`
- 有手工特征：`[instance_tensor, label_tuple, manual_feature, index]`

手工特征在初始化 `SelectionDataset(..., manual_feature=True)` 时预先批量计算（会比较慢，但训练中不再重复算）。

### 4.2 数据增强（8-fold）

当 `data_aug=True`，`SelectionDataset` 会对每个实例做一个几何 8 折增强：

- 实现：`neural-solver-selection/utils.py:augment_xy_by_8_fold()`
- 对 TSP 是 `(x,y)` 的翻转/交换组合
- 对 CVRP，会对坐标增强，同时把需求通道复制 8 份拼回去

增强后的 label（包括 cost/time/gap）简单复制 8 份，保持监督一致。

### 4.3 `collate_fn`：padding + mask

`neural-solver-selection/dataset.py:collate_fn()` 在 batch 维度把变长实例 pad 到 `max_length`，并返回关键张量：

- `batch_x`: `[B, maxN, node_dim]`
- `batch_y`: `[B]`（最优 solver 的索引）
- `batch_cost/batch_time/batch_gap`: `[B, num_solvers]`
- `lengths`: `[B]`（每个实例真实节点数）
- `ninf_mask`: `[B, maxN]`，pad 的位置是 `-inf`，真实节点位置是 `0`

后续 encoder 在 attention 里会把 `ninf_mask` 加到 logits 上，从而屏蔽 pad 节点。

---

## 5. Feature extraction：三种实现路径（论文 3.1）

### 5.1 手工特征：`manual_features()`

代码：`neural-solver-selection/dataset.py:manual_features(nodes)`

它是从另一篇关于 TSP difficulty 的工作改写而来，对点集计算统计特征，例如：

- 距离矩阵的 std、最近邻距离的 std
- 点集 centroid、平均 radius
- 距离取整后的 distinct count
- HDBSCAN 聚类得到的 cluster/outlier 比例、cluster 半径等
- CVRP 额外加 demand 的均值/方差（因此 CVRP 特征维度为 11；再加 scale 变成 12）

在作者框架里，手工特征作为 Feature extraction 的一种对照路线，对应后面的 `Naive_classifier`（见第 6.2 节）。

### 5.2 学习型编码器 1：`Naive_Encoder`

代码：`neural-solver-selection/model.py:Naive_Encoder`

这是一个“全连接 self-attention 编码器”，流程是：

1) 对每个节点做线性嵌入（TSP 输入 2 维坐标；CVRP 输入 3 维：坐标+需求）  
2) 堆叠若干层 `EncoderLayer`（multi-head attention + FFN + Add&Norm）  
3) 用 mask 把 pad 节点 embedding 置零，然后对节点维度做 mean pooling 得到 `graph_emb`

输出：`graph_emb ∈ R^{B×d}`，作为实例向量表示。

### 5.3 学习型编码器 2：`Encoder_h`（层次化 pooling）

代码：`neural-solver-selection/model.py:Encoder_h` 与 `Encoder_block_h`

这是论文重点强调的 OOD 泛化增强路线（通过分层/下采样构建多尺度表示）。它的核心结构是：

- 一个 `Encoder_h` 由多个 `Encoder_block_h` 堆叠（数量由 `block_num` 控制）。
- 每个 block 内：
  1) 先过若干层 `EncoderLayer` 得到节点 embedding；
  2) 做一次 readout：对节点 embedding 做 masked mean 与 masked max，再拼接并过 GELU，得到该层的 `graph_emb`；
  3) 做一次 downsampling：额外用一层注意力 `layer_score` 计算每个节点的代表性分数 `scores`，选 top `downsample_ratio` 的节点保留，并把分数加到 embedding 上（增强可分离性），形成下一层更小的节点集合。

这样，节点数会逐 block 减少，形成粗到细/细到粗的层次结构。编码器输出的实例表示用于后续选择模型。

> 论文描述里通常会把多层 readout 聚合成最终表示；代码里也有 `graph_emb_h` 的累加变量，但 `Encoder_h.forward()` 最终 `return` 的是最后一次 readout 的 `graph_emb`（见 `neural-solver-selection/model.py:176-196`）。理解与复现时，以代码实际行为为准。

---

## 6. Selection model：从实例表示到 solver 分数（论文 3.2）

### 6.1 基础版：`Selection_model`（固定 solver index）

代码：`neural-solver-selection/model.py:Selection_model`

基本流程是：

1) 用 encoder 得到 `graph_emb`（实例向量）
2) 拼接一个规模特征 `scales`（本质是 `N` 或有效节点数，用 `scales[:, None]` 加进来）
3) 过一个 MLP 输出 `output_dim=num_solvers` 维的分数向量

在 `Selection_model.__init__()` 里，作者根据 `pooling` 开关选择编码器：

- `pooling=False` → `Naive_Encoder`，`feature_dim = embedding_dim + 1`
- `pooling=True` → `Encoder_h`，`feature_dim = 2*embedding_dim + 1`

（`+1` 对应拼接的 `scales` 维度）

### 6.2 手工特征版：`Naive_classifier`

当配置 `train_params.manual_feature=True`，`run.py` 会改用：

- `neural-solver-selection/model.py:Naive_classifier`

它不再走图 encoder，而是直接把 `manual_features` 与 `scales` 拼起来，再用 MLP 输出分数。

这对应论文里 “Feature extraction 用手工特征” 的对照实验设置。

### 6.3 可选扩展：Neural Solver Features（`ns_feature=True`）

论文还讨论了“给每个 solver 一个可学习的 solver embedding/token”，让选择模型不仅看到“实例特征”，还看到“solver 特征”。作者实现的方式大致是：

1) 为每个 solver 选一小组“代表性实例”（representative instances）  
2) 用一个共享 encoder 把这些代表实例编码成向量  
3) 用 attention 把这组向量汇聚成一个 solver token  
4) 选择时用 `instance_feature` 与 `solver_token` 的融合/相似度输出该 solver 的分数

对应代码链路：

- 选代表实例：`neural-solver-selection/utils.py:representative(dataset, labels, ratio)`  
  - 对每个 solver k，仅在 “k 是最优 solver” 的实例子集里挑样本；
  - 用 `best_cost/second_best_cost` 的比值做排序，取最小的前 `ratio` 部分作为代表实例（直觉：这些实例上该 solver 相对优势更明显）。
- 计算代表实例特征：`neural-solver-selection/trainer.py:compute_representative()`  
  - 这些实例也走 `SelectionDataset + collate_fn`；
  - 用 `encoder_p`（动量 encoder）得到 embedding，并拼接 `scales`。
- 动量更新：`neural-solver-selection/trainer.py:_momentum_update_representative_encoder()`  
  - `encoder_p = m*encoder_p + (1-m)*encoder`，并且 `encoder_p` 不参与梯度更新。
- 从代表实例汇聚 token：`neural-solver-selection/model.py:representative_net`  
  - 先把 `model_token` 与代表实例特征拼起来做 self-attention；
  - 再做一次 cross-attention，把代表实例信息聚合到 token（最终输出一个向量）。
- 用 token 打分：`Selection_model.forward()`  
  - 先得到 `instance_feature = [graph_emb || manual/scales]`；
  - 对每个 solver i，拼 `instance_feature || model_token[i]`，过 `similarity` MLP 得到该 solver 分数。
- token 的更新入口：`Selection_model.update_tokens()`  
  - 在 `trainer.train_one_epoch()` 与 `trainer.test()` 里，如果启用 `representative_set`，每个 epoch/batch 会先计算 representative_feature，再 `model.update_tokens(...)`。

这一套机制对应论文“advanced neural solver features”的实现落地。

---

## 7. 训练目标：Classification vs Ranking（论文 3.2）

作者实现了两种训练损失（由配置 `train_params.loss` 决定）：

### 7.1 分类损失（CE/NLL）

把“选择”当成多类分类：

- 标签：`ind = argmin(cost)`（离线最优 solver 的索引）
- 实现：
  - `trainer.run()` 里 `self.criterion = torch.nn.NLLLoss()`
  - `train_one_epoch()` 里 `l = NLLLoss(log_softmax(y_pred), y)`

优点：目标简单直接；缺点：只强调 top-1，忽略“次优 solver”信息，可能导致选错时性能掉得更厉害（论文也在文字里提到这一点）。

### 7.2 排序损失：`RankingLoss`

代码：`neural-solver-selection/loss.py:RankingLoss`

它用离线 `cost` 向量提供更丰富的监督：不仅要把最优 solver 打到最高分，也希望模型学到一个较稳定的排序。

实现方式是一个“逐层剥离最优项”的 listwise 训练：

- 第 0 轮：在所有 solver 里把 `argmin(cost)` 当作 label，做一次 softmax/NLL
- 第 1 轮：从集合里去掉已经最优的那个 solver，再在剩下的 solver 中找当前最优（即全局 second-best），再做一次 softmax/NLL
- … 重复直到达到 `top_k` 轮（作者训练时设成 `num_solvers`，即把整个排序都监督到）

在 `trainer.run()` 里对应：

- `self.criterion = RankingLoss(num_ns, num_ns)`
- `train_one_epoch()` 里 `l = RankingLoss(y_pred, cost)`

---

## 8. 训练/测试主流程：`run.py` → `trainer.py`

### 8.1 配置加载与日志

入口：`neural-solver-selection/run.py`

- 如果 `--load` 存在：从 `train_logs/{load}/config.json` 读配置并恢复训练
- 否则：从 `--config_name` 读 YAML 配置
- 训练模式：创建 `train_logs/{config}_{loss}_{seed}`，写 `config.json`，日志写到 `log.csv`
- 测试模式（`--test_file` 非空）：把 `num_epochs=0`，输出写到 `results/{exp_name}.csv`

### 8.2 数据准备

`run.py` 里调用：

- `prepare_dataset(config['problem_type'], name=name)`  
  其中 `name` 在训练时是 `config['name']`，在测试时是 `args.test_file`（比如 `"TSPLIB"`/`"CVRPLIB"` 一类）。

然后构造：

- `SelectionDataset(..., manual_feature=..., data_aug=...)`

### 8.3 模型初始化

`run.py` 中模型的三种分支：

- `manual_feature=True`：`model = Naive_classifier(...)`
- 否则：`model = Selection_model(...)`
- 如果 `ns_feature=True`：先临时建一个 `Selection_model` 取出 encoder 作为 `encoder_representative`，并构造 `representative_set = representative(train_set, train_label)` 供 trainer 在训练/测试里更新 token

### 8.4 训练循环与 checkpoint

训练逻辑在 `neural-solver-selection/trainer.py:trainer.run()`：

- 先 `test()` 一次得到基线结果
- 每个 epoch：
  - `train_one_epoch()`
  - `test()`（含策略评估）
  - 写 CSV / 可选 wandb
  - 每隔 `save_interval`，如果 `top_1`（top-1 gap）变好则保存 `checkpoint_epoch_best.pt`

> 注意：这里保存“最好”的标准不是分类准确率，而是 `top_1` gap（见 `trainer.run()` 保存逻辑）。

---

## 9. Selection strategies：如何用分数“离线模拟”组合求解（论文 3.3）

代码入口：`neural-solver-selection/trainer.py:test()`

`test()` 做两件事：

1) 统计分类指标（acc/recall/precision）
2) 更重要：用离线 `gap/time` 向量 + 预测的分数向量，模拟不同选择策略的“最终 gap/time”

### 9.1 为什么可以离线模拟

对每个实例 `x`，离线标签给出了：

- `gap[x, i]`：如果选 solver i，gap 是多少
- `time[x, i]`：如果选 solver i，时间是多少

那么只要策略决定“选哪些 i”，就可以直接做：

- top-k 的 gap：`min_i gap[x, i]`
- top-k 的 time：`sum_i time[x, i]`

这就避免了测试时在线跑 solver 的成本，也能清晰控制“额外开销”来自哪里。

### 9.2 Top-1 / Top-k

实现片段（概念上）：

- `score_mat = softmax(y_pred)` 得到每个实例对每个 solver 的概率
- `topk_ind = score_mat.topk(k)` 得到要选的 solver 子集
- `topk_gap = gap_mat.gather(...).min(dim=1)`
- `topk_time = time_mat.gather(...).sum(dim=1)`

作者会输出 `k=1..4` 的结果，并把选择器推理耗时 `select_time` 加到 time 里，反映“额外开销”。

### 9.3 Rejection-based（置信度回退）

实现思路：

- 用 `score_mat.max(dim=1)[0]` 作为“置信度”（top-1 概率越大，越自信）
- 对实例按置信度排序
- 设定覆盖率（代码默认 0.8）：前 80% 高置信度实例用 top-1，后 20% 低置信度实例改用 top-k（通常 k=2）

这样可以在不显著增加平均时间的情况下，降低因为选错 top-1 导致的大 gap。

### 9.4 Top-p（自适应集合大小）

实现思路：

- 对每个实例，从 top-1 开始逐步扩大集合 `{top-1, top-2, ...}`
- 直到这些 solver 的概率和 ≥ p（代码默认 p=0.8）
- 用该集合做 `min gap / sum time`

它相当于把 “k” 变成随实例置信度变化的量：越不确定，集合越大。

### 9.5 两个对照基线：Single best 与 Oracle

`test()` 里会额外打印：

- **Single best**：在整个测试集上，选一个固定 solver，使其平均 gap 最小（这相当于“不做选择”的最优常数策略）。
- **Oracle**：对每个实例都选真实最优 solver（`min_i gap[x,i]`）；这是理论上限。

这两个值用来衡量选择器的空间：你距离 Oracle 差多远？比 single best 提升多少？

---

## 10. 一句话串起：作者代码到底在做什么

从工程角度，可以把作者实现浓缩成下面这条流水线：

1) 准备合成训练集实例 `dataset.pkl` 与离线标签 `raw_label.pkl`（每实例一条 `cost/time/gap` 向量）  
2) 训练一个 `instance encoder + MLP`，让它输出 solver 分数；用分类或排序损失对齐离线标签  
3) 测试时对每个实例输出分数，然后用 top-k / rejection / top-p 等策略把分数转成“要选的 solver 子集”，并在离线 `gap/time` 上计算整体效果（同时计入选择器推理时间）

这正是论文 “Feature extraction + Selection model + Selection strategy” 三组件框架在代码里的对应落地。

