# NSS 论文实现讲解：模型设计与训练流程

---

## 1. 先用一句话说明这篇论文到底在做什么

NSS 论文《Neural Solver Selection for Combinatorial Optimization》做的不是“训练一个 solver 去直接解 TSP / CVRP”，而是：

> 先准备一个 solver zoo，离线跑出每个实例上各个 solver 的结果，  
> 再训练一个 **selector**，让它看到实例以后，预测“这个实例更适合哪个 solver”。

所以它的本质是：

- 输入：一个实例
- 输出：每个 solver 的 compatibility score
- 训练方式：**监督学习**
- 推理方式：greedy / top-k / rejection / top-p 选择策略

这和你现在的 `2cmab2` 很不一样。

- NSS：离线全信息监督学习
- 你的 `2cmab2`：bandit / online 风格决策 + partial feedback

---

## 2. 论文里的三组件，在代码里分别对应什么

论文把整个框架拆成 3 个组件：

1. **Feature extraction**
2. **Selection model**
3. **Selection strategy**

在源码里，大致对应成下面这张表。

| 论文组件 | 代码里的对应 |
| --- | --- |
| Feature extraction | `model.py` 里的 `Naive_Encoder` / `Encoder_h` / `manual_features()` |
| Selection model | `Selection_model` / `Naive_classifier` |
| Selection strategy | `trainer.py::test()` 里的 top-1, top-k, rejection, top-p |

最核心的几个文件是：

- `run.py`：总入口
- `utils.py`：读数据、构造标签、辅助函数
- `dataset.py`：数据集包装、padding、手工特征
- `model.py`：编码器和 selector 主模型
- `loss.py`：ranking loss
- `trainer.py`：训练与评测

---

## 3. 这套实现首先不是 RL，而是标准监督学习

这一点要先完全想清楚。

论文主实现不是：

- 在线交互
- 每次选一个 solver 再拿 reward 更新
- 或者 policy gradient / REINFORCE

它做的是：

1. 先准备大量实例
2. 对每个实例，把 solver zoo 全部跑一遍
3. 记录每个 solver 的：
   - `cost`
   - `time`
   - `gap`
   - `ind`（最佳 solver 下标）
4. 用这些结果当监督信号，训练一个 selector

所以它是一个非常典型的：

\[
\text{instance} \rightarrow \text{score vector over solvers}
\]

的监督学习模型。

对应代码入口：

```python
# run.py
train_set, train_label, test_set, test_label = prepare_dataset(config['problem_type'], name=name)
train_dataset = SelectionDataset(...)
test_dataset = SelectionDataset(...)
```

源码位置：

- `run.py:89-94`
- `utils.py:48-91`

---

## 4. 监督数据和标签到底是怎么来的

### 4.1 原始实例数据

实例数据在：

- `datasets/TSPtrain/dataset.pkl`
- `datasets/TSPval/dataset.pkl`
- `datasets/TSPtest/dataset.pkl`
- `datasets/TSPLIB/dataset.pkl`
- `datasets/CVRPtrain/dataset.pkl`
- `datasets/CVRPval/dataset.pkl`
- `datasets/CVRPtest/dataset.pkl`
- `datasets/CVRPLIB/dataset.pkl`

我已经实际检查过这几个数据文件的对象结构。

#### TSP 的样本格式

TSP 的每条样本是一个 tensor：

\[
[1, N, 2]
\]

其中：

- `1`：batch 维
- `N`：节点数
- `2`：每个节点的 `(x, y)`

例如训练集中真实抽样结果就是：

```python
torch.Size([1, 362, 2])
```

#### CVRP 的样本格式

CVRP 原始样本不是 tensor，而是字典：

```python
{
    'loc':    [1, N, 2],
    'demand': [1, N],
    'depot':  [1, 1, 2]
}
```

然后在 `utils.py::process_instance_CVRP()` 中被统一整理成：

\[
[1, N+1, 3]
\]

三列分别是：

- `x`
- `y`
- `demand`

其中第 0 个点是 depot，需求会补一个 0。

对应代码：

```python
def process_instance_CVRP(instance):
    depot = instance['depot']
    loc = instance['loc']
    demand = instance['demand']

    xy = torch.cat((depot, loc), dim=1)
    demand = torch.cat((torch.zeros(1, 1), demand), dim=1)
    batch_x = torch.cat((xy, demand[:, :, None]), dim=2)
    return batch_x
```

源码位置：

- `utils.py:37-46`

---

### 4.2 标签数据

标签文件是 `raw_label.pkl`。

每个实例的标签格式是：

```python
{
    'cost': [...],
    'time': [...],
    'ind':  ...,
    'gap':  [...],
}
```

其中：

- `cost[i]`：第 `i` 个 solver 在该实例上的目标值
- `time[i]`：第 `i` 个 solver 的运行时间
- `gap[i]`：相对最优已知解的 gap
- `ind`：`argmin(cost)`，也就是该实例上的最优 solver 编号

我实际抽样检查过，格式和代码完全一致。

例如 TSP 的一条真实标签是：

```python
{
    'cost': [9.1560, 9.3641, 8.9111, 9.0352, 9.0667, 9.0042, 9.1334],
    'time': [2.8162, 1.3193, 1.6398, 3.1444, 2.5375, 2.1903, 2.2012],
    'ind': 2,
    'gap': [0, 0, 0, 0, 0, 0, 0]
}
```

对应构造逻辑在：

```python
train_label_set.append([
    train_raw_labels[key]['ind'],
    train_raw_labels[key]['cost'],
    train_raw_labels[key]['time'],
    train_raw_labels[key]['gap']
])
```

源码位置：

- `utils.py:79-85`

---

### 4.3 标签是如何从 solver 结果文件加工出来的

这个过程在：

- `datasets/process_raw_label.py`

它会读类似：

- `result_LEHD.txt`
- `result_ELG.txt`
- `result_bq.txt`
- `result_opt.txt`

这样的结果文件，然后按实例 id 把多个 solver 的结果聚起来。

核心逻辑是：

```python
labels[line[0]]['cost'].append(float(line[1]))
labels[line[0]]['time'].append(float(line[2]))
labels[line[0]]['gap'].append(...)
labels[k]['ind'] = np.argmin(np.array(labels[k]['cost']))
```

源码位置：

- `datasets/process_raw_label.py:35-59`

所以论文里的监督信号不是人工标注，而是：

> “solver zoo 对这个实例的离线跑分结果”

---

### 4.4 训练集、验证集、测试集、基准集到底是怎么构造的

这一部分是理解 NSS 最重要的地方之一。

论文源码里并不是只准备一个 `train` 和一个 `test`，而是准备了 4 类集合：

1. `train`
2. `val`
3. `test`
4. `LIB benchmark`

对应目录分别是：

- `datasets/TSPtrain`
- `datasets/TSPval`
- `datasets/TSPtest`
- `datasets/TSPLIB`
- `datasets/CVRPtrain`
- `datasets/CVRPval`
- `datasets/CVRPtest`
- `datasets/CVRPLIB`

我已经实际读取过这些文件，真实样本数如下：

| 数据集 | 样本数 |
| --- | --- |
| `TSPtrain` | 10000 |
| `TSPval` | 1000 |
| `TSPtest` | 1000 |
| `TSPLIB` | 49 |
| `CVRPtrain` | 10000 |
| `CVRPval` | 1000 |
| `CVRPtest` | 1000 |
| `CVRPLIB` | 100 |

所以可以把它们的角色理解成：

- `train`：真正参与梯度更新
- `val`：训练过程中每个 epoch 用来做验证和选 checkpoint
- `test`：合成分布下的正式测试集
- `LIB`：更像 benchmark / OOD 测试集

这说明 NSS 的“训练、验证、测试”不是从一个大表里临时随机切出来的，而是：

> 事先分别构造好、分别存盘、分别带标签

---

### 4.5 `prepare_dataset()` 是如何决定当前读哪个 split 的

这个逻辑都写在：

- `utils.py::prepare_dataset()`

关键代码是：

```python
with open(f'datasets/{problem_type}train/dataset.pkl', 'rb') as f:
    train_instance_set = pickle.load(f)

if "LIB" in name:
    test_file = f'datasets/{problem_type}LIB/dataset.pkl'
elif "test" in name:
    test_file = f'datasets/{problem_type}test/dataset.pkl'
else:
    test_file = f'datasets/{problem_type}val/dataset.pkl'
```

这里最关键的一点是：

- 训练集永远固定读 `train`
- 第二个返回值 `test_set`，其实是“当前想评测的集合”

也就是说，`prepare_dataset()` 返回的不是严格意义上的：

- `train + test`

而是：

- `train + current_eval_split`

---

#### 情况 1：训练模式

以 TSP 为例，配置文件里写的是：

```yaml
name: TSPtrain
```

因为：

- `"LIB" not in "TSPtrain"`
- `"test" not in "TSPtrain"`

所以会自动走：

```python
test_file = 'datasets/TSPval/dataset.pkl'
```

也就是说训练时真实读取的是：

- `train_set = TSPtrain`
- `test_set = TSPval`

这里变量名虽然叫 `test_set`，但它的实际语义是：

- **验证集 val**

CVRP 也是一样：

- `train_set = CVRPtrain`
- `test_set = CVRPval`

---

#### 情况 2：正式测试模式

如果你显式传：

```bash
python run.py --load ... --test_file TSPtest
python run.py --load ... --test_file CVRPtest
```

那么就会走：

```python
test_file = datasets/{problem_type}test/dataset.pkl
```

此时真正参与评测的是：

- `TSPtest`
- `CVRPtest`

---

#### 情况 3：基准库测试模式

如果你传：

```bash
python run.py --load ... --test_file TSPLIB
python run.py --load ... --test_file CVRPLIB
```

那么就会走：

```python
test_file = datasets/{problem_type}LIB/dataset.pkl
```

也就是：

- `TSPLIB`
- `CVRPLIB`

这两个集合才是论文里最重要的 benchmark 评测集。

所以一句话概括 `prepare_dataset()` 的行为就是：

> 它永远返回 `train + 一个当前要评测的集合`，  
> 而这个评测集合在训练阶段是 `val`，在正式测试阶段才是 `test / LIB`。

---

### 4.6 默认训练时，dataloader 实际看到多少样本

默认配置里：

```yaml
data_aug: True
```

而 `SelectionDataset` 在 `data_aug=True` 时会做 8-fold augmentation。

所以：

- 原始 `TSPtrain = 10000`
- 增强后训练样本数 `= 10000 x 8 = 80000`

CVRP 也一样：

- 原始 `CVRPtrain = 10000`
- 增强后训练样本数 `= 10000 x 8 = 80000`

但是：

- `val`
- `test`
- `LIB`

默认都 **不做 augmentation**。

所以训练阶段看到的是一个“增强后的大训练集”，而验证 / 测试阶段看到的都是原始样本。

如果再结合默认 batch size：

```yaml
train_batch_size: 64
```

那么每个 epoch 的训练步数大约是：

\[
80000 / 64 \approx 1250
\]

所以不要只看 `train=10000` 就以为每个 epoch 很小，实际训练时已经被 augmentation 扩大到了 8 倍。

---

### 4.7 `train / val / test / LIB` 四类集合分别承担什么角色

可以用这张表把它们彻底分清：

| 集合 | 作用 | 是否参与梯度更新 |
| --- | --- | --- |
| `train` | 训练 selector 参数 | 是 |
| `val` | 训练过程中监控表现、选模型 | 否 |
| `test` | 合成分布下的正式测试 | 否 |
| `LIB` | benchmark / 分布外测试 | 否 |

所以在 NSS 里：

- `val` 是训练时的选择依据
- `test / LIB` 是最终汇报性能时才用的集合

这点和你现在的 bandit / online 训练非常不一样，因为 NSS 的数据边界是非常清楚的：

- 训练数据和验证数据分开
- 测试集不参与训练
- benchmark 集也不参与训练

---

## 5. 训练入口 `run.py` 到底在做什么

整个入口文件其实很直白，可以概括成 6 步。

### 第 1 步：读配置

```python
with open(args.config_name, 'r', encoding='utf-8') as config_file:
    config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
```

然后把：

- `problem_type`
- `num_classes`
- `loss`
- `manual_feature`
- `ns_feature`

这些关键信息整理到 `config` 里。

源码位置：

- `run.py:33-49`

---

### 第 2 步：准备日志目录

如果是训练模式：

- 日志目录在 `train_logs/...`

如果是纯测试模式：

- 输出目录在 `results/...`

源码位置：

- `run.py:69-86`

---

### 第 3 步：读数据

```python
name = args.test_file if args.test_file is not None else config['name']
train_set, train_label, test_set, test_label = prepare_dataset(config['problem_type'], name=name)
```

这里有一个非常关键的实现细节：

当训练时，`name=config['name']`，例如：

- TSP 配置里 `name: TSPtrain`
- CVRP 配置里 `name: CVRPtrain`

而 `prepare_dataset()` 的逻辑是：

```python
if "LIB" in name:
    test_file = f'datasets/{problem_type}LIB/dataset.pkl'
elif "test" in name:
    test_file = f'datasets/{problem_type}test/dataset.pkl'
else:
    test_file = f'datasets/{problem_type}val/dataset.pkl'
```

这意味着：

- 训练时的 `test_set` 实际上是 **val 集**
- 不是论文里最终汇报的 `test` / `LIB`

这点很重要，因为很多人读 `run.py` 时会误以为 `test_dataset` 就是最终测试集。

源码位置：

- `run.py:89-91`
- `utils.py:57-77`

---

### 第 4 步：包成 `SelectionDataset`

```python
train_dataset = SelectionDataset(train_set, train_label, manual_feature=..., data_aug=...)
test_dataset = SelectionDataset(test_set, test_label, manual_feature=...)
```

这里可以打开两个可选开关：

- `manual_feature`
- `data_aug`

源码位置：

- `run.py:93-94`

---

### 第 5 步：构建模型

默认主线是：

```python
model = Selection_model(**config['model_params'])
```

只有当 `manual_feature=True` 时，才改成：

```python
model = Naive_classifier(**config['model_params'])
```

也就是说，论文主结果默认不是纯手工特征，而是图编码器版本。

源码位置：

- `run.py:104-108`

---

### 第 6 步：交给 `trainer.run()`

```python
trainer.run(train_dataset, test_dataset, representative_set, log_dir, load_path)
```

真正的训练循环、测试循环、top-k/rejection/top-p 评测，都在 `trainer.py` 里。

源码位置：

- `run.py:118-119`

---

## 6. `SelectionDataset` 到底返回什么

这个类是理解整个数据流的关键。

### 6.1 普通模式

如果 `manual_feature=False`，那么：

```python
return [self.dataset[index], self.labels[index], index]
```

也就是每条样本返回：

1. 实例本身
2. 标签 `[ind, cost, time, gap]`
3. 样本索引

源码位置：

- `dataset.py:47-51`

---

### 6.2 手工特征模式

如果 `manual_feature=True`，还会额外返回 `features[index]`：

```python
return [self.dataset[index], self.labels[index], self.features[index], index]
```

源码位置：

- `dataset.py:47-50`

---

### 6.3 训练增强

如果 `data_aug=True`，训练集会做 8-fold augmentation：

```python
aug_coords = augment_xy_by_8_fold(ins[:, :, :2])
```

然后 TSP 直接替换坐标，CVRP 则把需求列重复 8 份再拼回去。

所以训练时一条实例会扩成 8 条。

源码位置：

- `dataset.py:23-42`
- `utils.py:17-35`

---

## 7. `collate_fn()` 之后，一个 batch 长什么样

`collate_fn()` 会把不同规模实例 pad 到同一个 `N_max`。

输出是：

```python
[batch_x, batch_y, batch_cost, lengths, ninf_mask, batch_gap, batch_time, ...]
```

分别对应：

- `batch_x`: `[B, N_max, 2/3]`
- `batch_y`: `[B]`
- `batch_cost`: `[B, M]`
- `lengths`: `[B]`
- `ninf_mask`: `[B, N_max]`
- `batch_gap`: `[B, M]`
- `batch_time`: `[B, M]`
- `batch_feature`: `[B, 9/11]`，仅手工特征模式才有

这里的 `lengths` 在后面被当成：

- 规模特征 `scale`

使用。

对应代码：

```python
lengths = torch.tensor(lengths, dtype=torch.float32)
```

源码位置：

- `dataset.py:105-128`

---

## 8. 论文默认主线模型到底是什么

默认配置是：

```yaml
manual_feature: False
ns_feature: False
pooling: True
loss: rank
```

所以默认主线模型就是：

\[
\text{instance} \rightarrow \text{Encoder\_h} \rightarrow \text{instance feature} \rightarrow \text{MLP classifier} \rightarrow \text{scores}
\]

代码入口是：

```python
class Selection_model(nn.Module):
```

源码位置：

- `model.py:7-75`

---

### 8.1 从设计目标看，为什么默认主线是 `Encoder_h + scale + MLP`

如果先不看代码细节，只看模型设计目标，NSS 默认主线其实是在同时回答两个问题：

1. **怎样把一个可变规模的 TSP / CVRP 实例编码成固定长度向量？**
2. **怎样把这个固定长度向量映射成“每个 solver 的适配分数”？**

对应的设计答案就是：

1. 用图编码器 `Encoder_h` 负责实例表示
2. 用一个简单 MLP 负责 solver 打分

所以整个模型可以写成：

\[
\text{Instance Encoder} + \text{Solver Scoring Head}
\]

这和论文三组件里的前两部分正好一一对应：

- `Encoder_h`：feature extraction
- `MLP classifier`：selection model

这样的设计有 3 个直接好处。

#### 好处 1：输入规模 `N` 可以变化

TSP / CVRP 的节点数不是固定的。

例如真实数据里就可能出现：

- `N = 84`
- `N = 243`
- `N = 362`

如果直接把实例 flatten 成一个固定长度向量，模型会很不自然，也很难泛化到更大规模。

而 `Encoder_h` 通过：

- 节点 embedding
- self-attention
- pooling
- graph readout

把任意大小的图都压成固定长度表示 `[B,256]`。

这一步就是整个模型设计最核心的基础。

#### 好处 2：不仅用结构特征，还显式用规模特征

只靠图编码器得到的 `graph_emb`，更偏向表达：

- 节点空间分布
- 局部和全局结构
- 聚类和层次信息

但 solver 的适配性通常也和问题规模强相关。

例如：

- 某些 solver 更擅长小规模
- 某些 solver 在大规模 benchmark 上更稳

所以源码在 encoder 外部又额外拼了：

- `scale = N`

形成：

\[
[graph\_emb, scale]
\]

这相当于明确把“实例规模”这个信号送给选择头。

#### 好处 3：选择头尽量简单，把建模重点放在实例表示

默认选择头只是：

```python
Linear -> GELU -> Linear
```

也就是说，作者没有把最后一层做得特别复杂，而是把主要难点放在：

- “如何得到好的实例表示”

而不是：

- “如何设计一个复杂分类器”

所以默认主线很像：

\[
\text{强 encoder} + \text{轻量 scoring head}
\]

---

### 8.2 默认主线模型的整体结构图

默认主线模型可以画成下面这样：

```text
一个 batch 的实例
    -> collate_fn padding
    -> batch_x: [B, N_max, 2/3]
    -> lengths: [B]
    -> mask: [B, N_max]

batch_x, mask
    -> Encoder_h
    -> graph_emb: [B, 256]

lengths
    -> scale: [B, 1]

[graph_emb, scale]
    -> instance_feature: [B, 257]

instance_feature
    -> MLP classifier
    -> logits / scores: [B, M]
```

其中：

- TSP：`M = 7`
- CVRP：`M = 5`

这就是默认主线模型的总架构。

---

### 8.3 默认主线模型可以拆成 4 个模块

如果从代码角度看，默认主线模型其实可以拆成 4 个模块。

#### 模块 1：输入整理模块

对应：

- `SelectionDataset`
- `collate_fn`

负责把原始实例整理成统一 batch，并输出：

- `x`
- `scales`
- `mask`
- `cost`
- `gap`
- `time`

#### 模块 2：实例编码模块

对应：

- `Encoder_h`

输入：

\[
[B, N_{max}, 2/3]
\]

输出：

\[
[B, 256]
\]

#### 模块 3：特征融合模块

负责把图表示和规模特征拼起来：

\[
[B,256] + [B,1] \rightarrow [B,257]
\]

代码是：

```python
self.instance_feature = torch.cat((graph_emb, manual_features), dim=1)
```

在默认 `manual_features=None` 时，这里的 `manual_features` 实际上就是：

```python
scales[:, None]
```

#### 模块 4：solver 打分模块

负责把实例特征映射成每个 solver 的 score：

\[
[B,257] \rightarrow [B,128] \rightarrow [B,M]
\]

代码是：

```python
self.classifier = nn.Sequential(
    nn.Linear(feature_dim, model_params['embedding_dim']),
    nn.GELU(),
    nn.Linear(model_params['embedding_dim'], model_params['output_dim'])
)
```

---

### 8.4 默认主线模型的设计重点，不在 solver 内部，而在 selector 外部

这是理解 NSS 的一个核心视角。

NSS 默认主线模型并不会：

- 进入 solver 内部做联合建模
- 读取 solver 权重
- 学 solver 的逐步推理过程

它只做一件事：

\[
\text{instance} \rightarrow \text{solver score vector}
\]

也就是说，NSS 建模的对象不是：

- “如何解这个实例”

而是：

- “这个实例更适合由谁来解”

所以它的默认结构才会这么干净：

- 前面一个实例编码器
- 后面一个简单 MLP 选择头

这也是你在看 NSS 时，和看 `2cmab2` 时最大的视角差异。

---

## 9. 模型前向的总公式

如果只看默认主线，模型可以写成：

\[
I \rightarrow h(I) \rightarrow [h(I), N] \rightarrow g_\theta(I) \in \mathbb{R}^M
\]

其中：

- \(I\)：一个实例
- \(h(I)\)：图编码器输出
- \(N\)：实例规模
- \(g_\theta(I)\)：对 `M` 个 solver 的 compatibility scores

如果打开手工特征版本，则是：

\[
I \rightarrow f_{\text{manual}}(I) \rightarrow [f_{\text{manual}}(I), N] \rightarrow g_\theta(I)
\]

如果打开 `ns_feature=True`，则会再把 solver token 拼进去，变成 instance-solver pair scoring，但默认实验并不走这条线。

---

## 10. 默认前向传播的维度变化

这一部分最重要。

论文默认配置里：

- `embedding_dim = 128`
- `pooling = True`
- `block_num = 2`
- `encoder_layer_num = 2`

所以默认前向的维度流如下。

### 10.1 TSP

输入一个 batch：

\[
[B, N, 2]
\]

经过首层 embedding：

\[
[B, N, 2] \rightarrow [B, N, 128]
\]

经过层次编码器 `Encoder_h`：

\[
[B, N, 128] \rightarrow [B, 256]
\]

然后拼 `scale=[B,1]`：

\[
[B, 256] + [B, 1] \rightarrow [B, 257]
\]

再进 MLP：

\[
[B, 257] \rightarrow [B, 128] \rightarrow [B, 7]
\]

最后输出：

- TSP 的 7 个 solver score

---

### 10.2 CVRP

CVRP 先经 `process_instance_CVRP()` 变成：

\[
[B, N, 3]
\]

然后首层 embedding：

\[
[B, N, 3] \rightarrow [B, N, 128]
\]

经过 `Encoder_h`：

\[
[B, N, 128] \rightarrow [B, 256]
\]

拼 `scale=[B,1]`：

\[
[B, 256] + [B, 1] \rightarrow [B, 257]
\]

进 MLP：

\[
[B, 257] \rightarrow [B, 128] \rightarrow [B, 5]
\]

最后输出：

- CVRP 的 5 个 solver score

---

## 11. `Encoder_h` 是怎么编码图的

这是 NSS 最核心的模块。

### 11.1 第一层：节点 embedding

先把每个节点的原始输入映射到 128 维。

TSP：

```python
self.embedding = nn.Linear(2, embedding_dim)
```

CVRP：

```python
self.embedding = nn.Linear(3, embedding_dim)
```

所以：

- TSP：`[B,N,2] -> [B,N,128]`
- CVRP：`[B,N,3] -> [B,N,128]`

源码位置：

- `model.py:147-158`

---

### 11.2 第二层：多个 hierarchical blocks

默认有 2 个 `Encoder_block_h`：

```python
self.blocks = nn.ModuleList([Encoder_block_h(**model_params) for _ in range(model_params['block_num'])])
```

每个 block 里面做两件事：

1. 用若干个 attention layer 更新节点表示
2. 做一次 pooling downsample

源码位置：

- `model.py:157`

---

### 11.3 第三层：block 内部的 attention 更新

每个 block 默认有 2 个 `EncoderLayer`：

```python
self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
```

然后循环调用：

```python
for layer in self.layers:
    embs = layer(embs, mask)
```

这里的 `EncoderLayer` 本质就是：

- multi-head self-attention
- 残差 / 归一化
- feed-forward
- 残差 / 归一化

对应公式可以写成：

\[
H' = \text{Attention}(H)
\]

\[
\tilde H = \text{AddNorm}(H, H')
\]

\[
H'' = \text{FFN}(\tilde H)
\]

\[
H_{\text{out}} = \text{AddNorm}(\tilde H, H'')
\]

源码位置：

- `model.py:241-242`
- `model.py:271-308`

---

### 11.4 `EncoderLayer` 里面具体做了什么

先生成：

- `Q`
- `K`
- `V`

```python
q = reshape_by_heads(self.Wq(input1), head_num=head_num)
k = reshape_by_heads(self.Wk(kv), head_num=head_num)
v = reshape_by_heads(self.Wv(kv), head_num=head_num)
```

然后做 scaled dot-product attention：

```python
score = torch.matmul(q, k.transpose(2, 3))
score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
score_scaled = score_scaled + mask...
weights = nn.Softmax(dim=3)(score_scaled)
out = torch.matmul(weights, v)
```

再把多头输出合并回来：

```python
multi_head_out = self.multi_head_combine(out_concat)
```

然后经过：

- `Add_And_Normalization_Module`
- `Feed_Forward_Module`
- `Add_And_Normalization_Module`

源码位置：

- `model.py:288-308`
- `model.py:324-359`
- `model.py:361-411`

---

### 11.4.1 默认主线里，attention 到底作用在谁和谁之间

这里先直接回答你最关心的问题。

> 对 NSS 的默认主线来说，attention 不是“方法表示作为 K，实例表示作为 Q”，而是“同一个实例内部的节点之间做 self-attention”。

也就是说，在默认配置：

```yaml
ns_feature: False
```

时，attention 的对象是：

\[
\text{node}_1,\ \text{node}_2,\ \dots,\ \text{node}_N
\]

不是：

\[
\text{instance representation} \leftrightarrow \text{solver representation}
\]

所以如果你问：

> “是不是方法表示作为 K，实例表示作为 Q？”

对默认主线的答案是：

> **不是。**

默认主线根本没有“方法 token / solver token”参与 attention。

默认主线的 attention 只是：

- 每个节点看其他节点
- 聚合同一实例内部的结构信息
- 最后再把节点表示汇总成实例表示

---

### 11.4.2 默认主线里，`Q / K / V` 是怎么来的

在默认主线中，`EncoderLayer.forward()` 的输入是：

```python
input1: [B, N, 128]
```

这里的 `input1` 就是当前实例所有节点的 embedding。

代码是：

```python
if kv is None:
    kv = input1
q = reshape_by_heads(self.Wq(input1), head_num=head_num)
k = reshape_by_heads(self.Wk(kv), head_num=head_num)
v = reshape_by_heads(self.Wv(kv), head_num=head_num)
```

因为默认调用时没有传 `kv`，所以：

\[
kv = input1
\]

于是得到：

\[
Q = W_q H,\quad K = W_k H,\quad V = W_v H
\]

其中：

- \(H \in \mathbb{R}^{B \times N \times 128}\)

所以默认主线里：

- `Q` 来自节点表示
- `K` 来自同一批节点表示
- `V` 也来自同一批节点表示

这就是标准的 self-attention。

---

### 11.4.3 默认配置下，QKV 的维度是怎么变的

默认配置里：

- `embedding_dim = 128`
- `head_num = 8`
- `qkv_dim = 16`

所以每个 head 的 query / key / value 维度都是 16，一共 8 个 head。

#### 第一步：线性投影

```python
self.Wq = nn.Linear(128, 8 * 16, bias=False)
self.Wk = nn.Linear(128, 8 * 16, bias=False)
self.Wv = nn.Linear(128, 8 * 16, bias=False)
```

因此：

\[
[B, N, 128] \rightarrow [B, N, 128]
\]

这里数值上还是 128，但语义已经变成了：

\[
8 \times 16
\]

#### 第 2 步：拆成多头

`reshape_by_heads()` 之后：

\[
[B, N, 128] \rightarrow [B, 8, N, 16]
\]

所以：

- `q`: `[B, 8, N, 16]`
- `k`: `[B, 8, N, 16]`
- `v`: `[B, 8, N, 16]`

#### 第 3 步：计算注意力分数

```python
score = torch.matmul(q, k.transpose(2, 3))
```

维度是：

\[
[B, 8, N, 16] \times [B, 8, 16, N] \rightarrow [B, 8, N, N]
\]

这表示：

- 对每个样本
- 对每个 head
- 对每个节点
- 都会去看所有节点

也就是说，`score[b,h,i,j]` 表示：

- 第 `b` 个样本
- 第 `h` 个 head
- 第 `i` 个节点
- 对第 `j` 个节点的注意力相关性

#### 第 4 步：softmax 后对 `V` 加权求和

```python
weights = nn.Softmax(dim=3)(score_scaled)
out = torch.matmul(weights, v)
```

维度是：

\[
[B, 8, N, N] \times [B, 8, N, 16] \rightarrow [B, 8, N, 16]
\]

也就是：

- 每个节点都会拿到一组对所有节点的权重
- 然后用这组权重对所有节点的 `V` 做加权和

#### 第 5 步：合并多头

```python
out_transposed = out.transpose(1, 2)
out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
multi_head_out = self.multi_head_combine(out_concat)
```

维度变化是：

\[
[B, 8, N, 16] \rightarrow [B, N, 8, 16] \rightarrow [B, N, 128] \rightarrow [B, N, 128]
\]

最后再经过：

- 残差
- FFN
- 残差

所以一个 `EncoderLayer` 的整体输入输出维度保持不变：

\[
[B, N, 128] \rightarrow [B, N, 128]
\]

---

### 11.4.4 用一句话理解默认主线里的 attention

默认主线里的 attention 可以理解成：

> 让每个节点在更新自己表示时，去参考同一实例里的其他节点。

所以它真正负责的是：

- 建模节点之间的空间关系
- 建模局部与全局结构
- 让 encoder 更好地得到实例表示

它不是在做：

- 实例和方法之间的匹配

---

### 11.4.5 默认主线里，solver 信息到底什么时候进入模型

在默认主线中，solver 信息并不会以 token 的形式进入 attention。

默认主线只是在最后一步：

```python
probs = self.classifier(self.instance_feature)
```

直接输出一个长度为 `M` 的分数向量：

\[
[B,257] \rightarrow [B,M]
\]

所以默认主线的顺序其实是：

\[
\text{instance}
\xrightarrow{\text{attention encoder}}
\text{instance feature}
\xrightarrow{\text{MLP}}
\text{scores of all solvers}
\]

不是：

\[
\text{instance token}
\xleftrightarrow{\text{attention}}
\text{solver token}
\]

这一点一定要和 `ns_feature=True` 的扩展线区分开。

---

### 11.5 block 里的图 readout

每个 block 做完节点更新以后，不是直接丢掉，而是先读一个图级表示：

```python
mean_emb = self.masked_mean(embs, mask)
max_emb = self.masked_max(embs, mask)
graph_emb = self.nonlinear(torch.cat((mean_emb, max_emb), dim=1))
```

所以单个 block 的图表示维度是：

\[
[B, 128] \, || \, [B, 128] \rightarrow [B, 256]
\]

这里的 `||` 是拼接，不是相加。

源码位置：

- `model.py:244-247`

---

### 11.6 block 里的 pooling 是怎么做的

这部分对应论文的层次池化思想。

先用一个额外的 attention layer 产生 score embedding：

```python
score_embs = self.layer_score(embs, mask)
scores = self.act(self.p(score_embs))
scores = scores.squeeze(-1)
scores = scores + mask
```

也就是：

\[
H^{score} = \text{AttentionLayer}_{score}(H)
\]

\[
z = \tanh(W_{score} H^{score})
\]

然后对每个实例选 top-k 节点：

```python
score, ind = scores[i].topk(int(lengths[i] * self.model_params['downsample_ratio']), largest=True)
selected_emb = embs[i].take_along_dim(...)
selected_emb = selected_emb + score[:, None]
```

默认 `downsample_ratio = 0.8`，所以每个 block 都保留 80% 节点。

如果原来是 `N` 个点，那么：

- 第 1 个 block 后：大约变成 `0.8N`
- 第 2 个 block 后：大约变成 `0.64N`

源码位置：

- `model.py:249-269`

---

### 11.7 `Encoder_h` 的最终输出

在 `Encoder_h.forward()` 里，两个 block 跑完以后，还会再对最后一级节点表示做一次：

- mean pooling
- max pooling
- concat

```python
mean_emb = self.masked_mean(out, mask)
max_emb = self.masked_max(out, mask)
graph_emb = self.nonlinear(torch.cat((mean_emb, max_emb), dim=1))
```

按论文描述，本来是想把每一级 readout 累加成 `graph_emb_h`：

```python
graph_emb_h += graph_emb
```

但是当前源码最后返回的是：

```python
return graph_emb
```

而不是：

```python
return graph_emb_h
```

所以要非常注意：

> 当前仓库代码里，`Encoder_h` 实际返回的是“最后一级读出的图表示”，  
> 不是论文文字描述里那个“多级 readout 求和”的总表示。

源码位置：

- `model.py:176-195`

这也是论文描述和源码实现最容易混淆的一点。

---

## 12. `scale` 和手工特征是在哪里拼的

这是你之前很关心的一点。

在 NSS 默认主线里：

- 图编码器 `Encoder_h` 内部 **不拼 scale**
- scale 是在 `Selection_model.forward()` 的 **encoder 外部** 拼进去的

对应代码：

```python
graph_emb = self.encoder(points, mask)

if manual_features == None:
    manual_features = scales[:, None]
else:
    manual_features = torch.cat((manual_features, scales[:, None]), dim=-1)

self.instance_feature = torch.cat((graph_emb, manual_features), dim=1)
```

源码位置：

- `model.py:52-64`

这说明默认主线的特征融合顺序是：

\[
\text{graph encoder output} \rightarrow [\text{graph emb}, \text{scale or manual+scale}]
\]

而不是把 scale 当作图节点的一部分送进 encoder。

---

## 13. 默认选择头是什么结构

默认情况下 `ns_feature=False`，所以走的是固定 solver index 的 MLP：

```python
self.classifier = nn.Sequential(
    nn.Linear(feature_dim, model_params['embedding_dim']),
    nn.GELU(),
    nn.Linear(model_params['embedding_dim'], model_params['output_dim'])
)
```

默认维度就是：

\[
257 \rightarrow 128 \rightarrow M
\]

其中：

- TSP：`M=7`
- CVRP：`M=5`

输出的不是概率，而是 **logits / scores**。

源码位置：

- `model.py:33-37`

---

## 14. 默认训练目标到底是什么

论文实现支持两种损失：

1. `CE`
2. `rank`

对应代码：

```python
if self.train_params['loss'] == 'CE':
    l = self.criterion(F.log_softmax(y_pred, 1), y)
if self.train_params['loss'] == 'rank':
    l = self.criterion(y_pred, cost)
```

源码位置：

- `trainer.py:131-134`

---

### 14.1 CE：把问题看成分类

在这种模式下：

- 标签 `y` 就是 `ind = argmin(cost)`

也就是：

> “这个实例最优的 solver 是哪一个”

这就是普通多分类。

---

### 14.2 rank：把问题看成排序学习

这是论文默认主线。

Ranking loss 实现在：

```python
class RankingLoss(nn.Module):
    def forward(self, logits, costs):
        loss = 0
        for i in range(self.top_k):
            cur_cost, ind = costs.topk(self.num_solvers - i, largest=True)
            cur_label = cur_cost.min(dim=1)[1]
            cur_logits = torch.take_along_dim(logits, ind, 1)
            loss += F.nll_loss(F.log_softmax(cur_logits, 1), cur_label)
        return loss
```

源码位置：

- `loss.py:7-20`

这个实现第一次看会有点绕，因为它用了：

- `largest=True`
- 再配合 `cur_cost.min(dim=1)`

直观理解如下。

假设某个实例有 4 个 solver，它们的 cost 是：

\[
[10.0,\ 8.0,\ 7.0,\ 12.0]
\]

越小越好。

那么：

- 第 1 名是 solver 2
- 第 2 名是 solver 1
- 第 3 名是 solver 0
- 第 4 名是 solver 3

这段 loss 代码做的事情其实是：

1. 在全部 solver 中选出最优者
2. 去掉最优者，在剩下 solver 中选次优者
3. 再去掉次优者，继续选

也就是在逼模型学一个完整排序，而不是只学 top-1。

这和论文里 ranking loss 的思想是一致的：

\[
\max_\theta \sum_i \log
\frac{\exp(g_\theta(I)_{\phi_I(i)})}
{\sum_{j=i}^M \exp(g_\theta(I)_{\phi_I(j)})}
\]

它比 CE 更充分地利用了：

- 最优 solver
- 次优 solver
- 其余 solver 的相对强弱关系

这也是论文里 rank 通常优于 CE 的重要原因。

---

## 15. 训练循环到底是怎么走的

真实训练循环在：

- `trainer.py::run()`
- `trainer.py::train_one_epoch()`

---

### 15.1 `run()` 的整体流程

可以概括成：

1. 建 loss
2. 建 dataloader
3. 如果有 checkpoint 就加载
4. 先测一次 val
5. 每个 epoch：
   - 训练一轮
   - 再测一次 val
   - 写日志
   - 如果 val top-1 变好就保存 best checkpoint

对应代码：

```python
results = self.test(0, test_dataloader, representative_set)
for epoch in range(...):
    grad_norm, loss_mean = self.train_one_epoch(...)
    results = self.test(epoch, test_dataloader, representative_set)
    self.logger['file'].write(results)
```

源码位置：

- `trainer.py:55-104`

所以它是一个非常标准的：

> epoch 级监督学习训练 + epoch 结束后做验证

---

### 15.2 `train_one_epoch()` 的 batch 训练流程

每个 batch 里做：

1. 取出输入和标签
2. 前向得到 `y_pred`
3. 算 loss
4. `backward`
5. `optimizer.step`

对应代码：

```python
x = batch[0].to(self.device)
y = batch[1].to(self.device)
cost = batch[2].to(self.device)
scales = batch[3].to(self.device)
mask = batch[4].to(self.device)

y_pred = self.model(x, scales, manual_feature, mask)

if self.train_params['loss'] == 'CE':
    l = self.criterion(F.log_softmax(y_pred, 1), y)
if self.train_params['loss'] == 'rank':
    l = self.criterion(y_pred, cost)

l.backward()
self.optimizer.step()
```

源码位置：

- `trainer.py:114-149`

这就是最标准的 supervised training。

---

## 16. 测试阶段到底在做什么

测试函数在：

- `trainer.py::test()`

这个函数分成两层：

1. 先跑模型，得到每个实例的 score vector
2. 再基于 score vector 去评估不同 selection strategy

---

### 16.1 第一层：算分数

```python
y_pred = self.model(x, scales, manual_feature, mask)
score_mat.append(F.softmax(y_pred, 1).detach())
gap_mat.append(gap)
time_mat.append(time_cost)
```

也就是说，评测阶段真正用来做策略选择的是：

\[
\text{softmax}(y\_pred)
\]

不是原始 logits。

源码位置：

- `trainer.py:188-197`

---

### 16.2 top-1 / top-k

评测时先选分数最高的 top-k solver：

```python
_, topk_ind = score_mat.topk(k, 1, largest=True)
topk_gap = gap_mat.gather(1, topk_ind).min(dim=1)[0]
topk_time = time_mat.gather(1, topk_ind).sum(dim=1)
```

解释如下：

- `topk_ind`：模型选出来的 top-k solver
- `topk_gap`：这 `k` 个 solver 里实际表现最好的那个 gap
- `topk_time`：跑这 `k` 个 solver 的总时间

所以：

- `top_1` 对应 greedy selection
- `top_2/top_3/top_4` 对应论文里的 top-k strategy

源码位置：

- `trainer.py:225-234`

---

### 16.3 rejection-based selection

这个策略的思路是：

- 对高置信实例，用 top-1
- 对低置信实例，用 top-2

代码里置信度用的是：

```python
score_mat.max(dim=1)[0]
```

也就是 softmax 最大值。

然后按置信度排序：

```python
sort_ind = score_mat.max(dim=1)[0].sort(descending=True)[1].cpu().numpy()
```

默认 `cover_rates = [0.8]`，所以：

- 最高置信的 80% 实例：用 top-1
- 最低置信的 20% 实例：用 top-2

源码位置：

- `trainer.py:236-255`

---

### 16.4 top-p

top-p 的目标是：

> 找到一个最小的 solver 子集，使它们的预测概率和至少达到 \(p\)

代码逻辑是：

```python
for j in range(1, score_mat.shape[1] + 1):
    top_j, ind = score_mat[i].topk(j, largest=True)
    if top_j.sum() >= p:
        ...
        break
```

源码位置：

- `trainer.py:257-287`

不过这里要注意：

> 当前仓库里的 `top-p` 实现有一个细小的 off-by-one 风格问题，  
> 它在阈值第一次满足时，用到的是前一轮缓存的 `ind_`，不一定正好是“当前满足条件的最小集合”。

所以如果你是“理解论文主思想”，可以把它看成 top-p 策略实现。
如果你是“严格复现代码行为”，那它和论文想表达的 top-p 有一点小偏差。

---

## 17. 手工特征分支在做什么

如果 `manual_feature=True`，代码就不走图编码器了，而是走：

```python
model = Naive_classifier(**config['model_params'])
```

源码位置：

- `run.py:105-106`

这个分支里：

- TSP 用 9 维手工特征
- CVRP 用 11 维手工特征
- 再加 `scale`

所以输入维度是：

- TSP：`9 + 1 = 10`
- CVRP：`11 + 1 = 12`

对应代码：

```python
if self.model_params['problem_type'] == 'TSP':
    feature_dim = 10
else:
    feature_dim = 12
```

源码位置：

- `model.py:101-108`

---

### 17.1 手工特征具体包含什么

`manual_features()` 里实现了下面这些统计量：

TSP / CVRP 共有的 9 维：

1. 全距离矩阵标准差 `std`
2. 质心 `centroid_x`
3. 质心 `centroid_y`
4. 平均半径 `radius`
5. 距离值离散度 `count_distinct`
6. 最近邻距离标准差 `std_nNN`
7. 聚类比例 `cluster_ratio`
8. 离群点比例 `outlier_ratio`
9. 聚类半径 `radius_cluster`

CVRP 额外的 2 维：

10. `demand_mean`
11. `demand_std`

源码位置：

- `dataset.py:54-103`

---

## 18. `ns_feature=True` 这条线在做什么

这是论文里“solver feature / unseen solver generalization”的那条扩展线。

它的目标不是只靠 solver index，而是：

> 也给 solver 建一个表示，让模型能够在测试时利用一个新加入的 solver

代码结构是：

1. 先选每个 solver 的 representative instances
2. 用 encoder 把这些实例编码成 token
3. 用 `representative_net` 汇总成一个 solver token
4. 把 `instance_feature` 和 `solver_token` 拼起来，再算相似度分数

对应入口：

- `utils.py:93-122`
- `model.py:20-31`
- `model.py:77-94`
- `trainer.py:295-314`

不过这里要特别注意：

> 当前仓库版本里，这条分支并不是最稳的主线实现。

至少有两个明显问题：

1. `update_tokens()` 里只循环 `range(self.init_tokens.shape[0])`，而 `init_tokens` 当前是 `[1, feature_dim]`
2. `representative_net` 定义了 `att_layer_2`，但 forward 里第二次仍然调用的是 `att_layer_1`

也就是说：

- 这条路径更像论文里的探索性扩展实现
- 不是当前最可靠的默认主线

默认配置里也确实没有启用它：

```yaml
ns_feature: False
```

### 18.1 这条扩展线里的 attention，才更接近 token 间交互

如果打开 `ns_feature=True`，模型里才会出现更像“token 和 token 交互”的 attention。

关键代码是：

```python
model_tokens = model_token[None, None, :]
representative_features = representative_features.unsqueeze(0)
input1 = torch.cat((model_tokens, representative_features), dim=1)
out1 = self.att_layer_1(input1)
out2 = self.att_layer_1(out1[:, :1, :], kv=out1[:, 1:, :])
```

这里的语义是：

1. 先准备一个可学习的 summary / model token
2. 再准备若干个 representative instance feature
3. 把它们拼成一个 token 序列
4. 先做一次 self-attention
5. 再让 summary token 去读取 representative tokens，得到一个 solver token

所以这条扩展线里的 attention，确实已经不是“节点对节点”，而是：

- summary token
- representative instance tokens

之间的交互。

### 18.2 在 `representative_net` 里，`Q / K / V` 分别是谁

这里其实有两次 attention。

#### 第一次 attention：所有 token 一起做 self-attention

```python
out1 = self.att_layer_1(input1)
```

因为这里没有传 `kv=`，所以：

- `Q` 来自 `input1`
- `K` 来自 `input1`
- `V` 来自 `input1`

而 `input1` 里包含：

- 一个 summary / model token
- 多个 representative instance tokens

所以第一次 attention 是：

> summary token 和 representative tokens 一起做 self-attention

#### 第二次 attention：summary token 读取 representative tokens

```python
out2 = self.att_layer_1(out1[:, :1, :], kv=out1[:, 1:, :])
```

这里：

- `input1 = out1[:, :1, :]`，形状是 `[1, 1, d]`
- `kv = out1[:, 1:, :]`，形状是 `[1, R, d]`

因此：

- `Q` 来自 summary token
- `K` 来自 representative instance tokens
- `V` 来自 representative instance tokens

这一步就很像一个 cross-attention：

\[
Q = \text{summary token},\quad K,V = \text{representative instance tokens}
\]

它的作用不是给当前实例直接打分，而是：

> 把一组 representative instances 汇总成一个 solver token

### 18.3 `ns_feature=True` 的完整结构图

这一条扩展线最好拆成两段来看：

1. **先为每个 solver 构造一个 solver token**
2. **再用当前实例特征和 solver token 做 pair scoring**

如果按论文意图来画，完整结构图可以写成：

```text
阶段 A：为每个 solver 生成 solver token

solver k 的 representative instances
    -> 每个 representative instance 先过 instance encoder
    -> representative feature 序列: [R_k, 257]

一个可学习的 summary token
    -> [1, 257]

[summary token ; representative features]
    -> [1, R_k + 1, 257]
    -> Attention layer 1: 所有 token 做 self-attention
    -> [1, R_k + 1, 257]
    -> Attention layer 2: summary token 作为 Q，representative tokens 作为 K/V
    -> [1, 1, 257]
    -> squeeze
    -> solver token s_k: [257]


阶段 B：当前实例和每个 solver token 做打分

当前实例 x
    -> Encoder_h
    -> graph_emb: [B, 256]
    -> concat(scale)
    -> instance_feature: [B, 257]

对每个 solver token s_k
    -> expand 成 [B, 257]
    -> concat(instance_feature, s_k)
    -> [B, 514]
    -> similarity MLP
    -> [B, 1]

把所有 solver 的 [B, 1] 拼起来
    -> [B, M]
```

这里的维度为什么是这些数，来自 `Selection_model.__init__()`：

```python
if model_params['pooling']:
    feature_dim = 2 * model_params['embedding_dim'] + 1
```

默认 `embedding_dim=128`，所以：

\[
feature\_dim = 2 \times 128 + 1 = 257
\]

于是：

- `instance_feature` 是 257 维
- `solver token` 也是 257 维
- 拼接后是 514 维

对应代码：

```python
feature_dim = 2 * model_params['embedding_dim'] + 1
model_params_['embedding_dim'] = feature_dim
self.init_tokens = nn.Parameter(torch.Tensor(1, feature_dim))
feature_dim = 2 * feature_dim
```

源码位置：

- `model.py:13-31`

#### 阶段 A 的维度变化再展开一次

假设某个 solver 有 `R` 个 representative instances。

首先，这些 representative instances 会先经过 `encoder_p` 得到：

\[
[R, 257]
\]

这里 257 维来自：

- 图编码 256 维
- `scale` 1 维

对应代码：

```python
representative_feature.append(
    torch.cat((
        self.encoder_p(x, mask),
        scales[:, None]
    ), dim=1)
)
```

源码位置：

- `trainer.py:307-312`

然后进入 `representative_net`：

1. summary token：
   - `[257] -> [1, 1, 257]`
2. representative features：
   - `[R, 257] -> [1, R, 257]`
3. 拼接：
   - `[1, 1, 257] + [1, R, 257] -> [1, R+1, 257]`
4. 第一次 self-attention：
   - 输入输出都还是 `[1, R+1, 257]`
5. 第二次“summary token 读 representative tokens”：
   - `Q`: `[1, 1, 257]`
   - `K,V`: `[1, R, 257]`
   - 输出 `[1, 1, 257]`
6. `squeeze` 后得到：
   - solver token `[257]`

#### 阶段 B 的维度变化再展开一次

对当前 batch 的实例：

1. `instance_feature`
   - `[B, 257]`
2. 某一个 solver token `s_k`
   - `[257]`
3. 扩展到 batch：
   - `[257] -> [B, 257]`
4. 拼接：
   - `[B,257] + [B,257] -> [B,514]`
5. 送进 similarity MLP：
   - `[B,514] -> [B,128] -> [B,1]`
6. 对全部 solver 重复这一步
7. 最后拼起来：
   - `[B,1] x M -> [B,M]`

所以这条扩展线的最终形式可以写成：

\[
x \rightarrow h(x)\in\mathbb{R}^{257}
\]

\[
\text{solver }k \rightarrow s_k\in\mathbb{R}^{257}
\]

\[
\text{score}(x,k)=\text{MLP}([h(x), s_k])
\]

#### 注意：这是“论文意图上的完整结构图”，而不是当前代码完全可靠实现的保证

这里一定要补一句提醒。

当前代码里至少有两个问题会影响这张结构图在“多 solver”场景下的真实落地：

1. `self.init_tokens` 的形状是 `[1, 257]`，不是 `[M, 257]`
2. `update_tokens()` 循环的是 `range(self.init_tokens.shape[0])`

这意味着按当前代码字面行为，它未必真的会生成 `M` 个 solver token。

所以这张结构图更适合理解：

- 论文想表达的 `ns_feature=True` 架构
- 以及这份代码试图实现的设计

而不是说当前仓库这一版完全无误地把它实现好了。

### 18.4 但即便在 `ns_feature=True` 下，最终 instance-solver 打分也不是用 attention

这也是一个特别容易误解的点。

在 `ns_feature=True` 分支里，真正的最终打分不是：

- “实例表示作 Q，solver 表示作 K/V，再做一次 attention”

而是：

```python
torch.cat(
    (self.instance_feature, self.model_tokens[i][None, :].expand(batch_size, -1)),
    dim=-1
)
```

然后送进：

```python
self.similarity = nn.Sequential(
    nn.Linear(feature_dim, model_params['embedding_dim']),
    nn.GELU(),
    nn.Linear(model_params['embedding_dim'], 1)
)
```

所以最终的 instance-solver 打分仍然是：

- 先拼接
- 再过 MLP

不是 attention。

因此如果把你的问题完整回答，就是：

1. **默认主线 `ns_feature=False`：不是“方法表示作为 K，实例表示作为 Q”，因为根本没有方法 token 参与 attention**
2. **扩展线 `ns_feature=True`：只有在 `representative_net` 里，为了生成 solver token，才会出现 `Q=summary token, K/V=representative tokens` 这种更像 cross-attention 的结构**
3. **最终 instance-solver 打分阶段依然不是 attention，而是 concat + MLP**

---

## 19. 代码里几个很容易误读的点

### 19.1 `Encoder_h` 最终没有返回 `graph_emb_h`

代码里明明累计了多层 readout：

```python
graph_emb_h += graph_emb
```

但最后返回的是：

```python
return graph_emb
```

不是：

```python
return graph_emb_h
```

这意味着：

- 论文文字：多层层次信息求和
- 当前代码：最后一级 readout

这是理解源码时最要小心的点。

---

### 19.2 `representative_net` 第二层没有用 `att_layer_2`

定义时：

```python
self.att_layer_1 = EncoderLayer(**model_params)
self.att_layer_2 = EncoderLayer(**model_params)
```

但 forward 里：

```python
out1 = self.att_layer_1(input1)
out2 = self.att_layer_1(out1[:, :1, :], kv=out1[:, 1:, :])
```

第二次还是 `att_layer_1`。

这说明当前实现和注释想表达的“两层 attention”不完全一致。

---

### 19.3 `train_one_epoch()` 里的 `loss_mean` 不是严格 mean

代码里写的是：

```python
loss_mean += l.mean().item()
```

最后直接打印：

```python
print("Loss mean: {:.4f}".format(loss_mean))
```

但它没有再除以 batch 数。

所以这个 `Loss mean` 更准确地说是：

- 每个 batch loss 的累加和

不是严格意义上的 epoch 平均 loss。

---

### 19.4 运行目录必须对

`prepare_dataset()` 里用的是相对路径：

```python
with open(f'datasets/{problem_type}train/dataset.pkl', 'rb') as f:
```

所以如果你不在：

- `9nss论文/neural-solver-selection/`

这个目录下运行，代码会直接找不到数据文件。

这也是这套源码的一个实际使用注意点。

---

## 20. 用一句话总结默认主线的完整训练流程

可以把 NSS 默认主线总结成下面这条数据流：

\[
\text{离线实例数据}
\rightarrow
\text{solver zoo 全部跑分}
\rightarrow
\text{得到 } (cost, time, gap, ind)
\rightarrow
\text{图编码器提取实例表示}
\rightarrow
\text{MLP 输出每个 solver 的 score}
\rightarrow
\text{ranking / CE loss 训练}
\rightarrow
\text{测试时再做 greedy / top-k / rejection / top-p}
\]

如果写成更贴近代码的形式，就是：

```text
dataset.pkl + raw_label.pkl
    -> SelectionDataset
    -> collate_fn
    -> Selection_model
        -> Encoder_h
        -> concat(scale)
        -> MLP classifier
    -> RankingLoss / CE
    -> val set evaluation
    -> top-k / rejection / top-p metrics
```

---

## 21. 这篇论文的实现，最值得你记住的几个点

1. 它的核心不是 solver 本身，而是 **instance-level solver selector**
2. 它的训练不是 RL，而是 **离线监督学习**
3. 默认最重要的主线是：
   - `pooling=True`
   - `manual_feature=False`
   - `ns_feature=False`
   - `loss=rank`
4. 默认 encoder 是一个 **层次化图注意力编码器**
5. `scale` 是在 encoder 外部拼的，不是在图 encoder 内部拼的
6. ranking loss 不是只学 top-1，而是在利用完整 solver 排序
7. 测试时真正决定最终效果的，不只是 score 模型本身，还有：
   - greedy
   - top-k
   - rejection
   - top-p
8. 论文描述和当前仓库代码之间，有少数需要特别注意的小偏差，不能完全按论文文字想当然地理解源码

---

## 22. 如果你接下来要继续读 NSS 源码，建议的阅读顺序

最推荐的顺序是：

1. `run.py`
   - 先搞清楚入口怎么串起来
2. `utils.py`
   - 搞清楚数据和标签是怎么读出来的
3. `dataset.py`
   - 搞清楚 batch 里到底有什么
4. `model.py`
   - 重点看 `Selection_model`、`Encoder_h`、`Encoder_block_h`
5. `loss.py`
   - 搞清楚 ranking loss
6. `trainer.py`
   - 最后再看训练和评测策略

如果你的目标是“借鉴 NSS 去理解你自己的实现”，那最该重点对照的就是：

- `Selection_model.forward()`
- `Encoder_h.forward()`
- `Encoder_block_h.forward()`
- `RankingLoss.forward()`
- `trainer.test()`

---

## 23. 最后一句话

NSS 这套实现最本质的思想可以记成一句话：

> 不去训练一个万能 solver，  
> 而是利用不同 solver 在不同实例上的互补性，训练一个能看实例选 solver 的 selector。

这也是它和传统“单 solver 越训越强”的思路最不一样的地方。
