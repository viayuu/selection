# unified selector 数据包说明

本目录是当前 unified neural solver selector 使用的主数据目录。

当前数据已经整理成统一的 NSS-like 导出格式，并经过一轮完整审计：

- 问题总数：`18`
- 数据集总数：`56`
- 目录中包含：
  - `TSP`
  - `CVRP`
  - `ATSP`
  - `15` 个 `MVRP` 变体

额外说明：

- `TSP/CVRP` 当前使用的是 **EasyNCO init-only** 重新生成的标签，不再使用原始 NSS 论文中的完整迭代标签。
- `MVRP` 当前的 `dataset.pkl` 已升级为 **显式字段名 dict 格式**，不再使用匿名 tuple。

## 1. 包含的问题类型

当前 `18` 个问题为：

- `TSP`
- `CVRP`
- `ATSP`
- `OVRP`
- `VRPB`
- `VRPL`
- `VRPTW`
- `OVRPTW`
- `OVRPB`
- `OVRPL`
- `VRPBL`
- `VRPBTW`
- `VRPLTW`
- `OVRPBL`
- `OVRPBTW`
- `OVRPLTW`
- `VRPBLTW`
- `OVRPBLTW`

其中：

- `TSP/CVRP/ATSP` 是三类基础问题
- 其余 `15` 个是 `MVRP` 变体

## 2. 每类问题的方法集合

### TSP

方法为：

- `DACT`
- `DIFUSCO`
- `ELG`
- `GLOP`
- `INVIT`
- `LEHD`
- `LIH`
- `OMNI`
- `T2T`
- `UDC`

### CVRP

方法为：

- `DACT`
- `ELG`
- `GLOP`
- `ICAM`
- `INVIT`
- `LEHD`
- `OMNI`
- `UDC`

### ATSP

方法为：

- `GLOP`
- `MATNET`
- `MATPOENET`

### MVRP

所有 `15` 个 `MVRP` 变体共享同一组方法：

- `MTPOMO`
- `MVMOE`

## 3. 各数据集大小

### TSP

- `TSPtrain = 10000`
- `TSPval = 1000`
- `TSPtest = 1000`
- `TSPLIB = 49`

### CVRP

- `CVRPtrain = 10000`
- `CVRPval = 1000`
- `CVRPtest = 1000`
- `CVRPLIB = 100`

### ATSP

- `ATSPtrain = 10000`
- `ATSPval = 1000`
- `ATSPtest = 1000`

### MVRP

每个变体都包含：

- `train = 10000`
- `val = 1000`
- `test = 1000`

`MVRP` 的 `15` 个变体为：

- `OVRP`
- `VRPB`
- `VRPL`
- `VRPTW`
- `OVRPTW`
- `OVRPB`
- `OVRPL`
- `VRPBL`
- `VRPBTW`
- `VRPLTW`
- `OVRPBL`
- `OVRPBTW`
- `OVRPLTW`
- `VRPBLTW`
- `OVRPBLTW`

## 4. 各类问题的数据构造特征

这一节描述每类问题的数据是如何构造出来的，包括规模范围、分布口径和关键实例特征。

### 4.1 TSP

`TSP` 当前直接来自 NSS 的 varying-size 数据集。

#### split 与规模

- `TSPtrain`
  - `10000` 个实例
  - 规模范围：`50..499`
  - `450` 个 unique scales
- `TSPval`
  - `1000` 个实例
  - 规模范围：`50..499`
  - `404` 个 unique scales
- `TSPtest`
  - `1000` 个实例
  - 规模范围：`50..499`
  - `402` 个 unique scales
- `TSPLIB`
  - `49` 个实例
  - 规模范围：`51..1002`
  - `39` 个 unique scales

#### 分布与特征

- `train / val / test` 的分布口径来自 NSS manifest，标记为 `gaussian`
- 每个实例本质上是二维 Euclidean 节点坐标
- `TSPLIB` 是真实 benchmark 实例，不是合成高斯坐标数据

#### 训练时可见的实例特征

- 节点坐标
- 实例规模 `n`

### 4.2 CVRP

`CVRP` 当前也直接来自 NSS 的 varying-size 数据集。

#### split 与规模

- `CVRPtrain`
  - `10000` 个实例
  - 规模范围：`50..499`
  - `450` 个 unique scales
- `CVRPval`
  - `1000` 个实例
  - 规模范围：`50..499`
  - `409` 个 unique scales
- `CVRPtest`
  - `1000` 个实例
  - 规模范围：`50..499`
  - `398` 个 unique scales
- `CVRPLIB`
  - `100` 个实例
  - 规模范围：`101..1001`
  - `100` 个 unique scales

#### 分布与特征

- `train / val / test` 的分布口径来自 NSS manifest，标记为 `gaussian`
- 每个实例包含：
  - 一个 depot 坐标
  - 一组客户坐标
  - 一组归一化 demand
- `CVRPLIB` 是真实 benchmark 实例，不是合成高斯坐标数据

#### 训练时可见的实例特征

- `depot`
- `loc`
- `demand`
- 实例规模 `n`

### 4.3 ATSP

`ATSP` 是通过 `EasyNCO.data.ATSPGenerator` 生成的 dense varying-size 数据。

#### split 与规模

- `ATSPtrain`
  - `10000` 个实例
  - 规模范围：`20..100`
  - `81` 个 unique scales
  - shard 分配：
    - `20..56` 这 `37` 个尺度各 `124`
    - `57..100` 这 `44` 个尺度各 `123`
- `ATSPval`
  - `1000` 个实例
  - 规模范围：`20..100`
  - `81` 个 unique scales
  - shard 分配：
    - `28` 个尺度各 `13`
    - `53` 个尺度各 `12`
- `ATSPtest`
  - `1000` 个实例
  - 规模范围：`20..100`
  - `81` 个 unique scales
  - shard 分配：
    - `28` 个尺度各 `13`
    - `53` 个尺度各 `12`

#### 分布与特征

- `ATSP` 不是坐标型 Euclidean 数据
- 每个实例先随机生成一个 `n x n` 的整数 cost matrix
- 对角线强制设为 `0`
- 然后做 min-plus closure，直到矩阵稳定
- 最后整体除以 `1e6`，变成浮点 cost matrix

因此它的本质是：

- 非对称 cost matrix
- 不带显式节点坐标
- 不是 `uniform / gaussian` 这种点分布数据

#### 训练时可见的实例特征

- 一个 `n x n` 的非对称 cost matrix
- 实例规模 `n`

### 4.4 MVRP

`MVRP` 当前使用的是后续构造的多分布多规模版本。

#### 变体与 split

- 共 `15` 个变体
- 每个变体都有：
  - `train = 10000`
  - `val = 1000`
  - `test = 1000`

#### 规模

- `train`
  - 规模范围：`50..100`
  - 共 `51` 个尺度
  - 分配：
    - `50..53` 各 `197`
    - `54..100` 各 `196`
- `val / test`
  - 规模范围：`50..100`
  - 共 `51` 个尺度
  - 分配：
    - `31` 个尺度各 `20`
    - `20` 个尺度各 `19`

#### 坐标分布

`MVRP` 的 customer 坐标不是单一 uniform，而是 4 类混合分布：

- `uniform`
- `gaussian_mixture`
- `clustered`
- `ring`

其中：

- `train` 每个变体总量 `10000`
  - 分布总数接近全局均衡
  - 典型统计为：
    - `uniform = 2503`
    - `gaussian_mixture = 2499`
    - `clustered = 2499`
    - `ring = 2499`
- `val / test` 每个变体总量 `1000`
  - 全局严格均衡
  - 每类分布各 `250`

#### 其它实例特征

基础生成机制：

- depot 坐标单独采样
- customer 坐标按上述 4 类分布采样
- demand 先采样整数，再按 `demand_scaler` 归一化

不同约束通过以下方式体现在实例中：

- `B`
  - 通过 `node_demand` 中的负值表示 backhaul
- `L`
  - 通过显式字段 `route_limit`
  - 当前默认值为 `3.0`
- `TW`
  - 通过：
    - `service_time`
    - `tw_start`
    - `tw_end`
  - 当前时间窗生成采用更接近 Solomon 风格的 harder setting
  - 默认 `service_time = 0.2`
- `O`
  - 不是单独字段
  - 是问题定义的一部分
  - 所以 `OVRP` 与 `CVRP` 这类只差 `O` 的 pair，不能只靠实例字段区分

#### 训练时可见的实例特征

- 基础：
  - `depot_xy`
  - `node_xy`
  - `node_demand`
  - `capacity`
- 如果带 `L`
  - 额外有 `route_limit`
- 如果带 `TW`
  - 额外有 `service_time / tw_start / tw_end`

## 5. 每个数据集目录包含什么

每个数据集目录都包含这三类核心文件：

- `dataset.pkl`
- `raw_label.pkl`
- `results/result_*.txt`

典型目录结构如下：

```text
TSPtrain/
  dataset.pkl
  raw_label.pkl
  results/
    result_DACT.txt
    result_DIFUSCO.txt
    ...
```

## 6. `dataset.pkl` 的含义与格式

`dataset.pkl` 表示该数据集的实例列表。

它的顶层容器当前统一是：

- Python `list`

也就是说：

- `dataset.pkl[i]` 就是第 `i` 个实例
- 这个 `i` 与 `results/result_*.txt` 里的 `instance_id=i` 一一对应

不同问题类型的单实例格式如下。

### 6.1 TSP

单个实例是一个 tensor：

- 形状：`(1, n, 2)`

含义：

- `n` 个节点的二维坐标

可理解为：

```python
dataset[i] = tensor(shape=(1, n, 2))
```

### 6.2 CVRP

单个实例是一个 dict，字段为：

- `loc`
- `demand`
- `depot`

典型形状：

- `loc: (1, n, 2)`
- `demand: (1, n)`
- `depot: (1, 1, 2)`

可理解为：

```python
dataset[i] = {
  "loc": ...,
  "demand": ...,
  "depot": ...,
}
```

### 6.3 ATSP

单个实例是一个 tensor：

- 形状：`(1, n, n)`

含义：

- 一个非对称 cost matrix

可理解为：

```python
dataset[i] = tensor(shape=(1, n, n))
```

### 6.4 MVRP

单个实例是一个 dict。

基础字段为：

- `depot_xy`
- `node_xy`
- `node_demand`
- `capacity`

在不同变体中，还会按约束额外增加字段。

#### 基础类

适用于：

- `OVRP`
- `VRPB`
- `OVRPB`

字段为：

- `depot_xy`
- `node_xy`
- `node_demand`
- `capacity`

#### 带 `L` 的变体

适用于：

- `VRPL`
- `OVRPL`
- `VRPBL`
- `OVRPBL`

字段为：

- `depot_xy`
- `node_xy`
- `node_demand`
- `capacity`
- `route_limit`

#### 带 `TW` 的变体

适用于：

- `VRPTW`
- `OVRPTW`
- `VRPBTW`
- `OVRPBTW`

字段为：

- `depot_xy`
- `node_xy`
- `node_demand`
- `capacity`
- `service_time`
- `tw_start`
- `tw_end`

#### 同时带 `L + TW` 的变体

适用于：

- `VRPLTW`
- `OVRPLTW`
- `VRPBLTW`
- `OVRPBLTW`

字段为：

- `depot_xy`
- `node_xy`
- `node_demand`
- `capacity`
- `route_limit`
- `service_time`
- `tw_start`
- `tw_end`

补充说明：

- `B` 变体可以从 `node_demand` 中的负值看出来
- `L` 变体通过 `route_limit` 表示
- `TW` 变体通过 `service_time / tw_start / tw_end` 表示
- `O` 不是单独字段，而是问题类型定义的一部分

## 7. `results/result_*.txt` 的含义

`results/result_*.txt` 表示某一个方法在该数据集上的逐实例结果。

例如：

- `result_DACT.txt`
- `result_MATNET.txt`
- `result_MTPOMO.txt`

每行格式固定为：

```text
instance_id,no_aug_score
```

例如：

```text
0,8.123456
1,7.987654
2,8.543210
```

含义：

- 第 `0` 个实例的该方法 `no_aug_score = 8.123456`
- 第 `1` 个实例的该方法 `no_aug_score = 7.987654`

重要约定：

- `instance_id` 与 `dataset.pkl` 的实例索引严格一一对应
- 即 `dataset.pkl[i]` 对应 `result_*.txt` 中 `instance_id=i` 的那一行

## 8. `raw_label.pkl` 的含义

`raw_label.pkl` 是该数据集的统一标签汇总表。

它的顶层结构当前是：

- Python `dict`

键是实例 id 的字符串形式：

- `"0"`
- `"1"`
- ...
- `"N-1"`

也就是说：

```python
raw_label["0"]
raw_label["1"]
...
```

每个实例条目是一个 dict，包含：

- `cost`
- `time`
- `ind`
- `gap`

形如：

```python
raw_label["17"] = {
  "cost": [...],
  "time": [...],
  "ind": 3,
  "gap": [...]
}
```

各字段含义：

- `cost`
  - 该实例在各方法上的 cost 列表
- `time`
  - 该实例在各方法上的时间列表
- `ind`
  - `cost` 最小值对应的方法索引
- `gap`
  - 该实例在各方法上的 gap 列表

当前这套数据中，`cost` 是最关键字段，后续 selector 训练通常主要依赖它。

## 9. `raw_label.pkl` 与 `result_*.txt` 的关系

`raw_label.pkl` 中每个实例的 `cost` 向量，与同目录下各个 `result_*.txt` 是一一对应的。

更准确地说：

- 对某个实例 `i`
- `raw_label[str(i)]["cost"]` 是该实例在所有方法上的 cost 汇总
- 这些 cost 槽位与该数据集目录下的 `result_*.txt` 文件存在稳定的一一映射

这层对应关系已经做过审计，未发现乱序或错位。

## 10. 当前最重要的训练使用方式

如果后续用这套数据训练 unified selector，最常见的读取方式是：

1. 从 `dataset.pkl` 读取实例特征
2. 从 `raw_label.pkl` 读取该实例在所有方法上的 cost 向量与最佳方法索引
3. 或者直接从 `results/result_*.txt` 重建每个方法的逐实例 cost

当前这三者之间的对应关系已经核对通过：

- `dataset.pkl` 索引正确
- `result_*.txt` 索引正确
- `raw_label.pkl` 与 `result_*.txt` 映射正确
