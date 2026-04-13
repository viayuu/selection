# 2l2r 实现说明

`2l2r` 是一个和 `2cmab2` 并行的监督式排序版本。

核心变化只有一条：

- `2cmab2`：bandit 风格，只用当前选中的 arm 做更新
- `2l2r`：supervised learning-to-rank，同一实例上的所有可行 arm 一起进入 loss

当前实现仍然保留了原来的三层基础：

1. 统一动作空间  
   `TSP` 和 `CVRP` 共用一套全局 arm id，再用 `feasible_mask` 过滤不可行动作

2. 同一套 NSS 风格图编码器  
   `2l2r/encoders.py` 直接复用了当前 `2cmab2` 的实现  
   `TSP` 用 2 维输入 encoder，`CVRP` 用 3 维输入 encoder

3. 同一套数据构造协议  
   每个 `InstanceSample` 里都已经有：
   - `costs`
   - `rewards`
   - `ranks`
   - `best_arm`
   - `feasible_mask`

因此 `2l2r` 的训练目标会更直接：

- 给定一个实例
- 同时给所有可行 arm 打分
- 用 pairwise / listwise ranking loss 逼迫“更好的 arm 分数更高”

默认模型结构：

1. `HybridContextEncoder` 输出实例表示 `context(x)`，维度 `270`
2. 每个 arm 有一个 embedding `e(a)`，默认维度 `16`
3. 拼接后得到 `[context(x), e(a)]`，维度 `286`
4. 经过 `phi_net` 得到隐藏表示，默认维度 `64`
5. `score_head` 输出一个标量 `score(x,a)`

默认训练目标：

- `pairwise_logistic`

也就是如果某个实例上 `reward_i > reward_j`，就要求：

$$
score(x, a_i) > score(x, a_j)
$$

这样训练目标会比“回归某个绝对 reward 数值”更贴近“选最优方法”的最终任务。
