# 原论文详细解释：Neural Solver Selection for Combinatorial Optimization

## 论文基本信息

- **论文标题**: Neural Solver Selection for Combinatorial Optimization
- **发表会议**: ICML 2025
- **作者**: Chengrui Gao, Haopu Shang, Ke Xue, Chao Qian
- **代码仓库**: https://github.com/lamda-bbo/neural-solver-selection

---

## 一、论文核心思想与动机

### 1.1 No-Free-Lunch定理的实证发现

**问题背景**：
神经组合优化领域出现了大量优秀的求解器（POMO、Attention Model、LEHD、DIFUSCO等），但研究者发现一个重要现象：

**实证观察**：
1. **不存在万能求解器**：没有任何一个神经求解器能在所有实例上都表现最优
2. **性能互补性**：不同求解器在不同实例上各有优势
3. **分布敏感性**：修改实例分布可以完全反转求解器的优劣关系

**Figure 1的洞察**：
- Figure 1(a)：展示不同求解器在不同实例上的性能互补
- Figure 1(b)：展示分布变化对求解器排名的影响

### 1.2 论文的核心假设

**假设1**：神经求解器的性能具有实例依赖性
- 不同实例适合不同求解器
- 可以通过学习实现实例级的智能调度

**假设2**：求解器选择优于单一求解器
- 通过协调多个求解器，可以获得优于任何单一求解器的性能
- 关键是如何为每个实例选择最合适的求解器

### 1.3 与现有方法的区别

| 方法类型 | 代表工作 | 局限性 | 本论文优势 |
|---------|---------|--------|-----------|
| 单一求解器 | POMO, AM, LEHD | 无法适应所有实例 | 实例级自适应选择 |
| 集成学习 | Jiang et al. 2023 | 架构相同，多样性受限 | 多样化求解器池 |
| 种群训练 | Grinsztajn et al. 2023 | 推理时间长（需运行所有求解器） | 选择性调用，高效 |

---

## 二、三组件框架详解

### 2.1 整体架构图

```
输入: TSP/CVRP实例
    ↓
┌─────────────────────────────┐
│   Component 1: 特征提取      │
│   图注意力编码器             │
│   层次化图编码器（创新）      │
└─────────────────────────────┘
    ↓ 实例特征向量
┌─────────────────────────────┐
│   Component 2: 选择模型      │
│   MLP分类器                 │
│   损失函数: 分类/排名损失     │
└─────────────────────────────┘
    ↓ 求解器兼容性分数
┌─────────────────────────────┐
│   Component 3: 选择策略      │
│   Top-k, 拒绝策略, Top-p     │
└─────────────────────────────┘
    ↓ 选中的求解器
输出: 最优解（来自预存结果）
```

### 2.2 组件1: 特征提取

#### 2.2.1 图注意力编码器

**理论基础**：
- TSP/CVRP实例可以建模为全连接图
- 节点 = 城市/客户，边 = 潜在路线
- 使用多头注意力捕获节点间依赖关系

**数学表示**：
```
输入: x ∈ ℝ^(N×3) (CVRP: x,y,demand) 或 x ∈ ℝ^(N×2) (TSP: x,y)

初始嵌入:
H⁰ = xW, W ∈ ℝ^(3×d)

多层注意力更新:
H^l = AttentionLayer^l(H^(l-1)), l ∈ [L]

实例表征:
o = Mean(H^L)  # 平均池化
```

**代码实现**：
```python
# model.py: EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        
        # 多头注意力变换
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
```

#### 2.2.2 层次化图编码器（核心创新）

**创新动机**：
- 简单平均池化会丢失图的结构信息
- 不同抽象层次的特征对选择很重要
- 需要捕获图的层次化结构

**关键技术：可微分图池化**

**步骤1: 计算节点代表性分数**
```
H_score^l = AttentionLayer_score^l(H^l)
Z^l = tanh(H_score^l × W_score^l)
```
- `H_score^l`: 通过注意力层计算的分数嵌入
- `Z^l ∈ ℝ^(N^(l-1)×1)`: 每个节点的代表性分数

**步骤2: 选择代表性节点**
```
N^l = α × N^(l-1), α = 0.8
按Z^l降序排序，选择前N^l个节点
```

**步骤3: 可导优化**
```
H̃^l = H^l + Z^l × 1
```
- 将代表性分数融合到节点嵌入中
- 使池化操作可微分，支持反向传播

**步骤4: Readout操作**
```
o^l = σ(Mean(H^l) || Max(H^l))
```
- 均值池化 + 最大值池化
- 捕捉全局和局部特征

**步骤5: 多尺度特征融合**
```
o = Σ(l=1 to L+1) o^l
```
- 累加所有层次的特征
- 形成最终的层次化实例表征

**代码实现**：
```python
# model.py: Encoder_block_h
class Encoder_block_h(nn.Module):
    def forward(self, embs, mask, i):
        # 步骤1: 多层图注意力更新
        for layer in self.layers:
            embs = layer(embs, mask)
        
        # 步骤2: Readout操作
        mean_emb = self.masked_mean(embs, mask)
        max_emb = self.masked_max(embs, mask)
        graph_emb = self.nonlinear(torch.cat((mean_emb, max_emb), dim=1))
        
        # 步骤3: 可微分图池化
        score_embs = self.layer_score(embs, mask)
        scores = self.act(self.p(score_embs))  # tanh激活
        scores = scores.squeeze(-1)
        
        # 步骤4: 选择代表性节点
        for batch_idx in range(embs.shape[0]):
            num_selected = int(lengths[batch_idx] * self.model_params['downsample_ratio'])
            score, ind = scores[batch_idx].topk(num_selected, dim=-1, largest=True)
            selected_emb = embs[batch_idx].take_along_dim(ind[:, None].expand(-1, embs.shape[-1]), dim=0)
            # 关键：融合代表性分数
            selected_emb = selected_emb + score[:, None]
```

### 2.3 组件2: 选择模型

#### 2.3.1 模型结构

**MLP分类器**：
```
输入: [图特征(256维); 规模特征(1维)] = 257维
隐藏层: 257 → 128 (GELU激活)
输出层: 128 → M (M=求解器数量)
```

**代码实现**：
```python
# model.py: Selection_model
self.classifier = nn.Sequential(
    nn.Linear(257, 128),
    nn.GELU(),
    nn.Linear(128, 7)  # TSP: 7个求解器
)
```

#### 2.3.2 损失函数

**A. 分类损失**
```python
# 监督信号: 最优求解器索引
if self.train_params['loss'] == 'CE':
    l = self.criterion(F.log_softmax(y_pred, 1), y)
    # y_pred: [0.1, 0.8, 0.05, 0.03, 0.02]
    # y: 1 (第2个求解器最优)
```

**B. 排名损失（核心创新）**

**动机**：
- 分类损失只关注最优求解器
- 忽略了次优、第三优等求解器的相对关系
- 排名损失利用所有求解器的相对性能

**数学公式**：
```
max_θ E_I [Σ(i=1 to M) log (exp(g_θ(I)_φ_I(i))) / (Σ(j=i to M) exp(g_θ(I)_φ_I(j))))]
```

**直观解释**：
- `φ_I(i)`: 排名第i的求解器索引
- `g_θ(I)`: 选择模型的输出分数
- 目标: 让模型输出的分数顺序与真实性能排序一致

**代码实现**：
```python
# loss.py: RankingLoss
class RankingLoss(nn.Module):
    def forward(self, logits, costs):
        loss = 0
        for i in range(self.top_k):
            # 找到从第i名开始的求解器
            cur_cost, ind = costs.topk(self.num_solvers - i, largest=True)
            # 在这些求解器中找到最优的
            cur_label = cur_cost.min(dim=1)[1]
            # 提取对应的logits
            cur_logits = torch.take_along_dim(logits, ind, 1)
            # 计算损失：鼓励给最优求解器更高分数
            loss += F.nll_loss(F.log_softmax(cur_logits, 1), cur_label)
        return loss
```

**示例说明**：
```
假设有3个求解器，真实成本为 [10, 15, 20]

排名: φ_I = [0, 1, 2] (求解器0最优，求解器2最差)

多层次学习:
i=1: 在[0,1,2]中学习0最好
i=2: 在[1,2]中学习1比2好
i=3: 在[2]中学习2

优势: 充分利用所有求解器的相对关系
```

### 2.4 组件3: 选择策略

#### 2.4.1 贪心选择

**逻辑**：选择兼容性分数最高的求解器

**优点**：效率高（只运行1个求解器）
**缺点**：依赖预测准确性，容错性差

#### 2.4.2 Top-k选择

**逻辑**：选择前k个分数最高的求解器，取最优结果

**参数选择**：k=2（论文实验确定）

**代码实现**：
```python
# trainer.py
_, topk_ind = score_mat.topk(k, 1, largest=True)
topk_gap = gap_mat.gather(1, topk_ind).min(dim=1)[0]
topk_time = time_mat.gather(1, topk_ind).sum(dim=1)
```

**性能分析**：
- Top-1 (贪心): 1.86% gap, 1.33s
- Top-2: 1.51% gap, 2.56s
- Oracle: 1.24% gap, 8.93s

#### 2.4.3 拒绝策略

**核心思想**：基于置信度自适应选择

**置信度度量**：
```python
confidence = softmax(s)_1  # 最高分数的归一化概率
```

**策略**：
- 高置信度（80%）: 使用贪心选择（Top-1）
- 低置信度（20%）: 使用Top-2选择

**代码实现**：
```python
sort_ind = score_mat.max(dim=1)[0].sort(descending=True)[1]
threshold = int(num_instances * 0.8)  # 保留80%

reject_ind = sort_ind[threshold:]   # 低置信度
accept_ind = sort_ind[:threshold]  # 高置信度

gap_SR = torch.cat((topk_gap[reject_ind], top_1_gap[accept_ind]), dim=0)
```

#### 2.4.4 Top-p选择

**核心思想**：动态确定求解器数量

**逻辑**：
```
按分数降序排列: s_(1) ≥ s_(2) ≥ ... ≥ s_(M)
找最小k使得: Σ(i=1 to k) s_(i) / Σ(j=1 to M) s_(j) ≥ p
```

**参数**：TSP用p=0.5, CVRP用p=0.8

**代码实现**：
```python
for i in range(len(score_mat)):
    for j in range(1, score_mat.shape[1] + 1):
        top_j, ind = score_mat[i].topk(j, largest=True)
        if top_j.sum() >= p:  # 累积分数达到阈值
            times.append(time_mat[i][ind].sum().item())
            gaps.append(gap_mat[i][ind].min().item())
            break
```

**优势**：
- 自适应k值，无需手动调整
- 适配不同实例的分数分布

---

## 三、监督学习训练机制详解

### 3.1 监督数据的生成流程

#### 步骤1: 数据实例生成
```python
# datasets/data_utils.py
def generate_tsp_data_gaussian(dataset_size, problem_size, params):
    # 高斯混合分布采样
    num_modes = np.random.randint(params['num_modes_lower'], 
                                   params['num_modes_upper'])
    
    if num_modes == 0:
        # 均匀分布
        problems = torch.rand(size=(dataset_size, problem_size, 2))
    else:
        # 高斯混合分布
        for i in range(dataset_size):
            mix_proportion = np.random.rand(num_modes)
            nums = np.random.multinomial(problem_size, mix_proportion / sum(mix_proportion))
            
            xy = []
            for num in nums:
                # 采样均值和协方差
                center = np.random.uniform(0, 100, size=(1, 2))
                cov = [[var_x, cov_xy], [cov_xy, var_y]]
                nxy = np.random.multivariate_normal(mean=center, cov=cov, size=(num,))
                xy.extend(nxy)
            
            # 缩放到[0,1]×[0,1]
            xy = MinMaxScaler().fit_transform(xy)
            problems.append(xy)
    
    return problems
```

#### 步骤2: 求解器预运行

**候选求解器池**：
- TSP (7个): BQ, ELG, LEHD, T2T, T2T500, DIFUSCO, DIFUSCO500
- CVRP (5个): BQ, ELG, LEHD, Omni, MVMoE

**预运行脚本**：
```bash
# 为每个求解器在所有实例上运行
for solver in BQ ELG LEHD T2T DIFUSCO; do
    python run_${solver}.py --dataset TSPtrain
done

# 结果保存在 datasets/TSPtrain/results/
# result_bq.txt: "0,6.757,2.084" (实例ID,成本,时间)
# result_ELG.txt: "0,6.841,0.682"
# ...
```

#### 步骤3: 标签处理

**代码实现**：
```python
# datasets/process_raw_label.py
labels = {}
for instance_id in all_instances:
    labels[instance_id] = {
        'cost': [],     # 各求解器成本
        'time': [],     # 各求解器时间
        'gap': [],      # 各求解器最优性差距
        'ind': 0        # 最优求解器索引
    }

# 收集所有求解器结果
for solver in solvers:
    with open(f'result_{solver}.txt') as f:
        for line in f:
            instance_id, cost, time = line.strip().split(',')
            labels[instance_id]['cost'].append(float(cost))
            labels[instance_id]['time'].append(float(time))

# 计算最优性差距
for instance_id, data in labels.items():
    costs = np.array(data['cost'])
    optimal_cost = opts[instance_id]  # 专家求解器结果
    data['gap'] = [100 * (c - optimal_cost) / optimal_cost for c in costs]
    data['ind'] = np.argmin(costs)  # 最优求解器索引

# 保存标签
with open('raw_label.pkl', 'wb') as f:
    pickle.dump(labels, f)
```

### 3.2 训练流程详解

#### 步骤1: 数据加载
```python
# utils.py: prepare_dataset
def prepare_dataset(problem_type, name):
    # 加载实例数据
    with open(f'datasets/{name}/dataset.pkl', 'rb') as f:
        instance_set = pickle.load(f)
    
    # 加载标签数据
    with open(f'datasets/{name}/raw_label.pkl', 'rb') as f:
        raw_labels = pickle.load(f)
    
    # 构建训练样本
    label_set = []
    for i in range(len(instance_set)):
        key = str(i)
        label_set.append([
            raw_labels[key]['ind'],      # 最优求解器索引
            raw_labels[key]['cost'],     # 所有求解器成本
            raw_labels[key]['time'],     # 所有求解器时间
            raw_labels[key]['gap']       # 所有求解器gap
        ])
    
    return instance_set, label_set
```

#### 步骤2: 前向传播
```python
# trainer.py: train_one_epoch
def train_one_epoch(self, epoch, train_dataloader):
    for batch in train_dataloader:
        x = batch[0].to(self.device)          # [batch, nodes, 2]
        y = batch[1].to(self.device)          # [batch] 最优求解器索引
        cost = batch[2].to(self.device)       # [batch, num_solvers]
        
        # 前向传播
        y_pred = self.model(x, scales, manual_feature, mask)
        # y_pred: [batch, 7] 求解器兼容性分数
```

**模型内部流程**：
```python
# model.py: Selection_model.forward
def forward(self, points, scales, manual_features, mask):
    # 步骤1: 图编码器提取特征
    graph_emb = self.encoder(points, mask)  # [batch, 256]
    
    # 步骤2: 特征融合
    instance_feature = torch.cat((graph_emb, scales[:, None]), dim=1)  # [batch, 257]
    
    # 步骤3: MLP分类器
    probs = self.classifier(instance_feature)  # [batch, 7]
    
    return probs
```

#### 步骤3: 损失计算
```python
# 分类损失
if self.train_params['loss'] == 'CE':
    l = self.criterion(F.log_softmax(y_pred, 1), y)

# 排名损失
elif self.train_params['loss'] == 'rank':
    l = self.criterion(y_pred, cost)
```

#### 步骤4: 反向传播
```python
self.optimizer.zero_grad()
l.backward()
self.optimizer.step()
```

### 3.3 训练配置

**超参数设置**：
- 优化器: Adam
- 学习率: 1×10⁻⁴
- 权重衰减: 1×10⁻⁶
- 训练轮数: 50 epochs
- 批量大小: 64 (训练), 8 (测试)
- 随机种子: 5个不同种子

**配置文件示例**：
```yaml
# config_TSP.yml
train_params:
  num_classes: 7
  num_epochs: 50
  train_batch_size: 64
  learning_rate: 0.0001
  weight_decay: 0.000001
  loss: rank  # 或 CE
  data_aug: true

model_params:
  pooling: true  # 使用层次化编码器
  embedding_dim: 128
  encoder_layer_num: 2
  block_num: 2
  head_num: 8
  output_dim: 7
```

---

## 四、测试阶段机制详解

### 4.1 关键理解：不实际求解！

**重要澄清**：测试时不会运行任何TSP/CVRP求解算法

**原因**：
1. 这个项目是**元学习系统**，不是求解器
2. 核心是学习"选择"，而不是"求解"
3. 所有求解器结果都已预存在文件中

### 4.2 测试流程完整示例

**实例**: TSPLIB中的pr2392实例（2392个城市）

**步骤1: 加载预存数据**
```python
# datasets/TSPLIB/raw_label.pkl
instance_0 = {
    'cost': [15432.5, 15489.3, 15398.7, ...],  # 7个求解器成本
    'time': [12.5, 8.2, 7.8, ...],              # 7个求解器时间
    'gap': [1.2, 1.8, 0.5, ...],                # 相对于最优解的gap
    'ind': 2                                    # 最优求解器：LEHD
}
```

**步骤2: 模型预测**
```python
# 加载实例坐标
coords = load_instance('TSPLIB/pr2392.tsp')  # [1, 2392, 2]

# 图编码器提取特征
graph_emb = encoder(coords, mask)  # [1, 256]

# MLP分类器预测
scores = classifier(torch.cat([graph_emb, scale]))  # [1, 7]
# scores = [0.05, 0.12, 0.85, 0.08, 0.03, 0.02, 0.15]
#          LEHD分数最高
```

**步骤3: 选择策略应用**
```python
# Top-1选择
selected_solver = argmax(scores)  # = 2 (LEHD)
final_cost = cost[2]  # = 15398.7
final_time = time[2]  # = 7.8s

# Top-2选择
top2_solvers = topk(scores, 2)  # = [2, 6]
final_cost = min(cost[2], cost[6])
final_time = time[2] + time[6]
```

**步骤4: 性能评估**
```python
# 最优性差距
gap = (final_cost - optimal_cost) / optimal_cost * 100%
# optimal_cost来自专家求解器LKH

# 推理时间
total_time = model_inference_time + solver_time
```

### 4.3 多策略评估代码

```python
# trainer.py: test
def test(self, epoch, test_dataloader):
    score_mat = []  # 收集所有预测分数
    gap_mat = []    # 收集所有预存gap
    time_mat = []   # 收集所有预存时间
    
    for batch in test_dataloader:
        # 预测
        y_pred = self.model(x, scales, manual_feature, mask)
        score_mat.append(F.softmax(y_pred, 1).detach())
        
        # 预存数据
        gap_mat.append(gap)
        time_mat.append(time_cost)
    
    # 合并所有batch
    score_mat = torch.cat(score_mat, dim=0)  # [num_instances, 7]
    gap_mat = torch.cat(gap_mat, dim=0)
    time_mat = torch.cat(time_mat, dim=0)
    
    # Top-k评估
    for k in [1, 2, 3, 4]:
        _, topk_ind = score_mat.topk(k, 1, largest=True)
        topk_gap = gap_mat.gather(1, topk_ind).min(dim=1)[0]
        topk_time = time_mat.gather(1, topk_ind).sum(dim=1)
        print(f"Top-{k}: {topk_gap.mean():.4f}%, {topk_time.mean():.4f}s")
```

---

## 五、实验结果分析

### 5.1 主要性能指标

**最优性差距**：
```
Optimality Gap = (c_I(σ̂) - c_I(σ*)) / c_I(σ*) × 100%
```
- `c_I(σ̂)`: 待评估方法得到的解的成本
- `c_I(σ*)`: 专家求解器的最优已知解（TSP用LKH，CVRP用HGS）

### 5.2 实验结果汇总

#### TSP合成数据
| 方法 | Gap (%) | Time (s) |
|-----|---------|---------|
| DIFUSCO (最佳单个) | 2.33 | 1.45 |
| 本框架 (rank+greedy) | 1.86 | 1.33 |
| 本框架 (rank+top2) | **1.51** | 2.56 |
| Oracle | 1.24 | 8.93 |

#### TSPLIB
| 方法 | Gap (%) | Time (s) |
|-----|---------|---------|
| T2T (最佳单个) | 1.95 | 1.78 |
| 本框架 (rank+top2) | **1.07** | 2.71 |
| Oracle | 0.74 | 9.12 |

#### CVRP合成数据
| 方法 | Gap (%) | Time (s) |
|-----|---------|---------|
| Omni (最佳单个) | 6.82 | 2.12 |
| 本框架 (rank+top2) | **4.82** | 3.45 |
| Oracle | 4.64 | 10.23 |

### 5.3 消融实验结果

#### 特征提取方法对比
| 特征提取 | TSP Gap | CVRP Gap | TSPLIB Gap |
|---------|---------|----------|------------|
| 手工特征 | 2.15 | 5.89 | 1.85 |
| 图注意力编码器 | 1.92 | 5.34 | 1.52 |
| 层次化编码器 | **1.86** | **4.82** | **1.07** |

**发现**：
- 手工特征虽然简单，但已经有效
- 学习特征优于手工特征
- 层次化编码器在分布外数据上表现最佳

#### 损失函数对比
| 损失函数 | TSP Gap | TSPLIB Gap |
|---------|---------|------------|
| 分类损失 | 1.92 | 1.45 |
| 排名损失 | **1.86** | **1.07** |

**发现**：
- 排名损失在分布外泛化上明显优于分类损失
- 说明学习相对排序很重要

---

## 六、技术创新点总结

### 6.1 架构创新

1. **层次化图编码器**
   - 可微分图池化
   - 多尺度特征融合
   - 提升分布外泛化能力

2. **排名损失**
   - 充分利用所有求解器的相对关系
   - 增强鲁棒性和泛化性

3. **多种选择策略**
   - 平衡性能与效率
   - 自适应选择机制

### 6.2 工程创新

1. **预运行+监督学习**
   - 避免训练时实时运行求解器
   - 提升训练效率

2. **批量实验管理**
   - 自动生成训练/测试脚本
   - 多GPU并行训练

3. **模块化设计**
   - 清晰的组件分离
   - 灵活的配置系统

### 6.3 实验验证

1. **多个数据集**
   - 合成数据 (TSP/CVRP)
   - 标准基准 (TSPLIB/CVRPLIB)
   - 大规模扩展 (N∈[500,2000])

2. **多个评估指标**
   - 最优性差距
   - 推理时间
   - 分类准确率

3. **多次随机种子**
   - 5个不同种子
   - 报告均值和标准差

---

## 七、代码实现与论文对应关系

### 7.1 特征提取组件

| 论文位置 | 代码文件 | 类/函数 |
|---------|---------|---------|
| 3.1 图注意力编码器 | model.py | EncoderLayer |
| 3.1 层次化编码器 | model.py | Encoder_h, Encoder_block_h |
| 附录A.1 ReZero归一化 | model.py | Add_And_Normalization_Module |

### 7.2 选择模型组件

| 论文位置 | 代码文件 | 类/函数 |
|---------|---------|---------|
| 3.2 MLP分类器 | model.py | Selection_model.classifier |
| 3.2 分类损失 | trainer.py | CrossEntropyLoss |
| 3.2 排名损失 | loss.py | RankingLoss |

### 7.3 选择策略组件

| 论文位置 | 代码文件 | 类/函数 |
|---------|---------|---------|
| 3.3 Top-k选择 | trainer.py | test()中的topk循环 |
| 3.3 拒绝策略 | trainer.py | test()中的rejection逻辑 |
| 3.3 Top-p选择 | trainer.py | test()中的topp逻辑 |

### 7.4 数据与实验

| 论文位置 | 代码文件 | 功能 |
|---------|---------|------|
| 4.1 数据生成 | datasets/data_utils.py | 高斯混合分布采样 |
| 4.1 标签处理 | datasets/process_raw_label.py | 求解器结果聚合 |
| 4.1 训练 | run.py, trainer.py | 训练循环实现 |
| 4.1 评估 | trainer.py | 多策略评估 |

---

## 八、总结

### 核心贡献
1. **首个通用神经求解器选择框架**
2. **层次化图编码器**提升特征质量
3. **排名损失**增强学习效果
4. **多种选择策略**平衡性能与效率

### 实验验证
- TSP: 提升0.82% (合成), 0.88% (TSPLIB)
- CVRP: 提升2.00% (合成), 0.71% (CVRPLIB)
- 时间开销: 仅增加毫秒级选择时间

### 技术特点
- **监督学习**: 基于预存结果的离线学习
- **元学习系统**: 学习选择，而非求解
- **模块化设计**: 清晰的三组件框架
- **工程优化**: 高效的训练和评估流程

---

**文档生成时间**: 2025年1月  
**对应论文版本**: ICML 2025  
**代码版本**: https://github.com/lamda-bbo/neural-solver-selection
