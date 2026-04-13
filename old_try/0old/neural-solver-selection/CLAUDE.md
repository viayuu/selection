# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是ICML 2025论文"Neural Solver Selection for Combinatorial Optimization"的官方实现。该项目提出了**首个通用神经求解器选择框架**，通过协调多个神经求解器，为每个问题实例分配最合适的求解器。

### 核心贡献
- **实例级求解器选择**：基于no-free-lunch定理，不同求解器在不同实例上具有互补性
- **三组件框架**：特征提取、选择模型、选择策略
- **显著性能提升**：在TSPLIB和CVRPLIB上分别提升0.88%和0.71%
- **高效推理**：仅需少量额外时间开销

## 核心技术原理

### 🎯 **项目本质：神经求解器的智能调度器**
这个项目是一个**元学习系统**，其核心不是求解TSP/CVRP问题，而是学习**如何为每个实例选择最适合的神经求解器**。

**关键理解**：
- ❌ **不实际求解**：测试时不会运行任何TSP/CVRP求解算法
- ✅ **学习选择**：训练一个选择模型，预测哪个预存求解器最适合当前实例
- ✅ **利用预存结果**：基于各神经求解器在训练集上的预运行性能进行监督学习

### 📊 **监督学习的完整机制**

#### **1. 监督数据来源与结构**
```python
# 数据来源：预运行所有神经求解器得到的性能数据
datasets/
├── TSPtrain/results/
│   ├── result_bq.txt      # BQ求解器在所有训练实例上的成本和时间
│   ├── result_ELG.txt     # ELG求解器在所有训练实例上的成本和时间
│   ├── result_LEHD.txt    # LEHD求解器在所有训练实例上的成本和时间
│   └── result_opt.txt     # 专家求解器(LKH)的最优已知解
└── raw_label.pkl          # 处理后的监督标签数据

# 监督标签数据结构（raw_label.pkl）
labels = {
    "instance_0": {
        'cost': [6.757, 6.841, 6.755, 6.789, 6.812, 7.023, 6.945],  # 7个求解器的路径成本
        'time': [2.084, 0.682, 0.452, 1.123, 2.345, 1.789, 0.987],  # 7个求解器的运行时间
        'gap':  [0.0, 1.2, 0.0, 0.5, 0.8, 4.0, 3.2],              # 相对于最优解的gap
        'ind':  2                                                      # 最优求解器索引(LEHD)
    }
}
```

#### **2. 监督学习训练的模型组件**
监督学习主要训练两个核心组件：

**A. 特征提取器（图编码器）**：
```python
# 训练目标：学习从TSP/CVRP实例中提取区分性特征
class Encoder_h(nn.Module):
    def __init__(self):
        # 🔍 可训练的嵌入层：将坐标映射到高维空间
        self.embedding = nn.Linear(2, 128)  # TSP: (x,y) → 128维特征

        # 🔍 可训练的多头注意力：学习节点间的依赖关系
        self.blocks = nn.ModuleList([Encoder_block_h() for _ in range(2)])

        # 🔍 可训练的图池化：层次化特征聚合
        self.layer_score = EncoderLayer()
        self.p = nn.Linear(128, 1)  # 节点重要性评分

# 监督信号：好的特征表示应该能区分适合不同求解器的实例
# 实例A(集中分布) → 特征A → 预测求解器2(LEHD)最优
# 实例B(分散分布) → 特征B → 预测求解器5(DIFUSCO)最优
```

**B. 选择模型（MLP分类器）**：
```python
# 训练目标：学习实例特征→求解器兼容性的映射
class Selection_model(nn.Module):
    def __init__(self):
        # 🔍 可训练的MLP分类器
        self.classifier = nn.Sequential(
            nn.Linear(257, 128),  # 256维图特征 + 1维规模特征
            nn.GELU(),
            nn.Linear(128, 7)    # 输出7个求解器的适配分数
        )

# 监督信号：学习让输出的求解器分数与真实性能排序一致
# 输入：256维实例特征 + 1维规模特征 = 257维
# 输出：7个求解器的适配分数 [0.05, 0.12, 0.85, 0.08, 0.03, 0.02, 0.15]
# 目标：第3个分数最高（因为LEHD确实在这个实例上表现最好）
```

#### **3. 损失函数与监督信号**
项目支持两种监督学习损失：

**A. 分类损失（CrossEntropy Loss）**：
```python
# 监督目标：让模型预测的最优求解器与真实最优求解器一致
if self.train_params['loss'] == 'CE':
    l = self.criterion(F.log_softmax(y_pred, 1), y)
    # y_pred: [0.1, 0.8, 0.05, 0.03, 0.02]  # 预测求解器1最适合
    # y: 1  # 实际最优求解器是1(ELG)
    # loss: 计算预测误差
```

**B. 排名损失（Ranking Loss）**：
```python
# 监督目标：让模型输出的求解器分数与真实性能排序一致
if self.train_params['loss'] == 'rank':
    l = self.criterion(y_pred, cost)

class RankingLoss(nn.Module):
    def forward(self, logits, costs):
        # 多层次排名学习：不仅学习最优，还学习次优、第三优等
        for i in range(self.top_k):
            # 找到从第i名开始的求解器
            cur_cost, ind = costs.topk(self.num_solvers - i, largest=True)
            # 在这些求解器中找到最优的
            cur_label = cur_cost.min(dim=1)[1]
            # 让模型给最优求解器更高分数
            loss += F.nll_loss(F.log_softmax(cur_logits, 1), cur_label)
```

### 🔄 **完整的训练流程详解**

#### **阶段1：数据准备**
```python
# run.py:91 - 加载监督数据
train_set, train_label, test_set, test_label = prepare_dataset(config['problem_type'], name=name)

# utils.py:66-85 - 构建监督样本对
for i in range(len(train_instance_set)):
    key = str(i)
    train_label_set.append([
        train_raw_labels[key]['ind'],      # 最优求解器索引（分类监督）
        train_raw_labels[key]['cost'],     # 所有求解器成本（排序监督）
        train_raw_labels[key]['time'],     # 求解器运行时间
        train_raw_labels[key]['gap']       # 最优性差距
    ])
```

#### **阶段2：模型前向传播**
```python
# trainer.py:127 - 核心训练过程
y_pred = self.model(x, scales, manual_feature, mask)

# model.py内部流程：
def forward(self, points, scales, manual_features, mask):
    # 步骤1: 图编码器提取实例特征
    graph_emb = self.encoder(points, mask)  # [batch, 256] 学习好的特征表示

    # 步骤2: 特征融合
    instance_feature = torch.cat((graph_emb, scales[:, None]), dim=1)  # [batch, 257]

    # 步骤3: 求解器兼容性预测
    probs = self.classifier(instance_feature)  # [batch, 7] 7个求解器的适配分数

    return probs
```

#### **阶段3：损失计算与参数更新**
```python
# trainer.py:134-179 - 监督学习优化
self.optimizer.zero_grad()
if self.train_params['loss'] == 'CE':
    l = self.criterion(F.log_softmax(y_pred, 1), y)    # 分类监督
elif self.train_params['loss'] == 'rank':
    l = self.criterion(y_pred, cost)                   # 排名监督

l.backward()  # 反向传播
self.optimizer.step()  # 更新所有可训练参数

# 可训练参数包括：
# 1. 图编码器参数：nn.Linear(2, 128), 多头注意力权重, 图池化网络
# 2. MLP分类器参数：nn.Linear(257, 128), nn.Linear(128, 7)
```

### 🧪 **测试阶段的运作机制**

#### **关键理解：测试时没有实际求解新TSP问题！**
测试过程完全是基于预存结果的"选择验证"：

```python
# trainer.py:171-179 - 测试数据加载（全是预存结果）
for batch in test_dataloader:
    x = batch[0]          # [batch, nodes, 2] TSP实例坐标
    y = batch[1]          # [batch] 最优求解器索引（预存标签）
    cost = batch[2]       # [batch, 7] 预存的求解器成本矩阵
    time_cost = batch[6]  # [batch, 7] 预存的求解器时间矩阵

    # 核心预测：基于实例特征预测求解器适配分数
    y_pred = self.model(x, scales, manual_feature, mask)

    # 选择策略：基于预测分数选择预存的求解器结果
    _, topk_ind = score_mat.topk(2, 1, largest=True)
    topk_gap = gap_mat.gather(1, topk_ind).min(dim=1)[0]  # 使用预存gap
```

**具体例子 - TSPLIB实例0的测试过程**：
```python
# 预存数据：
instance_0_coords = [[0.1, 0.2], [0.8, 0.9], [0.3, 0.4], ...]  # 52个城市坐标
pre_stored_costs = [6.757, 6.841, 6.755, 6.789, 6.812, 7.023, 6.945]  # 7个求解器成本
pre_stored_times = [2.084, 0.682, 0.452, 1.123, 2.345, 1.789, 0.987]  # 7个求解器时间

# 模型预测：
predicted_scores = [0.05, 0.12, 0.85, 0.08, 0.03, 0.02, 0.15]  # LEHD分数最高

# 选择结果：
selected_solver = 2  # LEHD
final_cost = pre_stored_costs[2]  # 6.755（使用预存结果）
final_time = pre_stored_times[2]  # 0.452（使用预存结果）
```

### 📈 **项目的技术创新点**

#### **1. 元学习架构**
- **输入**：TSP/CVRP实例的几何特征
- **输出**：最适合的神经求解器选择
- **学习目标**：实例特征→求解器性能的映射关系

#### **2. 层次化图编码器**
- **创新点**：通过可微分图池化实现多尺度特征学习
- **技术优势**：能够捕获不同抽象层次的图结构信息

#### **3. 排名损失学习**
- **创新点**：不仅学习最优求解器，还学习求解器的相对性能排序
- **技术优势**：更强的泛化能力和鲁棒性

#### **4. 多种选择策略**
- **Top-k选择**：选择前k个预测分数最高的求解器
- **拒绝策略**：基于置信度自适应选择策略
- **Top-p选择**：动态确定求解器数量

### 🎯 **项目的实际价值**

这个项目解决了神经求解器选择的核心问题：
- **挑战**：不同神经求解器在不同实例上具有互补性能
- **解决方案**：学习"看图识求解器"的能力
- **效果**：TSPLIB上提升0.88%，CVRPLIB上提升0.71%
- **开销**：仅增加选择模型的推理时间（毫秒级）

**最终实现了一个智能的神经求解器调度系统，能够根据问题实例的特征自动选择最适合的求解器，从而获得优于任何单个求解器的性能。**

## 常用命令

### 训练模型
```bash
# TSP训练
python run.py --config_name config_TSP.yml --loss rank --seed 2024 --gpu_id 0

# CVRP训练
python run.py --config_name config_CVRP.yml --loss rank --seed 2024 --gpu_id 0

# 批量训练（使用脚本）
bash run_experiment.sh
```

### 测试模型
```bash
# 基本测试
python run.py --gpu_id 0 \
--load config_TSP.yml_rank_2024 \
--test_file TSPLIB \
--exp_name config_TSP.yml_rank_TSPLIB

# 批量测试（使用脚本）
bash run_test.sh
```

### 生成实验脚本
```python
# 生成训练脚本
python experiments/generate_experiments_shell.py

# 生成测试脚本
python experiments/generate_test_shell.py
```

## 详细技术实现细节

### 一、数据准备与预处理（论文4.1+附录A.2/A.3）

#### 1. 数据集构建
##### （1）合成数据集生成
- **生成对象**：TSP和CVRP实例，均通过高斯混合分布采样节点坐标
- **采样规则**：
  - 高斯混合成分数\(c \sim U(0,15)\)（\(c=0\)时为均匀分布），节点随机分配到各成分
  - 每个成分的均值\(\mu=(x_\mu,y_\mu) \sim U(0,1)\times U(0,1)\)，方差\(var_x,var_y \sim U(1,100)\)，协方差\(cov \sim [-\sqrt{var_x \cdot var_y}, \sqrt{var_x \cdot var_y})\)，节点坐标采样后缩放至\([0,1]\times[0,1]\)
  - 实例规模\(N \in [50,500]\)（训练集），测试集规模一致，扩展实验中\(N \in [500,2000]\)
- **CVRP额外设置**：
  - 车辆容量采用两种分布，各50%概率选择：
    1. 规模相关分布：\(Q=30+\lceil\frac{N}{5}\rceil\)
    2. 三角分布：先采样上限\(ub \sim U(20,\frac{N}{2})\)、众数\(m \sim U(5,ub)\)、下限\(lb \sim U(3,m)\)，再从\(T(lb,m,ub)\)采样容量
  - 节点需求\(m_i \sim U(1,10)\)，并按车辆容量\(Q\)归一化
- **数据集划分**：
  - 训练集：10,000个TSP实例+10,000个CVRP实例，应用8倍实例增强（Kwon et al., 2020）
  - 测试集：1,000个TSP实例+1,000个CVRP实例（合成数据）
  - 分布外/大规模验证集：TSPLIB（选\(N \leq 1002\)实例）、CVRPLIB Set-X（\(N \in [100,1000]\)）

##### （2）实例增强方式
采用Kwon et al. (2020)的POMO增强方法，保持实例本质特征不变，轻微扰动数据，用于提升模型泛化性

#### 2. 神经求解器池构建
##### （1）候选求解器列表
初始候选为8个开源SOTA神经求解器：Omni、BQ、LEHD、DIFUSCO、T2T、ELG、INViT、MVMoE

##### （2）求解器筛选流程（附录A.3）
- **筛选目标**：移除贡献极低的求解器，构建紧凑求解器池
- **贡献评估指标**：\(\mathcal{A}(s_i)=\mathbb{E}_I[\mathcal{P}_I(S)-\mathcal{P}_I(S/s_i)]\)，其中\(\mathcal{P}_I(\cdot)\)为求解器池在实例\(I\)上的最优性差距，\(S\)为当前候选池，\(s_i\)为待评估求解器
- **筛选步骤**：
  1. 计算每个求解器的贡献值\(\mathcal{A}(s_i)\)
  2. 移除贡献值最低的求解器
  3. 重复步骤1-2，直至所有剩余求解器的\(\mathcal{A}(s_i) \geq 0.01\%\)（预设阈值）
- **最终池规模**：TSP保留7个求解器，CVRP保留5个求解器

##### （3）求解器调用配置（附录A.2.2）
- 所有求解器采用贪心解码避免随机波动
- POMO-based求解器（如Omni）的POMO规模设为100，增强次数设为8
- 扩散模型类求解器（如DIFUSCO）的去噪步数设为50，2-opt迭代次数设为100

### 二、特征提取模块实现（论文3.1+附录A.1）

#### 核心前提
TSP和CVRP实例均表征为全连接图：节点=城市/客户+仓库，边=节点间潜在路线，节点属性=TSP（x,y）/CVRP（x,y,m），采用图神经网络（GNN）提取特征

#### 1. 图注意力编码器（Graph Attention Encoder）
##### （1）网络结构与参数
- 嵌入维度\(d=128\)
- 图注意力层数量=4层
- 注意力头数=8个
- Feed-Forward（FF）网络隐藏维度=512
- 激活函数：注意力层无明确说明，FF层采用ReLU

##### （2）分步实现流程
1. **初始嵌入**：对CVRP实例原始特征\(x \in \mathbb{R}^{N \times 3}\)（x,y,m），通过线性层映射为初始节点嵌入：\(H^0 = xW\)，其中\(W \in \mathbb{R}^{3 \times d}\)；TSP实例原始特征\(x \in \mathbb{R}^{N \times 2}\)（x,y），线性层权重\(W \in \mathbb{R}^{2 \times d}\)
2. **多层注意力更新**：每层注意力层由"多头注意力子层+FF子层"组成，含残差连接与ReZero归一化（附录A.1），更新公式：
   \[
   \hat{h}_i = h_i^{l-1} + \alpha^l \cdot MHA_i^l(h_1^{l-1}, h_2^{l-1}, ..., h_N^{l-1})
   \]
   \[
   h_i^l = \hat{h}_i + \alpha^l \cdot FF(\hat{h}_i)
   \]
   其中\(\alpha^l\)为ReZero归一化的可学习参数，MHA为多头注意力，因实例为全连接图，MHA覆盖所有节点对（含自连接），等效于自注意力机制
3. **实例表征聚合**：对最后一层输出的节点嵌入\(H^L \in \mathbb{R}^{N \times d}\)进行平均池化，得到实例表征向量\(o \in \mathbb{R}^d\)

#### 2. 分层图编码器（Hierarchical Graph Encoder）
##### （1）网络结构与参数
- 嵌入维度\(d=128\)
- 编码器块数量\(L=2\)块，每块含2个图注意力层
- 池化降维比例\(\alpha=0.8\)（每块保留前一层80%的节点）
- 激活函数：tanh（用于代表性分数计算）、sigmoid（用于Readout操作）

##### （2）分步实现流程
1. **初始嵌入**：与图注意力编码器一致，通过线性层将原始特征映射为\(H^0 \in \mathbb{R}^{N \times d}\)
2. **编码器块迭代（每块3步）**：
   - 步骤1：图注意力更新：通过2个图注意力层更新当前节点嵌入（同上述注意力层公式）
   - 步骤2：图池化（可导）：
     ① 计算代表性分数：通过额外注意力层生成分数嵌入\(H_{score}^l = AttentionLayer_{score}^l(H^l)\)，再通过线性层映射为标量分数：\(Z^l = \tanh(H_{score}^l W_{score}^l)\)，其中\(W_{score}^l \in \mathbb{R}^{d \times 1}\)，\(Z^l \in \mathbb{R}^{N^{l-1} \times 1}\)
     ② 节点筛选：按\(Z^l\)降序排序，选择前\(N^l = \alpha \cdot N^{l-1}\)个节点
     ③ 可导优化：将代表性分数与节点嵌入融合：\(\tilde{H}^l = H^l + Z^l \cdot \mathbb{1}\)，其中\(\mathbb{1} \in \mathbb{R}^{1 \times d}\)为全1向量
   - 步骤3：Readout操作：对池化后的节点嵌入进行均值池化+最大值池化，拼接后通过sigmoid激活：\(o^l = \sigma(Mean(H^l) \parallel Max(H^l))\)，得到当前块的层级特征\(o^l \in \mathbb{R}^{2d}\)（拼接后维度），再通过线性层映射回\(d\)维
3. **多尺度特征聚合**：堆叠2个块后，对最后一层节点嵌入执行Readout操作得到\(o^{L+1}\)，将所有层级特征求和：\(o = \sum_{l=1}^{L+1} o^l\)，最终实例表征\(o \in \mathbb{R}^d\)

#### 3. 实例规模融合
选择模型输入为"实例表征+实例规模N"：N为实例节点数，直接作为额外特征与编码器输出的\(d\)维向量拼接，形成最终输入特征（维度\(d+1=129\)）

### 三、选择模型实现（论文3.2）

#### 1. 模型结构
- **核心组件**：图编码器（上述两种之一）+ 多层感知机（MLP），端到端串联
- **MLP结构**：输入层（129维，实例表征+N）→ 隐藏层1（无明确维度，论文仅提"MLP"）→ 隐藏层2（无明确维度）→ 输出层（M维，M为求解器池规模：TSP=7，CVRP=5）
- **输出**：M维适配分数向量，分数越高表示求解器越适合当前实例

#### 2. 训练数据与监督信息
- **训练数据集**：含数千个合成COP实例（论文4.1明确为10,000个+8倍增强）
- **监督信息**：每个实例的"求解器目标函数值"——让所有候选求解器在该实例上运行，记录各自的目标函数值（TSP路径长度、CVRP总运输成本），以此推导监督信号

#### 3. 损失函数
##### （1）分类损失（Classification Loss）
- **监督信号**：每个实例的"最优求解器索引"（目标函数值最小的求解器），作为分类标签
- **损失函数**：交叉熵损失（论文明确提及）
- **训练目标**：最小化模型预测的适配分数与真实标签的交叉熵，学习"实例特征→最优求解器"的映射

##### （2）排序损失（Ranking Loss）
- **监督信号**：每个实例的"求解器优劣排序"\(\phi_I: [M] \to [M]\)，其中\(\phi_I(i)\)为排名第i的求解器索引（按目标函数值升序排列）
- **损失函数目标**：最大化模型输出分数符合真实排序的对数概率，公式：
  \[
  \max_{\theta} \mathbb{E}_I\left[\sum_{i=1}^M log \frac{exp(g_\theta(I)_{\phi_I(i)})}{\sum_{j=i}^M exp(g_\theta(I)_{\phi_I(j)})}\right]
  \]
  其中\(g_\theta\)为带参数\(\theta\)的选择模型，\(I\)为问题实例
- **训练目标**：让模型输出的适配分数顺序与真实求解器优劣顺序一致，利用所有求解器的相对关系提升鲁棒性

### 四、选择策略实现（论文3.3）

#### 1. 贪心选择（Greedy Selection）
- **核心逻辑**：直接选择选择模型输出的适配分数最高的求解器
- **执行流程**：对单个实例，计算适配分数向量→取最大值对应的求解器索引→调用该求解器并输出结果
- **特点**：效率最高（仅调用1个求解器），鲁棒性依赖选择模型预测精度

#### 2. Top-k选择（Top-k Selection）
- **核心参数**：\(k=2\)（论文4.1明确超参数）
- **执行流程**：对单个实例，计算适配分数向量→按分数降序排序→选择前k个求解器→调用所有k个求解器→取目标函数值最小的结果作为最终输出
- **特点**：精度高于贪心策略，时间成本为贪心策略的k倍（k=2时约2倍）

#### 3. 基于拒绝的选择（Rejection-based Selection）
- **核心逻辑**：按选择模型的置信度自适应选择贪心或Top-k策略
- **关键步骤**：
  1. **置信度计算**：将适配分数通过softmax归一化，取排名第一的求解器的归一化概率作为置信度：\(c = softmax(s)_1\)，其中\(s\)为适配分数向量
  2. **阈值设定**：通过验证集确定阈值\(\tau\)，拒绝置信度最低的20%实例（论文4.1超参数）
  3. **策略执行**：若\(c \geq \tau\)，采用贪心策略；若\(c < \tau\)，采用Top-k策略（k=2）
- **特点**：平衡精度与效率，仅20%实例调用Top-k

#### 4. Top-p选择（Top-p Selection）
- **核心参数**：TSP的\(p=0.5\)，CVRP的\(p=0.8\)（论文4.1明确）
- **执行流程**：
  1. **分数排序**：将适配分数按降序排列，得到\(s_{(1)} \geq s_{(2)} \geq ... \geq s_{(M)}\)
  2. **累积占比计算**：计算归一化后的累积占比\(\sum_{i=1}^k s_{(i)} / \sum_{j=1}^M s_{(j)}\)
  3. **求解器选择**：找到最小的k，使得累积占比≥p，选择前k个求解器→调用并取最优结果
- **特点**：自适应k值，无需手动调整，适配不同实例的分数分布

### 五、训练配置实现（论文4.1+附录A.2.3）

#### 1. 优化器与超参数
- **优化器**：Adam优化器
- **学习率**：\(1 \times 10^{-4}\)
- **权重衰减**：\(1 \times 10^{-6}\)
- **训练轮数（Epoch）**：50轮
- **批量大小（Batch Size）**：论文未明确，附录无补充（仅提及"训练5个模型"）
- **随机种子**：5个不同随机种子（用于结果的均值与标准差计算）

#### 2. 训练流程
1. **数据加载**：加载合成训练集（10,000个实例+8倍增强）、验证集（1000个合成实例）
2. **模型初始化**：初始化图编码器与MLP，设置优化器与损失函数
3. **训练循环**：
   - 每轮迭代：前向传播计算适配分数→计算损失→反向传播更新参数
   - 模型选择：通过验证集性能选择最优模型（论文提及"最终模型根据验证集性能选择"）
4. **结果汇总**：训练5个不同随机种子的模型，最终报告性能的均值与标准差

#### 3. 实例Tokenization的动量更新（论文4.3）
- **适用场景**：神经求解器特征提取（用于泛化到 unseen 求解器）
- **实现逻辑**：实例编码器参数\(\theta'\)采用动量更新：\(\theta' \leftarrow m \cdot \theta' + (1-m) \cdot \theta\)，其中\(m=0.99\)（动量系数），仅\(\theta\)通过反向传播更新，\(\theta'\)用于生成稳定的实例token
- **目的**：确保实例表征的稳定性，提升对 unseen 求解器的泛化能力

### 六、评估实现（论文4.1+附录A.2.3）

#### 1. 性能指标
##### （1）最优性差距（Optimality Gap）
- **核心指标**，公式：
  \[
  \text{Optimality Gap} = \frac{c_I(\hat{\sigma}) - c_I(\sigma^*)}{c_I(\sigma^*)}
  \]
  其中：
  - \(\hat{\sigma}\)：待评估方法（框架/单个求解器）输出的解
  - \(\sigma^*\)：专家求解器的最优已知解（TSP用LKH，CVRP用HGS，论文4.1明确引用Helsgaun, 2017; Vidal, 2022）
  - \(c_I(\cdot)\)：实例I的成本函数（路径长度/总运输成本）
- **结果表示**：百分比，数值越小性能越好

##### （2）平均推理时间
- **统计范围**：选择模型运行时间 + 选中求解器的运行时间
- **报告形式**：均值（论文表格中含标准差，基于5个随机种子）

#### 2. 评估数据集
- **合成测试集**：1000个TSP/CVRP实例（N∈[50,500]）
- **基准数据集**：TSPLIB（N≤1002）、CVRPLIB Set-X（N∈[100,1000]）
- **扩展数据集**：N∈[500,2000]的大规模合成实例（附录A.10）

#### 3. 评估流程
1. 对每个评估实例，通过选择模型计算适配分数
2. 按选定策略选择求解器并调用
3. 计算该实例的最优性差距与推理时间
4. 汇总所有实例的指标，得到平均最优性差距与平均推理时间

### 七、神经求解器特征提取实现（论文4.3+附录）

#### 1. 零样本泛化的具体实现
- **步骤1**：对目标求解器，在合成数据集上运行，筛选"该求解器表现最优"的实例
- **步骤2**：按"求解器目标函数值/次优求解器目标函数值"的比值升序排序，选择前1%实例作为该求解器的"代表性实例"
- **步骤3**：用分层图编码器对代表性实例编码，得到实例token向量
- **步骤4**：通过2层Transformer模型处理token向量，输出该求解器的特征表征
- **目的**：无需微调选择模型，即可将新求解器纳入池中，实现零-shot泛化

#### 2. 消融实验相关实现
- **特征提取对比**：手动特征（Smith-Miles et al., 2010）、图注意力编码器、分层图编码器（论文4.3+表3）
- **损失函数对比**：分类损失、排序损失（表1、表2）
- **选择策略对比**：4种策略的超参数调优（附录A.8），如拒绝策略的k=2/3/4、Top-p的p=0.4~0.95

## 实验结果

### 性能对比
- **TSP合成数据**：1.51% vs DIFUSCO 2.33%（提升0.82%）
- **CVRP合成数据**：4.82% vs Omni 6.82%（提升2.00%）
- **TSPLIB**：1.07% vs T2T 1.95%（提升0.88%）
- **CVRPLIB**：5.39% vs ELG 6.10%（提升0.71%）

### 组件效果分析
- **特征提取**：层次化编码器 > 图注意力编码器 > 手工特征
- **损失函数**：排名损失在分布外泛化上优于分类损失
- **选择策略**：Top-k和拒绝策略提供最佳性能-时间权衡

## 核心架构实现

### 主要组件
- **Selection_model (model.py)**: 主选择模型，支持层次化编码和神经求解器特征
- **trainer (trainer.py)**: 训练和测试流程，支持排名损失和交叉熵损失
- **SelectionDataset (dataset.py)**: 数据集类，支持数据增强和手动特征
- **run.py**: 主入口，处理配置、日志、数据准备

### 模型类型
- **Selection_model**: 基于Transformer的选择模型，支持层次化池化
- **Naive_classifier**: 基于手动特征的简单分类器
- **Encoder_h**: 层次化图编码器（论文核心创新）
- **Naive_Encoder**: 基础图注意力编码器

### 配置系统
- 使用YAML配置文件：`config_TSP.yml`、`config_CVRP.yml`
- 支持两种损失函数：`rank`（排名损失）、`CE`（交叉熵）
- 可配置模型参数：嵌入维度、注意力头数、层数等

### 特性支持
- **层次化编码**: 通过下采样实现多层图表示（`pooling: True`）
- **神经求解器特征**: ns_feature模式支持代表性特征学习
- **数据增强**: 8倍旋转对称增强（`data_aug: True`）
- **手动特征**: 可选的传统优化特征（`manual_feature: True`）

## 文件结构
```
├── run.py                    # 主入口
├── model.py                  # 选择模型架构
├── trainer.py                # 训练和测试逻辑
├── dataset.py                # 数据集处理
├── utils.py                  # 工具函数
├── loss.py                   # 损失函数
├── config_*.yml             # 配置文件
├── run_*.sh                 # 执行脚本
├── experiments/             # 实验生成脚本
├── datasets/                # 数据处理相关
└── train_logs/             # 训练日志和检查点
```

## 数据格式
- **TSP**: 节点坐标 (x, y)
- **CVRP**: 节点坐标 + 需求 (x, y, demand)
- 标签: 最佳求解器索引和性能指标
- 支持TSPLIB和CVRPLIB标准测试集

## 训练设置
- 默认batch_size: 64 (训练), 8 (测试)
- 学习率: 1e-4
- 训练轮数: 50 epochs
- 优化器: Adam
- GPU训练支持

## 测试评估
支持多种评估策略：
- Top-k选择
- Top-p概率阈值选择
- 拒绝策略（基于置信度）
- 与最优单个求解器对比

---

## 八、详细代码实现解析：三组件框架对应关系

本节将详细解析代码实现，并明确标识每个代码模块与论文三组件框架（特征提取、选择模型、选择策略）的对应关系。

### 1. 特征提取组件 (Feature Extraction)

特征提取组件负责从组合优化问题实例中提取有效特征，包括图神经网络编码器和数据预处理。

#### 1.1 数据预处理与特征工程 (dataset.py)

##### 数据增强：提升泛化能力
```python
def augment_xy_by_8_fold(problems):
    """
    [特征提取] 数据增强函数
    论文引用：Kwon et al. (2020)的POMO增强方法
    目的：通过8种对称变换增加数据多样性，提升模型泛化性

    Args:
        problems: (batch, problem, 2) 节点坐标张量

    Returns:
        aug_problems: (batch*8, problem, 2) 增强后的坐标张量
    """
    x = problems[:, :, [0]]  # 提取x坐标
    y = problems[:, :, [1]]  # 提取y坐标

    # [特征提取] 8种对称变换：保持欧几里得距离不变性
    dat1 = torch.cat((x, y), dim=2)           # 原始坐标 (x,y)
    dat2 = torch.cat((1 - x, y), dim=2)       # 水平翻转 (1-x,y)
    dat3 = torch.cat((x, 1 - y), dim=2)       # 垂直翻转 (x,1-y)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)   # 双重翻转 (1-x,1-y)
    dat5 = torch.cat((y, x), dim=2)           # 对角线翻转 (y,x)
    dat6 = torch.cat((1 - y, x), dim=2)       # 对角线+水平翻转 (1-y,x)
    dat7 = torch.cat((y, 1 - x), dim=2)       # 对角线+垂直翻转 (y,1-x)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)   # 完全翻转 (1-y,1-x)

    # [特征提取] 将8种变换结果拼接，实现8倍数据增强
    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    return aug_problems
```

##### 手工特征提取：传统优化特征
```python
def manual_features(nodes):
    """
    [特征提取] 手工特征提取函数
    论文引用：Smith-Miles et al. (2010)的传统优化特征
    目的：提取问题的统计和结构特征，用于基于规则的特征提取方法

    Args:
        nodes: (problem, 2) 或 (problem, 3) 节点坐标，TSP为(x,y)，CVRP为(x,y,demand)

    Returns:
        features: 手工提取的特征向量
    """
    nodes = nodes.squeeze(0)
    problem_type = 'TSP'
    if nodes.shape[-1] == 3:
        problem_type = 'CVRP'
        demands = nodes[:, 2]  # CVRP需求信息
        nodes = nodes[:, :2]   # 提取坐标部分

    # [特征提取] 计算距离矩阵 - 图结构的基础
    dist_mat = (nodes[:, None, :] - nodes[None, :, :]).norm(p=2, dim=-1)

    # [特征提取] 距离统计特征：捕获问题实例的全局特性
    std = torch.std(dist_mat.reshape(1, -1).squeeze(0), dim=-1, keepdim=True)   # 距离标准差
    centroid = nodes.mean(dim=0, keepdim=True)  # 节点重心
    radius = (nodes - centroid).norm(dim=1).mean(dim=0, keepdim=True)   # 平均半径

    # [特征提取] 图结构特征：反映图的拓扑性质
    count_distinct = (torch.bincount(torch.round(dist_mat * 100).int().reshape(1, -1).squeeze(0)) == 0).sum()
    nNN, _ = torch.min(dist_mat + 1e3 * torch.eye(nodes.shape[0], device=dist_mat.device), dim=-1)
    std_nNN = torch.std(nNN)    # 最近邻距离的标准差

    # [特征提取] 聚类特征：识别节点的空间分布模式
    results = HDBSCAN().fit(nodes)  # 使用HDBSCAN聚类算法
    cluster_ratio = np.max(results.labels_) / nodes.shape[0]    # 聚类比例
    outlier_ratio = (results.labels_ == -1).sum() / nodes.shape[0]   # 离群点比例

    # [特征提取] 根据问题类型构造特征向量
    if problem_type == 'TSP':
        features = torch.zeros(9)  # TSP: 9维特征
    else:
        features = torch.zeros(11) # CVRP: 11维特征（包含需求特征）

    # 填充特征值
    features[0] = std                                    # 距离标准差
    features[1:3] = centroid                            # 重心坐标
    features[3] = radius                                # 平均半径
    features[4] = count_distinct                        # 不同距离值数量
    features[5] = std_nNN                               # 最近邻距离标准差
    features[6] = cluster_ratio                         # 聚类比例
    features[7] = outlier_ratio                         # 离群点比例
    features[8] = torch.tensor(radius_cluster)           # 聚类内平均半径

    # CVRP特有的需求特征
    if problem_type == 'CVRP':
        features[9] = demands.mean()    # 平均需求
        features[10] = demands.std()    # 需求标准差

    return features
```

#### 1.2 图神经网络编码器 (model.py)

##### 层次化图编码器：论文核心创新
```python
class Encoder_h(nn.Module):
    """
    [特征提取] 层次化图编码器 - 论文核心技术创新
    论文3.1: Hierarchical Graph Encoder
    目的：通过图池化捕获COP实例的层次化结构特征

    创新点：
    1. 可微分图池化：通过学习选择代表性节点
    2. 多尺度特征融合：结合不同抽象层次的图表示
    3. Readout机制：聚合全局和局部特征
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']

        # [特征提取] 节点嵌入层：将原始坐标映射到高维空间
        if model_params['problem_type'] == 'TSP':
            self.embedding = nn.Linear(2, embedding_dim)  # TSP: (x,y) -> embedding_dim
        if model_params['problem_type'] == 'CVRP':
            self.embedding = nn.Linear(3, embedding_dim)  # CVRP: (x,y,demand) -> embedding_dim

        # [特征提取] 多个编码器块的堆叠，实现层次化处理
        self.blocks = nn.ModuleList([Encoder_block_h(**model_params)
                                   for _ in range(model_params['block_num'])])
        self.nonlinear = nn.GELU()  # Readout层的激活函数

    def masked_mean(self, embs, mask):
        """[特征提取] 带掩码的均值池化"""
        emb_mask = torch.where(mask == float('-inf'), 0, 1)
        embs = embs * emb_mask[:, :, None].expand_as(embs)
        mean_emb = embs.sum(dim=1) / emb_mask.sum(-1)[:, None]
        return mean_emb

    def masked_max(self, embs, mask):
        """[特征提取] 带掩码的最大值池化"""
        embs = embs + mask[:, :, None].expand_as(embs)
        max_emb = embs.max(dim=1)[0]
        return max_emb

    def forward(self, data, mask):
        """
        [特征提取] 前向传播：实现层次化特征提取

        Args:
            data: (batch, problem, 2/3) 原始节点坐标
            mask: (batch, problem) 注意力掩码

        Returns:
            graph_emb_h: (batch, 2*embedding_dim) 层次化图表示
        """
        # [特征提取] 步骤1: 初始节点嵌入
        out = self.embedding(data)  # (batch, problem, embedding_dim)

        # [特征提取] 步骤2: 层次化嵌入处理
        i = 0
        graph_emb_h = 0.  # 累积各层的图表示
        for block in self.blocks:
            # 每个编码器块返回：当前层图表示、池化后节点嵌入、更新后掩码
            graph_emb, out, mask = block(out, mask, i)
            graph_emb_h += graph_emb  # [特征提取] 累积多尺度特征
            i += 1

        # [特征提取] 步骤3: 最终层Readout操作
        mean_emb = self.masked_mean(out, mask)  # 均值池化
        max_emb = self.masked_max(out, mask)    # 最大值池化
        graph_emb = self.nonlinear(torch.cat((mean_emb, max_emb), dim=1))
        graph_emb_h += graph_emb  # 添加最终层表示

        return graph_emb_h

class Encoder_block_h(nn.Module):
    """
    [特征提取] 层次化编码器块 - 实现可微分图池化
    目的：通过注意力更新和节点选择，构建层次化的图表示
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        encoder_layer_num = self.model_params['encoder_layer_num']

        # [特征提取] 多层图注意力网络
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

        # [特征提取] 节点重要性评分网络
        self.layer_score = EncoderLayer(**model_params)
        self.p = nn.Linear(model_params['embedding_dim'], 1)  # 评分映射层
        self.act = nn.Tanh()  # 激活函数
        self.nonlinear = nn.GELU()  # Readout激活函数

    def padding_concate(self, list_):
        """[特征提取] 批处理填充：处理不同长度的序列"""
        lengths = torch.tensor([t.shape[0] for t in list_], device=list_[0].device)
        max_len = lengths.max().item()
        mask = torch.zeros(len(list_), max_len)
        for i, t in enumerate(list_):
            list_[i] = F.pad(t, (0, 0, 0, max_len - lengths[i]))[None, :, :]
            mask[i, lengths[i]: max_len] = float('-inf')
        embs = torch.cat(list_, dim=0)
        return embs, mask

    def forward(self, embs, mask, i):
        """
        [特征提取] 编码器块前向传播

        Args:
            embs: (batch, problem, embedding_dim) 输入节点嵌入
            mask: (batch, problem) 注意力掩码
            i: 当前块的索引

        Returns:
            graph_emb: (batch, 2*embedding_dim) 当前块的图表示
            selected_embs: (batch, selected_problem, embedding_dim) 池化后节点嵌入
            selected_mask: (batch, selected_problem) 池化后掩码
        """
        # [特征提取] 步骤1: 多层图注意力更新
        for layer in self.layers:
            embs = layer(embs, mask)

        # [特征提取] 步骤2: 当前块Readout操作
        mean_emb = self.masked_mean(embs, mask)
        max_emb = self.masked_max(embs, mask)
        graph_emb = self.nonlinear(torch.cat((mean_emb, max_emb), dim=1))

        # [特征提取] 步骤3: 可微分图池化
        score_embs = self.layer_score(embs, mask)  # 计算节点代表性分数
        scores = self.act(self.p(score_embs))      # tanh激活归一化
        scores = scores.squeeze(-1)
        scores = scores + mask                     # 应用掩码

        # [特征提取] 步骤4: 选择代表性节点
        selected_embs_list = []
        lengths = (mask == 0).sum(dim=-1, keepdim=True)
        for batch_idx in range(embs.shape[0]):
            # 根据池化比例选择节点数量
            num_selected = int(lengths[batch_idx] * self.model_params['downsample_ratio'])
            score, ind = scores[batch_idx].topk(num_selected, dim=-1, largest=True)
            selected_emb = embs[batch_idx].take_along_dim(ind[:, None].expand(-1, embs.shape[-1]), dim=0)

            # [特征提取] 关键创新：将代表性分数与节点嵌入融合，使池化可微分
            selected_emb = selected_emb + score[:, None]
            selected_embs_list.append(selected_emb)

        # [特征提取] 步骤5: 批处理填充和掩码生成
        selected_embs, selected_mask = self.padding_concate(selected_embs_list)

        return graph_emb, selected_embs, selected_mask
```

##### 图注意力层：基础编码单元
```python
class EncoderLayer(nn.Module):
    """
    [特征提取] 图注意力层 - 基础编码单元
    论文基础：Veličković et al. (2018) + Kool et al. (2019)
    目的：通过多头自注意力机制更新节点表示
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # [特征提取] 多头注意力线性变换层
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)  # 查询变换
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)  # 键变换
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)  # 值变换
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)  # 输出变换

        # [特征提取] Transformer架构组件
        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)  # 残差连接+归一化1
        self.feedForward = Feed_Forward_Module(**model_params)                   # 前馈网络
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)  # 残差连接+归一化2

    def forward(self, input1, mask=None, edges=None, kv=None):
        """
        [特征提取] 图注意力层前向传播

        Args:
            input1: (batch, problem, embedding_dim) 输入节点嵌入
            mask: (batch, problem, problem) 注意力掩码
            kv: 键值张量（用于交叉注意力）

        Returns:
            out3: (batch, problem, embedding_dim) 更新后的节点嵌入
        """
        head_num = self.model_params['head_num']
        if kv is None:
            kv = input1  # 自注意力模式

        # [特征提取] 多头注意力计算
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)      # 查询向量
        k = reshape_by_heads(self.Wk(kv), head_num=head_num)         # 键向量
        v = reshape_by_heads(self.Wv(kv), head_num=head_num)         # 值向量

        # [特征提取] 缩放点积注意力
        out_concat = multi_head_attention(q, k, v, rank2_ninf_mask=mask)

        # [特征提取] 多头结果合并
        multi_head_out = self.multi_head_combine(out_concat)

        # [特征提取] 残差连接和归一化
        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)                                    # 前馈网络
        out3 = self.addAndNormalization2(out1, out2)                     # 第二个残差连接

        return out3
```

### 2. 选择模型组件 (Selection Model)

选择模型组件基于提取的特征，学习实例与求解器之间的兼容性关系。

#### 2.1 主选择模型
```python
class Selection_model(nn.Module):
    """
    [选择模型] 主选择模型 - 实现实例到求解器的映射
    论文3.2: Selection Model
    目的：基于实例特征预测每个求解器的兼容性分数

    核心功能：
    1. 特征融合：结合图编码特征、手工特征、实例规模
    2. 兼容性预测：输出每个求解器的适配分数
    3. 神经求解器特征：支持零样本泛化
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        # [选择模型] 根据配置选择编码器类型
        if model_params['pooling']:
            self.encoder = Encoder_h(**model_params)  # 层次化图编码器
            feature_dim = 2 * model_params['embedding_dim'] + 1  # 均值+最大值+规模
        else:
            self.encoder = Naive_Encoder(**model_params)  # 基础图注意力编码器
            feature_dim = model_params['embedding_dim'] + 1  # 单一表示+规模

        # [选择模型] 神经求解器特征扩展机制（论文4.3创新）
        if self.model_params['ns_feature'] == True:
            model_params_ = model_params.copy()
            model_params_['embedding_dim'] = feature_dim
            # [选择模型] 代表性网络：学习求解器特征表示
            self.representative_net = representative_net(**model_params_)
            self.init_tokens = nn.Parameter(torch.Tensor(1, feature_dim))
            self.init_tokens.data.uniform_(-1, 1)
            self.model_tokens = []
            feature_dim = 2 * feature_dim  # 实例特征+求解器特征
            # [选择模型] 相似性计算网络：计算实例-求解器兼容性
            self.similarity = nn.Sequential(
                nn.Linear(feature_dim, model_params['embedding_dim']),
                nn.GELU(),
                nn.Linear(model_params['embedding_dim'], 1))
        else:
            # [选择模型] 传统MLP分类器：固定求解器索引模式
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, model_params['embedding_dim']),
                nn.GELU(),
                nn.Linear(model_params['embedding_dim'], model_params['output_dim']))

    def acquire_feature(self, points, scales, mask):
        """
        [选择模型] 获取实例特征表示
        目的：单独提取图编码特征，用于特征分析
        """
        graph_emb = self.encoder(points, mask)
        instance_feature = torch.cat((
            graph_emb,
            scales[:, None]
        ), dim=1)
        return instance_feature

    def update_tokens(self, representative_features):
        """
        [选择模型] 更新神经求解器特征token
        目的：为每个求解器计算代表性特征向量
        """
        self.model_tokens = []  # 重置token列表
        for i in range(self.init_tokens.shape[0]):
            self.model_tokens.append(self.representative_net(self.init_tokens[i], representative_features[i]))

    def forward(self, points, scales, manual_features, mask, return_feature=False):
        """
        [选择模型] 前向传播：预测求解器兼容性分数

        Args:
            points: (batch, problem, 2/3) 节点坐标
            scales: (batch,) 实例规模
            manual_features: (batch, feature_dim) 手工特征
            mask: (batch, problem) 注意力掩码
            return_feature: 是否返回特征表示

        Returns:
            probs: (batch, num_solvers) 求解器兼容性分数
        """
        # [选择模型] 步骤1: 图编码器提取实例特征
        graph_emb = self.encoder(points, mask)

        # [选择模型] 步骤2: 特征融合
        if manual_features == None:
            manual_features = scales[:, None]  # 仅使用实例规模
        else:
            # [选择模型] 融合手工特征和实例规模
            manual_features = torch.cat((manual_features, scales[:, None]), dim=-1)

        # [选择模型] 构建最终实例特征：图嵌入 + 手工特征 + 实例规模
        self.instance_feature = torch.cat((
            graph_emb,
            manual_features
        ), dim=1)

        # [选择模型] 步骤3: 求解器兼容性分数计算
        if self.model_params['ns_feature'] == True:
            # [选择模型] 神经求解器特征模式：支持零样本泛化
            probs = []
            batch_size = self.instance_feature.shape[0]
            for i in range(len(self.model_tokens)):
                # 计算实例特征与每个求解器特征的相似度
                instance_solver_feature = torch.cat((
                    self.instance_feature,
                    self.model_tokens[i][None, :].expand(batch_size, -1)
                ), dim=-1)
                probs.append(self.similarity(instance_solver_feature))
            probs = torch.cat(probs, dim=-1)
        else:
            # [选择模型] 固定求解器索引模式
            probs = self.classifier(self.instance_feature)

        return probs
```

#### 2.2 损失函数：监督学习机制
```python
class RankingLoss(nn.Module):
    """
    [选择模型] 排名损失函数 - 论文3.2核心创新
    目的：学习求解器的相对性能排序，而非仅识别最优求解器

    优势：
    1. 鲁棒性：利用所有求解器的相对关系
    2. 数据利用率：充分利用性能信息
    3. 泛化性：更好的分布外泛化能力
    """
    def __init__(self, num_solvers, top_k):
        super().__init__()
        self.num_solvers = num_solvers  # 求解器总数
        self.top_k = top_k              # 考虑前k个排名

    def forward(self, logits, costs):
        """
        [选择模型] 排名损失计算

        Args:
            logits: (batch_size, num_solvers) 模型预测的兼容性分数
            costs: (batch_size, num_solvers) 各求解器的真实成本

        Returns:
            loss: 排名损失值
        """
        loss = 0
        # [选择模型] 多层次排名学习：不仅学习最优，还学习次优、第三优等
        for i in range(self.top_k):
            # [选择模型] 步骤1: 找到从第i名到最后的求解器
            cur_cost, ind = costs.topk(self.num_solvers - i, largest=True)
            # [选择模型] 步骤2: 在这些求解器中找到最优的（成本最小的）
            cur_label = cur_cost.min(dim=1)[1]
            # [选择模型] 步骤3: 提取对应的模型预测分数
            cur_logits = torch.take_along_dim(logits, ind, 1)
            # [选择模型] 步骤4: 计算交叉熵损失，鼓励模型给最优求解器更高分数
            loss += F.nll_loss(F.log_softmax(cur_logits, 1), cur_label)
        return loss
```

### 3. 选择策略组件 (Selection Strategies)

选择策略组件基于选择模型的输出，决定实际调用哪些求解器。

#### 3.1 评估与策略实现 (trainer.py)

##### 多策略评估系统
```python
def test(self, epoch, test_dataloader, representative_set=None):
    """
    [选择策略] 测试评估函数 - 实现多种选择策略
    论文3.3: Selection Strategies
    目的：评估不同选择策略的性能

    策略类型：
    1. 贪心选择 (Greedy Selection)
    2. Top-k选择 (Top-k Selection)
    3. 基于拒绝的选择 (Rejection-based Selection)
    4. Top-p选择 (Top-p Selection)
    """
    # 收集所有预测结果和真实性能
    score_mat = []  # [选择策略] 模型预测的兼容性分数
    gap_mat = []    # [选择策略] 各求解器的最优性差距
    time_mat = []   # [选择策略] 各求解器的运行时间
    num_instances = 0.

    # [选择策略] 前向传播收集预测结果
    self.model.eval()
    for batch in test_dataloader:
        # ... 数据预处理 ...
        y_pred = self.model(x, scales, manual_feature, mask)
        score_mat.append(F.softmax(y_pred, 1).detach())  # [选择策略] 兼容性分数
        time_mat.append(time_cost)                        # [选择策略] 运行时间
        gap_mat.append(gap)                               # [选择策略] 最优性差距

    # [选择策略] Top-k选择策略评估
    k_list = [1, 2, 3, 4]
    for k in k_list:
        # [选择策略] 选择兼容性分数最高的k个求解器
        _, topk_ind = score_mat.topk(k, 1, largest=True)
        # [选择策略] 取这些求解器中的最优结果
        topk_gap = gap_mat.gather(1, topk_ind).min(dim=1)[0]
        # [选择策略] 计算总运行时间（选择时间+求解器时间）
        topk_time = time_mat.gather(1, topk_ind).sum(dim=1)
        print("Top-{} gap mean: {:.4f}%, {:.4f}s".format(k, topk_gap.mean(), topk_time.mean() + select_time))

    # [选择策略] 基于拒绝的选择策略评估
    if k >= 2:
        # [选择策略] 按置信度排序：使用最大softmax响应作为置信度度量
        sort_ind = score_mat.max(dim=1)[0].sort(descending=True)[1].cpu().numpy()
        threshold = int(num_instances * 0.8)  # [选择策略] 保留80%高置信度实例

        # [选择策略] 低置信度实例：使用Top-k策略
        reject_ind = sort_ind[threshold:]
        # [选择策略] 高置信度实例：使用贪心策略
        accept_ind = sort_ind[:threshold]

        # [选择策略] 组合结果
        gap_SR = torch.cat((topk_gap[reject_ind], top_1_gap[accept_ind]), dim=0)
        time_SR = torch.cat((topk_time[reject_ind], top_1_time[accept_ind]), dim=0)
        print("Rejection 20%: {:.4f}%, {:.4f}s".format(gap_SR.mean(), time_SR.mean() + select_time))

    # [选择策略] Top-p选择策略评估
    p_values = [0.8]
    for p in p_values:
        times = []
        gaps = []
        for i in range(len(score_mat)):
            # [选择策略] 寻找最小求解器子集，使得归一化分数和达到p
            for j in range(1, score_mat.shape[1] + 1):
                top_j, ind = score_mat[i].topk(j, largest=True)
                if top_j.sum() >= p:  # [选择策略] 累积分数达到阈值p
                    times.append(time_mat[i][ind].sum().item())
                    gaps.append(gap_mat[i][ind].min().item())
                    break
        print("Top p {}%: {:.4f}%, {:.4f}s".format(100 * p, np.array(gaps).mean(), np.array(times).mean() + select_time))
```

##### 神经求解器特征计算：零样本泛化
```python
def compute_representative(self, representative_set):
    """
    [选择策略] 计算神经求解器代表性特征 - 论文4.3创新
    目的：实现零样本泛化，支持新的求解器无需重新训练

    流程：
    1. 选择每个求解器的代表性实例
    2. 使用动量编码器生成稳定的实例表示
    3. 聚合形成求解器特征向量
    """
    representative_feature = []
    representative_data = representative_set[0]  # 各求解器的代表性实例
    representative_label = representative_set[1]

    for i in range(len(representative_data)):
        # [选择策略] 为每个求解器创建数据集
        dataset = SelectionDataset(representative_data[i], representative_label[i])
        dataloader = DataLoader(dataset, collate_fn=collate_fn,
                               batch_size=len(dataset), generator=torch.Generator(device=self.device))

        for batch in dataloader:
            x = batch[0].to(self.device)
            scales = batch[3].to(self.device)
            mask = batch[4].to(self.device)
            with torch.no_grad():
                # [选择策略] 使用动量编码器生成稳定的实例表示
                instance_embedding = self.encoder_p(x, mask)
                # [选择策略] 融合实例规模信息
                representative_feature.append(
                    torch.cat((instance_embedding, scales[:, None]), dim=1))

    return representative_feature
```

#### 3.2 代表性实例选择算法
```python
def representative(dataset, labels, ratio=0.01):
    """
    [选择策略] 代表性实例选择算法 - 论文4.3核心算法
    目的：为每个神经求解器选择表现最好的代表性实例

    算法流程：
    1. 筛选每个求解器表现最优的实例
    2. 计算性能优势比率（最优/次优）
    3. 选择比率最小的前1%作为代表性实例
    """
    representative_data = []
    representative_label = []

    # [选择策略] 为每个求解器选择代表性实例
    for k in range(len(labels[0][1])):  # 遍历所有求解器
        data_per_solver = []
        label_per_solver = []
        gaps = []  # 性能优势度量
        idx_data = []

        # [选择策略] 步骤1: 收集该求解器表现最优的实例
        for i in range(len(dataset)):
            if labels[i][0] == k:  # 如果该实例的最优求解器是k
                cost = torch.tensor(np.array(labels[i][1]))
                cost, _ = cost.topk(2, largest=False)  # 找到最优和次优性能
                # [选择策略] 计算性能比率：最优/次优，越小表示优势越大
                gaps.append(cost[0] / cost[1])
                idx_data.append(i)

        # [选择策略] 步骤2: 选择性能优势最大的前1%实例
        num = int(len(gaps) * ratio)
        gaps = torch.tensor(np.array(gaps))
        top_gaps, idx_sel = gaps.topk(num, largest=False)  # 选择比率最小的实例

        # [选择策略] 步骤3: 构建代表性实例集
        idx = [idx_data[j] for j in idx_sel]
        for j in idx:
            data_per_solver.append(dataset[j])
            label_per_solver.append(labels[j])

        representative_data.append(data_per_solver)
        representative_label.append(label_per_solver)

    return [representative_data, representative_label]
```

### 4. 训练与动量更新机制

#### 4.1 动量更新：稳定特征学习
```python
@torch.no_grad()
def _momentum_update_representative_encoder(self):
    """
    [选择模型] 动量更新关键编码器 - 论文4.3技术创新
    目的：确保实例表征的稳定性，提升零样本泛化能力

    动量更新公式：θ' ← m·θ' + (1-m)·θ
    其中 m=0.99 为动量系数
    """
    for param, param_p in zip(self.model.encoder.parameters(), self.encoder_p.parameters()):
        # [选择模型] 动量更新：平滑参数更新轨迹
        param_p.data = param_p.data * self.m + param.data * (1.0 - self.m)
```

#### 4.2 训练循环：三组件协同训练
```python
def train_one_epoch(self, epoch, train_dataloader, representative_set=None):
    """
    训练循环：三组件框架的协同训练
    """
    # [特征提取] 计算代表性实例特征（用于神经求解器特征）
    representative_feature = self.compute_representative(representative_set) if representative_set is not None else None

    self.model.train()
    for batch in tqdm(train_dataloader):
        # [特征提取] 数据准备：坐标、标签、成本、规模、掩码
        x = batch[0].to(self.device)          # 节点坐标
        y = batch[1].to(self.device)          # 最优求解器标签
        cost = batch[2].to(self.device)       # 所有求解器的成本
        scales = batch[3].to(self.device)     # 实例规模
        mask = batch[4].to(self.device)       # padding mask

        # [选择模型] 前向传播：特征提取 + 兼容性预测
        if representative_set is not None:
            self.model.update_tokens(representative_feature)
        y_pred = self.model(x, scales, manual_feature, mask)

        # [选择模型] 损失计算：排名损失或分类损失
        self.optimizer.zero_grad()
        if self.train_params['loss'] == 'CE':
            l = self.criterion(F.log_softmax(y_pred, 1), y)  # 分类损失
        if self.train_params['loss'] == 'rank':
            l = self.criterion(y_pred, cost)                  # 排名损失

        # [选择模型] 反向传播和参数更新
        l.backward()
        self.optimizer.step()

        # [选择模型] 动量更新（神经求解器特征模式）
        if representative_set is not None:
            self._momentum_update_representative_encoder()
            representative_feature = self.compute_representative(representative_set)
```

这个详细的代码实现解析明确展示了每个模块与论文三组件框架的对应关系，并通过详细注释说明了代码的技术创新点和实现原理。

### 九、数据生成与实验管理系统

#### 1. 数据生成系统 (datasets/)

##### (1) 合成数据生成流程 (data_utils.py)
```python
# [特征提取] 数据准备模块：生成符合论文4.1要求的合成数据
def generate_tsp_data_gaussian(dataset_size, problem_size, params):
    '''
    生成高斯混合分布的TSP数据，对应论文4.1的数据集构建
    params: ['var_lower', 'var_upper', 'num_modes_lower', 'num_modes_upper']
    '''
    if params['num_modes_lower'] == params['num_modes_upper']:
        num_modes = params['num_modes_lower']
    else:
        # 高斯混合成分数c ~ U(0,15)，实现论文中的分布采样
        num_modes = np.random.randint(params['num_modes_lower'], params['num_modes_upper'])

    if num_modes == 0:
        # c=0时为均匀分布，符合论文规范
        problems = torch.rand(size=(dataset_size, problem_size, 2)).numpy().tolist()
    else:
        problems = []
        for i in range(dataset_size):
            # 按论文采样混合比例和节点分配
            mix_proportion = np.random.rand(num_modes)
            nums = np.random.multinomial(problem_size, mix_proportion / np.sum(mix_proportion))
            xy = []
            for num in nums:
                # 实现论文中的方差和协方差采样
                if params['no_cov']:
                    var = np.random.uniform(params['var_lower'], params['var_upper'])
                    cov = [[var, 0], [0, var]]
                else:
                    var_x = np.random.uniform(params['var_lower'], params['var_upper'])
                    var_y = np.random.uniform(params['var_lower'], params['var_upper'])
                    # 协方差采样：cov ~ [-√(var_x·var_y), √(var_x·var_y)]
                    cov_xy = np.random.uniform(-np.sqrt(var_x * var_y), np.sqrt(var_x * var_y))
                    cov = [[var_x, cov_xy], [cov_xy, var_y]]
                # 均值采样：μ = (x_μ, y_μ) ~ U(0,1)×U(0,1)，缩放至[0,100]
                center = np.random.uniform(0, 100, size=(1, 2))
                nxy = np.random.multivariate_normal(mean=center.squeeze(), cov=cov, size=(num,))
                xy.extend(nxy)
            # 坐标缩放至[0,1]×[0,1]，符合论文要求
            xy = np.array(xy)
            xy = MinMaxScaler().fit_transform(xy).tolist()
            problems.append(xy)

    return problems

def generate_vrp_data(dataset_size, problem_size, config, distribution, capacity_type):
    '''[特征提取] CVRP数据生成：节点坐标+车辆容量+需求'''
    # 节点坐标复用TSP生成逻辑
    node_xy = generate_tsp_data(dataset_size, problem_size, config, distribution)
    # 仓库位置：均匀分布
    depot_xy = torch.rand(size=(dataset_size, 1, 2))

    if capacity_type == 'scale':
        # 规模相关分布：Q=30+⌈N/5⌉，实现论文第一种容量分布
        demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(0, 9).int() + 1).float()
        capacities = torch.ceil(torch.tensor(30 + problem_size / 5)).repeat(dataset_size)

    elif capacity_type == 'triangular':
        # 三角分布：实现论文第二种容量分布
        choice = random.choice(['a', 'b', 'c', 'd'])
        if choice == 'a':
            demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(0, 9).int() + 1).float()
        elif choice == 'b':
            demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(4, 9).int() + 1).float()
        elif choice == 'c':
            demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(0, 99).int() + 1).float()
        elif choice == 'd':
            demand = (torch.FloatTensor(dataset_size, problem_size).uniform_(49, 99).int() + 1).float()

        # 三角分布采样：T(lb,m,ub)，参考CVRPLIB Set-X设置
        route_length = torch.tensor(np.random.triangular(3, 6, 25, size=dataset_size))
        capacities = torch.ceil(route_length * demand.sum(1) / problem_size)

    # 数据组装：loc(节点坐标)+demand(归一化需求)+depot(仓库位置)
    data = {
        'loc': node_xy,
        # 需求按车辆容量Q归一化：m_i/Q
        'demand': (demand / capacities[:, None]).float(),
        'depot': depot_xy
    }
    return data
```

##### (2) 批量数据生成配置 (generate_data.py + data_config.yml)
```python
# [特征提取] 论文数据集构建的实现：10,000个训练实例
def main():
    with open('data_config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)

    # 支持多种分布类型，论文主要使用Gaussian
    distribution = config['distribution'].split(',')
    capacity_type = config['capacity'].split(',')

    # 实例规模采样：N ∈ [50,500]，符合论文训练集设置
    seed_everything(config['seed'])
    problems = []

    for d in distribution:
        for c in capacity_type:
            for i in tqdm(range(int(config['dataset_size'] / (len(distribution) * len(capacity_type))))):
                if config['problem_size_lower'] == config['problem_size_upper']:
                    problem_size = config['problem_size_lower']
                else:
                    # 随机采样实例规模
                    problem_size = np.random.randint(config['problem_size_lower'], config['problem_size_upper'])

                if config['problem_type'] == 'TSP':
                    xy = generate_tsp_data(1, problem_size, config, d)
                elif config['problem_type'] == 'CVRP':
                    xy = generate_vrp_data(1, problem_size, config, d, c)
                problems.append(xy)

    save_dataset(problems, config['save_path'] + '/dataset.pkl')
```

**YAML配置文件映射**：
```yaml
# data_config.yml - 对应论文4.1的数据生成参数
save_path: TSPtrain              # 保存路径
seed: 4321                       # 随机种子
dataset_size: 1000000            # 数据集大小（这里包含增强）
problem_type: TSP                # 问题类型
problem_size_lower: 50           # 最小规模
problem_size_upper: 500          # 最大规模 N ∈ [50,500]
distribution: Gaussian           # 高斯混合分布
Gaussian_params:
  no_cov: False                  # 允许协方差
  var_lower: 1.0                 # 最小方差
  var_upper: 100.0               # 最大方差
  num_modes_lower: 0             # 最小混合成分数
  num_modes_upper: 15            # 最大混合成分数 c ~ U(0,15)
```

##### (3) 求解器标签处理 (process_raw_label.py)
```python
# [选择模型] 监督信号生成：为每个实例生成求解器性能标签
if __name__ == "__main__":
    # 定义候选求解器池，对应论文筛选后的求解器列表
    # TSP: ['bq', 'ELG', 'LEHD', 'T2T', 'T2T500', 'DIFUSCO', 'DIFUSCO500']
    # CVRP: ['bq', 'ELG', 'LEHD', 'Omni', 'MVMoE']
    methods = ['bq', 'ELG', 'LEHD', 'Omni', 'MVMoE']

    # 初始化标签字典：每个实例包含成本、时间、最优性差距、最优求解器索引
    labels = {}

    # 读取专家求解器最优解（TSP用LKH，CVRP用HGS）
    if args.compute_gaps:
        opts = {}
        opt_file_name = f'{args.dataset}/results/result_opt.txt'
        with open(opt_file_name, 'r') as f:
            data = f.readlines()
            for line in data:
                line = line.strip().split(',')
                # 记录每个实例的最优已知解成本
                opts[line[0]] = float(line[1])

    # 收集所有候选求解器的性能
    all_gaps = []
    for method in methods:
        file_name = f'{args.dataset}/results/result_{method}.txt'
        gaps = []
        with open(file_name, 'r') as f:
            data = f.readlines()
            for line in data:
                line = line.strip().split(',')
                instance_name = f'{line[0]}'
                # 记录求解器成本
                labels[instance_name]['cost'].append(float(line[1]))
                # 记录求解器运行时间
                labels[instance_name]['time'].append(float(line[2]))

                if args.compute_gaps:
                    # 计算最优性差距：(cost - opt) / opt × 100%
                    if opts[instance_name] == 0:
                        opts[instance_name] = float(line[1])
                    gap = 100 * (float(line[1]) - opts[instance_name]) / opts[instance_name]
                    labels[instance_name]['gap'].append(gap)
                    gaps.append(gap)
                else:
                    labels[instance_name]['gap'].append(0)

        print(f"{method} average gap: {np.mean(gaps):.4f}%")
        all_gaps.append(gaps)

    # 计算求解器池的最优性能：min_{s∈S} gap_I(s)
    all_gaps = np.array(all_gaps)
    print(f"Ensemble best gap: {np.mean(np.min(all_gaps, axis=0)):.4f}%")

    # 确定每个实例的最优求解器：argmin_s cost_I(s)
    inds = []
    for k, v in labels.items():
        # 最优求解器索引：成本最小的求解器
        labels[k]['ind'] = np.argmin(np.array(labels[k]['cost']))
        inds.append(np.argmin(np.array(labels[k]['cost']))

    # 统计各求解器的最优实例数量
    wins = []
    for i in range(len(methods)):
        wins.append(np.sum(np.array(inds) == i))
    print(f"Solver wins: {wins}")

    # 保存标签文件，用于选择模型训练
    data_file = f'{args.dataset}/raw_label.pkl'
    with open(data_file, 'wb') as f:
        pickle.dump(labels, f)
```

#### 2. 实验管理系统

##### (1) 训练脚本生成 (experiments/generate_experiments_shell.py)
```python
# [实验管理] 批量训练实验生成，对应论文5个不同随机种子的训练设置
config_name = ['config_TSP.yml', 'config_CVRP.yml']  # 两种问题类型
loss = ['rank']                                        # 排名损失
seed = ['54321', '4321', '2024', '216', '924']        # 5个随机种子

shell = []
i = 0
gpus = [0, 1]  # 多GPU并行训练

for c in config_name:
    for l in loss:
        for s in seed:
            # 生成训练命令：python run.py --config_name config --loss loss --seed seed --gpu_id gpu_id &
            shell.append(f"python run.py --config_name {c} --loss {l} --seed {s} --gpu_id {gpus[i % len(gpus)]} &\n")
            i += 1
            # GPU负载均衡：每两个任务后等待，避免GPU过载
            if i % len(gpus) == 0:
                shell.append(f"wait;\n")

# 保存为run_experiment.sh脚本
with open(f"../run_experiment.sh", "w") as f:
    for line in shell:
        f.writelines(line)
```

##### (2) 测试脚本生成 (experiments/generate_test_shell.py)
```python
# [实验管理] 批量测试实验生成，对应论文多个测试集的评估
config_name = ['config_TSP.yml', 'config_CVRP.yml']
loss = ['rank']
seed = ['54321', '4321', '2024', '216', '924']

shell = []
i = 0
gpus = [0, 1]

for c in config_name:
    for l in loss:
        for s in seed:
            i += 1
            if 'TSP' in c:
                # TSP测试：合成测试集 + TSPLIB基准集
                shell.append(f"python run.py --gpu_id {gpus[0]} --load {c}_{l}_{s} --test_file TSPtest --exp_name {c}_{l}_TSPtest &\n")
                shell.append(f"python run.py --gpu_id {gpus[1]} --load {c}_{l}_{s} --test_file TSPLIB --exp_name {c}_{l}_TSPLIB &\n")
            else:
                # CVRP测试：合成测试集 + CVRPLIB基准集
                shell.append(f"python run.py --gpu_id {gpus[0]} --load {c}_{l}_{s} --test_file CVRPtest --exp_name {c}_{l}_CVRPtest &\n")
                shell.append(f"python run.py --gpu_id {gpus[1]} --load {c}_{l}_{s} --test_file CVRPLIB --exp_name {c}_{l}_CVRPLIB &\n")
            shell.append(f"wait;\n")

# 保存为run_test.sh脚本
with open(f"../run_test.sh", "w") as f:
    for line in shell:
        f.writelines(line)
```

##### (3) 实验执行脚本分析
```bash
# run_experiment.sh - 实际生成的训练脚本示例
python run.py --config_name config_TSP.yml --loss rank --seed 2025 --gpu_id 0 &
python run.py --config_name config_CVRP.yml --loss rank --seed 2025 --gpu_id 1 &
# 实现了论文中的批量训练：5个随机种子 × 2种问题类型 = 10个训练任务

# run_test.sh - 实际生成的测试脚本示例
python run.py --gpu_id 1 --load config_TSP.yml_rank_2024 --test_file TSPtest --exp_name config_TSP.yml_rank_TSPtest &
python run.py --gpu_id 0 --load config_TSP.yml_rank_2024 --test_file TSPLIB --exp_name config_TSP.yml_rank_TSPLIB &
wait;
python run.py --gpu_id 1 --load config_CVRP.yml_rank_2024 --test_file CVRPtest --exp_name config_CVRP.yml_rank_CVRPtest &
python run.py --gpu_id 0 --load config_CVRP.yml_rank_2024 --test_file CVRPLIB --exp_name config_CVRP.yml_rank_CVRPLIB &
wait;
# 实现了论文中的多数据集评估：合成测试集 + 标准基准集
```

#### 3. 训练日志与结果分析

##### (1) 训练监控日志 (log.csv)
```csv
# 列名对应论文4.1的评估指标
acc,top_1,top_2,top_3,top_4,cover_80,top-p,time_top_1,time_top_2,time_top_3,time_top_4,time_cover_80,time_top-p

# acc: 分类准确率（最优求解器预测准确度）
# top_k: Top-k选择策略的最优性差距（k=1,2,3,4）
# cover_80: 拒绝策略（80%置信度阈值）的最优性差距
# top-p: Top-p选择策略的最优性差距
# time_*: 对应策略的平均推理时间

# 示例数据：
0.293,2.171,1.615,1.468,1.352,1.942,1.615,1.416,2.900,4.441,6.096,1.758,2.902
```

**日志指标映射**：
- `top_1`: 贪心选择策略性能（直接选择最高分数的求解器）
- `top_2`: Top-2选择策略性能（选择前2个求解器取最优）
- `cover_80`: 拒绝策略性能（80%实例用贪心，20%用Top-2）
- `top-p`: Top-p策略性能（累积分数达到阈值p的求解器子集）

##### (2) 配置文件存储 (config.json)
```json
{
  "name": "TSPtrain",
  "problem_type": "TSP",
  "cuda_device_num": 3,
  "seed": 2024,
  "train_params": {
    "num_classes": 7,           // TSP求解器池大小
    "num_epochs": 50,           // 训练轮数（论文4.1）
    "train_batch_size": 64,     // 训练批量大小
    "test_batch_size": 8,       // 测试批量大小
    "learning_rate": 0.0001,    // 学习率 1×10^-4
    "weight_decay": 1e-06,      // 权重衰减 1×10^-6
    "loss": "rank",             // 排名损失
    "manual_feature": false,    // 不使用手工特征
    "ns_feature": false,        // 不使用神经求解器特征
    "data_aug": true            // 使用8倍数据增强
  },
  "model_params": {
    "pooling": true,            // 使用层次化编码器
    "downsample_ratio": 0.8,    // 图池化比例
    "embedding_dim": 128,       // 嵌入维度
    "encoder_layer_num": 2,     // 每块注意力层数
    "block_num": 2,             // 编码器块数量
    "head_num": 8,              // 注意力头数
    "qkv_dim": 16,              // 查询键值维度
    "ff_hidden_dim": 512,       // 前馈网络隐藏维度
    "norm": "rezero",           // ReZero归一化
    "output_dim": 7             // 输出维度（求解器数量）
  }
}
```

#### 4. 实验复现流程

##### 完整的实验执行流程：
1. **数据准备阶段**：
   ```bash
   cd datasets/
   python generate_data.py                    # 生成合成训练数据
   python process_raw_label.py --dataset TSPtrain  # 处理求解器标签
   ```

2. **训练阶段**：
   ```bash
   python experiments/generate_experiments_shell.py  # 生成训练脚本
   bash run_experiment.sh                               # 执行批量训练
   ```

3. **测试阶段**：
   ```bash
   python experiments/generate_test_shell.py      # 生成测试脚本
   bash run_test.sh                                 # 执行批量测试
   ```

4. **结果分析**：
   - 检查`train_logs/*/log.csv`获取训练过程指标
   - 分析测试输出获取最终性能数据
   - 对比论文Table 1, 2的结果

### 十、代码设计的核心优势

#### 1. 模块化与可扩展性
- **清晰的组件分离**：特征提取、选择模型、选择策略各自独立实现
- **配置驱动设计**：所有超参数通过YAML文件管理，便于实验调优
- **可插拔架构**：支持不同编码器、损失函数、选择策略的灵活组合

#### 2. 实验管理优化
- **批量实验支持**：自动生成多GPU、多种子的训练和测试脚本
- **完整的日志系统**：训练过程监控和结果记录的自动化
- **标准评估流程**：支持多种选择策略的同时评估

#### 3. 代码质量与工程实践
- **高效的批处理**：动态padding处理不同规模的实例
- **内存优化**：动量更新机制减少参数存储开销
- **GPU加速**：充分利用PyTorch的并行计算能力

#### 4. 论文实现的完整性
- **严格对应论文框架**：每个代码组件都有明确的论文章节对应
- **技术创新的完整实现**：层次化编码、排名损失、零样本泛化等
- **实验的可复现性**：完整的训练配置和评估流程

## 十一、深度学习框架与架构使用

### 1. 深度学习框架分类

深度学习框架分为两个层面：

#### **软件框架** (Software Frameworks)
- **PyTorch** (本项目使用)
- TensorFlow
- JAX
- PaddlePaddle
- MXNet
- Caffe

#### **架构框架** (Architectural Frameworks/Models)
- **MLP** (Multi-Layer Perceptron，多层感知机)
- **CNN** (Convolutional Neural Network，卷积神经网络)
- **RNN** (Recurrent Neural Network，循环神经网络)
- **Transformer** (注意力机制架构)
- **GNN** (Graph Neural Network，图神经网络)
- **Autoencoder** (自编码器)
- **GAN** (Generative Adversarial Network，生成对抗网络)

### 2. 本项目使用的架构框架

基于代码分析，这个神经求解器选择项目主要使用了以下三种架构框架：

#### **GNN (Graph Neural Network)** - 核心架构

##### 层次化图编码器 (Hierarchical Graph Encoder)
```python
class Encoder_h(nn.Module):
    """
    层次化图编码器 - GNN架构的核心实现
    将TSP/CVRP实例建模为全连接图进行特征提取
    """
    def __init__(self, **model_params):
        # 图嵌入层：将节点坐标映射到高维空间
        if model_params['problem_type'] == 'TSP':
            self.embedding = nn.Linear(2, embedding_dim)  # TSP: (x, y)
        if model_params['problem_type'] == 'CVRP':
            self.embedding = nn.Linear(3, embedding_dim)  # CVRP: (x, y, demand)

    def forward(self, data, mask):
        # GNN的核心：通过图池化实现层次化特征学习
        for block in self.blocks:
            graph_emb, out, mask = block(out, mask, i)
            graph_emb_h += graph_emb  # 多尺度图表示聚合
```

##### 图注意力网络
```python
class EncoderLayer(nn.Module):
    """
    图注意力层 - GNN架构的核心组件
    实现多头自注意力机制处理图结构数据
    """
    def __init__(self, **model_params):
        # 多头注意力的线性变换
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

    def forward(self, input1, mask=None):
        # GNN注意力机制：节点间信息传递
        out_concat = multi_head_attention(q, k, v, rank2_ninf_mask=mask)
        multi_head_out = self.multi_head_combine(out_concat)
```

#### **Transformer** - 注意力机制架构

##### 多头注意力机制
```python
def multi_head_attention(q, k, v, rank2_ninf_mask):
    """
    Transformer的核心：多头注意力机制
    """
    # 注意力权重计算
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])

    # 缩放点积注意力 (Transformer的核心创新)
    if rank2_ninf_mask is not None:
        attn = attn + rank2_ninf_mask
    attn = torch.softmax(attn, dim=-1)

    # 注意力加权的值计算
    output = torch.matmul(attn, v)
    return output
```

##### Transformer编码器层
```python
class EncoderLayer(nn.Module):
    """
    完整的Transformer编码器层架构
    """
    def __init__(self, **model_params):
        # Transformer的三个核心组件
        self.Wq = nn.Linear(...)      # 查询变换
        self.Wk = nn.Linear(...)      # 键变换
        self.Wv = nn.Linear(...)      # 值变换
        self.multi_head_combine = nn.Linear(...)  # 输出变换

        # Transformer架构的残差连接和归一化
        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)  # 前馈网络
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)
```

##### ReZero归一化 (Transformer的现代改进)
```python
class Add_And_Normalization_Module(nn.Module):
    """
    Transformer的ReZero归一化 - 现代改进版本
    相比传统的LayerNorm更加稳定和高效
    """
    def __init__(self, **model_params):
        if model_params["norm"] == "rezero":
            # ReZero：可学习的归一化参数
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)

    def forward(self, input1, input2):
        if isinstance(self.norm, nn.Parameter):
            # Transformer + ReZero: input + α * output
            back_trans = input1 + self.norm * input2
        return back_trans
```

#### **MLP (Multi-Layer Perceptron)** - 分类器架构

##### 兼容性分类器
```python
class Selection_model(nn.Module):
    """
    选择模型：GNN + Transformer + MLP的混合架构
    """
    def __init__(self, **model_params):
        # MLP分类器：将图表示映射到求解器兼容性分数
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, model_params['embedding_dim']),  # 第一层
            nn.GELU(),                                              # 激活函数
            nn.Linear(model_params['embedding_dim'],
                      model_params['output_dim'])                  # 输出层
        )
```

##### 神经求解器特征的相似性网络
```python
if self.model_params['ns_feature'] == True:
    # MLP相似性计算：实例特征与求解器特征的兼容性
    self.similarity = nn.Sequential(
        nn.Linear(feature_dim, model_params['embedding_dim']),  # 隐藏层1
        nn.GELU(),                                              # 激活函数
        nn.Linear(model_params['embedding_dim'], 1)             # 输出层：相似度分数
    )
```

### 3. 架构框架的融合设计

#### GNN + Transformer + MLP 的混合架构
```python
def forward(self, points, scales, manual_features, mask):
    # 1. GNN组件：层次化图编码器
    graph_emb = self.encoder(points, mask)  # GNN特征提取

    # 2. 特征融合
    self.instance_feature = torch.cat((graph_emb, manual_features), dim=1)

    # 3. MLP组件：兼容性预测
    if self.model_params['ns_feature'] == True:
        # 神经求解器特征模式：MLP相似性计算
        probs = []
        for i in range(len(self.model_tokens)):
            instance_solver_feature = torch.cat((
                self.instance_feature,
                self.model_tokens[i][None, :].expand(batch_size, -1)
            ), dim=-1)
            probs.append(self.similarity(instance_solver_feature))  # MLP
    else:
        # 固定求解器模式：MLP分类器
        probs = self.classifier(self.instance_feature)  # MLP

    return probs
```

### 4. **本项目没有使用的架构框架**

#### **CNN (卷积神经网络)**
- CNN主要用于图像处理和网格数据
- 本项目处理的是图结构数据，不适合CNN的卷积操作

#### **RNN (循环神经网络)**
- RNN主要用于序列数据处理
- 虽然节点序列可以被看作序列，但项目采用了更适合的GNN架构

### 5. 架构选择的合理性

#### **GNN的选择理由**
- TSP/CVRP问题天然适合图建模
- 城市客户是节点，潜在路线是边
- GNN能有效捕获图的结构信息和节点关系

#### **Transformer的选择理由**
- 多头注意力机制增强GNN的表达能力
- 能够建模节点间的复杂依赖关系
- ReZero归一化提升训练稳定性

#### **MLP的选择理由**
- 作为分类器，将提取的特征映射到求解器选择
- 简单高效，适合最后的决策任务

### 6. PyTorch框架在项目中的具体体现

#### 核心特性应用
- **动态计算图**：支持复杂的控制流和变长序列处理
- **自动求导**：简化了反向传播的实现
- **GPU加速**：无缝的CUDA支持和设备管理
- **丰富的神经网络层**：Linear、Multi-head Attention等
- **完善的生态系统**：torchmetrics、DataLoader等工具

#### 主要PyTorch组件
- **nn.Module**：所有神经网络模块的基类
- **torch.optim.Adam**：优化器实现
- **torch.utils.data.DataLoader**：数据加载和批处理
- **torchmetrics**：评估指标计算
- **nn.Parameter**：可学习参数管理

## 总结

本项目采用了 **GNN + Transformer + MLP** 的混合架构设计：
- **GNN** 作为核心架构处理图结构数据
- **Transformer** 增强注意力机制和特征表达能力
- **MLP** 作为分类器进行最终的选择决策

这种架构组合充分发挥了各架构的优势，特别适合组合优化问题的特征提取和求解器选择任务。项目基于PyTorch框架实现，充分利用了其动态计算图、自动求导和GPU加速等特性。

---

## 十二、用户创新项目：双层神经求解器选择框架

### 项目概述

基于对原论文"Neural Solver Selection for Combinatorial Optimization"的深入分析，用户提出了一个创新性的**双层神经求解器选择框架**，通过组件解耦的方式实现更精细化的算法选择，从选择完整求解器转向选择求解器的核心组件。

### 核心理论基础

**No-Free-Lunch定理的深度应用**：
- 单一神经求解器不可能在所有组合优化实例上都表现最优
- 每种算法都有其偏向性和适用范围
- **关键洞察**：神经求解器的pipeline可以分解为初始化和迭代两个独立阶段，且不同阶段对数据的需求和适用性完全不同

### 原论文架构分析与局限性

#### 原论文架构特点
- **单层Gate设计**：一个选择器直接选择完整的神经求解器模型
- **映射关系**：将多对多关系（实例↔求解器）转化为一对一关系
- **学习范式**：监督学习，预先对所有实例-求解器组合进行性能排序
- **关键假设**：一个模型的完整pipeline适用于所有数据实例

#### 原论文的根本缺陷
**核心假设不成立**：原论文隐含假设"一个模型的pipeline适用于所有数据"，但实际上：
- 神经求解器的pipeline可以分解为**初始化**和**迭代优化**两个阶段
- 不同阶段对数据特征的需求完全不同
- 初始化阶段更关注全局结构，迭代阶段更关注局部优化

### 创新方案：双层选择架构

#### 架构设计图
```
原始数据实例
    ↓
┌─────────────┐
│   Gate 1    │ ← 初始化选择器
│ (POMO/RELD  │
│ /LEHD/指针)  │
└─────────────┘
    ↓ 初始化解
    ↓
┌─────────────┐
│   Gate 2    │ ← 迭代选择器
│ (2-opt/大规模 │
│ 搜索/RRC)    │
└─────────────┘
    ↓ 最终解
```

#### Gate 1 - 初始化选择器
**核心功能**：根据问题实例特征选择最优的初始化算法

- **候选算法**：POMO、RELD、LEHD、指针网络等
- **选择依据**：
  - 实例规模（节点数量）
  - 空间分布特征（集中vs分散）
  - 图结构特征（聚类系数、连通性）
- **输入**：原始问题实例（节点坐标、需求等）
- **输出**：高质量的初始解

#### Gate 2 - 迭代选择器
**核心功能**：基于初始解选择最优的迭代优化策略

- **候选算法**：传统2-opt、大规模搜索、RRC等
- **选择依据**：
  - 初始解质量评估
  - 剩余优化空间估计
  - 计算时间预算
- **输入**：Gate 1的初始解 + 原始问题实例
- **输出**：经过优化的最终解

### 方法论创新：从监督学习到强化学习

#### 监督学习的局限性分析
1. **组合爆炸问题**：
   - n个初始化 × m个迭代 = n×m种可能组合
   - 随着算法增加，标注成本呈指数增长

2. **标注成本高昂**：
   - 需要预先运行所有算法组合获得性能标签
   - 每个新算法都需要与现有算法组合测试

3. **扩展性差**：
   - 难以适应新算法的加入
   - 无法泛化到新的数据分布

#### 强化学习的核心优势
1. **智能探索能力**：
   - 通过试错学习发现最优组合策略
   - 自动平衡探索与利用

2. **样本效率提升**：
   - 在线学习，无需预标注所有组合
   - 动态调整策略适应不同实例

3. **自适应性强**：
   - 能够根据实例特征动态调整选择策略
   - 支持新算法的无缝集成

### 技术实现方案

#### 1. 强化学习架构选择
**推荐算法**：Actor-Critic系列 (A2C/A3C/PPO)

**选择理由**：
- **连续动作空间**：适合算法参数的连续调节
- **稳定性好**：Critic网络提供稳定的价值估计
- **探索效率**：Actor网络实现智能的策略探索

#### 2. 状态空间设计
```python
状态空间 = [
    # 实例特征
    实例特征向量 (图嵌入 + 统计特征),

    # 初始解质量特征 (Gate 2专用)
    初始解路径长度,

    # 历史信息
    历史算法选择记录,
    历史性能表现,

    # 环境上下文
]
```

#### 3. 动作空间设计
**分层动作结构**：

- **Gate 1动作空间**：
  ```python
  action_gate1 = {
      'algorithm': [POMO, RELD, LEHD, PointerNet],
  }
  ```

- **Gate 2动作空间**：
  ```python
  action_gate2 = {
      'algorithm': [2opt, LargeScaleSearch, RRC],
  }
  ```

- **联合动作**：(action_gate1, action_gate2) 的组合选择

#### 4. 奖励函数设计
**简化但有效的奖励机制**：
```python
reward = -final_path_length
```

**设计理念**：
- **单一目标**：专注优化求解质量（路径长度）
- **直观性**：路径越短，奖励越高（负值越小）
- **计算简便**：避免复杂的多目标权衡
- **可扩展**：后续可添加时间、计算资源等惩罚项

#### 5. 基准策略（Baseline）机制详解

**Baseline的核心作用**：
- **问题**：直接使用路径长度作为奖励会导致方差很大，训练不稳定
- **解决**：使用相对比较代替绝对值，关注"比平均水平好多少"

**数学原理**：
```python
# 原始奖励函数
R = -path_length

# Baseline计算（使用batch内均值）
B = mean(batch_path_lengths)

# 优势函数
A = R - B
```

**具体计算示例**：
```python
# 假设当前batch有4个实例
path_lengths = [100.5, 98.2, 102.1, 99.8]
rewards = [-100.5, -98.2, -102.1, -99.8]  # 负号因为越短越好

# 计算baseline
baseline = np.mean(path_lengths)  # = 100.15

# 计算优势函数
advantages = rewards - baseline  # [-0.35, 1.95, -1.95, -0.35]

```

#### 6. 损失函数设计
**多组件联合优化**：
```python
Total_Loss = Loss_advantage + λ1·Loss_gate1 + λ2·Loss_gate2
```

**损失组件说明**：
- **Loss_advantage**：Actor-Critic的标准优势函数损失
- **Loss_gate1**：初始化选择器的策略损失
- **Loss_gate2**：迭代选择器的策略损失
- **权重平衡**：λ1, λ2调节两个gate的相对重要性

### 实际求解需求确认

**与原论文的本质区别**：
- **原论文**：基于预存结果的"选择验证"，不实际求解新问题
- **本项目**：需要**真正求解**新的TSP/CVRP实例，通过强化学习优化实际求解过程

**求解流程确认**：
1. Gate 1选择初始化算法并执行，获得初始解
2. Gate 2选择迭代优化算法并执行，获得最终解
3. 根据最终解的质量计算奖励信号
4. 更新两个选择器的策略参数

