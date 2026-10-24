# 实例、方法表示与输入信息：跨版本代码机制复核

开始复核：2026-10-23；成稿：2026-10-24，Asia/Shanghai。目录沿用任务指定日期。

## 结论与优先级

**当前证据不足以支持立即再训一个新架构。** 更准确的问题不是“有没有关系建模”，而是“什么关系被作为可更新的内容保留，什么只改变消息权重，以及留下的关系是否足以区分候选优势”。规模是合法且已保留的输入，不建议遗忘规模、均匀化方法频率或削弱强信号来制造覆盖。

本次新增了一个有边界的代码结论：**基线 ATSP 编码器存在可解析的拓扑不可区分类；完整矩阵进入前向不等于其关系结构可以被读出。** 六节点 CPU 示例证实了这一现象，同时 R92 raw-statistics 对同一对图产生完全相同的输入。它没有证明实际数据存在同类碰撞，更没有证明碰撞决定当前 solver winner。R102 的 MatNet 身份通道是重要反例，不能把这条机制外推到所有版本或十八题。

优先级：先把这个可证伪的表示限制与真实候选差异之间的联系说清；没有联系就停止沿它追加网络。文末仅保留一个条件性的研究对象，不作为立刻启动训练的建议。没有新的主线架构获准、没有性能改善承诺。

### 范围与证据状态

- 已读前版 [REPORT.md](/public/home/shiys/0selection2/docs/research/deep_direction_review_20261023/REPORT.md:1) 及两份历史 CSV，避免把路线监督、候选工作区、私有模型、预训练或关系网络当成未尝试。
- 本文读当前源码、冻结协议和已完成小报告；没有加载训练 checkpoint、预测数组或原始大数据集，没有训练、solver 执行、GPU、launcher、提交或活动文件修改。唯一执行是随机初始化 `InstanceEncoder` 的六节点 CPU 静态示例。
- R105/R106 只读设计源码，按本任务启动时的“运行中”处理；不读其训练曲线、不推断效能、不接管运行。R102 历史报表来自 R102 自己的恢复目录，不是 R106 的实验结果。
- 路径/行号指本次工作区源码，不把当前文件自动认作每个历史 checkpoint 的逐字节源码快照；历史效能仅引用既有报告，没有重新认证全链路。
- 本子任务由主线程以gpt-6-astra/max调用，本地turn_context同样记录；不作服务端后端身份认证。本文是交给主线程综合的代码机制材料。

## 1. 模型实际看到了什么？

以下是 R77/R84A 的实际继承路径，不从张量宽度猜测语义。`NewSplitDataset -> RealTrainDataset -> UnifiedProblemDataset.__getitem__ -> build_batch -> ScenarioCache.batch`；R77/R84 通过 `make_r45_model` 构造的是 **handcrafted + ID 的 ScoreDifferenceSelector**，不是源码编码器。

| 输入对象 | 实际进入前向的值 | 没有进入或须限定的内容 | 源码定位 |
| --- | --- | --- | --- |
| TSP | 每个真实节点的 FP32 `x,y`；后续补零至八槽；真实 `n` 和 padding mask | 基线没有物理 node0 标记、节点序号或位置嵌入。R88 才显式增加一个角色位；不能把坐标集合与原始排列当成同一信息 | [data.py:102](/public/home/shiys/0selection2/code/unified_selector/data.py:102)、[r88_model.py:57](/public/home/shiys/0selection2/code/V4/r88_model.py:57) |
| CVRP | depot 坐标、customer 坐标、已有归一化 demand、`is_depot`；depot demand=0，补齐其余槽 | 这里不重新除 capacity，也不读取单独 capacity 字段；不能因此认定归一化需求缺少容量语义 | [data.py:114](/public/home/shiys/0selection2/code/unified_selector/data.py:114) |
| 15 种 MVRP | `x,y,signed_demand,is_depot,route_limit,service_time,tw_start,tw_end`；L/TW 按任务启用；depot 的后四槽为零 | 原始 `capacity` 不读取；未启用的字段置零。实际读取比 README 的“训练可见 capacity”更窄。是否丢掉有用变化取决于上游归一化/契约，本轮没有扫描全数据证明 | [data.py:127](/public/home/shiys/0selection2/code/unified_selector/data.py:127)、[README.md:342](/public/home/shiys/0selection2/data/README.md:342) |
| ATSP | **完整有向 FP32 成本矩阵**、真实 `n`、mask；没有对称化或构造假坐标 | 进入编码器后，节点初值只有行/列均值、最小、最大、绝对非对称度及对角；完整矩阵另走 attention bias。不是“只输入八个统计” | [data.py:109](/public/home/shiys/0selection2/code/unified_selector/data.py:109)、[V4Model.py:236](/public/home/shiys/0selection2/code/V4/V4Model.py:236) |
| 任务/规模 | problem ID、坐标/矩阵 kind、五个 C/O/B/L/TW bit、十六项 descriptor、`log1p(n)/6`、实例统计 | `n` 包含 CVRP/MVRP depot，TSP/ATSP 不含虚构 depot。O 不需要单独节点字段，因为任务条件已输入；descriptor 也不是唯一任务接口 | [registry.py:58](/public/home/shiys/0selection2/code/unified_selector/registry.py:58)、[V4Model.py:122](/public/home/shiys/0selection2/code/V4/V4Model.py:122) |
| 生成 metadata | 新数据 `coord_dist` 从真实 metadata 读取并进入 batch | R77/R84 的 `ignore_coord_dist=True` 使它不进入条件嵌入；source index、capacity_type 不进入网络；metadata 中的 scale 已由真实 `n` 提供。metadata 被用于冻结划分/分层，不等于参与预测 | [new_split_data.py:95](/public/home/shiys/0selection2/code/V4/new_split_data.py:95)、[r84_data_release.py:53](/public/home/shiys/0selection2/code/V4/r84_data_release.py:53)、[protocol.json:301](/public/home/shiys/0selection2/code/V4/runs/R84_new90k_attention/A_baseline/protocol.json:301) |
| 候选与 mask | manifest 中的合法 pool/global IDs；22-ID 全局空间保持稳定；每个合法局部候选的 mask 为真 | 全局 `mask` 与局部 `solver_mask` 不同。删掉的候选根本不构造 token，不是模型预测了零次。`reld_mt` 仍映射原 RELD_MOEL ID，不合并两列成本 | [new_split_data.py:21](/public/home/shiys/0selection2/code/V4/new_split_data.py:21)、[r76_cache.py:189](/public/home/shiys/0selection2/code/V4/r76_cache.py:189)、[dual_stream.py:78](/public/home/shiys/0selection2/code/V4/dual_stream.py:78) |
| 监督 | native/FP64 winner、合法候选成本产生的 pair/risk 目标用于 loss | `inputs_only` 删除 costs、ind、performance/comparison/regret 目标；基线前向不读取当前实例的求解结果。该函数是已知标签键过滤，不是任意未来字段的通用防泄漏证书 | [pairwise_objective.py:10](/public/home/shiys/0selection2/code/V4/pairwise_objective.py:10)、[r76_experiment.py:158](/public/home/shiys/0selection2/code/V4/r76_experiment.py:158) |

**尺度核对：**上述 loader/cache 不把每个实例的 xy 再中心化、旋转、除包围盒或除实例平均距离，原始存储坐标仍走 `coord_proj`。距离 bias 和局部几何有自己的无量纲化，不能把支路的尺度不变性误称为整个网络的尺度不变性。ATSP 节点行列统计也保留原矩阵数值尺度。源数据若已经归一化，这里不会恢复上游单位。见 [r84_prepare.py:170](/public/home/shiys/0selection2/code/V4/r84_prepare.py:170)、[V4Model.py:223](/public/home/shiys/0selection2/code/V4/V4Model.py:223)。

### R77/R84A 的完整计算图

| 阶段 | 实际计算/状态 | 需要避免的误读 | 源码定位 |
| --- | --- | --- | --- |
| 节点初值 | `Linear(raw8)`；坐标另加几何残差；ATSP 为 `Linear(row/col8)` | 非零原始字段没有被均值统计替换 | [V4Model.py:270](/public/home/shiys/0selection2/code/V4/V4Model.py:270) |
| 几何残差 | k=4/8/16 的半径、均距、distance-CV、anisotropy，距离按实例平均边长归一；再用 TRAIN-only mean/std 标准化；与 raw8、log-n 拼接后投影相加 | 12 维几何不是完整图，但原节点并未因此删除；末层零初始化不代表训练后仍为零。FP64 几何计算最终缓存 FP32 | [local_geometry.py:7](/public/home/shiys/0selection2/code/V4/local_geometry.py:7)、[local_geometry.py:49](/public/home/shiys/0selection2/code/V4/local_geometry.py:49)、[r84_prepare.py:259](/public/home/shiys/0selection2/code/V4/r84_prepare.py:259) |
| condition + 节点自注意力 | 拼入 CLS、condition、stats 三个 token；四层带距离/有向矩阵 bias 的 self-attention 与 FFN | condition 确实参与消息；`node_only=True` 仅在四层后丢弃三个特殊 token，不是从未使用它们，也不使用另一条 `h_cls/summary_proj` 读出 | [V4Model.py:288](/public/home/shiys/0selection2/code/V4/V4Model.py:288) |
| 方法初值 | 学习型 solver ID + 34 位静态特征的 MLP，LayerNorm | factory 名称含 R45 不等于使用 code embedding；`SolverMemoryEncoder` 的 problem/arm embedding 不是这条实际路径 | [dual_stream.py:10](/public/home/shiys/0selection2/code/V4/dual_stream.py:10)、[solver_code_encoder.py:135](/public/home/shiys/0selection2/code/V4/solver_code_encoder.py:135) |
| 双流 | 两层同步 cross-update：nodes 读 solvers，同时 solvers 读上一层 nodes；各自 residual、FFN、norm | 有实例-方法双向交互，且可经节点间接传递候选集合信息。方法 token 到此已随实例改变，不是固定 prototype | [dual_stream.py:31](/public/home/shiys/0selection2/code/V4/dual_stream.py:31) |
| pooling / decoder | 更新后节点 masked mean/max -> 128维 h；h 作**单个 query**读取所有合法方法，得到共同 context c | 不是只保留 h：每个方法仍有其自身的交互后 token z_m。协议旧键 `query_num=4` 不能推断这条路径有四个 query | [dual_stream.py:93](/public/home/shiys/0selection2/code/V4/dual_stream.py:93) |
| shared score head | 同一个 MLP 对每个 `[h,c,z_m]` 输出 u_m；pair=u_i-u_j，再 maximin | 共享头不等于相同输出。此处不是独立两两比较器；固定合法池时 argmax(maximin(u_i-u_j)) 与 argmax(u) 一致，但概率/训练梯度不等价 | [dual_stream.py:98](/public/home/shiys/0selection2/code/V4/dual_stream.py:98)、[pairwise_selector.py:11](/public/home/shiys/0selection2/code/V4/pairwise_selector.py:11) |

## 2. 七个关键事实及反例约束

### F1. 可见字段比“只有几何摘要”丰富，比“原始输入全部可见”窄

全体坐标、带符号需求、L/TW 数值、有向矩阵、任务条件和规模均有真实消费者。另一方面，节点执行次序/完整排列身份、独立 capacity、某些生成 metadata 不进入基线。**遗漏字段不自动等于遗漏与 winner 有关的信息**：归一化需求可能已经包含容量比；生成标签可能只描述可由几何估计的分布；固定部署可以被 ID 区分。对这些字段只能确认“接口未读取”，不能确认“造成平台”。

### F2. 泛化的共同疑点是关系内容的组织，不是关系不存在

基线有距离偏置、全局节点 self-attention、双向候选交互；R56 有持久有向边状态；R63 有候选专属节点工作区；R92 有真实有向粗图。因此不能再说“没有关系建模/没有 solver 交互”。更窄的疑点是：几何与约束关系在多数继承路径中主要经节点值聚合，资源/路径组合没有作为显式数值状态保留。**这通常是学习难度假设，不是普遍的不可表达定理**；坐标和约束已在网络中，网络可能学出这些关系。

### F3. ATSP 的 bias-only 关系入口确有一个可证明的限制

构造两张六节点完整有向图：对角为0，其余边默认2。图A的1成本边为 `0->1->2->3->4->5->0`；图B为 `0->1->2->0` 和 `3->4->5->3`。每节点的八个初始统计相同，每行 bias 的多重集相同，全图统计相同，但1成本边组成的拓扑不同。

在精确算术、eval 模式下，每层所有物理节点的值向量相同。矩阵结构只改变这些相同值的权重排列；给特殊 token 的总权重也相同，故下一层仍无法区分。该归纳不依赖训练是否成功，也不靠“小维度必然丢信息”的猜测。之后同一合法池的双流与读出不能恢复没有形成的差异。

CPU 实测使用实际 `InstanceEncoder`，d=128、四层、四头、原公式、SDPA关闭；seed 2/7/19 随机参数，未加载模型/标签：

| 检查量 | A/B 差异 |
| --- | ---: |
| 原始矩阵 max-abs | 1.0 |
| 初始八维节点统计 / 全图统计 max-abs | 0 / 0 |
| attention bias max-abs | 0.72958982 |
| 四层后节点 max-abs，三个 seed | 7.15e-7 / 9.54e-7 / 7.15e-7 |
| R92 `features()` 的357列 max-abs | 0 |
| 1成本边的三步闭合游走数 `trace(A^3)` | 图A=0，图B=6 |

这是浮点级近等加解析等价论证，不把数值误差声称为逐 bit 相等。R92 的统计定义见 [r92_tree_features.py:141](/public/home/shiys/0selection2/code/V4/r92_tree_features.py:141)。更宽/更深的同类基线不打破该对称性；R94 的 ATSP 本就没有 donor，R101 的部署学生也未换结构。**反例边界：R102/106 的 MatNet 用不同节点的 one-hot TO 身份构造值向量，不能套此证明；真实 ATSP 未必含这些等统计结构；拓扑不同也不保证候选 winner 不同。**

### F4. 方法表示曾发生真正的输入不可区分，但不能套到当前 ID 分支

R46保留全部 source role/chunk token并与节点交互，再按实例动态池化；没有 solver ID、候选位置或私有头。既有报告记录 DIFUSCO/DIFUSCO500、T2T/T2T500 的完整输入包相同，旧 TSP TEST 的1000例均分别得到同 logits，导致两种高ID方法的288个 native winner 不可被选中。代码与历史观测一致，见 [solver_code_tokens.py:13](/public/home/shiys/0selection2/code/V4/solver_code_tokens.py:13)、[R46 RESULTS.md:45](/public/home/shiys/0selection2/code/V4/runs/R46_code_token_interaction/RESULTS.md:45)。

当前 ID+34 分支区分这些部署，因此不能用这个历史碰撞解释其现有零选。34位字段中的0是“未标注”，不是已证实“没有该能力”；它们不直接表达训练规模、具体权重或执行预算，但 ID 可对固定部署学习常数与条件差异。[solver_features.py:1](/public/home/shiys/0selection2/code/V4/solver_features.py:1) 因而“静态方法描述不充分”是跨部署迁移的风险，不是当前固定池不可学习的证明。

### F5. 换读出、加私有头、加 donor，并未分别隔离同一个假设

R84 direct-attention保留同一实例编码、两层双流和mean/max，仅把 decoder/context/score MLP 换成 h 对方法 keys 的多头点积分数。R87在原 `[h,c,z_m]` 上加私有非线性残差，未提供新的节点关系入口。两者阴性结果不能排除早期关系表征不足；也不能证明 early encoder 是唯一原因。R104仅改变宽深，仍保持运算类型。[r84_attention_selector.py:55](/public/home/shiys/0selection2/code/V4/r84_attention_selector.py:55)、[r87_model.py:84](/public/home/shiys/0selection2/code/V4/r87_model.py:84)、[r104_model.py:19](/public/home/shiys/0selection2/code/V4/r104_model.py:19)

donor 路线也不同：R94把 frozen 数值节点早融合；R101只用 TRAIN hint 改变初始化，部署不增加输入；R102扩展到 TSP/ATSP 并引入 MatNet 节点身份。它们不是三个重复的“加预训练”。R94 ReLD 只读取 depot xy、customer xy/signed-demand/TW endpoints，**不读 service、L、O**；主网络仍保留这些原始字段，不能说整个模型丢了它们。[r94_donor.py:178](/public/home/shiys/0selection2/code/V4/r94_donor.py:178)

### F6. 全任务私有化确实试过，但不能反证所有共享表示问题

R69不是私有头，而是18套完整 selector，按原 problem ID 路由；初始 trunk继承R67/R68路径，单任务分支只在自己的更新中训练。它针对跨任务参数干扰，却不改变各分支的原始输入、节点消息类型和mean/max读出。旧池、继承初始化、每参数更新史与NEW90k不同，因此不是当前基线的等预算“唯一去共享”因果排除。[r69_model.py:13](/public/home/shiys/0selection2/code/V4/r69_model.py:13)、[r68_model.py:25](/public/home/shiys/0selection2/code/V4/r68_model.py:25)

反过来，也不能继续把共享当成必然根因：历史完整私有模型的泛化没有胜出，且R92树模型、R97-100冻结上下文并不依赖同一神经SGD路径。共同错误可以来自信息、函数类、监督对象、学习过程的不同组合。

### F7. TRAIN零选排除了一个过窄故事，没有识别唯一原因

在已锁定、同一模型的TRAIN中，足量正例仍全漏，说明“只是遇到未见数据才失败”不是充分解释，也不是单纯没有正例曝光。它**不意味着**输出没有监督梯度、标签随机、最优分类器必须给每方法一个区域、模型缺乏任意拟合能力，或optimizer一定坏了。基线CE、pair、risk、R-Drop联合目标也不是要求最大化TRAIN Top1的无约束记忆器。[pairwise_objective.py:63](/public/home/shiys/0selection2/code/V4/pairwise_objective.py:63)

只有额外证明同一输入/函数类/目标允许更低经验风险，才有资格把差距进一步指向学习过程；当前checkpoint没有达到100% TRAIN Top1本身不是这种证明。

## 3. 各变化究竟保留与改变了什么？

| 版本 | 已实际实现的机制 | 没有隔离/仍继承的限制 | 代码入口 |
| --- | --- | --- | --- |
| R46 | source chunks + role embeddings；动态节点/source交互；最后每方法pool；无ID | 同源码包部署不可区分；不是测试“只加源码是否有效”的单变量对照 | [r46_model.py:14](/public/home/shiys/0selection2/code/V4/r46_model.py:14) |
| R56 | 四层节点/有向边共同更新，edge bias/gate及持久edge residual | 15 MVRP上的MOEL/MTL二分类专家，旧完整池Top2 gate；最后只pool节点；边更新读本边与端点，不是显式两条边的路径资源组合 | [r56_model.py:14](/public/home/shiys/0selection2/code/V4/r56_model.py:14)、[r56_model.py:129](/public/home/shiys/0selection2/code/V4/r56_model.py:129) |
| R63 | 每节点×候选工作区；稀疏有向图、8轮GRU；A读初始邻居，B读前轮邻居；每候选pool | 不可再说“没有候选专属状态”。但消息以distance gate调制projected节点值，没有显式剩余容量/时间/路径控制态；A/B实际40/15E不等预算 | [r63_model.py:42](/public/home/shiys/0selection2/code/V4/r63_model.py:42)、[r63_graph.py:54](/public/home/shiys/0selection2/code/V4/r63_graph.py:54) |
| R69 | 完整task-private selector bank | 去共享不更换表征算子；旧数据/继承trunk并非本轮同数据独立对照 | [r69_model.py:32](/public/home/shiys/0selection2/code/V4/r69_model.py:32) |
| R84 B/D | 原编码与双流，h-Q / method-K直接点积读出 | 未更改编码入口；头、maximin概率链接等共同改变，不是冻结同一表示只测一个线性读出上限 | [r84_attention_selector.py:33](/public/home/shiys/0selection2/code/V4/r84_attention_selector.py:33) |
| R87 | 共享/方法private非线性utility residual，参数数目匹配 | 同一个编码器、方法交互和共享原score head；不是每方法一整套encoder | [r87_model.py:51](/public/home/shiys/0selection2/code/V4/r87_model.py:51) |
| R88 | TSP canonical anchor vs physical node0，填原slot3，非TSP不变 | 只测一个角色位，不覆盖完整执行排列、所有起点、seed或checkpoint语义；非TSP变化不能归因于自己的输入语义变了 | [r88_model.py:57](/public/home/shiys/0selection2/code/V4/r88_model.py:57) |
| R92 A/B | directed GNN，32/8 clusters，原节点质量归一粗化；hard/learned assignment，B含link/entropy辅助 | 真正更换instance encoder；仍以distance及exp(-distance)两标量关系作in/out聚合；TW在节点/条件中，未显式进入可行关系运算；最终输出8簇再接原双流 | [r92_model.py:61](/public/home/shiys/0selection2/code/V4/r92_model.py:61)、[r92_model.py:243](/public/home/shiys/0selection2/code/V4/r92_model.py:243) |
| R92 C/D | 357维确定性input统计，每题树模型；C winner分类、D gap回归 | 不是原始全图直接交给树。分位数、网格、相关性与行列统计丢掉不少节点对应关系；保留coord_dist，与神经基线也非完全同输入 | [r92_tree_features.py:17](/public/home/shiys/0selection2/code/V4/r92_tree_features.py:17) |
| R94 | 16 routing接口的frozen ReLD128，LN+linear后在第一原始层前与raw/geometry相加 | 不覆盖TSP/ATSP；不是具体候选各自的全推理状态；FP32输出量化FP16，非仿射LN舍弃节点channel均值等信息，但raw路径仍在 | [r94_model.py:110](/public/home/shiys/0selection2/code/V4/r94_model.py:110) |
| R101 | TRAIN-only LN-node-MSE hint预训练，迁移encoder；第二阶段/部署回到原模型 | 改初始化与学习经历，不增加部署输入或状态；不是证明teacher所有信息都被迁移、保留或用于winner辨别 | [r101_pretrain.py:72](/public/home/shiys/0selection2/code/V4/r101_pretrain.py:72)、[r101_model.py:8](/public/home/shiys/0selection2/code/V4/r101_model.py:8) |
| R102 | routing预训练相同；TSP ELG128、ATSP FROM256+TO256，额外donor随机/预训练对照；分组projection | 有ATSP固定one-hot身份，不能继承“值全部相同”的证明；拼接512维再投影128并非全网络必然丢失winner信息的证明 | [r102_model.py:98](/public/home/shiys/0selection2/code/V4/r102_model.py:98)、[r102_donor.py:303](/public/home/shiys/0selection2/code/V4/r102_donor.py:303) |
| R104 | d128->256或instance4/joint2->8/4；原运算路径 | 宽深没有改变编码的不变性；不否定所有容量，也不支持继续盲目扩容 | [r104_model.py:27](/public/home/shiys/0selection2/code/V4/r104_model.py:27) |
| R105，运行中 | 每个instance FFN替为四个dense专家；A gate读零输入，B读post-attention状态；四专家都算 | 改条件计算，不增加原始字段/关系值入口；A的router特征权重不活跃。只记设计，不判断是否改善 | [r105_model.py:75](/public/home/shiys/0selection2/code/V4/r105_model.py:75) |
| R106，运行中 | 同三种native encoder在线运行，A冻结、B可微调；native保持eval但B autograd开启；不使用外部nco缓存 | 不再是frozen feature无法适应的对照；同源raw/融合/下游保留，ATSP身份亦保留。只记设计，不能说微调成功或失败 | [r106_model.py:103](/public/home/shiys/0selection2/code/V4/r106_model.py:103)、[r106_donor.py:211](/public/home/shiys/0selection2/code/V4/r106_donor.py:211) |

**交集需要降格表述：**这些实验没有一个完全相同、已被证明致因的“共同瓶颈层”。多数后期神经变体继承相同raw接口、任务条件、基线双流及读出框架；R63、R69、R92分别打破其中不同环节。可保留的共同研究问题是静态输入中的局部/全局关系能否经济地支持候选相对优势，而不是所有历史模型都只有同一个128维向量，或都受同一个optimizer问题支配。

## 4. 十八题并不只有一个平台

只取一个固定已完成模型 R102/A 的 selected-best TRAIN，不拼接版本，也不替主线程重新选定“当前最佳模型”。每题81,000例；native/FP64 ind；合法候选池。下面由现成小CSV重排计数，不重读预测。

| 任务 | 合法候选数 | TRAIN Top1(%) | 有正例且严格零选的方法组数 | 这些零选组的winner例数 |
| --- | ---: | ---: | ---: | ---: |
| TSP | 8 | 45.9815 | 3 | 8,083 |
| CVRP | 10 | 48.9099 | 0 | 0 |
| ATSP | 5 | 78.1136 | 1 | 233 |
| OVRP | 6 | 41.4296 | 2 | 1,226 |
| VRPB | 3 | 81.5889 | 0 | 0 |
| VRPL | 6 | 38.1802 | 2 | 5,350 |
| VRPTW | 6 | 80.8420 | 3 | 492 |
| OVRPTW | 6 | 84.1728 | 1 | 71 |
| OVRPB | 3 | 88.6765 | 0 | 0 |
| OVRPL | 6 | 42.1691 | 2 | 1,207 |
| VRPBL | 3 | 80.7531 | 0 | 0 |
| VRPBTW | 3 | 80.0593 | 1 | 97 |
| VRPLTW | 6 | 80.8901 | 3 | 504 |
| OVRPBL | 3 | 88.9173 | 0 | 0 |
| OVRPBTW | 3 | 83.5531 | 1 | 1 |
| OVRPLTW | 6 | 84.0901 | 1 | 83 |
| VRPBLTW | 3 | 80.0975 | 1 | 82 |
| OVRPBLTW | 3 | 83.5790 | 1 | 3 |

来源：[method_selection_train_val_test.csv](/public/home/shiys/0selection2/code/V4/runs/R102_full_type_pretrained_fusion/preparation/postprocess_recovery/report_view/A/full_model_statistics/method_selection_train_val_test.csv:1)；逐题Top1与 [train_metrics_by_problem.csv](/public/home/shiys/0selection2/code/V4/runs/R102_full_type_pretrained_fusion/preparation/postprocess_recovery/report_view/A/full_model_statistics/train_metrics_by_problem.csv:1) 一致。B题已删除的 RouteFinder、MoSES_RF、MoSES_CaDA 是“已删除（不参与）”，没有纳入零选计数。RELD_MTL同样不参与。

具体边界：TSP/DIFUSCO有6,806个TRAIN winner但选0；VRPL/MTPOMO与MVMOE分别2,709、2,641但均选0。它们不能被“正例只有一两个”解释。相反，OVRPBTW/MTPOMO只有1个winner，不宜用同样强度称其训练信号充分。TSP/LEHD有14,279个winner，选25次、正确6次，是近零而不是严格零。CVRP没有严格零选组，却只有48.91% TRAIN Top1：改善零选数不是普适质量指标。

十八题可迁移的对象是**有向移动、局部替代、资源/约束耦合怎样改变方法相对优势**，不是把所有任务视为同一种二分类。TSP重在空间连接，ATSP方向关系不可省略，CVRP容量与depot相关，MVRP按O/B/L/TW组合启用不同运算；共享基本算子不意味着共享同一个充分统计或同一个难度结论。高Top1的三候选B题与低Top1的多候选题也不是同难度的重复证据。ALL约71%只是宏平均现象，不是统一信息上限。

## 5. 三种机制如何分开？

| 机制 | 必须证明什么 | 当前证据与不能推出的结论 |
| --- | --- | --- |
| 输入信息缺失 | 在允许输入相同的条件下，相关执行状态/候选区别确实被遗漏，并影响目标 | R46相同源码包是实证实例；基线节点序号、capacity字段、metadata未使用只是接口事实，尚不是当前全18 winner的必要信息证明 |
| 编码/函数类限制或不易学 | 原输入有区别，但算子强制合并；或有限学习过程很难构造相关组合 | ATSP反例证明一个严格子类限制；一般坐标/约束任务只能提出“组合难学”的假设。低维、低秩、attention存在或不存在、线性probe失败都不能单独认证信息丢失 |
| 学习过程未达到可用解 | 同输入、同函数类及同目标下存在更好的可达经验解，而当前训练未得到 | TRAIN零选及小TRAIN-VAL gap不足；监督项、采样、优化与正则会共同作用。R101测试hint初始化，R104测试容量，均没有分离上述所有因素 |

因此本次最强结论是局部计算机制的存在性，不是“optimizer不用改”或“输入不足是唯一原因”。如果现有冻结表示已能在未参与拟合的实例上分辨漏选方法，而原决策仍失败，应优先研究读出/目标；如果只增加易学的结构但候选优势仍不可辨，应放弃把这种结构当成瓶颈。不能用当前错误分数自己定义“不可学区域”，再以这些区域验证当前分数。

## 6. 只保留一个条件性研究对象

### 显式路径组合与资源状态，而非再加一个 attention 分支

**尚不建议立即训练。** 先明确新增对象：不是新节点向量、更多donor、更多轮邻居聚合，而是保留“同一中间节点连接的两条/三条有向边”的对应关系，以及这段连接上的数值资源和由已核实部署语义定义的控制状态。比如 TW 的 `max(到达时间,窗口开始)`、service累加、L剩余量、已核实的 signed-load 累积/边界状态、O的返库规则，不能被一个未经解释的edge embedding命名替代。

**B 语义边界：**R58 仍暂缓；当前新池删除 RF/MoSES 并未发布统一 classical-B 契约。局部资源状态只能遵循该数据已锁定/核实的 signed-load 部署语义，不能默认经典“配送先于取货”；未恢复或未核实的部分不添加阶段状态、过滤条件或硬规则，不以新特征修改任务定义。

以下只是可执行定义的边界，未实现、未测量效能：

| 项目 | 明确定义 |
| --- | --- |
| 输入 | 原始xy/ATSP矩阵、真实n、需求、depot、L/TW、任务约束、合法方法ID+原34位描述；保留绝对值及无量纲值。无需当前实例标签、solver路线或decoder输出；不删除规模信号 |
| 状态 | 有向边状态及共享中间节点的局部片段 `(i,j,k)`；保留端点对应、方向/闭合标记、长度、容量需求、最早服务时刻/窗口余量、已核实的 signed-load 状态（不预设配送/取货阶段）和返库所需长度。无约束任务关闭相应状态；TSP/ATSP也保留两边组合与三步闭合证据 |
| 更新 | 固定两轮短片段组合，不选择整条路线、不做beam/搜索。时间更新例如 `arrival=t+d_ij; service_start=max(arrival,a_j); departure=service_start+s_j`，窗口检查服务开始是否超界；资源仅按已锁定/核实的任务部署契约组合。可学习残差读取这些明确数值，而非让edge只充当消息gate。需保持重复节点/返库语义，不把局部通过叫作全局可行证明 |
| 读出 | 对保留下来的片段内容作候选条件化证据汇总，兼顾出现质量和少量极端瓶颈，再产生完整合法池的utility。保留方法ID，避免R46相同描述合并部署；task-specific残差允许同一局部事件对不同问题/方法作用不同。不把“候选条件化”本身当新贡献 |
| 监督 | 仍只用冻结TRAIN的native winner和既有FP64成本比较目标监督最终选择。确定性片段字段可以直接保留，不需要新solver标签；不把整例winner伪装成每个片段的因果标签。损失换名不算本方向的新机制 |
| 成本 | 稀疏k邻域并保留depot关联边，片段数约 `O(N k^2)`，固定轮数下内容更新约 `O(N k^2 d)`；距离构造可仍为 `O(N^2)`，候选读出另计。可流式聚合避免全量 `N^3` 状态。截断也可能漏掉全局关键关系，经济性与充分性均未证明 |

**与已做的准确分界：**

- R56已有持久有向edge状态，但其edge更新是本边及端点的函数，没有显式共享中间节点的两边组合或资源算术；它还是旧池局部专家，不能替代当前全18验证。
- R61已有保存路线的successor/END预训练并迁移encoder，不是未经训练的空白起点；新增对象不是再预测一遍路线，而是在部署中保留输入可计算的片段数值/控制态。[r61_data.py:1](/public/home/shiys/0selection2/code/V4/r61_data.py:1)
- R63已有候选专属workspace与GRU递归，新增对象不能仅叫“工作记忆”；需要从不透明节点记忆变为上述有固定组合语义的关系状态。
- R92已有有向粗化和质量保持，新增对象不是多尺度pool；必须保留粗化前具体关系的连接对应与约束组合，不只是平均边权。
- R101已有node-hint初始化，新增对象不是再做一次teacher alignment；这里要求部署计算图改变并保留解释明确的状态。R106在线微调正在覆盖另一个假设，不重复提议。
- 额外去重：R55已做“预测计费边差->距离求和”，不能把它重命名为解释性表示；本对象不预测solver的边计数。[r55_model.py:15](/public/home/shiys/0selection2/code/V4/r55_model.py:15) R99甚至已有两步传播 `P(P attributes)`，但不是保存路径对应/资源可行态，不能把“加两跳”当新东西。[r99_relations.py:61](/public/home/shiys/0selection2/code/V4/r99_relations.py:61)

这些差别只说明与历史计算图不完全重复，**不说明具有科研价值或能够改善选择**。若局部资源/结构在不同winner之间没有稳定条件差异，或需要接近完整求解才能获得差异，就应放弃。不能把精确路径算法的存在性定理当成selector效果保证。

### 最小后续研究与停止条件

本轮已完成的六节点CPU性质检查是一个非循环、无需标签的低开销证伪：它否定“只要完整矩阵被传入，基线就能区分任意拓扑”的强说法；没有反复调probe或拟合head。为了复现，按F3两张矩阵同时调用 `InstanceEncoder.eval()`、`_matrix_features_and_bias()`、`_matrix_stats()` 和 R92 `features()` 即可；无需任何checkpoint、数据缓存、GPU或solver。

**下一步只需主线程判断一个问题：有无已保存、可明确解释的候选优势证据，与这种被合并的关系对象相关？** 可优先查现有案例，不新解实例、不做全库碰撞审计。不要求完全相等才有意义，但不能用“同图像看着相似/当前分数接近”替代有定义的关系差别。如果现有案例不能建立这座桥，应把该构想保留为未验证假设，当前不再训练；如果能建立，才值得设计一个保留/删除该明确对象的匹配表示比较，而不是一次同时改encoder、loss、donor与budget的综合包。

本报告不另建共享 `source_coverage.csv`，避免多人写同一清单；实际覆盖、继承路径及新机制已经由上述源码表给出。本文唯一新增文件为本报告。
