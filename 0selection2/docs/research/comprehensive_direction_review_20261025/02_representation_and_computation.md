# 镜头二：实例表示、计算结构与信息获取

日期：2026-10-25。职责：论文与科学路线研究，不是代码审查、结果认证或启动实验的指令。本地会话请求记录为 `gpt-6-astra / max`；这不是对服务端模型身份的独立认证。本文只交付本专题与文献 CSV。

## 核心判断

**当前缺的不是一个尚未尝试过的“图网络大类”。** 历史已经覆盖可训练历史记忆、候选节点递归、完整任务私有模型、联合性能分布、原生编码器迁移及连通资源路径。新名字不能抹去这些邻近实验。多数路线仍在解决同一个问题：用有限计算，从静态实例中摊销预测固定候选的相对表现；但它们并非全都共享一个可证明致命的表示瓶颈。

**撤回“下一步优先增强关系表示”的宽泛建议。** R107 已实际连接两条边、保留方向与闭合信息，并组合局部时间、需求和计费路径。现在没有找到一个有项目证据的对象，能够同时满足“完整 R107 仍混淆”“新表示能区分”“区分会改变相对 solver 优势”。表达力定理本身不补上后两个条件。

**尚值得有限考虑的差异，是把历史 TRAIN 当作推理时拟合局部规律的数据，而不只是加权读出的记忆。** 下文给出 R47 特定聚合操作丢失“位置与结果对应关系”的具体构造，以及保留该对应关系的局部函数求解式。它是项目内可辨认的新计算对象，不是保证突破 71% 的新架构，也不是首次提出检索、局部回归或 ICL。

## 一、证据范围与历史账本

依据当前 [REQUEST](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/REQUEST.md)、[AGENTS](/public/home/shiys/0selection2/AGENTS.md)、[机械索引](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/artifact_index.csv)、[指标摘录](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/reported_all_metrics.csv)及实际模型/报告理解机制。索引是 **96 个运行目录、78 个编号，不是 96 次独立训练**。以下数字为已发布结果的研究性引用，本次未复跑、重算预测或认证。

legacy18、NEW70114、NEW90k 的数据和候选池不同，不能把跨时代 Top1 提升归因于结构。当前 NEW90k 每题 81,000/4,500/4,500，18 题、seed2、逻辑 batch1024；保留八个 B 问题删候选及单一 `reld_mt` 规则。TEST/LIB 已参与研发，不称全新盲测。R110 以实际 [RESULTS](/public/home/shiys/0selection2/code/V4/runs/R110_method_conditioned_ce/RESULTS.md)为准；旧专题末尾的 running 已过期。R109/R111 按本次状态快照仍在运行，不计入负结果。

| 历史机制 | 实际改变及不能再称为新方向的内容 |
|---|---|
| R35-R44 | 单任务继续训练、小样本拟合、多尺度几何、性能回归、关系图与 solver query、成对解码、约束/规模条件适配均已有相邻尝试。R36 小样本能拟合不是全分布可泛化证据；R39 已有性能建模，R42 已有关系/候选查询。依据 [R35-R79 历史账本](/public/home/shiys/0selection2/docs/research/deep_direction_review_20261023/ASTRA_HISTORY_R35_R79.md)。 |
| R45-R46：代码表示 | R45 比较手工描述+ID 与代码表示+ID；R46 把代码 token 带入交互且移除 ID。相同源码、不同预算候选出现身份碰撞，不意味着代码语义天然能解释部署性能；重新加入 ID 也不等于新信息机制。见 [R46 结果](/public/home/shiys/0selection2/code/V4/runs/R46_code_token_interaction/RESULTS.md)。 |
| R47：FULL18 可训练记忆 | 同题 TRAIN bank；共享可训练 key，排除关联 base，选中 reference 的 key 重新参与梯度；value 含历史 gap、winner 和候选表示，另有 key 差修正。不是二分类，也不是固定近邻投票。缓存摘要不等于检索不可训练。见 [实际模型](/public/home/shiys/0selection2/code/V4/r47_model.py)。 |
| R50-R51：冻结编码与短程行为 | R50 是旧 OVRPTW 两 ReLD 对照，冻结原生编码器，不解码路线。R51 已读取两候选短程解码的进度、容量、可行动作、熵等，并研究续跑。它们不是全18一般性结论。见 [R50](/public/home/shiys/0selection2/code/V4/runs/R50_pretrained_solver_encoder/RESULTS.md)、[R51](/public/home/shiys/0selection2/code/V4/runs/R51_solver_behavior_probe/RESULTS.md)。 |
| R53-R56：专家、结构监督、边状态 | R53 有二方法专家/旧 Top2 门控；R54 以两个 ReLD 后继关系辅助监督；R55 直接把预测计费边次数差与距离相乘形成成本差；R56 有持续有向边状态和 attention bias/gate，但不是完整 EGT。后几项核心是旧 MVRP 二方法问题，不能外推成全18结构已经被彻底否定。见 [R54](/public/home/shiys/0selection2/code/V4/runs/R54_route_supervision/RESULTS.md)、[R55](/public/home/shiys/0selection2/code/V4/runs/R55_route_cost_difference/RESULTS.md)、[R56](/public/home/shiys/0selection2/code/V4/runs/R56_relational_encoder/RESULTS.md)。 |
| R61-R63：预训练与递归 | R61 用保存路线预训练、仅迁移 encoder；R62 用跨来源可观察性能比较预训练，再回原完整池。R63 已有候选×节点工作区、稀疏出入邻居、8 轮 GRU，A/B 区别包含是否反馈前轮邻居状态；预算不等。不能再推荐笼统“候选条件化+迭代推理”。见 [R61](/public/home/shiys/0selection2/code/V4/runs/R61_behavior_pretraining/RESULTS.md)、[R62](/public/home/shiys/0selection2/code/V4/runs/R62_portfolio_performance_pretraining/RESULTS.md)、[R63](/public/home/shiys/0selection2/code/V4/runs/R63_recurrent_workspaces/PLAN.md)。 |
| R69、R71 | R69 是按已知问题 ID 路由的 **18 个完整 selector**，从继承编码器微调，不是只换私有头。R71 在这类任务私有骨干上建模全协方差成本差混合分布，再积分逐候选成为最小值的概率；分布建模与候选相关性不是空白。见 [R69](/public/home/shiys/0selection2/code/V4/runs/R69_task_specific_finetuning/RESULTS.md)、[R71](/public/home/shiys/0selection2/code/V4/runs/R71_joint_performance_mixture/RESULTS.md)。 |
| R84、R87、R88 | R84 有直接 attention 读出；R87 私有残差位于池化后；R88 对比 TSP 规范锚与物理 node0，非 TSP 输入未因此改变。不能把已存在的候选 token 当成“不具备候选条件化”。见 [R84](/public/home/shiys/0selection2/code/V4/runs/R84_new90k_attention/comparison_artifacts/RESULTS.md)、[R87 模型](/public/home/shiys/0selection2/code/V4/r87_model.py)、[R88 模型](/public/home/shiys/0selection2/code/V4/r88_model.py)。 |
| R92：层级 | 已有硬/可学习的两级 32/8 簇对象及完整18题统计特征树模型。加一个层次池化名字，不构成新机制。见 [说明](/public/home/shiys/0selection2/code/V4/runs/R92_broad_learning/README.md)。 |
| R94、R101、R102：迁移 | R94 把随机/预训练 ReLD 节点向量早融合，覆盖 CVRP+15 MVRP，selector 仍评全18。R101 两臂均做节点 hint 预训练，比较 R94 随机/预训练教师缓存，再仅迁移 encoder；其它模块回到共同新初始化。R102 两臂都保留 ReLD 预训练，增量比较 TSP/ATSP donor 的随机/预训练状态，不是“全部随机对全部预训练”。见 [R94 模型](/public/home/shiys/0selection2/code/V4/r94_model.py)、[R101 方案](/public/home/shiys/0selection2/code/V4/runs/R101_hint_pretraining/PLAN.md)、[R102 模型](/public/home/shiys/0selection2/code/V4/r102_model.py)。 |
| R97、R98、R100：ICL | 历史 TRAIN、357维统计摘要、冻结表格先验已经试过；分别是 winner 分类、逐方法预期相对 gap、成对上下文后 Borda 汇总。R98 已使用 TabICLv2 回归器。三者不是同一个决策目标，也不能说“用历史标签”仍未尝试。见 [R97](/public/home/shiys/0selection2/code/V4/runs/R97_tabular_in_context/RESULTS.md)、[R98](/public/home/shiys/0selection2/code/V4/runs/R98_performance_context/RESULTS.md)、[R100](/public/home/shiys/0selection2/code/V4/runs/R100_pairwise_context/RESULTS.md)。 |
| R99：关系输入 | 行归一化与平衡 Sinkhorn 关系消息，32维投影注入原节点流；已经有输入可计算的全局竞争关系，不是实际 solver 轨迹。见 [模型](/public/home/shiys/0selection2/code/V4/r99_model.py)。 |
| R105、R106 | R105 在层内用四个 FFN 专家，经静态/输入门控混成一条节点流，不是四个完整 encoder。R106 将原生 donor 从冻结改为可微微调，保留原始输入早融合；“让预训练可训练”已经落地。见 [R105](/public/home/shiys/0selection2/code/V4/r105_model.py)、[R106](/public/home/shiys/0selection2/code/V4/r106_model.py)。 |
| R107：连通关系 | 两边共享中间节点，保留 shortcut/closure、方向、时间窗等待、需求前缀极值和计费路径；两轮关系更新注入节点层。A 是独立边递归，B 才联合路径；相同完整张量数不是相同活跃自由度/FLOPs。见 [模型](/public/home/shiys/0selection2/code/V4/r107_model.py)。 |
| R109、R110、R111 | R109 共享四层后增加候选条件节点块，比较 tied/private，仍在跑。R110 同路线使用一次前向 raw-utility CE，已完成。R111 是族私有节点特征与联合概率 `p(m)=p(f)q(m;f)`，其中 q 是族内条件概率，不是硬选根节点后只保留一族；仍在跑。见 [R109](/public/home/shiys/0selection2/code/V4/r109_model.py)、[R110 结果](/public/home/shiys/0selection2/code/V4/runs/R110_method_conditioned_ce/RESULTS.md)、[R111](/public/home/shiys/0selection2/code/V4/r111_model.py)。 |

这份账本记录“试过什么”，不是把继承微调、不同年代候选池、不同预算当作独立同条件反证。R74 的已尝试信息获取机制另在第七节说明。

## 二、到底反复共享了什么假设？

原始主干并非只读取少量统计量。坐标类问题有原始节点字段及局部几何；ATSP 虽用行列统计初始化节点，**完整有向矩阵仍进入 attention bias**。典型路径是四层实例更新、两层节点/候选双向交互，再读取节点 mean/max 摘要 `h`、公共 context `c` 和实例条件化候选向量 `z_m`：

\[
u_m=F_\theta(h(x),c(x),z_m(x)),\qquad \hat m=\arg\max_{m\in M(x)}u_m.
\]

候选向量不是纯静态 ID；标量差分的 maximin 与 utility 的 argmax 一致，不代表两者 softmax 的概率或梯度一致。不能只画一个共享池化瓶颈，就抹掉候选通道或损失差别。见 [主干](/public/home/shiys/0selection2/code/V4/V4Model.py)、[双流](/public/home/shiys/0selection2/code/V4/dual_stream.py)。

反复出现的是三项偏好：用有限层和有限状态提前压缩实例；多数时候以所有实例共享的参数摊销预测；假定几何/资源结构中存在容易迁移的性能规律。R69 改变参数共享范围，R63 改变候选状态及递归，R47 改变推理输入，R51 改变可观察量；它们不属于同一严格等价类。

这也解释一个可能而未被证实的 TRAIN 少选机制：令 Z 包括模型决策前的全部实例和候选状态，而非仅 h。即使某方法有真实赢家，只要每个 Z 区域内其条件胜率都低于另一个方法，基于 Z 的最优 argmax 也可以永远不选它。丢失区分信息或把不同局部规律平均起来都可能造成此现象；有训练样本不保证存在已学出的决策区域。但这既不证明当前模型确实如此，也不证明完整原始输入 X 的 Bayes 上限。零 argmax 更不等于该候选没有梯度。

真正的目标也不是最优路线本身。对固定部署可作概念分解：

\[
C_m(x)=a(x)+r_m(x),\quad C_m-C_n=r_m-r_n.
\]

其中 `a` 可以取共同困难度或最优值，比较时完全消去。能准确识别簇、瓶颈、资源紧张甚至最优路线，不自动识别某个有限预算、固定 checkpoint 的失败残差。反过来，低维特征若对残差差值充分，就不必重建整张图。需要的是**决策充分性**，不是实例的无损编码。

只有当结构量 `T(x)` 改变了候选间的响应，结构才有选择价值。例如两个求解器面对同一瓶颈都同等变差，瓶颈预测再准也不改变 winner。称“需要高维非线性”之前，应说明它作用于哪两个候选的哪个不同过程；不是要求把所有 solver 完整模拟一遍。

另一个容易混淆的假设是不变性。问题最优值对重编号不变，不保证实际部署的首节点、并列打破和有序解码也不变；但目前没有全18的部署敏感性证据。R88 只检查过一个锚角色，R102/106 的 MatNet 身份初始化又不符合“所有节点完全同质”的前提。**不因此推荐给全18盲加序号**，更不引入数据行号、来源 ID 或不可预知随机结果。原始坐标输入本来也不保证旋转不变。

## 三、十篇一手论文：方法实际说了什么

下表只摘与判断有关的机制；章节及实际浏览范围同步保存在 [02_sources.csv](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/02_sources.csv)。本次读取的是原文 HTML/PDF 核心方法，不是用摘要代替方法；未逐篇精读全部附录或复核全部实验。

| 来源与实际阅读 | 方法要点 | 对本项目的约束 |
|---|---|---|
| S01 [GSN](https://arxiv.org/html/2006.09252v3)，2020预印本，实读 v3；§3.1-3.3 | 按节点/边的子图轨道计数，进入消息函数；表达力结论有指定子图条件。 | 完全加权图不能直接照搬无类型 motif；更能分图不等于更能分 solver 胜负。 |
| S02 [EGT](https://www.cs.rpi.edu/~zaki/PaperDir/SIGKDD22.pdf)，KDD 2022；§3.2-3.3，PDF第3-4页 | 独立边通道递推，为 attention 提供偏置和门控，并构造动态中心性。 | R56 已靠近核心机制；补齐原文部件不是全18收益证据。 |
| S03 [LSPE](https://arxiv.org/html/2110.07875v2)，ICLR 2022；§3.1、§4.3 | 结构与位置分别更新；随机游走返回概率提供位置初始化。 | 图内位置不是原生执行角色；R92/R107 之后不能凭 PE 名字重启结构方向。 |
| S04 [CLRS](https://proceedings.mlr.press/v162/velickovic22a/velickovic22a.pdf)，ICML 2022；§3、§4.1，PDF第3、6-7页 | 编码、递归处理、解码并反馈算法 hints；测试反馈预测 hints。原基线仍用 hint 数决定步数。 | 算法状态有明确语义，不等于任意 GRU；不得借入真实查询轨迹或 oracle 停止长度。 |
| S05 [InfoMin](https://proceedings.neurips.cc/paper/2020/file/4c2e5eaae9152079b9e95845750bb9ab-Paper.pdf)，NeurIPS 2020；§3.1-3.3、命题3.1 | 减少两视图互信息的前提，是各视图保留全部下游标签信息。 | 保持路线最优值的增强，不自动保持固定部署 winner；不能无条件承诺对比预训练有效。 |
| S06 [TabR](https://arxiv.org/html/2307.14338v2)，ICLR 2024；§3.1-3.2，式(1)-(5) | 共享 key 的距离检索，标签 value 加 query/key 差修正；原文明确受 DNNR 启发。 | R47 已包含这些关键思想；“可学习检索/局部修正”不能作为新提案。 |
| S07 [TabICLv2](https://arxiv.org/html/2602.11139v1)，2026预印本；§3、§4、§5 | 标签感知表格嵌入、列/行/ICL 分层处理和合成数据先验；回归输出分位数。 | 它是关于表格函数的先验，不自带路线性能规律；R98 已用 v2，不能建议只升级名称。 |
| S08 [DNNR](https://proceedings.mlr.press/v162/nader22a/nader22a.pdf)，ICML 2022；§2-3、算法1，PDF第2-3页 | 在邻域估计局部梯度，用 Taylor 预测代替仅平均标签；与每查询拟合一个平面的局部回归有区别。 | 支持区分“读记忆”与“从记忆估计函数”；不是其回归结果可直接迁移到18题分类的证明。 |
| S09 [EDDI](https://www.nowozin.net/sebastian/papers/ma2019eddi.pdf)，ICML 2019；§3.1-3.3，PDF第3-5页 | Partial VAE 对已观察变量集合建模，按目标后验的信息收益选择下一变量。 | 熵下降不等于选方法的成本下降；需要另给停止与付费规则。实际读作者站副本。 |
| S10 [SATzilla](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/08-JAIR-Satzilla.pdf)，JAIR 2008；§1.2、§2、§3.3 | 经验性能预测纳入特征提取、探测和超时回退的代价。 | 可借鉴成本核算，不能照搬包含预求解器及失败后换 solver 的整套执行制度。 |

这些论文的共同启发不是“再堆一种 Transformer”，而是明确中间对象：结构轨道、持续边状态、算法状态、历史函数局部导数、观察后验。对象必须对当前决策有用，不能只对论文原任务有用。

## 四、直接回答 R107 的反问

R107 的 `phi(edge_ij, edge_jk, composed, previous_path)` 不是两个无关边统计的简单拼接。每个源点有界选择邻居，构造同一中间点的 `i-j-k`，在实例层1和3之后两次汇入节点流。`composed` 已包括 `shortcut_ik`、`closure_ki`、闭合三角长度、反向路径、等候、窗余量和有符号需求前缀。把“保留连接性、方向、局部资源组合”再写成建议，属于重复。

它仍不等于完整可行路线计算：需求前缀不是经认证的车载负载；没有按资源过滤路径，也不在中间 depot 自动重置；有界三节点路径不能显式表示所有长程互斥。但这些是**所计算对象的边界**，不是已证明的选择失败原因。更重要的是，原始节点/矩阵仍经过其它网络分支。两个实例的局部路径统计相同，不足以推出整个 R107 输出必相同。

因此不提供一个未经证实的“更长路径/高阶子图/约束超图”方案。若使用旧 ATSP 六环与两三环的示例，也不能直接迁移成 R107 反例：它的闭合字段可能区分三环；原生 MatNet 路线还有不同身份初始化。局部模块的反例必须和完整前向的反例区分。

重新考虑关系表示至少需连起三个论证环节：给出当前合法实例中的具体混淆对象；说明拟加入的量为何改变已部署候选间的相对损失；说明完整前向为何仍未有效表征或计算该对象。并不要求增加原始输入之外的信息，改变可计算性也可以，但必须具体。若只拿出更强 WL/子图计数定理，没有候选性能联系，本专题仍建议暂停。R63 已有递归工作区、R69 已有完整私有模型、R71 已有联合分布，也分别阻止“多想几步”“独立学任务”“学相关性”成为未经细化的新主张。

## 五、预训练与辅助任务为何可能学得很好却选不好？

先保留有利证据，不把结果写成统一失败。下表引用当前 NEW90k 完整模型的 ALL TEST，采用 native/FP64 ind 而非 FP32 argmin；Top1 与 actual regret 均为逐题等权百分数。不同实验之间不是因果对照，不汇总成所谓独立证据票数。

| 对照 | A Top1 / regret | B Top1 / regret | 能支持的有限判断 |
|---|---:|---:|---|
| R94 随机/预训练 ReLD 早融合 | 71.290 / 0.398 | 71.578 / 0.379 | 有小幅转移信号，不能宣布预训练完全无用。 |
| R101 随机/预训练教师节点 hint | 71.322 / 0.393 | 71.435 / 0.391 | 两臂都预训练；此教师知识对照没有形成大幅全18跃升。 |
| R106 冻结/微调原生 donor | 71.523 / 0.382 | 71.160 / 0.391 | 解除冻结不是已获支持的通用修复。 |
| R107 独立边/连通路径 | 71.260 / 0.395 | 71.307 / 0.394 | 已有显式局部组合仍未明显改变总体平台。 |
| R110 tied/private 节点，纯 CE | 71.307 / 0.406 | 71.233 / 0.399 | 私有化不自动增加赢家召回；Top1 与成本也不完全同向。 |

R110 固定36组 TEST 有4,598个真实 winner，A/B 分别找回70/44个；B 相对 A 救回21、丢失47、新错选41。不能把21解释成全部救回，也不能把“选过更多方法”当改善。两臂约230万/1022万参数，不是等参数、等 FLOPs 对照。该结果削弱“池化前私有化足够解决问题”，但不裁决仍在跑的 R109/R111。

辅助目标不忠实有两种不同情况。第一，辅助任务和最终决策没有强制的计算联系：R54/R61 的路线辅助头会丢弃；R101 则直接回归教师节点表示，部署没有教师或 hint loss。其辅助误差可以下降而 score 不使用相关能力。第二，即便强制经结构计算成本差，误差仍可能大过决策 margin。R55 已经尝试后一条，不能再当新方案。以每条计费无向边只计一次、D 表示次数差为例：

\[
\Delta C=\sum_{i<j}D_{ij}d_{ij},\qquad
|\widehat{\Delta C}-\Delta C|
\leq \|\widehat D-D\|_1\|d\|_\infty.
\]

当误差小于非零的 `|Delta C|` 时，可保证该成对符号正确；范数按只计一次的边向量定义。平均边预测好，不保证关键少数边误差小；原生并列 winner 仍须遵守冻结标签规则。类似地，预测绝对成本的高解释度可能只拟合了公共项 `a(x)`，而没有拟合 `r_m-r_n`。这是本文的数学解释，不是声称已测出当前误差来源。

据此暂不推荐更多随机图预训练、更长 hint 序列、通用对比增强或仅更换 donor。它们增加的是关于其它目标的经验及可计算性，不自动增加部署相对表现的信息。恢复该方向需要明确迁移量与候选 margin 的联系，而不是只呈现辅助 loss 曲线。R101 的节点表示蒸馏不是 CLRS 的逐步算法轨迹监督；后者有精确状态语义，也不能直接充当当前 selector 的免费教师。

## 六、唯一保留的新机制：推理时拟合上下文局部函数

### 6.1 与 R47 的实质差别在哪里？

从实际代码抽象，合法 mask 相同的候选 `m` 的 R47 记忆为：

\[
b_m=\sum_i w_i V(z_m,g_{im},y_{im})
    +\sum_i w_i T(k_q-k_i,z_m).
\]

这保留历史结果与 key 差的两个聚合量，却没有显式的“key 差乘历史结果”的局部交叉矩。**固定 query、key 与模型参数时**，若若干 reference 等距，交换其完整 `(gap 向量,winner)` 记录，两个和均不变，之后的非线性头也不能恢复该对应关系。R47 的 key 可训练，能改变距离；因此这不是对它全部可学习函数的表达力不可能定理。

一个可手算的示意是 query 在二维原点，三个等距 reference 位于：

\[
v_1=(1,0),\quad v_2=(0,1),\quad v_3=(-3/5,-4/5).
\]

三条历史记录分别由 A/B/C 获胜。等距加权记录的多重集不因交换标签位置而变。局部仿射插值却在 query 得到对应权重 `(1/4,1/3,5/12)`，因为它们和为1且加权坐标为0；交换 `v1` 与 `v3` 的记录，会改变最高分候选。这是零正则插值极限，足够小的正则强度由连续性保留严格次序。三点只是简化说明，等权记录交换的不变性可嵌入更大的固定 K 邻域。**这是计算机制见证，不是发现了真实数据中的这三个实例，也不是泛化收益证明。**

因此新对象不叫“更好的 memory”，而叫 **上下文坐标与历史结果的局部对应关系**。TabR 本来就借鉴过 DNNR，局部近邻函数估计也不是新发明；项目内的区别在于每次推理实际解出一个依赖 support 数据的函数，而不是由全局固定 value/correction 网络一次加权读出。

### 6.2 一个明确、覆盖全18的单方法定义

输入仍是当前原始实例、问题身份、合法候选池，以及已授权的同题冻结 TRAIN 上下文。设可学习实例表示为 `z=E_theta(x)`，从 TRAIN 取邻域 `S_q`，排除 query 本身及相同 base 的全部关联副本。邻域内用负平方距离 softmax 得到非负权重 w。历史标签 `Y_i` 是合法池上的原生 winner one-hot；当前 query 的标签绝不进入拟合矩阵。

\[
\phi_i=[1;(z_i-z_q)],\quad
\Phi_q=[\phi_i^\top]_{i\in S_q},\quad
W_q=\operatorname{diag}(w_{qi}),
\]
\[
B_q=(\Phi_q^\top W_q\Phi_q+\lambda I)^{-1}
       \Phi_q^\top W_qY_q^{\rm support},\quad
s_q=e_0^\top B_q,\quad
\hat m=\arg\max_{m\in M(q)}s_{qm}.
\]

`lambda>0` 保证该局部求解有定义；`s` 是排序分数，不冒充校准概率。全局学习可用 TRAIN 内 query/support 排除式 episode，让 query 标签仅监督最终分类，并通过该求解学习 `E_theta`。所有18题使用同一“编码、检索、局部拟合、一次 argmax”机制，各题只用自己的合法池和 TRAIN，不把 ReLD 二分类门控移植给其它题。

这里实际改变三件事：中间对象从压缩 behavior 向量变成局部系数 `B_q`；推理参数由上下文共同决定；最终判定直接由局部函数输出，不再仅给旧头加一个残差。没有执行任何候选 solver，更没有把多个求解结果择优输出。

相对 R47，新量是 `Phi^T W Y` 与 support-support 几何共同确定的解，而非“多读几个邻居”。相对 R97/R98/R100，不依赖冻结的合成表格先验及固定357维摘要，不做逐方法 gap 回归或成对 Borda。相对 R63，迭代/求解作用于历史监督形成的局部函数，不是候选节点潜状态。相对 R69/R71，不新增任务私有 encoder 或性能混合密度。它不声称比上述整个函数类更强，只提出一项可证伪的计算偏好。

### 6.3 为什么值得考虑，又为什么很可能失败？

值得考虑的是部署误差规律可能随实例区域变动，而固定压缩头把这些局部变化平均掉；局部函数明确使用“哪些方向的变化伴随哪个候选变好”，不要求先模拟路线。即使只对一部分有充分邻域支持的区域成立，也可能跨 TSP/CVRP/ATSP/MVRP 复用计算规则。

失败原因同样具体：几何邻近未必意味着性能规律相近；高维局部拟合可能病态；winner 函数在微小 margin 处不平滑；罕见候选可能在局部 support 中根本没有支持；表示已丢掉的信息，局部求解不能创造出来。低全局参数量也不代表低计算，保存/搜索上下文和每查询矩阵求解均有成本，直接求解量级含 `O(Kd^2+d^3+Kd|M|)`，还不含检索。

**否决事实**：在既定 TRAIN/VAL 排除规则下，打乱位置与历史记录的配对后优势不消失，说明方法没有利用所宣称的局部对应关系；只保留常数项的同邻域预测同样有效，说明局部函数假说没有增量；若发现收益依赖未排尽的同 base 副本，纠正后消失，则不能成立；真实赢家救回被新误选及 regret 抵消，或收益只在旧 ReLD 对局出现，则不能支持全18主张。以上是未来科学反驳条件，不是本次新增工程任务。

**授权边界**：历史 TRAIN 标签上下文本身已有许可，不申请 query 标签、外部 solver 轨迹、新划分或新候选。任何新增训练、缓存建设和推理计时仍须另批；不能以本报告视为获准。潜在收益应来自同口径验证，不凭 TEST 再选局部规则。本路线优先级仅高于无对象的结构换名，不高到可以跳过失败条件。

## 七、静态选择与短程信息获取的真正边界

输入可计算的路径、谱量、松弛量都可能改善可计算性，但若 `F(X)` 是确定函数，则 `I(Y;F(X)|X)=0`；增加字段不是自动增加原始实例之外的信息。历史 TRAIN 提供未压缩的经验上下文，帮助有限模型估计未知规律；也不提供 query 的真实求解结果。不能把信息论上的新观察与有限网络更容易算出的统计量混为一谈。

短程 solver 观察的不同在于它直接访问当前实例上的实际算法行为；固定程序/随机状态下仍可能是输入的昂贵确定函数，有未观察随机状态时则还可能携带该状态信息。无论如何，它突破当前“query 不给求解结果”的静态前向边界，必须另批，不能把它称为免费特征工程。

而且这条方向并非未试。R51 的旧二分类对照，内部验证有改善、原始验证反而退步；报告端到端耗时约 `152.857/127.794=1.1961` 倍。R74 已尝试全18自适应前缀：按 TSP/CVRP/ATSP/MVRP 使用相应原生模型，两级增量观察与 gate，训练目标设计为精确三叶期望 CE；它被暂停/终止，没有完整结论。不能把“按不确定性选择是否看一段原生行为”包装成新的大方向。机制依据 [R74 实际说明](/public/home/shiys/0selection2/code/V4/R74_README.md)及 [R51 实际报告](/public/home/shiys/0selection2/code/V4/runs/R51_solver_behavior_probe/RESULTS.md)。

若未来重新讨论，真正需要改变的是 **有成本的决策价值与停止规则**，而不只是再加 entropy token。令 `s` 为当前已观察状态，`R(s)=min_m E[ell(m,Y)|s]`，其中损失口径事先明确，则一次观察的价值是：

\[
\operatorname{VOI}(a|s)=R(s)-\mathbb E_{o_a|s}R(s,o_a).
\]

只有获准的预算允许、且预期决策收益超过明确的观察代价，才继续；否则立即选择唯一方法。这里对风险的估计不是当前真实标签。EDDI 的信息收益提供方法参照，不保证 regret 收益；与 R74 的区别在成本/决策风险及停止语义，而非自适应探测本身首次出现。

完整部署时间必须记为：

\[
T=T_{\rm selector}+\sum T_{\rm probe}+T_{\rm chosen,remaining}.
\]

只有已证明与原始部署轨迹、状态及预算一致的被选方法才能抵扣其已执行前缀；其它方法的探测不能抵扣。若必须从头运行，最后一项就是完整运行时间。若续跑改变了原始候选行为，也不能继续套用原来的冻结成本标签。输出仍只能是一个最终方法，绝不以多方法完整求解后的最小成本冒充 selector 性能。

本次不把这列为立即推荐：新增观察须用户明确授权，现有 R51/R74 又不足以证明收益超过代价。若未来观察只降低熵、不改变有价值的最终选择，或收益不抵探测成本，应停止；取得全18、同部署、计费后的净收益证据，才有理由重开。不能以一篇信息获取论文跳过这些反证。

## 八、交给综合者的两项选择

1. **优先暂停泛结构扩张。** 暂停以更高阶图、更多专家、更深递归、更多 donor 为主要理由的新版本；不触碰正在运行的 R109/R111。保留现有输入与单方法决策路径，不增加观察和计算。撤回依据是 R107 已覆盖关键局部组合、R63/R69/R71 已覆盖邻近计算机制、R110 完整结果削弱了私有化的充分性。反转条件是出现具体、部署相关且尚未被这些机制有效表征或计算的区分对象，不是新术语或单一辅助指标改善。
2. **若继续，有限考虑上下文局部函数求解。** 它有可写出的新中间对象、针对 R47 固定聚合的明确差别、全18一致的单方法输出及可失败条件；没有新增 query 求解信息。它仍可能失败，不能承诺大幅涨点。除这项计算机制外，本专题没有找到第二条证据足够且不重复的全18大结构路线，不为凑满三项而推荐。

71% 附近的平台是现象，不是 Bayes 上限、随机标签证明或所有模型同病的证明；TRAIN 少选也不能单归因于泛化过拟合。最重要的研究纪律是让新方案指出自己改变了哪一种可用信息、哪一个中间对象及哪一步决策，并允许证据让我们停止，而不是无限追加结构。
