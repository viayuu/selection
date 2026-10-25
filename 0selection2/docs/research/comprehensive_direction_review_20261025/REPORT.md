# 从零选择与71%平台，走向一个有终点的研究问题

完整科学综合，2026-10-25。研究对象：冻结NEW90k、全部18题、一次选择一个合法方法。

## 执行摘要

### 现在最值得相信什么？

**模型学到了条件先验和部分实例信号，但现有方法尚未稳定提取剩余的、可迁移的候选相对优势。** 这是目前最有根据的工作描述，不是“71%就是上限”，也不是已经确诊某个共同结构缺陷。零选择只是其中一部分：v2的R94B参照中，固定36个零/近零组占错误的19.46%，其余错误占80.54%。只救LEHD、DIFUSCO或两种ReLD，不能解释或解决整个ALL平台。

**这次应撤回两项旧优先建议，而不是再包装一次。** v2保留的连通关系计算已由R107实质触及，条件先验残差/密度比已由R108完成；它们没有带来所期待的联合改善。R105输入路由、R106原生编码器微调、已完成并提交的R110方法私有节点加纯CE，进一步削弱了“换一种共享方式、解冻或去掉复合损失便足够”的期待。不是所有指标都变差，也不是所有机制从此不可能有效；但继续优先追逐这些宽泛解释，证据已不足。

### 下一步只保留两项，且只有一项是新机制

| 顺序 | 研究主线 | 什么结果改变路线 |
| --- | --- | --- |
| 1，在途收束，不新增版本 | 等R109/R111的完整锁定结果，裁决“候选私有计算/族条件监督”能否形成可迁移的净决策收益。 | 若仍只交换错误，停止以共享干扰为默认理由追加私有层、私有头或另一种分组；若有联合收益，沿真实受益路径解释，而非宣称普遍解决。 |
| 2，唯一新增机制候选 | **每个query用合法历史TRAIN邻域拟合一个局部函数**，使邻居的表示位置与winner结果共同决定最终方法分数。 | 需要优于同邻域的局部常数读出，并在全18上产生有价值的改选；若没有，停止这一轮局部拟合路线，不自动升级邻居数、阶数或另一个ICL模型。 |

第二项不是“第一次使用历史标签”，也不是绕过R47。R47已经是FULL18可训练行为记忆；R97/98/100已经用全部81k历史TRAIN做冻结表格上下文。此次剩下的具体差别，是显式求解由**整个邻域几何和位置-结果交叉矩**决定的局部系数，而不是分离聚合标签value与key差修正。它仍可写成历史标签的几何加权读出，并不支配R47全部可学习函数类。

**支持强度应如实写低：当前没有直接项目证据证明这个局部函数信号存在。** 其价值是提出了一个不同、可计算、可反驳的对象，而不是论文已有大幅收益可以搬来。邻域如果只反映相近规模而不反映相对solver优势，或者winner边界不平滑，这条路线很可能失败。甚至某方法在局部support中没有winner时，所提纯one-hot ridge不能把它选为第一。不能靠“使用更多历史信息”的口号回避这些限制。

### 如何让01、02、03形成一条线，而非三份清单？

01提供**相对当前策略的净收益见证**，02提供待检验的局部计算对象，03要求**负结果能真正结束一类建议**。见证不单独占一个新方向，也不变成训练前无限审核：一次完整局部函数研究本身，就能同时回答“对应关系是否被使用”和“是否造成有价值的改选”。不要求旧分数先证明新表示能学会什么。

判定以同一完整模型为单位：自然Top1、固定目标真实救回与丢失、新误选、meanCost、actual regret、清单外错误及全部18题一起看。只增加选择次数不是成功；只降一点成本而降低准确率是取舍，不称共同突破；也不要求每题每指标全部严格上升。收益的来源与代价要明示，不能事后换目标。

**本报告没有启动任何实验。** 保留81k/4.5k/4.5k冻结划分、logical batch1024、seed2和候选池，不恢复删除候选、不重做R58、不加query标签或在线solver。R109/R111按本轮提供的状态仍待完整交付，R110已完成，旧post_R110文末的未完成时态作废。TEST/LIB已参与研发；本报告是同族LLM的暂定科学判断，不是独立复现实证。31条一手论文阅读记录去重为30篇，记录的是相关章节阅读，不声称30篇均逐页读完。

## 1. 研究全景与证据范围

### 1.1 覆盖的是机制，不是把版本数当证据票数

本次全文阅读了[第一轮综述](/public/home/shiys/0selection2/docs/research/deep_direction_review_20261023/REPORT.md)、[v2综述](/public/home/shiys/0selection2/docs/research/deep_direction_review_v2_20261023/REPORT.md)，以及三个完成专题：[决策与可辨识性](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/01_decision_and_identifiability.md)、[表示与计算](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/02_representation_and_computation.md)、[学习与优化](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/03_learning_and_optimization.md)。同时读取近期结果和关键前向，目的仅是辨认科学机制。

[artifact_index](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/artifact_index.csv)索引96个目录、78个编号，不等于96次独立训练。R1-R34仍是partial；R35-R104由两次综述及历史账本覆盖，本轮没有逐行重审全部代码。R105-R110以实际完成报告更新，R109/R111不纳入阴性计数。[reported_all_metrics](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/reported_all_metrics.csv)是机械报告行；本轮不复算预测、不认证完整性或显著性。

| 机制族 | 已做的关键内容 | 仍不能从历史推出什么 |
| --- | --- | --- |
| 输出与目标 | R39性能回归；R43差分；R78校准/成本头；R81跨实例AUC；R82条件排序；R85 OVR/EQL/cost/subset；R91链接；R93纯CE；R108密度比 | 既非“从未用成本”，也不能由某代理失败证明标签不可预测。R78部分项不收敛，不能全记成完整阴性。 |
| 表示与关系 | R37-38几何；R42关系/solver query；R56有向边状态；R63候选节点递归；R92层级；R99关系输入；R107连通路径 | 更强图表达力不自动识别部署性能；旧局部实验不等于全18同条件排除所有结构。 |
| 共享与专家 | R53局部专家；R64 PCGrad；R69完整18题task-private；R87池化后私有残差；R105 FFN专家；R109/110候选节点；R111族条件特征 | 私有状态、私有参数、完整梯度隔离不同；但不能据此无限否认已有私有化实验的反证价值。 |
| 历史上下文 | R47同题TRAIN可训练key与行为值；R97 winner分类；R98逐方法期望gap；R100严格成本pair+Borda | 普通检索、历史标签、可训练memory、表格ICL均非空白。局部函数的几何-标签联合求解仍是不同的具体偏好。 |
| 预训练与计算监督 | R50原生冻结编码；R54路线辅助；R55结构计费差；R61/62路线/性能预训练；R67 SupCon；R94早融合；R101 hint；R102其它donor；R106微调 | 学好共同困难度、路线或教师向量，不保证学好候选间性能残差；R94等小幅正结果也不能被抹掉。 |
| 优化与数据 | R36/40/49记忆和长训；R65 ScheduleFree；R68 SAM；R70 mixup；R73伪标签；R77样本规模；R80 CE/batch/LR；R89 Muon；R95重要性采样；R104容量 | 优化器不是全无效果；独立样本、曝光次数、更新数和活跃参数时间不同。有限阴性不等于普遍不可能，但降低同类路线优先级。 |
| 联合分布与额外观察 | R71全协方差成本差混合并积分冠军概率；R51短程行为；R74全18自适应前缀 | 不是首次联合性能建模或自适应探测；R74无完整结论，查询求解观察又超出当前授权。 |

机制细节与局部/全18、legacy/新数据区别见[旧R35-R79账本](/public/home/shiys/0selection2/docs/research/deep_direction_review_20261023/ASTRA_HISTORY_R35_R79.md)、[旧R80-R104账本](/public/home/shiys/0selection2/docs/research/deep_direction_review_20261023/ASTRA_HISTORY_R80_R104.md)及三个专题。本表不把继承微调、复用历史控制和不同预算当成独立匹配实验。

### 1.2 固定问题与正确的风险对象

令`t`为题型，`M_t`为当前合法候选池，`X`为部署时允许观察的实例，`Y`为native/FP64 ind winner，`C_m`为该固定候选部署的成本，`C_* = min_m C_m`。本项目当前相对regret定义使用正的oracle成本。三个总体目标分别为：

\[
\begin{aligned}
R_{01}(d)&=\frac1{18}\sum_t\mathbb E_t[\mathbf1\{d(X)\ne Y\}],\\
R_C(d;t)&=\mathbb E_t[C_{d(X)}],\\
R_r(d)&=\frac1{18}\sum_t\mathbb E_t[(C_{d(X)}-C_*)/C_*].
\end{aligned}
\]

因此，冠军后验、期望原始成本、期望相对成本不是同一个统计对象。`E[C_m/C_* given X]`不能替换成两个条件期望的比。ALL按题等权；跨题raw meanCost只作描述，不赋予不同任务共同物理尺度。`vs_SBS/vs_Oracle`是均值比，actual regret是逐实例比值后平均。SBS采用TRAIN拟合口径，不用事后最优方法冒充部署先验。

当前候选规则是八个带B题删除RouteFinder、MoSES_RF、MoSES_CaDA，删除RELD_MTL，原RELD_MOEL原值更名`reld_mt`，不取两种ReLD的逐例最小值。旧47%左右与新71%左右同时改变数据和候选竞争，不能当作同任务的二十多个百分点算法进步。

## 2. 相比v2，哪些结论真的改变了？

### 2.1 最近五项完成结果，改变的是建议而非数据口径

下表都是各自对照的TEST ALL，Top1和actual regret单位为百分数。跨行不是匹配比较；小差异未在本轮独立作不确定性认证。

| 已完成对照，A/B | Top1 A / B | actual regret A / B | 对旧建议的更新 |
| --- | ---: | ---: | --- |
| R105 静态/输入FFN门控 | 71.237037 / 71.339506 | 0.393442 / 0.392972 | 有小配置收益；固定36组正确18/20，没有广泛恢复。普通输入MoE不再优先。 |
| R106 冻结/微调native donor | 71.523457 / 71.160494 | 0.381866 / 0.390786 | 解除冻结不是已获支持的修复。保留R94的迁移正证据，不等于所有预训练无效。 |
| R107 独立边/连通路径 | 71.260494 / 71.307407 | 0.394675 / 0.394479 | 已触及v2的核心关系建议；固定36组正确21/18。撤回泛关系编码的优先地位。 |
| R108 先验CE/密度比 | 71.338272 / 71.217284 | 0.400252 / 0.399144 | v2备用方向已经完成，不再列为未试；不是Top1与成本联合改善。 |
| R110 tied/private节点，纯CE | 71.307407 / 71.233333 | 0.406464 / 0.399449 | 私有节点加CE不足以解决问题；成本小幅取舍不能抵充真实恢复证据。 |

来源：[R105 A](/public/home/shiys/0selection2/code/V4/runs/R105_conditional_representation_experts/A_static_gate/full_model_statistics/RESULTS.md)、[R105 B](/public/home/shiys/0selection2/code/V4/runs/R105_conditional_representation_experts/B_input_gate/full_model_statistics/RESULTS.md)、[R106-R108汇总](/public/home/shiys/0selection2/code/V4/runs/continuation_20261024_results/RESULTS_BRIEF.md)、[R108恢复报告](/public/home/shiys/0selection2/code/V4/runs/R108_conditional_density_ratio/preparation/report_only_recovery/RESULTS.md)、[R110完成报告](/public/home/shiys/0selection2/code/V4/runs/R110_method_conditioned_ce/RESULTS.md)。汇总中旧发布状态不替代后续完成记录。

R110尤其不能只报TEST较低regret。B的TRAIN/VAL/TEST Top1为71.264472/70.896296/71.233333，A为71.616941/71.103704/71.307407；VAL regret是B更差。固定36组TEST有4,598个真实winner，A/B正确70/44，B救回21、丢失47，另有41个新误选。TEST meanCost从11.374706635到11.374359128，确有小幅下降，不能说所有指标被支配。但这显然不是“漏选被恢复”的结果。

A/B参数约230万/1022万，锁定更新为43,200/21,600，并非等参数或等FLOPs因果对照。**能削弱的是“此私有化配方已经足够”与其优先可信度，不能证明所有私有化无效，更不能证明71%上限。** R110完成并提交`370e0b49a6783036ba0899fdbc995f3d2230a27f`；[旧post_R110](/public/home/shiys/0selection2/docs/research/post_r110_broad_mechanism_20261025.md)文末的未完成描述不再适用。

### 2.2 新发现与旧知识分开

| v2已经知道 | 本轮实际增量 | 对研究动作的影响 |
| --- | --- | --- |
| winner频率不保证存在当前表示可识别的获胜区域；零选择不等于零梯度 | 将问题改成相对当前策略的可部署净收益见证，不必先证明Bayes区域 | 不再重复后验常数例子；直接问一次改选多救谁、伤谁、成本怎样。 |
| 关系、先验残差、私有计算是候选解释 | R105-R110已覆盖关键近邻，尤其R107、R108、R110提供新反证 | 正式撤回宽泛优先建议，而非把它们移入“再改一点”的长队列。 |
| AUC、成本和冠军分类目标不相同 | 成本pair胜率不能直接当互斥冠军pair后验；校准不等于输入充分 | 不为R100补个coupling名字就声称恢复冠军概率；残差分析也不充当充分性认证。 |
| 低TRAIN拟合可能涉及优化、表示和目标 | 有限假设类的proper-loss最优不保证最佳argmax；当前WD直接收缩上界很小 | 既不能用CE正确宣告不可学，也不能以强WD或不干净CE无限续命。 |
| R47已学历史记忆，固定表格ICL已试 | 明确位置-结果交叉矩、联合几何解及其受限反例，并进一步写成等效标签权重 | 仅保留一个具体局部函数假说，不宣称首次检索或函数类严格更强。 |

## 3. 零选择：一张多层因果假说图

下面的箭头是研究中的可能路径，不是已识别的因果效应。`[O]`表示已有报告观测，`[H]`表示解释假说，`[D]`表示定义或代数关系。

```text
[D] 冻结候选/部署/原生winner口径
       -> [O] 每方法全局与局部winner支持
                         |
[D] 允许输入X + 历史TRAIN -> [H] 可迁移的相对优势规律
                         -> [H] 表示保留/局部计算可达
                         -> [H] 监督与参数更新形成优势特征
                         -> [H] 读出利用并在全池正确比较
                         -> [D] argmax或联合叶子argmax
                         -> [O] 0/少量选择、救回、误选与成本

竞争解释：局部条件下本来不是最优动作；特征存在但读出不用；
          邻域支持不足；有限目标取舍；尚未学到而非无法学到。
```

### 3.1 支持、可辨识、形成、利用四个环节不能合并

| 层次 | 已观察到什么 | 主要假说与最强反方 |
| --- | --- | --- |
| 支持 | R84A的TSP TRAIN中LEHD有14,279个winner、选择7次；DIFUSCO有6,806个winner、选择0次 | 不是都可叫极端长尾。但全局正例多不保证局部边界密度足够；CVRP/RouteFinder在900个条件单元中只有323个含正例。 |
| 可辨识 | R103同题同精确N保频次置换使R94B VAL从71.4284降到67.7134 | 已有实例信号；不能说只背先验。反方是置换未固定全部生成条件，也没有证明每个漏选方法都存在可恢复区域。 |
| 特征形成 | 当前多个模型TRAIN也少选；R110私有化未改善 | 可能缺少优势相关特征或学习路径。反方：输出失败不证明隐藏特征不存在；参数更多也未必有更多有效学习时间。 |
| 读出与竞争 | R94B的LEHD winner进Top3为87.80%，DIFUSCO为26.95%；margin AUC为0.6709/0.6123 | 二者失败形态不同，统一偏置不合理。反方：AUC是跨实例列排序，不能保证同实例全池可盈利切换。 |
| 决策后果 | R110 B选择/恢复变化伴随救回21、失去47、新误选41 | 选择频率变化不能当学习成功。双方都错的改选可改变成本而不改变Top1。 |

数据来源为[v2第2-4节](/public/home/shiys/0selection2/docs/research/deep_direction_review_v2_20261023/REPORT.md)、[R103](/public/home/shiys/0selection2/code/V4/runs/R103_conditional_signal_map/RESULTS.md)和R110报告。R94B的逐例结构不自动外推成R110的逐例结构。

### 3.2 非零梯度为何仍不足？

对自然CE，正确类输出梯度是`p_y-1`，被严重漏判时并不小。参数梯度还要经过输出对参数的Jacobian：

\[
\nabla_\theta\ell=J_\theta^\top(p-e_Y).
\]

因此输出有学习压力，不保证需要的实例方向已表示、能同时由有限共享参数拟合，或被读出采用。[Gradient Starvation原论文](https://proceedings.neurips.cc/paper_files/paper/2021/file/0987b8b338d6c90bbedd8631bc499221-Paper.pdf)在特定线性化双特征条件下给出机制；不是当前模型的确诊。

反方必须同时保留：[DFR原文第4-6节](https://arxiv.org/html/2204.02937)在若干伪相关基准中发现，失败的预测背后仍有可读出的核心特征。它不能证明我们已有这些特征，也不能照搬其使用平衡留出标签重训头的制度。其价值是阻止“漏选必因特征未形成”成为唯一故事。R67已分阶段学表示和重置读出，R78已做若干固定读出处理，不能将它们抹去。

### 3.3 可靠概率与完整输入充分性不是一回事

`E[e_Y given q(X)] = q(X)`描述预测向量可靠；`q(X)=P(Y given X)`才是完整输入后验。粗化输入后的条件概率可以可靠但丢掉可决策信息，[Vaicenavicius等的Proposition 1](https://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf)支持这个区别。低ECE不能证明输入已经用足。

反过来，按粗分数箱发现输入相关残差，也不能立刻证明遗漏了新输入。若`d0`由完整预测状态决定，设`W_k=1[Y=k]-1[Y=d0]`、`r_k(q)=E[W_k given q]`、`m_k(q)=q_k-q_d0`，则

\[
W_k-m_k(q)=[W_k-r_k(q)]+[r_k(q)-m_k(q)].
\]

第二项是预测状态内部的校准失配，可能与箱内输入变化相关。故01的残差见证适合作为寻找可改进规则的工具，而不是把“校准还是表示”一次性因果分类的认证。后文直接使用动作净收益，避开这层不必要承诺。

## 4. ALL平台：另一张图，不能由零选择图代替

```text
[D] 每题分布与候选池 -> [O] 强条件先验、题间难度差异
         + 允许的实例信号 -> [H] 有限样本中的相对性能规律
         + 共享计算/目标/更新路径 -> [H] 可迁移决策函数
                                      |
                                      v
[O] 清单内少选错误 + [O] 清单外常选方法之间的错误
                                      |
                     [O] 新模型纠正旧错，同时破坏旧对
                                      |
                     [D] 18题等权Top1/成本/regret聚合
                                      -> [O] 约71%平台

[D] 多轮VAL/TEST反馈 -> 研究路线选择
                     -> 结论外推的不确定性，而非自动证明单模型过拟合
```

### 4.1 平台包含大量非零选择错误，而且并非都无成本

v2的R94B共有23,022个TEST错误，固定36组占4,481，其外18,541。错误中所选成本超过oracle 1%的有9,820个，占错误42.65%、贡献regret的81.58%；精确零成本错误只有116个。不能将主要差距解释为浮点并列，也不能把成本gap当后验概率margin。

TSP、CVRP、OVRP、VRPL、OVRPL五题承担55.6251%的错误，却只承担39.8157%的regret。R94B相对TRAIN条件多数规则，在TSP/CVRP分别已有9.3778/15.0889个百分点Top1收益，在VRPL只有0.0889个百分点。问题之间既有已用实例信息的强弱差别，也有错误成本差别。研究应覆盖全部合法池，用这些差异解释收益来源，而不是将全18问题改写成一个ReLD对局。

### 4.2 为什么换了模型，却常常只换错误？

| v2固定预测比较 | 纠正 / 伤害 | 可支持的判断 |
| --- | ---: | --- |
| R94B对R89B | 1,756 / 1,628 | 真实增益存在，但大量被反向错误抵消。 |
| R102A对R94B | 1,114 / 1,104 | 输出不同不等于有容易部署的互补性。 |
| 三者共同错 | 20,842 | 占R94B错误90.53%；支持共享困难的工作假说，不识别其唯一原因。 |

知道答案后逐例挑三者中正确输出可得到74.2691%，这是**给定模型集合的答案感知覆盖统计**，不是可部署ensemble、可学习上限或突破预期。固定36组全救且无伤害得到77.1099%也是诊断算术，不可部署。两者都不能用来为下一方法承诺涨点。

可能的解释包括：相近表示/训练偏好遗漏了相同规律；有限样本下剩余优势不稳定；目标偏向同一类稳定低成本动作；模型改变的是估计噪声而非新信息。反方是这些版本共享数据、seed、部分代码及研发史，不是独立证据票数，也不是所有算法函数类的搜索。没有证据将其中任一解释提升为Bayes上限。

### 4.3 排除几个已失去优先级的万能解释

**并非所有模型严重过拟合。** R40旧场景出现高TRAIN低VAL，R49小子集能记忆；当前R89B/R94B/R102A锁定TRAIN-VAL差约0.44-0.52个百分点，许多漏选在TRAIN已出现。小gap不证明收敛或不可学，但不能套用旧过拟合故事。

**不是只要再换选型指标。** v2所读保存VAL轨迹中，成本锁定点与最高Top1点差最多0.0654个百分点；R93两臂只差0.041975/0.030864。目标确有差别，却没有证据藏着一条高很多的Top1曲线。本报告不据此重选任何checkpoint。

**不是当前WD把权重大幅压回原点。** 以R110等当前协议的AdamW显式乘法衰减为例：

\[
D_S=\prod_{s=1}^{S}(1-\lambda\alpha_s),\qquad
1-D_S\le\sum_s\lambda\alpha_s
\le57600\times10^{-4}\times10^{-4}=0.000576.
\]

即上界0.0576%，实际warmup/cosine积分更小。这是协议数值推导，不是参数测量；不排除路径依赖或小梯度方向的相对影响，也不是与WD=0完整轨迹的差异界。[L2动力学论文](https://papers.nips.cc/paper_files/paper/2020/file/32fcc8cfe1fa4c77b5c58dafd36d1a98-Paper.pdf)的无限宽齐次/梯度流等条件不能直接套给这里的AdamW。但“大幅直接收缩导致平台”这一朴素解释应退出。

**也不是把曝光次数叫成更多信息就会有效。** R77同49,680更新下，10k到70,114独立样本带来Top1 70.5691到71.2252的真实小收益；当前81k、57,600总更新终点的平均曝光约40.454次/样本。R80已按曝光匹配改变batch和更新数。新独立数据可能有价值，但本轮没有生成新标签授权，不能从这些数字推出无限长训、加数据或最终grok的承诺。

## 5. 文献怎样映射到已做与未做，而非替代项目证据

下面只保留影响取舍的桥梁。论文栏是原文提供的结论或方法；“项目延伸”不是原论文已经证明我们的模型应当改进。具体阅读章节、版本和未读范围保留在[01来源表](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/01_sources.csv)、[02来源表](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/02_sources.csv)、[03来源表](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/03_sources.csv)。

### 5.1 决策理论给的是对象与条件，不是涨点保证

| 一手论文及原结论 | 项目已做的最近机制 | 未被证明的迁移与取舍 |
| --- | --- | --- |
| [Tewari-Bartlett](https://www.jmlr.org/papers/volume8/tewari07a/tewari07a.pdf)讨论多分类校准；[H-consistency](https://proceedings.neurips.cc/paper_files/paper/2022/file/051f3997af1dd65da8e14397b6a72f8e-Paper-Conference.pdf)保留假设类和minimizability gap | R91、R93、R108、R110改变目标/链接 | proper loss的理想正确性不等于有限类最佳argmax；仍不构成再试CE变体的理由。 |
| [模型校准](https://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf)区分可靠性层次；[Multicalibration](https://proceedings.mlr.press/v80/hebert-johnson18a/hebert-johnson18a.pdf)连接可计算子群残差与弱学习 | R78校准、R103信号诊断、R108先验残差 | 用于寻找动作见证，不把有限子群可靠性或非零残差当输入充分/净收益证明。 |
| [Agarwal排序界](https://jmlr.org/papers/volume15/agarwal14b/agarwal14b.pdf)是proper代理到二部排序的单向关系；[Wu-Lin-Weng](https://jmlr.org/papers/volume5/wu04a/wu04a.pdf)耦合互斥类别条件概率 | R81/82跨实例排序；R100成本pair+Borda | AUC高不保证全池argmax正确；成本pair胜率不能直接输入冠军概率coupling。 |
| [Audibert-Tsybakov](https://arxiv.org/pdf/0708.2321)给附条件的后验margin分类界；[Label Shift](https://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf)需要类条件分布不变和可辨识 | 近并列讨论、旧新候选池、R108条件先验 | 成本gap不等于后验margin；删候选改变winner事件，不是一般的label-shift校正问题。 |
| [Logit Adjustment](https://arxiv.org/html/2007.07314)针对balanced error；[JTT](https://proceedings.mlr.press/v139/liu21f/liu21f.pdf)强调worst-group重加权 | R85平衡/OVR、R95逆概率采样 | 方法等权不是自然Top1；R95保目标的采样与JTT改目标不同，但未试某种权重不自动值得优先。 |
| [SPO原文v5](https://arxiv.org/pdf/1710.08005v5)构造决策损失代理，并在特定成本分布/可行域条件下一致 | R39、R71、R78、R82、R85已利用成本 | 有限候选下可写成cost-augmented分类；不是新监督信息，也不保证冠军Top1，不作为第三方向。 |

有限函数类的限制值得具体说明：若两个等概率输入的真实二分类后验是`(0.9,0.6)`，函数类只允许`A=(0.9,0.4)`和`B=(0.55,0.55)`，population CE分别为0.539594、0.648005，而准确率为65%、75%。这是03的数学构造，不是项目数据。它说明CE在受限类里偏好的函数可以不是Top1最好的函数；既不能据此宣布该换loss，也不能用“CE proper”排除学习问题。

R100的对象错误也不只是术语：它估计的是包括第三方法冠军行的`P(C_i<C_j given X, strict pair)`；经典coupling要求的是`P(Y=i given Y in {i,j},X)`。即使所有成本pair胜率精确给出，冠军向量仍未必唯一。01第4.2节的三个候选排序分布反例已说明这一点。R71确实建模过联合成本差并积分冠军概率，不能再把“考虑相关性”列为首次。

### 5.2 表示论文必须连接部署相对优势

| 一手机制 | 实际历史近邻 | 本轮取舍 |
| --- | --- | --- |
| [GSN轨道计数](https://arxiv.org/html/2006.09252v3)、[EGT持续边通道](https://www.cs.rpi.edu/~zaki/PaperDir/SIGKDD22.pdf)、[LSPE结构/位置分离](https://arxiv.org/html/2110.07875v2) | R56、R63、R88、R92、R99、R107 | 更能区分图，不等于更能区分部署solver；没有具体剩余对象，暂停泛高阶图/位置路线。 |
| [CLRS算法状态与反馈](https://proceedings.mlr.press/v162/velickovic22a/velickovic22a.pdf)、[InfoMin保留下游信息的视图条件](https://proceedings.neurips.cc/paper/2020/file/4c2e5eaae9152079b9e95845750bb9ab-Paper.pdf) | R54、R61、R63、R67、R101 | 递归要有状态语义；最优值不变的增强不保证部署winner不变。辅助成功不能替代最终选择收益。 |
| [TabR标签value与key差修正](https://arxiv.org/html/2307.14338v2)、[DNNR局部导数估计](https://proceedings.mlr.press/v162/nader22a/nader22a.pdf)、[TabICLv2表格先验](https://arxiv.org/html/2602.11139v1) | R47、R97、R98、R100 | 唯一条件性保留的是每query联合几何的局部拟合，不是首次检索，不是升级TabICL版本，也不冒称照搬DNNR。 |
| [EDDI观察信息收益](https://www.nowozin.net/sebastian/papers/ma2019eddi.pdf)、[SATzilla特征/探测成本](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/08-JAIR-Satzilla.pdf) | R51短行为、R74全18自适应前缀 | 信息收益不等于决策收益；现阶段不获准query求解观察，授权后仍必须计费，且不是首次自适应探测。 |

为何不继续泛关系编码？R107已经使用共享中间节点的`i-j-k`路径，含方向、shortcut/closure、局部等待、窗余量、需求前缀与计费路径，并两次注入节点流。其边界是真实的：它不是完整可行路线，不能显式枚举所有长程互斥。但“没有显式算全”不等于“完整前向必然混淆”，因为原始节点及矩阵还经过其它分支。02因此撤回宽泛推荐，本综合采纳这一撤回。

重开结构方向需要一个能连接三个环节的对象：当前完整前向确实难以利用的区别、它怎样改变不同候选的响应、拟议计算怎样保留它。不是要求先证明万能表示理论，也不是再建一套前置审核；**现在尚无这样的具体对象，所以不占下一研究名额。**

### 5.3 学习动力学解释需同时容纳反方

| 一手文献 | 项目约束 | 本轮判断 |
| --- | --- | --- |
| [Gradient Starvation](https://proceedings.neurips.cc/paper_files/paper/2021/file/0987b8b338d6c90bbedd8631bc499221-Paper.pdf)、[Lazy Training](https://papers.neurips.cc/paper/8559-on-lazy-training-in-differentiable-programming.pdf) | 特征耦合/线性化有条件；目前没有当前网络处于NTK-lazy的轨迹证据 | 保留形成困难的可能，不以低TRAIN或强读出就确诊starvation/lazy。 |
| [DFR](https://arxiv.org/html/2204.02937)、[Feature Sieve](https://proceedings.mlr.press/v202/tiwari23a/tiwari23a.pdf) | 其它任务可有未被使用的特征；简单特征也可能有效 | 不能盲抹规模/容量，不拿VAL标签重训头；R67/R101已覆盖部分分阶段思路。 |
| [PCGrad](https://papers.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)、[L2动力学](https://papers.nips.cc/paper_files/paper/2020/file/32fcc8cfe1fa4c77b5c58dafd36d1a98-Paper.pdf) | 有害冲突还涉及幅度/曲率；当前WD的直接收缩量很小 | R64已有实测干预，R69完整私有化已试；不以负余弦或WD存在作为继续理由。 |
| [Double Descent](https://arxiv.org/html/1912.02292)、[随机标签记忆](https://arxiv.org/pdf/1611.03530v2) | 非单调训练现象和记忆能力都依赖设置 | R40/R49不是当前全81k可迁移性的证明，不把无限长训当等待奇迹。 |

仍应保留历史正证据：R65的匹配ScheduleFree对照有1.1667个百分点旧数据Top1收益；R67有0.6111；当前R89混合Muon有0.1877；R101教师hint差别有0.1123。它们足以反对“所有训练技巧无效”，不足以把下一批训练技巧重新提到最高优先级。研究取舍不是给整个方法家族判永久死刑，而是停止没有新理由的优先追加。

最后，[Dwork等的自适应数据分析](https://arxiv.org/pdf/1506.02629v2)说明控制反馈可以取得特定保证；我们的历史没有事后继承其Thresholdout制度。承认TEST参与研发，与判断某次TRAIN-VAL是否过拟合是不同问题。现在不自动新建划分，未来独立确认需另行安排。

## 6. 一把共同尺子：直接见证净决策收益

### 6.1 不必先证明Bayes区域

取一个事先固定的完整参照策略`d0`，以及只使用合法前向信息的新策略`d1`。对每个评估实例定义：

\[
W=\mathbf1\{d_1(X)=Y\}-\mathbf1\{d_0(X)=Y\},\qquad
G=(C_{d_0(X)}-C_{d_1(X)})/C_*.
\]

`mean(W)`是Top1变化，`mean(G)`是actual regret下降，均先题内再18题等权。成本改善另看`mean(C_d0-C_d1)`的逐题值。它们在训练时可以由已有TRAIN监督定义，但**部署时绝不读取当前query的W、G、Y或成本**。

若新规则只在输入可计算区域`h(X)=1`将动作改成`k(X)`，则

\[
\Delta\mathrm{Acc}=\mathbb E[h(X)(\mathbf1\{Y=k(X)\}-\mathbf1\{Y=d_0(X)\})].
\]

正值反驳“当前策略在这个规则类内已经最优”，不用证明`k`是真实Bayes冠军，甚至不用要求其precision大于50%。第三个方法仍可能更好。负值或未找到见证也只否定已检查的策略类，不证明所有原始输入规则都无价值。

### 6.2 必须保留四种事件，而非只有“救回/误选”

| 旧动作 / 新动作 | Top1增量 | 成本可能怎样 | 应怎样表述 |
| --- | ---: | --- | --- |
| 错 / 对 | +1 | 新动作达到原生winner对应成本 | 真正纠正；固定目标winner内另记救回。 |
| 对 / 错 | -1 | 可能为零成本并列，也可能显著变差 | 真正伤害；不能被更多选择次数掩盖。 |
| 错 / 错 | 0 | 可变好、变坏或不变 | 新false pick可能在这里，不等于又损失一个Top1。 |
| 对 / 对 | 0 | 固定winner口径下保持正确 | 不是新增恢复。 |

因此R110的`21-47=-26`只是固定目标winner正确数的变化，不能再减41便声称全体Top1净变化。固定36组是旧R84A VAL清单，后续不随结果移动；清单外和完整18题必须同时保留。某单一零组被救回而其它方法恶化，不能称全局解决；一个方案若在常选方法之间带来真实收益，也不应因为零组数量没变而被否定。

### 6.3 怎样使用见证，而不制造无限前置诊断？

**最终采取一个整体新决策规则，就先评价它自身的W/G，而不是先训练另一个“能否改选”的gate。** 第8节局部函数模型的argmax本身就是合法规则，能直接回答01的问题。已有分数/元数据若显示稳定的正收益子群，可用于解释，但不成为新表示研究的前置许可；旧表示找不到见证不能否决新表示。

仅在解释需要时，按预先固定的任务、候选及差异事件汇总，不从TEST反复挖阈值、再把最好的区域当部署收益。也不把正的预测残差相关直接当动作收益：若旧概率差为`m_k=q_k-q_d0<=0`，需要残差优势足够抵消这个差，才能有正的`E[h W_k]`。01的受限策略类遗漏价值可以限制“不可能”措辞，但本次不另做理论认证项目。

## 7. 优先事项一：让R109/R111结束共享机制这一轮追问

这是**现有实验的科学收束**，不是新模型提案，不改变它们的协议、选型或运行。状态采用本轮[running_snapshot](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/running_snapshot.csv)及用户完成通知：R110已完成，R109/R111尚未有本报告可引用的完整结果。

### 7.1 数学对象、监督和最终路径

**R109的问题**是同一原始图经过候选条件的节点计算后，独立参数是否让优势证据更易形成。每方法从共享节点状态与其solver token形成`H_m`，经过两层tied或private节点更新得到摘要`h_m`，共同score给出`u_m`，最后全池argmax。训练仍是已锁定的两view复合目标：CE、comparison、relative risk与R-Drop；不是纯冠军NLL。原始输入和候选身份决定分支，query结果不进入前向。[R109方案](/public/home/shiys/0selection2/code/V4/runs/R109_method_conditioned_nodes/PLAN.md)

**R111的问题**不是分组本身增加了标签信息，而是条件监督能否让族内优势学习更集中：

\[
p_m(X)=\pi_{f(m)}(X)q_{m\mid f(m)}(X),\qquad
\ell=-\log\pi_{f(Y)}(X)-\log q_{Y\mid f(Y)}(X),\qquad
d(X)=\arg\max_{m\in M_t}p_m(X).
\]

根读共享post-joint表示；非退化族用两层私有节点计算形成族内证据，最终仍共享pool/score等读出。所有合法族都算，单方法族条件概率为1，不用真实winner选运行分支，也不是硬选一个族后丢掉其它族。真实族私有节点参数得到该例条件NLL梯度，其它族私有节点参数没有这项直接梯度；共享编码、context、读出及部分族相关参数仍耦合。[R111方案](/public/home/shiys/0selection2/code/V4/runs/R111_family_conditional_features/PLAN.md)、[实际前向](/public/home/shiys/0selection2/code/V4/r111_model.py)

最近历史是R63候选节点递归、R69完整task-private、R105共享节点流的FFN专家，以及R109/110候选私有节点。R111真正改变的是**族内私有节点与正确族条件监督的联合路径**，不是第一次候选条件化或第一次私有模型。R109对R110主要改变整个训练目标与view制度；因此若结果不同，不能归功于其中一个损失系数。

### 7.2 成功和失败应支持到哪一层？

可能成功：共享的后期节点变换不足以同时形成不同候选/族的优势，条件监督减少了对私有特征的无关直接更新；新增证据最终经完整池比较转成净收益。

可能失败：共享前段已丢失所需信息；条件分支支持更少；可辨识规律很弱；共享读出未使用它；根给真实族的质量太低；或者这些不是主要瓶颈。当前R110、R64和R69使“大多数缺口都由共享压制造成”的解释可信度下降，而不是仍处在未受检验状态。

R111有一个精确但有限的诊断：固定根与其它族输出，若

\[
\pi_{f(Y)}(X)<\max_{j:f(j)\ne f(Y)}p_j(X),
\]

则真实族内怎样重分配概率都无法使真实winner胜出。它可解释族内改善为何没转成全池改善，却使用了真实答案，不能作为部署路由或成绩上限。发现这种情形也**不自动授权再做一个私有根版本**；要优先尊重现有整体路线是否已有净收益。

负结果的实际停止对象是：没有新机制对象或可迁移正证据，却继续以“还有共享部分没拆干净”为理由追加私有层/头/族划分。它不宣称所有参数分解数学上无效。03保留的学习时间/任务顺序问题也不升为第三方向：当前没有优势特征先形成、随后被其它题更新系统性抹掉的证据；单看负梯度余弦或TRAIN曲线不够，R64/R69/R80已构成反方历史。

## 8. 优先事项二：唯一新增机制是局部函数，不是泛ICL

**科学问题：相同题型的历史实例中，表示位置的局部变化与完整候选池winner之间，是否存在全局压缩读出尚未利用的可迁移对应关系？** 不是问“加历史标签有没有用”，也不是只为某对ReLD设一个专家。这里的邻居是**历史训练实例**，坐标是**实例表示空间**，不是一张路线图里的邻接节点。

优先级是条件性、探索性的：若R109/R111给出清晰联合收益，应先收束那条已见收益的路径；若它们没有，局部函数是本轮唯一保留的新机制候选。没有直接局部信号正证据，因此不把它称为最可能突破平台的方法，只认为它比继续改名结构/损失更有明确可反驳的内容。

### 8.1 完整数学定义：学习什么，推理时算什么？

令`z=E_theta(X)`为从原始合法输入得到的低维实例表示。科学起点沿用现有实例编码、TRAIN-only归一化和mean/max等摘要，再加一个共享投影；**不同时新增donor、高阶关系网络、族划分或性能标签**。表示及投影可以学习，不固定为R97的357维统计表，也不以已训练checkpoint冒充从头初始化。

对query `q`，从同题同合法池的冻结TRAIN取`K`个表示邻居，排除query及同base关联副本。检索只用表示和排除标识，不用query结果。设支持集为`S_q`，`Y_i`为其中历史实例的原生winner one-hot：

\[
w_{qi}=\frac{\exp(-\|z_i-z_q\|^2)}{\sum_{j\in S_q}\exp(-\|z_j-z_q\|^2)},\qquad
\phi_i=[1;(z_i-z_q)],\qquad \Phi_q=[\phi_i^\top]_{i\in S_q}.
\]

用`W_q=diag(w_q)`和`lambda>0`定义一次局部加权ridge：

\[
B_q=\arg\min_B\sum_{i\in S_q}w_{qi}\|\phi_i^\top B-Y_i\|_2^2+\lambda\|B\|_F^2
=(\Phi_q^\top W_q\Phi_q+\lambda I)^{-1}\Phi_q^\top W_qY_S.
\]

最终直接使用局部函数在query处的值：

\[
s_q=e_0^\top B_q,\qquad d_1(q)=\arg\max_{m\in M_t}s_{qm},\qquad e_0=(1,0,\ldots,0)^\top.
\]

`s`是排序分数，可以越出概率区间，不冒称校准后验，不再经另一个旧分类头绕过局部函数。全部合法方法在同一向量里比较，最后只运行所选一个方法。

为避免“有一个内层求解，但外层任务未定义”，本综合将初始学习目标具体化为**自然频率one-hot平方监督**：

\[
\mathcal L_{\rm outer}(\theta)=\frac1{18}\sum_t
\mathbb E_{q\sim D_t^{\rm TRAIN}}\|s_\theta(q;S_q)-e_{Y_q}\|_2^2.
\]

query标签只在这个监督损失出现，不进入`Phi`、`W`或support标签矩阵。固定邻域索引内，通过线性求解向query/support表示反传；top-K索引本身是离散选择，不声称处处可微。自然题内样本权重不改成候选均衡，也不让选择频率追随winner频率。

外层平方目标是综合者为这条提案作的明确化选择，**不是第三个“新loss”贡献，也不宣称比CE优越**。它和内层的one-hot回归语义一致，不再附加将回归分数转换为CE logits的温度或尺度选择；具体有限函数类依然不保证最佳Top1。若比较局部常数与局部仿射，两者应保持同一外层目标，不能把损失差别算作局部几何的证据。未来选型与完整评估规则须在实施前固定，本报告不改任何当前模型的checkpoint规则。

可打印的决策流程是：

```text
q -> raw input encoder -> z_q
S_q = same-task TRAIN neighbors, excluding related base instances
Phi = [ones, z_support - z_q]
Y_support = historical native-winner one-hot rows
A = Phi.T W Phi + lambda I
solve A B = Phi.T W Y_support
scores = first row of B
return one legal method with maximum score

Training only: compare scores with the query's one-hot target.
Deployment: no query label, cost, route, solver probe, or answer-aware gate.
```

### 8.2 相比R47，实质不同在哪，又没有强到哪里？

R47已有同题TRAIN、共享可训练key、选中reference两侧key梯度、历史gap/winner、key差修正，以及加入原score的128维行为向量。其graph summary是周期刷新的detached缓存，不能因此称它固定检索或不学习。实际模型对每方法的聚合可抽象为：

\[
b_m=\sum_iw_i V(z_m,g_{im},y_{im})+\sum_iw_i T(k_q-k_i,z_m).
\]

局部函数显式使用两个耦合统计量：`Phi.T W Phi`与`Phi.T W Y`。固定query、key、参数及一致mask时，R47对等距reference交换完整结果记录保持上述聚合不变；局部拟合一般会改变，因为结果和方向的位置对应改变了。依据为[R47实际模型](/public/home/shiys/0selection2/code/V4/r47_model.py)及[机制说明](/public/home/shiys/0selection2/code/V4/R47_README.md)。

02的简化构造可直接检查：query在原点，三点为`(1,0)`、`(0,1)`、`(-3/5,-4/5)`，到query等距。仿射插值在query处的系数是`(1/4,1/3,5/12)`，它们和为1、加权坐标为0。因此交换第一与第三点的A/C winner记录，会交换最高分方法；等距分离聚合仍不变。严格次序在足够小的正ridge下保持。**这是固定表示下的计算见证，不是真实项目样本，更不是泛化收益证据。** 可训练key可能主动打破等距，故不能推出R47整个函数类无法学到同一决策。

还必须修正“拟合函数不再是加权读出”的过强说法。把上式展开：

\[
s_q=a_q^\top Y_S,\qquad
a_q^\top=e_0^\top(\Phi_q^\top W_q\Phi_q+\lambda I)^{-1}\Phi_q^\top W_q.
\]

它仍是历史标签的线性加权读出，但`a_q`由**整个support的联合几何**决定，可以带负权，不只是独立query-key距离的非负softmax。其新意是特定计算偏好和监督对应关系，不是信息凭空增加，也不是数学上严格强于所有深度记忆网络。若任何收益都能由同一已学表示上的局部常数预测解释，这个新机制主张就应降级。

[TabR第3.2节](https://arxiv.org/html/2307.14338v2)本来就受DNNR启发；[DNNR第2-3节](https://proceedings.mlr.press/v162/nader22a/nader22a.pdf)还明确区分“每query拟合一个局部平面”与它自己的“对邻居分别估导数再Taylor预测”。本提案属于前一类可微局部ridge，不借DNNR的实验或误差界宣称当前全18分类已得到证明。

### 8.3 为什么邻域弱信号下它可能失败？

需要的不是embedding距离近，而是相对winner规律在该表示附近有可估计变化。令`eta_m(z)`表示固定表示下的条件winner概率，只作解释，则

\[
s_m-\eta_m(z_q)
=\left(\sum_i a_i-1\right)\eta_m(z_q)
+\sum_i a_i[\eta_m(z_i)-\eta_m(z_q)]
+\sum_i a_i[Y_{im}-\eta_m(z_i)].
\]

三项依次是常数收缩偏差、局部函数逼近误差、标签残差。无正则且局部线性系统适定时，仿射再现可消去一阶变化的偏差；有ridge时只是近似。这个好处需要邻域内规律可平滑描述，而固定部署winner可能对细小结构跳变。

在额外假定表示/权重固定、support标签条件独立时，最后一项方差至多`sum_i a_i^2 / 4`。负权和病态几何可以放大它。因此“更多方向”并不总胜过常数平均。这个方差式不是对同一TRAIN标签学习出的表示的现成泛化界，也不假定本项目已满足独立条件；它只是指出收益必须超过额外估计噪声。

还有一个严格的支持限制：某方法`m`若在support中没有winner，其`Y_S`列全零，所以`B_m=0`、`s_m=0`。对于这里的`lambda I`、正权和one-hot目标，总分满足

\[
\sum_m s_m=e_0^\top(M+\lambda I)^{-1}Me_0\in(0,1),
\qquad M=\Phi^\top W\Phi.
\]

至少一个有支持的方法分数为正，因此无支持方法不会成为argmax。于是该定义的Top1不超过“query真实winner在support标签中出现”的覆盖率。这是**机制必要条件的推导**，不是已测得的项目覆盖率，也不是可部署oracle成绩。它说明局部winner缺失时，不能期待ridge靠更好的求解器凭空救回该方法。追加全局先验残差或成本目标会变成另一方法，本轮不自动扩张提案来逃避此限制。

### 8.4 现有证据到底支持多大期待？

| 最近历史 | 已有事实 | 对局部函数的含义 |
| --- | --- | --- |
| R47 FULL18 trainable memory | legacy TEST约47.36%，仅历史对照；已有位置修正与结果记忆 | 不能称检索未试，也没有据此证明对应关系有收益；年代不同不构成对新90k局部ridge的严格否定。 |
| R97冻结winner上下文 | TEST 71.449383%，regret 0.394650%；相对R89B纠正1,759、伤害1,735 | 大上下文能用，但净差很小；仅历史标签暴露不是大改进的正证据。 |
| R98/R100固定表格上下文 | TEST Top1为71.216049/71.316049，regret 0.386229/0.389043 | 成本目标和pair信号有取舍，不能把局部one-hot拟合伪装成重复的gap/Borda。 |
| R103与v2条件先验比较 | 存在实例信号，但未识别“同邻域斜率比常数更好” | 只提供可能性，不提供局部平滑、足够支持或新增方向可迁移的直接证据。 |

来源：[R97](/public/home/shiys/0selection2/code/V4/runs/R97_tabular_in_context/RESULTS.md)、[R98](/public/home/shiys/0selection2/code/V4/runs/R98_performance_context/RESULTS.md)、[R100](/public/home/shiys/0selection2/code/V4/runs/R100_pairwise_context/RESULTS.md)。R98已经使用TabICLv2回归器；R97使用81k全上下文，不可改写成只有少量示例。上述结果不是局部ridge正证据，也不支持泛ICL必然无效。

**综合取舍是允许一个明确问题，而不是推荐一条无限搜索路线。** 现有事实已否决“历史记忆就是遗漏的普遍答案”“局部函数必然恢复无支持winner”“只换表格模型便突破”的强叙事。较弱假说仍未被直接检验：联合几何系数能否在现有支持内减少不必要的局部平均偏差，并把这些改进转成全池净正确。

### 8.5 计算代价不是一句“小头”可以带过

令`N_t=81000`，表示维数`d`，support数`K`，方法数`M_t`。忽略编码器，精确全bank距离搜索是`O(N_t d)`；构建局部系统、一次分解和所有方法右端求解的量级为：

\[
O(Kd^2+d^3+KdM_t+d^2M_t).
\]

`d`在此略去常数截距维。标签右端可以共享同一矩阵分解，不为每方法重跑encoder或候选solver。表示bank需保存`O(N_t d)`坐标及历史标签；推理还需query编码、检索与局部解，不能仅报告矩阵求解时间。

训练定义中的选中support由同一当前编码器得到；若两侧都反传，其编码代价也必须计入。缓存可用于加速，但陈旧坐标、停止梯度或近似检索都是实现近似，不能悄悄视为精确式子。锁定推理表示可预计算合法TRAIN bank，其构建和存储成本仍存在。`K<d+1`时ridge保证有解，却不增加实际独立支持信息；维数越高不自动更好。

本报告不指定GPU、运行时长或超参数网格，也没有性能计时。`d/K/lambda`应成为后续有限实现定义的一部分，而非看到TEST后不断扩张的逃生通道。当前logical1024、seed2和自然样本语义不因此改变。

### 8.6 一个有限问题，如何接上净收益见证？

唯一必要的机制对照是**相同表示入口、自然监督和最终单方法路径下，局部常数与局部仿射的区别**。常数形式为`sum_i w_i Y_i/(1+lambda)`，argmax等价于同邻域加权winner频率。它与局部仿射共享学习信号，能防止把一般训练差别误认成位置-结果对应关系的作用；不是同时展开K、阶数、损失、编码器和gate的大矩阵。

两者各自产生一个完整策略。直接用第6节的W/G比较它们及事先固定的历史完整参照，并报告固定36与清单外、18题和ALL。不需要先用旧模型找到“正Bayes区域”，也不追加只在答案好的样本启用新头的路由器。如果完整策略已体现可用收益，再用固定位置、交换完整历史结果记录的敏感性检查问“是否真的依赖配对”；敏感性本身不作完整因果认证。

**继续的依据**是局部仿射相对常数有可迁移的动作增益，并且相对完整参照的真实恢复、伤害和成本显示值得保留的整体改进。若收益集中于几题，准确写清局部范围；不能冒称全18普遍改善，也不靠逐题挑模型拼接成绩。

**终止的依据**包括：仿射并不优于常数；只降TRAIN平方误差而无VAL决策增益；收益依赖同base泄漏；支持不足使关键方法不可选；或救回被相反伤害及代价抵消。若只有cost改善而Top1下降，可如实记录成本取舍，但不能作为此次“自然Top1与漏选恢复”主张的成功。若完全消除位置-结果配对仍不失优势，则撤回“局部函数对应关系是收益来源”的机制解释。

这些结果出来后，不自动升级为高阶多项式、更多邻居、GP、另一种表格基础模型或新的gate。必须先有不同的具体信号对象，才能重新开启科学提案；否则本轮到此结束。

## 9. 结果出来后，怎样选择而不是永远继续？

下表是科学决策分支，不是执行授权。“联合收益”指在预先声明的主目标下有真实可迁移改善，且完整报告恢复、伤害、成本与全18取舍；不要求每个单元格都严格改善，也不让事后挑一个指标替换原主张。

| 完整结果模式 | 能支持的解释 | 下一步 | 不允许的续命方式 |
| --- | --- | --- | --- |
| R109/R111都无联合收益，主要交换错误 | 晚期参数共享/非winner族压力不是已获支持的主要修复点 | 结束当前私有化主线；若还要研究，仅保留第8节一个新问题 | 逐层声称“还没彻底隔离”，再追加根/池化/score私有化。 |
| R109有收益，R110没有 | 方法条件计算与复合训练制度存在可用交互 | 保留成功的完整机制包，解释其全18净收益 | 把交互直接归功于某一loss项，立刻拆成大量系数搜索。 |
| R111有收益，而R109/110没有 | 族条件特征、监督分配及乘积决策的联合路径可能有用 | 优先收束该路径；局部函数暂缓，避免两条线同时扩张 | 宣称完全梯度隔离已被证实，或自动推广所有分组。 |
| R109/R111均有收益 | 此计算/条件学习家族包含有价值路径 | 比较统一完整策略和成本后选一条主线解释 | 按题拼接两者最优输出，或把同族结果当独立重复。 |
| 仅TRAIN、族内指标或选择次数增加 | 局部变化没有转成当前所需泛化动作 | 不作为继续同类训练的默认依据 | 用根上限或剩余共享为任何阴性免责。 |
| 局部仿射胜常数，但不胜完整参照 | 对应关系可能有用，但尚不足以改进整体selector | 保留窄机制结论，暂不推进为完整替代方案 | 以oracle选好样本接回旧模型，报告答案感知union。 |
| 局部仿射有完整净收益且配对作用成立 | 支持当前表示与支持范围内的局部函数偏好 | 保留同一单方法路径，准确限定受益任务和幅度 | 把小提升叫突破71%瓶颈，或宣称原始输入已充分。 |
| 局部仿射无增量，或收益依赖泄漏/误选代价 | 此具体邻域函数假说未获支持 | 结束本轮新增机制路线 | 无条件加K、阶数、标签或在线solver。 |

一个容易误解的中间结果是“TRAIN很好、VAL不好”。对第8节，它优先支持局部函数估计/表示过适配，而不是立刻说明需要更多层；对R111，它说明条件拟合不等于迁移。相反，TRAIN也无改进不证明Bayes上限，只使该有限学习路径缺少支持。这两类结果都不要求先完成所有可能诊断才允许结束。

## 10. 停止、延后与授权边界

### 10.1 现在停止优先追加什么？

| 类别 | 本轮处置 | 何种事实才值得重开 |
| --- | --- | --- |
| 泛关系、高阶图、更多专家/递归 | 停止以架构名称作为主要理由的新版本；R107、R63、R105已有近邻 | 有具体部署相关区分对象，而不是只有表达力定理或辅助loss变好。 |
| 密度比、平衡、focal/OVR、SPO+或另一概率链接 | 不列为独立优先方向；R108已完成，R93/110已试纯CE | 有指向特定学习路径的可迁移证据，而非仅证明理想损失一致。 |
| 解冻/更换donor、通用hint/对比预训练 | 延后；保留R94/R101的有限正证据 | 明确新增量如何改变候选相对损失，而非只拟合共同困难度。 |
| 梯度手术、任务顺序、长训、减WD | 不作为第三方向；等待现有共享机制回答 | 优势相关方向确被其它更新抹去等定向证据，而非负余弦、低TRAIN或WD存在。 |
| 普通检索、固定表格ICL升级 | 不重演R47/R97/98/100 | 第8节已给出唯一有限例外；例外失败不能自动扩张整个路线。 |
| 更少零组、均匀选择、局部ReLD专家 | 不作为全18成功目标 | 必须转成自然分布的净正确和成本收益，保留全部合法池。 |

“停止”指科研优先级，不是杀掉健康任务、删掉候选、撤销既有结果或否认所有潜在数学可能。

### 10.2 超出当前授权的思路放在这里，不进入前三

**新独立标签或外部确认集。** 更多独立信息可能帮助薄弱优势区域，独立评估可缓解自适应TEST研发的外推问题；但目前没有定位到应补哪种数据能产生净收益，也没有新标签/划分授权。R58的旧场景修复仍延后，不将旧语义疑问用作所有当前错误的解释。

**付费短程行为观察。** 若未来用户改变静态前向边界，问题应是观察的决策价值，而非熵下降：

\[
\mathrm{VOI}(a\mid s)=\min_m\mathbb E[\ell(m,Y)\mid s]
-\mathbb E_{o_a\mid s}\min_m\mathbb E[\ell(m,Y)\mid s,o_a].
\]

只有预期收益抵得过明确代价才值得观察，最后仍只选一个方法。总时间需包括selector、所有probe、被选方法剩余执行；未被选方法的probe不能抵扣，只有部署状态/轨迹相同的合法续跑才可能抵扣被选前缀。R51已做短行为，R74已做全18自适应前缀但无完整结论；新意不能再称“自适应探测”，而应是计费后的决策价值与停止规则。当前不推荐执行，也不申请在线solver作为第8节失败的自动补救。

## 11. 结论与阅读索引

**这轮真正改变的是：从“还有哪一类网络没试”转向“哪一种尚未利用的对应关系，能以一个完整策略产生正净收益”。** R105-R110要求我们撤回旧建议的一部分，而不是通过改名维持原判断。R109/R111应给当前共享/条件监督问题一个终点；若还要新增研究，只保留由合法TRAIN邻域联合几何决定的局部函数，并承认它尚无直接项目正证据。

这个结论既不是悲观的不可学宣判，也不是新架构的收益承诺。若两个在途机制和一个局部函数假说都没有支持，合理终点是**暂停当前信息制度下没有新对象的重复方法搜索**，而不是宣称证明了71%上限，更不是擅自追加标签与solver。若其中有真实收益，就沿那个具体决策路径继续，而非把小提升泛化为普遍突破。

| 要核对的内容 | 入口 |
| --- | --- |
| 两次旧研究与原定量参照 | [第一轮REPORT](/public/home/shiys/0selection2/docs/research/deep_direction_review_20261023/REPORT.md)、[v2 REPORT](/public/home/shiys/0selection2/docs/research/deep_direction_review_v2_20261023/REPORT.md) |
| 三镜头完整论证与阅读范围 | [01正文](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/01_decision_and_identifiability.md)、[02正文](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/02_representation_and_computation.md)、[03正文](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/03_learning_and_optimization.md)及各自sources.csv |
| 近期完整模型指标 | [R110总报告](/public/home/shiys/0selection2/code/V4/runs/R110_method_conditioned_ce/RESULTS.md)、[A全18结果](/public/home/shiys/0selection2/code/V4/runs/R110_method_conditioned_ce/A_tied_ce/full_model_statistics/RESULTS.md)、[B全18结果](/public/home/shiys/0selection2/code/V4/runs/R110_method_conditioned_ce/B_private_ce/full_model_statistics/RESULTS.md)；原报告保留逐题Top1、meanCost、SBS/Oracle、比值和regret |
| 本轮科学取舍与机器可读方向 | [SYNTHESIS_DIALOGUE](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/SYNTHESIS_DIALOGUE.md)、[research_directions.csv](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/research_directions.csv) |

本轮没有复跑模型、重算预测或独立认证统计显著性；数学例子与推导均标为解释而非实验。三个专题共31条一手论文阅读记录，标题规范化去重30篇，SPO重复；主综合者全文阅读三个专题及来源记录，仅对关键原论文相关段落作定向复核，不将他人的章节阅读声明冒充自己逐页精读全部论文。请求角色为`gpt-6-astra/max`，[agent_trace](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/agent_trace.json)记录请求而非服务端模型身份独立认证。结论保持同族LLM provisional；PDF制作与机械来源检查不改变这一证据等级。
