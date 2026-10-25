# R108 之后：方法专属图计算，而不是再次调整选择频次

研究请求日期：2026-10-24；按最后一轮 Sol 执行主线程补充修订于 2026-10-25。保留原指定文件名。

**结论：保留一个主机制，即池化前的方法条件化节点 refinement，并比较其更新参数共享或按方法私有。原复合目标与纯 CE 是这个机制的两个目标条件，不另算第二个研究方向。** 这比再提出一个缺少独立动机的成本模型更可信。Sol 执行主线程最后明确不必增加不够强的第二条建议，本稿不为凑数增加方案。

范围仅为文献研究、机制判断和失败边界；Sol 负责协议、实现、执行、审查及实测分析。本稿不指定预算、gate、资源安排或启动命令。按 Sol 执行主线程补充，已有 spawn-agent 请求回执记录 `model=gpt-6-astra`、`reasoning_effort=max`。这是指定模型与努力等级的调用请求记录，不是服务端后端身份的独立认证；本稿不另作后端认证，也不声称未使用指定模型。

## 1. 先给需要立即使用的判断

**Sol 执行主线程最后补充的版本值得保留，没有从现有机制材料中发现足以直接否决的概念性问题，但不是已证明能突破平台。** 采用的是“原四层共享编码不变，再增加两层 refinement”，不是最初讨论的“前两层共享、后两层私有”。这里的“不变”是架构保留，不是冻结权重。

| Sol 执行主线程补充的对象 | 本稿判断 |
| --- | --- |
| A：每个合法方法都有自己的节点状态，但两层更新权重在方法间共享 | 已经是方法条件化图计算，不能称为没有方法专属状态的控制。 |
| B：相同分支计算，更新权重按 global method ID 私有 | 新问题是方法间参数共享方式，不是首次加入方法身份、候选状态或候选池化。 |
| R109：上述 A/B 使用原 R77 复合目标 | 研究私有化在原目标条件下的作用，不从历史成绩预判方向。 |
| R110：相同图模型 A/B 使用单一全池 CE | 与 R109 合起来研究表征专门化和目标组合的交互；不是把 R93 的纯 CE 重新命名为新机制。 |

这些名称和目标条件来自 Sol 执行主线程正在形成的工程方案，不归因为人类用户直接提出的架构细节；本稿没有制定或批准执行协议。如果纯 CE 下私有化表现不同，最多先说明其作用依赖目标条件，不能直接锁定 pair、risk、KL 中哪一项有害。若前向次数或概率链接也随之改变，比较的是目标及其训练实现的组合，不能称为只移除了一个损失项。

Sol 最新提供的 R106/R107/R108 完成状态与结果只作为研究上下文：它们尚未显示所需突破。本稿没有重算指标、检查 checkpoint、读取预测数组或核验 GPU 状态，也不把三个版本的结果合并成“所有表征或所有损失方向均已排除”。

## 2. 最小机制：每个方法先形成自己的节点证据，再共同竞争

以下按 Sol 执行主线程补充的实际连接关系作数学说明，不是源码审计或额外接口要求。令 `X` 为原有实例输入，`J(X)` 为冻结的合法候选池；`m` 是候选的真实 global ID，不是当前实例的 winner。

保留 R88 的原始输入与四层共享节点编码，以及 ID+34 和原 joint stream。若使用 R94 前端，也只复用既定输入/融合路径，不把更换 donor 同时包装进这项机制：

\[
H_4=E_{1:4,\theta}(X),\qquad
s_m=E_{\mathrm{ID+34}}(m).
\]

这里 `s_m` 是 ID+34 生成的初始候选 token，不是经过 joint 的候选状态；refinement 发生在 joint 之前。对每个合法候选都构造节点状态，并在本分支全局池化之前执行两次图更新：

\[
U_m^{(0)}=H_4+\mathbf 1 P(s_m)^\top,\qquad
U_m^{(\ell)}=T_{\phi_{\ell,m}}
 \bigl(U_m^{(\ell-1)};B(X),M(X)\bigr),\quad \ell=1,2.
\]

`T` 是完整节点 attention/FFN 更新，含原有类型的残差及归一化；`B(X)`、`M(X)` 是输入可计算的关系 bias 和节点 mask。方法专属的 attention 可以改变哪些节点信息被汇聚，FFN 可以改变汇聚后计算什么，而不是仅给最终标量加偏置。

\[
h_m=P_h\left[\operatorname{mean}_{M} U_m^{(2)};
                    \operatorname{max}_{M} U_m^{(2)}\right],\qquad
\widetilde s_m=s_m+h_m.
\]

全部增强后的候选 token 再进入原 joint stream，随后生成共同 context 和最终分数：

\[
(H_{\rm joint},s^{\rm joint}_{1:K})
 =\mathcal J(H_4,\widetilde s_{1:K},X),\qquad
c=\mathcal C(H_{\rm joint},s^{\rm joint}_{1:K},M(X)),\qquad
u_m=f_{\rm old}([h_m,c,s_m^{\rm joint}]).
\]

这里 `P_h` 和 `f_old` 可保持共享。`h_m` 既直接进入本方法的原 score MLP，也通过 `s_m+h_m` 进入 joint 的 solver/node 更新及共同 context，因而可以间接影响其它方法的分数。保留逐方法 `h_m` 的直接读出，不等于禁止其参与共享交互。**这不是 branch-isolated 架构，不存在由参数私有化自动得到的梯度独立性。**

**A/B 唯一需要在概念上区别的是：** A 的 `phi_(ell,m)=phi_ell`；B 的 `phi_(ell,m)` 按 global method ID 分开，同一方法在它参与的不同问题中继续使用同一份参数。B 的各分支从与 A 相同的 prototype 数值开始是合理的，不要求随机制造初始差异；不同方法的输入状态和监督梯度可以使其随后分化。相同初始函数不等于相同训练动力学，也不保证必然形成有用的分工。

全池 CE 的基本形式为：

\[
p_m(X)=\frac{\exp u_m(X)}{\sum_{j\in J(X)}\exp u_j(X)},\qquad
L_{\rm CE}=-\log p_y(X),\qquad
\widehat y=\arg\max_{m\in J(X)}u_m(X).
\]

它在完整合法池的共同尺度上学习，不是各私有分支单独设置阈值后竞争。**使用 CE 不等于已证明有限模型的概率校准。** 上式描述 utility-CE；如果原复合目标使用 maximin logits 的概率链，不应借这个写法静默更换它。原有 utility 差值的 maximin 可以保持同一 argmax，但其 softmax 概率和梯度未必相同。

### 变化究竟发生在什么梯度上？

对这里的实际连接关系，私有分支参数 `phi_m` 可以经 joint/context 影响全部分数。utility-CE 的链式法则是：

\[
\nabla_{\phi_m}L_{\rm CE}
 =\sum_{j\in J(X)}(p_j-\mathbf 1[y=j])\nabla_{\phi_m}u_j.
\]

即使 `j != m`，`h_m -> enhanced token -> joint/context -> u_j` 路径也允许非零导数。早稿的单项 `(p_m-1[y=m])*grad(u_m)` 只适用于没有这些交叉路径的假想隔离分支，不是本方案的梯度公式。

共享参数时，可将同一套权重在各分支的出现暂记为不同参数槽，再通过链式法则合并：

\[
\nabla_{\phi}L_{\rm CE}\big|_{\rm tied}
 =\sum_{m\in J(X)}
   \left.\nabla_{\phi_m}L_{\rm CE}
   \right|_{\phi_1=\cdots=\phi_K=\phi}.
\]

私有化拆开的是不同分支对同一套图更新权重的参数绑定，不是每个分支接收到的跨分数监督。每个私有参数块本身仍可能累积来自不同分数、实例和问题的相互竞争的梯度；不能声称“不再有跨方法梯度相加”，更不能由此证明当前冲突有害。上述公式仅说明 utility-CE，原复合目标还有其自身的链式梯度。

共有的 `H4`、ID/条件投影、joint stream、readout 仍接收跨方法信号；joint/context 的交互与 CE 的概率归一化同时保留。因而准确称谓是“后段图更新参数专门化，并保留全池联合交互”，不是“方法分支梯度独立”或“消除了多任务干扰”。

已有 CE 在一个漏选 winner 上的 logit 梯度接近 `-1`，不是因为方法很少被选就没有监督。本方向试图改变这些误差信号形成节点特征的路径，不能把零选择反推为梯度消失或梯度饥饿已经查实。

## 3. 历史去重：最接近的是 R63，而不只是 R87

下表只比较已读机制说明，不将历史配置当作当前模型的同数据因果排除。

| 已有机制 | 与本建议的关系及剩余区别 |
| --- | --- |
| R63：每节点×候选 workspace，逐候选池化，原 shared decoder/score head | **实质近邻。** 已有独立候选状态、pooling 和直接选择路径。其 gated-message/GRU 更新 cell 在候选之间共享，A/B 比较初始或演化邻居信息；当前区分后段 attention/FFN 参数是否按方法私有，并保留原 joint stream。因此不能声称首次候选专属图计算。 |
| R87：在原 `[h,c,z_m]` 上增加 shared/private nonlinear utility residual | 私有化发生在共享图表征及池化之后；当前将可训练方法专属运算放在节点关系更新中，作用位置不同。不能据此反推 R87 已证明早期共享有害。 |
| R105：四层 instance encoder 的每个 FFN 都换成四专家混合 | **不是四套完整 encoder。** 专家输出混成一套公共节点流，attention 仍属原共享路径；静态/输入 gate 控制 FFN 混合，未保留每个方法自己的 attention/FFN 参数及后续节点流。当前是原模块化方向的另一种具体共享边界。 |
| R69：按问题 ID 路由的完整 task-private bank | 分离不同问题，不是在同一问题内让各候选方法拥有后段图更新。当前仍在 full18 中跨题复用同一方法参数，不能声称解决了全部问题间竞争。 |
| R92：硬/学习图粗化，另有全18统计特征树分类/回归 | 改变关系组织或整体学习器，不等于方法私有节点变换。当前不新增 cluster graph 或统计特征，也不继承其表达能力保证。 |
| R85：OVR、EQL式平衡、非winner成本 KL、候选子池 | 改监督或比较上下文，没有建立本建议的每方法后段图参数；但说明“让罕见方法得到不同训练信号”本身不是未尝试的发现。 |
| R93：原共享模型上的纯 utility CE | 是重要近邻目标控制，但没有和这里的方法私有图计算交叉。R110 的研究意义在交互，不在宣布纯 CE 是新目标。 |
| R94/R106：预训练节点融合或原生编码器适配 | 改已有数值表示的来源/适配，不按候选方法分开后段更新；不将 donor 更强等同于选择证据更强。 |
| R107/R108：显式相连路径、条件密度比 | 分别改变关系内容和监督参数化。当前不再加入短路径特征，也不再通过先验加残差重写冠军概率。 |

关键本地来源：[R63机制说明](/public/home/shiys/0selection2/code/V4/R63_README.md)、[R87研究边界](/public/home/shiys/0selection2/code/V4/runs/R87_private_method_residual/GOAL.md)、[R105模型说明](/public/home/shiys/0selection2/code/V4/runs/R105_conditional_representation_experts/MODEL_PREP.md)、[既有表示专题](/public/home/shiys/0selection2/docs/research/deep_direction_review_v2_20261023/REPRESENTATION_AND_INPUTS.md)、[R80-R102机制索引](/public/home/shiys/0selection2/docs/research/experiment_mechanism_index_R80_R102.md)。

## 4. 为什么覆盖 full18，以及哪里可能失败

所有合法方法都从同一输入生成自己的 refinement 分支，不按真实 winner 路由、不只激活零选方法、不设 ReLD 二选一触发器。方法参数在其合法的全部问题上接受 winner 与 nonwinner 样本的学习信号，因此主要方法之间的错误也属于这一机制的作用范围。它针对完整池的实例区分，不以选择直方图更均匀为目标。

沿用每题 TRAIN 81,000、validation 4,500、TEST 4,500 及既有输入；八个 B 问题的删除规则、单一 `reld_mt` 和 native winner 口径不变。没有新数据、求解器重跑、路线标签、当前 query 成本输入或 TEST 调参。已有 TEST 参与过研发，不重新称为盲测。

| 失败边界 | 为什么重要 |
| --- | --- |
| 原四层输出已丢失决定性信息 | 后段私有网络不能从相同可见状态恢复不存在的信息。保留输入关系 bias 也不自动打破既有对称性；已有 ATSP 反例不能靠“换一套方法权重”就宣布解决。 |
| 共享实际有益，私有参数缺少支持 | 每个方法需要学习新的图变换；稀疏优势区域可能只被记忆，或失去跨方法共享统计效率。一个方法有大量 winner 不证明其优势可以泛化。 |
| 私有网络仍只学到条件常数 | 不同权重、不同激活或非零选择次数不等于有用的实例差异。一个更复杂的类别偏置也是可能结局。 |
| 干扰仍在共享前段、joint/context 或跨问题更新 | 私有分支仍有跨分数梯度路径，后段分开不能排除这些瓶颈。纯 CE 也不自动让剩余共享部分形成缺失特征。 |
| 更多参数解释了差异 | B 的有效自由度增加；不能仅凭私有化获益断言原因必是梯度冲突缓解。A 与原基线的差异还包含新增深度和候选计算。 |
| Top1 与真实成本并不一致 | 全池 CE 学冠军概率，不保证少数更昂贵的误选不会增加。减少零选择或提高训练拟合不等于改善选择风险。 |

两层 refinement 对全部合法候选执行，节点状态存储通常随 `K*N*d` 增长；稠密 attention 计算还含候选倍数的 `N^2` 项，实际显存取决于实现。这里没有免费计算、等参数或等 FLOP 承诺。低秩方法适配可以属于同一共享机制家族，但不是全私有 B 的同义词，本稿不再增加该变体。

这项研究不会增加输入包含的 Bayes 信息。它改变有限网络的参数共享、可达表示和统计效率，是否更容易形成有用优势证据仍未知。既不能由当前平台推出不可学，也不能由“权重按方法私有”推出必然可学。

## 5. 为什么本轮不把密集成本监督列为第二条推荐

**成本目标仍有研究价值，但“用所有 TRAIN 成本而不是只用 winner”不是新的项目机制。** 初步想法必须在以下近邻面前收缩，不能只换 loss 名称。

| 已有工作 | 对新建议的约束 |
| --- | --- |
| R39 | 已有独立分类/性能头、CE 加 centered-cost MSE，且正式决策直接取性能头 argmin。因此“让成本头直接参与选择、避免被分类头绕过”本身也已经试过。 |
| R55 | 以计费边差重建为必经瓶颈，距离加权后直接决定旧 ReLD pair；不是完整池的无路线风险学习，但已试过监督中介直接决定输出。不能重复提路线差瓶颈，也不能用局部失败排除 full18。 |
| R71 | 已建模联合成本差分布，以混合密度 NLL 学习，再积分全池冠军概率。它不是均值风险回归，但足以否定“以前没有利用联合成本信息”。 |
| R85C | 已用非winner相对成本 profile 的条件 KL，并保留原复合目标。不是只监督冠军一个标签；成本转 soft target 再换温度不是本轮下一机制。 |
| R92D / R98 | 已涉及 relative-gap 回归及每方法期望相对 gap 的上下文学习；更换回归器不能自动构成新的决策理论。 |

来源：[R39机制与既有分析](/public/home/shiys/0selection2/code/V4/runs/R39_performance_model/RESULTS.md)、[R55机制](/public/home/shiys/0selection2/code/V4/runs/R55_route_cost_difference/RESULTS.md)、[R71机制](/public/home/shiys/0selection2/code/V4/R71_README.md)、[R85文献与目标定义](/public/home/shiys/0selection2/docs/research/r85_zero_selection_literature.md)、[既有损失专题](/public/home/shiys/0selection2/docs/research/deep_direction_review_v2_20261023/LOSS_AND_OPTIMIZATION.md)。这里只采用机制描述，不对这些报告作结果认证。

一个容易误判为新方法的改写是“所有 pair 的成本差 MSE”。令 `e_m = predicted_r_m - target_r_m`，在同一合法池内有精确恒等式：

\[
\sum_{m<n}(e_m-e_n)^2
 =K\sum_m(e_m-\bar e)^2.
\]

所以等权完整 pair 差回归只是 centered MSE 的缩放，并未凭 pair 数量增加独立监督。改为相对 Oracle 成本改变的是风险尺度；它不自动消除弱方法的大误差占据拟合能力，也不能保证前两名的次序变准。

另一个必须保留的区别是：冠军概率最大、条件期望成本最低、条件期望相对 regret 最低，通常不是同一决策。成本回归可能合理改善一种风险，却降低 native Top1；不能用“更接近 Bayes”承诺两者同时增长。TRAIN 成本只作为监督，推理仍不能读取 query 的真实成本或 Oracle。

决策导向 surrogate，例如 SPO+，确实不等于普通成本 MSE，也不要求重新运行原 routing solver；在有限方法集合上其决策运算可退化为候选比较。但“能写出不同公式”尚不足以将其提升为当前第二优先。其理论一致性有分布前提，不是任意有限神经网络、近并列标签或混合目标下的保证。当前没有足够具体的项目桥梁，说明它比已试风险/比较监督更能形成缺失的实例证据。[Elmachtoub 与 Grigas，作者原文](https://arxiv.org/html/1710.08005v5)

**因此，本轮采用 Sol 执行主线程补充的同一表征机制下复合目标/纯 CE 的比较，不再附加新的成本头、密度模型、配额或损失组合。** 这是优先级判断，不是宣布成本学习已经穷尽。

## 6. 一手文献支持到哪里为止

| 一手来源与本次访问范围 | 可以借鉴 | 不能外推 |
| --- | --- | --- |
| Liu、Johns、Davison，CVPR 2019，[End-To-End Multi-Task Learning With Attention](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.html)。本次核实官方论文页与摘要，PDF 请求被拒绝；不声称重新读完正文。 | 共享主网络与任务专属特征级 attention 可以共存；专门化可以发生在中间特征，而非只在最终输出头。 | 本项目的方法是互斥决策候选，不是论文中的多个独立任务标签。该论文不是“每候选私有 Transformer 可提高 solver Top1”的直接证据，也未证明本方案的两层位置最优。 |
| Elmachtoub、Grigas，[Smart Predict, then Optimize，作者 arXiv v5](https://arxiv.org/html/1710.08005v5)，[正式发表入口](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3922)。本次读到作者正文的引言、预测-决策定义及其对一致性条件的说明。 | 区分预测误差与决策损失；使用成本信息时必须说清最终优化的决策对象。 | 连续、关于均值对称等条件不能默认为当前条件成本分布成立；更不能把成本风险一致性当成冠军准确率提升定理。 |

已有 [条件模块化研究](/public/home/shiys/0selection2/docs/research/post_r104_conditional_modularity.md) 已讨论 MMoE、共享边界与专家塌缩；本稿将当前方案定位为该方向的具体推进，不声称新发明了条件模块。也参考 [第一轮综合](/public/home/shiys/0selection2/docs/research/deep_direction_review_20261023/ASTRA_SYNTHESIS.md) 和 [第二轮报告](/public/home/shiys/0selection2/docs/research/deep_direction_review_v2_20261023/REPORT.md) 的证据边界，未把其旧运行状态当作当前状态。

后续补充检索遇到搜索服务 TLS/502、连接失败及原站访问失败，未把未取得正文的论文列为已全文核查，也未补写未经查证的最新文献结果。这是一份有边界的机制判断，不是穷尽式文献综述。

**最终优先判断：推进“完整合法池上的方法专属图更新”这一条概念线，准确区分状态专属、参数专属和目标条件。它有实质区别，也有很近的历史先例；现在有理由研究它，没有理由提前宣称它解决了零选择或约 71% 平台。**
