# 镜头三：训练目标、优化动力学、数据利用与可迁移判别

日期：2026-10-25。性质：论文与科学思路专题，不是代码审查、结果认证或实验启动方案。

本会话记录的模型为 `gpt-6-astra`、`reasoning_effort=max`，会话ID为 `01a17167-d7ee-7ed1-afc6-85b6ba176b47`，对应 [agent_trace.json](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/agent_trace.json) 的第三镜头；这是会话配置证据，不是对服务端实际推理实现的独立认证。遵守当前 goal paused 状态，仅阅读历史材料、论文与相关机制实现，未运行新实验、重算模型预测、改标签或划分。本文与 `03_sources.csv` 是本镜头仅有的写入。

## 1. 先回答六个问题

1. **自然分布 Top1 与少选方法的真实恢复不必冲突，但“让每个方法都被选中”与 Top1 最优可能冲突。** 要找的是输入可识别的优势区域，不是让选择频率追随 winner 频率。方法在总体上有很多 winner，不等于它在当前可用特征的某个条件区域拥有最高后验。
2. **proper loss 的理论正确性既不能证明现有 CE 已学好，也不能证明再换一个一致损失就能突破。** Fisher consistency 是理想条件风险结论；有限假设类中的最优 CE 与最优 Top1 可以不同。另一方面，R91、R93、R108、R110 已大幅削弱“只要去掉复合损失/概率链接就能解决”的解释。
3. **R40 的过拟合与当前低 TRAIN 拟合不是同一个现象。** R40 证明旧场景能沿某条轨迹记忆更多训练样本而损害泛化；R49 的小样本 memorization probe 只证明局部可记忆。它们都不证明当前 81k/题的正确判别已经可迁移，也不证明当前 71% 是 Bayes 上限。
4. **“共享干扰”“特征饥饿”“读出偏置”“训练时间不足”必须拆开。** 有非零梯度不等于优势特征在生长；负梯度余弦不等于有害干扰；TRAIN 与 VAL 接近不等于收敛；参数私有化也不等于信息、梯度和优化器状态全部隔离。
5. **同信息上的损失和优化器替换已不是高优先级新方向。** PCGrad、ScheduleFree、SupCon 两阶段、SAM、mixup、教师伪标签、匹配数据规模、续训/校准、CE/batch/LR、AUC、条件排序、focal、OVR/EQL/cost profile/subset、Muon、纯 CE/dropout0、重要性采样、hint 预训练、条件密度比和节点私有纯 CE 都有实质历史。部分有小幅匹配收益，不宜统称“全部无效”；但也不能再称尚未探索。
6. **只保留两个有边界的科学方向，不制造第三个新名字。** 第一是等待已在运行的 R109/R111 对“部分私有节点计算与条件监督”的完整回答；第二是有条件地保留自然风险下学习时间/顺序与可迁移优势形成的研究问题。后者不是追加训练建议，缺少定向证据时应降级。若 R109/R111 仍只是交换错误，就停止把同类共享干扰解释作为默认续训理由。

## 2. 证据口径与历史覆盖

### 2.1 本文如何使用已有材料

优先采用同目录的 [artifact_index.csv](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/artifact_index.csv)、[reported_all_metrics.csv](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/reported_all_metrics.csv) 和 [running_snapshot.csv](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/running_snapshot.csv) 作为机械入口，再阅读相关原报告与目标实现理解机制。它们是现存文件的汇总，不自动构成新的统计验证。

对照两轮旧深调研：[第一轮总报告](/public/home/shiys/0selection2/docs/research/deep_direction_review_20261023/REPORT.md)、[R35-R79 历史](/public/home/shiys/0selection2/docs/research/deep_direction_review_20261023/ASTRA_HISTORY_R35_R79.md)、[R80-R104 历史](/public/home/shiys/0selection2/docs/research/deep_direction_review_20261023/ASTRA_HISTORY_R80_R104.md)、[第二轮总报告](/public/home/shiys/0selection2/docs/research/deep_direction_review_v2_20261023/REPORT.md) 和 [第二轮损失与优化专题](/public/home/shiys/0selection2/docs/research/deep_direction_review_v2_20261023/LOSS_AND_OPTIMIZATION.md)。本镜头补的是机制辨析和理论边界，不重复另外两个镜头的表征架构目录。

LEGACY18、NEW70114、NEW90k 分开解释。当前 NEW90k 每题 TRAIN/VAL/TEST 为 81,000/4,500/4,500，18 题、seed2、逻辑 batch1024；保留冻结候选池、B 问题删除规则、单一原 MOEL 的 `reld_mt` 名称及 native/FP64 `ind` winner。ALL 为逐题等权，不将跨题混合 raw cost 当成有共同物理尺度的总风险。表中 Top1 和 actual regret 数值单位均为 `%`，变化用百分点表示。

**状态覆盖旧叙述：R110 已完成；R109/R111 在本次 snapshot 中仍为 training。** 不采纳 [post_r110 旧专题](/public/home/shiys/0selection2/docs/research/post_r110_broad_mechanism_20261025.md) 中已经过时的 running 判断，也不把未完成 R109/R111 写成阴性。旧 TEST/LIB 已参与研发，本报告不是盲测结论；不据此替当前版本重选 checkpoint。

### 2.2 不是一串“失败”：最相关的历史对照

以下为报告中的观察，不是本轮独立复算；`A/B` 只在各行注明的场景内比较。

| 历史与场景 | 实际改变及主要观察 | 对今天的约束 |
| --- | --- | --- |
| R35-R40，LEGACY18 | 单题续训、小样本记忆、几何残差和高更新密度均已有历史。R40 A/B 终点 TRAIN=74.122/70.647，VAL=45.011/45.883；A 在第56轮停止，B完成60轮。 | 证明可以出现严重拟合/泛化分离；不是纯 batch 因果对照，也不是当前数据的训练上限。 |
| R47、R49、R53，LEGACY18 | R47 是 FULL18 学习式 TRAIN 行为记忆。R49 的 OVRPTW 二方法 2k/8k 子集在相同4k更新后 TRAIN=100/83.025，内部留出约52.1/50.5。R53 独立 pair expert TRAIN/VAL 约57/56；接回全系统 TEST=47.7611/47.7111，未过 R45A=47.9111。 | 记忆路线、二方法专家均已尝试；R49 不是81k全任务可拟合的证据，R53 也不能证明标签不可预测。 |
| R60-R63、R71，LEGACY18 | 扩充成本、路线/性能预训练、候选专属节点状态、联合成本分布已有记录；部分版本没有完整 TEST。 | “先预测性能”“加辅助监督”“候选特征独立”不是从未提出；不把未完成版本合并为完整阴性证据。 |
| R64 PCGrad，LEGACY18 | 同一参数点取3题各256例，A平均梯度，B对共享部分投影；TEST=47.3778/47.4111，只多6例正确。负梯度对约43.7%，投影相对变化约0.1126。 | 有梯度冲突且投影确实改变更新，但不足以解释主要性能缺口；不是“没真正做过多任务梯度处理”。 |
| R65 ScheduleFree，LEGACY18 | 匹配新训 A AdamW/B ScheduleFree，TEST=46.4722/47.6389；B 用平均参数推理。 | 有1.1667个百分点的匹配收益，不能说优化器无效；仍未过当时 R60B=48.2444。 |
| R67 SupCon，LEGACY18 | 第一阶段同 winner、同题的对比学习；第二阶段丢弃投影器及整个评分 MLP，换新读出并全量微调。TEST A/B=47.4889/48.1000。 | 表征/读出分阶段并非未试；同 winner 拉近不保证保留区分不同优势子区域的信息。 |
| R68 SAM、R69 task-private、R70 latent mixup，LEGACY18 | R68 从同一预训练 trunk 起步，SAM B=46.6444，AdamW A=46.8611。R69 完整18题私有 bank B=46.3222，终点 TRAIN/VAL=59.9856/46.4944。R70 mixup=46.7167。 | 已有完整任务私有化，不应重命名再提。R69 各支路活跃更新少于共享模型，参数量不同；不能当完美因果隔离。 |
| R73 教师，LEGACY18 | 在162万扩展输入上使用教师共识硬伪标签；学生 TEST=48.0056，对照共识=48.2222。 | 多输入不等于多独立真实 winner 信息；不是仅改变训练样本数的实验。 |
| R77，NEW70114 | 同 batch1024、同49,680总更新，10k/70,114每题 TRAIN：TEST=70.5691/71.2252，regret=0.412741/0.394014。 | 更多独立样本有实际收益；两边重复曝光次数不同、TRAIN归一化各自拟合、选中更新不同。 |
| R78，NEW70114 | 截距校准前两项有不收敛/不确定部分；后续续训、冻结概率成本头、固定集成、成本邻域目标未满足原标准。成本头降低零选择数却使 VAL Top1约70.894降到67.948。 | 校准、续训及固定读出成本学习不是空白；不能把整个 R78 写成“均已收敛的完整模型阴性”。 |
| R80，NEW70114 | CE-only、batch512、LR3e-4 三支 TEST约71.2266/71.3151/71.1753。batch512 支将更新数加倍至99,360，按曝光量匹配进度。 | 单纯更大 LR、移除辅助损失、固定样本曝光下增加更新数都已试；不能以“epochs不够”绕过历史。 |
| R81 AUC、R82 CMR，NEW70114 | 同方法 winner-vs-nonwinner AUC；同题/规模/分布/容量上下文内按相对 regret 排序；TEST约71.1510/71.2780。 | 已做跨实例、条件化监督；两者不等同自然分布 calibrated argmax，也不是全无条件 hard negative。 |
| R84 focal，NEW90k | A baseline=71.2358；C 是在原目标上追加 gamma2 focal，非替换CE，TEST=71.2716，regret反升。R90 的 A 选中 TRAIN/VAL=71.2439/71.0852。 | 小拟合间隙是真的；focal 不构成突破，但不能说已单独验证所有 focal-only 形式。后者也不足以成为新大方向。 |
| R85 OVR/EQL/cost profile/subset，NEW90k | 四支 TEST=71.2321/71.2111/71.3111/71.4370；后两支 regret约0.389713/0.390487。原 pair/risk/KL 仍在。 | 覆盖输出竞争、梯度平衡、非winner成本与候选子集，不能压缩成“只调了权重”；部分成本收益不等于充分恢复少选方法。 |
| R89 hybrid Muon，NEW90k | A AdamW/B 混合 Muon TEST=71.2321/71.4198，regret=0.394236/0.389646；Muon 只作用于指定隐藏矩阵。 | 有小幅匹配收益，不是全模型 Muon；不足以独立证明平台来自优化器几何。 |
| R91 link、R93 plain，NEW90k | R91 A只改CE链接，B改全部概率链接，但保留pair。R93 真正单次前向 raw-utility CE，dropout A=.1/B=0，保留WD。 | “CE没有真正试干净”不成立；R93 的保存评估CE仍是maximin诊断，不能与其优化的raw CE直接混读。 |
| R95 importance，NEW90k | 自适应抽样且对整项损失乘逆概率，TEST A/B=71.2494/71.2543，regret B更高。 | 保持经验目标的采样未有联合收益；不等于所有改变风险的难例重加权已试，更不意味着后者自然合理。 |
| R101 hint，NEW90k | 12,800 hint 预训练后再57,600 selector更新；仅迁移 encoder。TEST A/B=71.3222/71.4346，TRAIN=71.3688/71.4851。 | 目标特征预热及阶段切换已有历史；有小收益，不是 inference 时获得求解器答案。 |
| R108 density ratio，NEW90k | TRAIN A/B=71.6171/71.4957，TEST=71.3383/71.2173；原特征、TRAIN条件先验加残差，对照CE与密度比目标。 | 第二轮曾保留的密度比路线已经完成，不再列为未试新方向。 |
| R110 node-private + CE，NEW90k | A tied/B private TRAIN=71.6169/71.2645，VAL=71.1037/70.8963，TEST=71.3074/71.2333；regret=0.406464/0.399449。 | 私有节点加纯CE没有解决低拟合；B成本略好而Top1/目标恢复更差，不能说B所有指标都被支配。 |

关键原文入口：[R40](/public/home/shiys/0selection2/code/V4/runs/R40_discriminative_training/RESULTS.md)、[R49](/public/home/shiys/0selection2/code/V4/runs/R49_existing_data_diagnosis/RESULTS.md)、[R53](/public/home/shiys/0selection2/code/V4/runs/R53_pair_specialist/RESULTS.md)、[R64](/public/home/shiys/0selection2/code/V4/runs/R64_pcgrad_full18/RESULTS.md)、[R65](/public/home/shiys/0selection2/code/V4/runs/R65_schedule_free_full18/RESULTS.md)、[R67机制](/public/home/shiys/0selection2/code/V4/R67_README.md)、[R69](/public/home/shiys/0selection2/code/V4/runs/R69_task_specific_finetuning/RESULTS.md)、[R77](/public/home/shiys/0selection2/code/V4/runs/R77_sample_size_batch1024/RESULTS.md)、[R78回顾](/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/final_delivery/zero_selection_retrospective/RETROSPECTIVE.md)。新数据各行的完整文件映射见前述 artifact index；R81 的机制入口是 [method-conditioned AUC README](/public/home/shiys/0selection2/code/V4/R81_README.md)，不要仅凭同编号不同目录名称混淆。

### 2.3 选型目标不是一个可以无限复用的解释

不少版本实际按完整 VAL `macro_vs_sbs_pct` 最低选中，而非 Top1 或 actual regret。三者不等价，因此应准确描述模型的选择压力。但第二轮已比较保存的 VAL 点：R84A、R89B、R91B 的成本选中点同时也是最高 Top1 点；R93A/B 改看最高 Top1 只高 **0.041975/0.030864 个百分点**。这只覆盖保存的点，不是全连续轨迹证明，却足以显著削弱“平台主要是选错 checkpoint”的解释。R64 等旧版本另有 Top1 选型，R53 用过 actual regret，也不能把所有历史都归为一种规则。

本文不据此重选、拼接或重测任何模型。结论是：**要承认目标不完全一致，但不要把已观察到的低 TRAIN 拟合和方法漏选一笔归咎于 checkpoint 指标。**

## 3. 自然准确率、真实恢复与成本到底是什么关系？

### 3.1 少选不等于统计意义上的稀有，也不等于必然误判

固定任务与合法候选集，令 `X` 为推理可得输入，`Y` 为固定口径 winner，

\[
\eta_m(x)=P(Y=m\mid X=x),\qquad
\delta_{01}^{*}(x)=\arg\max_m\eta_m(x).
\]

这里的概率表达输入信息与数据生成的不确定性，不宣称实际标签是随机噪声。即使 winner 来自确定成本表，有限表示或样本也可能不足以恢复对应函数。

**本文推导的反例，不是项目事实：**若所有可分辨区域均有后验 `(0.45,0.35,0.20)`，第二方法拥有35%的真实 winner，却永远不是自然0-1风险最优动作。让它被选中必然降低该例的最优准确率。因此“winner支持充分”排除了极端样本稀缺，却没有证明存在可恢复决策区域。

反过来，若模型本来没学会某区域 `S`，将旧动作 `a(x)` 改为方法 `m` 的准确率变化为

\[
\Delta\mathrm{Acc}
=\mathbb E[\mathbf1\{X\in S\}(\eta_m(X)-\eta_{a(X)}(X))].
\]

若 `S` 中真实优势可识别，少选方法恢复与自然准确率完全可以同向；不存在普遍的不可能定理。实际成对预测上，净正确数就是“旧错新对”减“旧对新错”。**新误选数不等于新损害数**：旧错新错也算新方法误选，却不额外降低Top1；它仍可能显著改变成本。

项目已提供两种必要提醒。R84A 的 TSP TRAIN 中 LEHD 有14,279个 winner 却只有7次选择，DIFUSCO有6,806个 winner而0选择，不能统称罕见类别。第二轮的同题、同规模置换使 R94B VAL Top1 从71.4284降到67.7134，说明实例信号已经被利用，不是纯先验分类器；这仍不保证模型利用到了这些少选方法的优势。metadata cell 内没有成为多数类，也不等于更细的几何条件内不存在优势区域。

### 3.2 三种最优动作不相同

令 `C_m` 为合法候选成本，`C_*` 为实例 oracle 成本，假定使用当前正成本的相对 regret 定义。则

\[
\begin{aligned}
\delta_{01}^{*}(x)&=\arg\max_mP(Y=m\mid x),\\
\delta_{C}^{*}(x)&=\arg\min_m\mathbb E[C_m\mid x],\\
\delta_{r}^{*}(x)&=\arg\min_m\mathbb E[C_m/C_*\mid x].
\end{aligned}
\]

第三行不是 `E[C_m|x]/E[C_*|x]`。知道谁更常赢不等于知道输的时候代价多大；甚至精确的 winner 后验也不足以恢复后两种动作。宏平均 `vs_SBS` 则对每题的期望成本除以该题固定 TRAIN-SBS 成本，在无限容量下不改题内最优成本动作，但在共享有限容量下改变题间拟合资源的分配。ALL 不应靠混合不同题 raw cost 抹平这一点。

因此有两种诚实结果：一是未有联合改善，只在目标之间作交换；二是修复此前未学到的共同判别，Top1、真实恢复和成本共同改善。不能把前者包装成后者。R110 的 B 相对 A 就是具体例子：TEST Top1下降约0.0741个百分点，actual regret改善约0.007015个百分点；固定36目标组中4598个真实 winner 的召回从1.5224%降到0.9569%，成对救回21、失去47，另有41个新误选。它否定的是联合突破，不否认成本的小改善。[R110完整报告](/public/home/shiys/0selection2/code/V4/runs/R110_method_conditioned_ce/RESULTS.md)

### 3.3 为什么 class balancing 不能自动解决这个问题

对 winner 加权 CE 的逐点最优分布为

\[
\min_q-\sum_mw_m\eta_m\log q_m
\quad\Longrightarrow\quad
q_m^*=\frac{w_m\eta_m}{\sum_jw_j\eta_j}.
\]

不做相应反校正时，其 argmax 已不再是自然 microaccuracy 的 Bayes 动作。Menon 等的 logit adjustment 明确针对 **balanced error**，核心决策与 `eta_m/pi_m` 有关，不能将其 Fisher consistency 直接移植成自然 Top1 保证。自然任务的逐题等权也不等于每个方法的等权。[P02，§3、§5.1-5.2](https://arxiv.org/html/2007.07314)

这不意味着 class-balanced 训练在有限网络里永远无益；它可能改善表示后再恢复自然先验。但那是有限学习路径假设，不是“按理论应当提高当前指标”。R78 校准、R85 OVR/EQL、R108 条件先验残差已有近邻，继续同类改写需要超出“少数类会被压制”的新证据。

## 4. Proper loss 正确，为什么有限学习仍可能错？

### 4.1 一致性的承诺范围

标准 CE 在给定输入处满足

\[
\mathcal L_{\rm CE}(q\mid x)
=H(\eta(x))+\mathrm{KL}(\eta(x)\Vert q(x)).
\]

无限制地最小化条件风险可恢复后验；分类校准讨论 surrogate 风险趋近理想最优时，分类风险是否也趋近最优。Tewari-Bartlett 的 §2 定义与 Theorem 2 处理的是这一层，不承诺一个固定 Transformer、有限样本、AdamW有限步一定达到最优。[P01](https://www.jmlr.org/papers/volume8/tewari07a/tewari07a.pdf)

**本文的有限假设类反例：**两个等概率输入的二分类真实后验为 `(0.9,0.6)`，假设类只允许两种正类概率预测 `A=(0.9,0.4)`、`B=(0.55,0.55)`。

| 候选函数 | population CE | 自然准确率 |
| --- | ---: | ---: |
| A | 0.539594 | 65% |
| B | 0.648005 | 75% |

CE 在这个受限类中更喜欢 A，尽管 B 的 Top1 更高。此例不证明项目应换 loss，只说明“CE proper，所以低Top1必定是不可预测”的推理错误。

H-consistency 文献进一步显式保留假设类与 minimizability gap：

\[
M_{\ell,\mathcal H}
=\inf_{h\in\mathcal H}R_\ell(h)
-\mathbb E_X\inf_{h\in\mathcal H}R_\ell(h\mid X)\ge0.
\]

逐个输入可以理想选择的函数，不一定能由同一个共享参数函数同时实现。Awasthi 等 §2-3 给出相应条件风险到总体风险的界；其具体结果依赖损失族、对称性/完备性等条件及特定假设类，不能把论文里的线性或单隐层结论当成本项目 softmax Transformer 的证书。[P03](https://proceedings.neurips.cc/paper_files/paper/2022/file/051f3997af1dd65da8e14397b6a72f8e-Paper-Conference.pdf)

### 4.2 当前复合目标确有不同压力，但这已被多条实验触及

历史 baseline 以 `z_m=u_m-max_{j!=m}u_j` 做 maximin 分数，argmax 与原 utility 相同，却不是对所有候选减同一个常数，故 CE 概率和梯度不同。二候选时相当于放大 logit 差；多候选不能简单称统一温度。链接本身不禁止某方法成为最大值。

原每个 dropout view 的主要项为 `.35 CE + .35 comparison + .02 risk`，两view平均，再加 ramp 到 `.35` 的对称 KL。risk 的实现尺度使其相当于 `2 min(E_p r,1)`；不是“0.02很小可忽略”。pair 采用严格FP64成本比较、排除ties，组均值归一化；不能说它天然按候选对数平方膨胀。以上是旧专题和实现的机制事实，不是新缺陷判断。

这些目标当然不共同保证拟合自然 winner 后验：pair 想恢复排序，risk 想降低期望相对损失，KL要求两次扰动一致。但 R91 链接控制、R93 真纯CE及dropout0、R108 单次密度比、R110 节点私有纯CE都未形成联合突破。因此证据支持“目标不等价”，**不支持“复合损失是足以解释整个平台的唯一原因”**。

R93 的锁定 TRAIN Top1 A/B=71.459945/71.434088，VAL=71.125926/71.109877；纯 CE 后仍是低拟合平台，而非 TRAIN 突然接近100、只剩泛化失败。[TRAIN A](/public/home/shiys/0selection2/code/V4/runs/R93_plain_classification/A/full_model_statistics/train_metrics_by_problem.csv)、[TRAIN B](/public/home/shiys/0selection2/code/V4/runs/R93_plain_classification/B/full_model_statistics/train_metrics_by_problem.csv)

### 4.3 Decision-focused learning 是否是新出口？

SPO+ 的学习信号是“预测成本诱导的动作是否造成真实决策损失”，而不是普通成本拟合。对有限候选选择，将可行域取为候选 one-hot 的凸包，令 `y=argmin c_m`，其形式化简为

\[
\ell_{\rm SPO+}(\hat c,c)
=\max_j(c_j-2\hat c_j)+2\hat c_y-c_y.
\]

这是从原论文 §3.2 定义代入简单形得到的本文推导；相应信号只涉及真实winner与 cost-augmented 最强竞争者，不需要在这里另跑组合优化求解器。它可能改变有限模型的误差分配，却没有创造新的 TRAIN 成本信息。[P13，Definition 3.5、Proposition 3.6](https://arxiv.org/html/1710.08005v5)

同文 §4 Theorem 4.2 的 Fisher consistency 需要条件成本分布中心对称、连续性、唯一条件均值最优动作及非空内部等假设。有限候选简单形只有相对内部；即使用降维处理这一点，当前条件成本分布也没有满足其余假设的证据，不能照搬定理。即使满足，目标也是成本决策，不是少选方法召回或自然Top1。R71 联合成本、R78 成本头、R82 条件regret排序、R85 cost profile 已覆盖大量同源信号，**不把再换成 SPO+ 当成第三条高优先级新路线**。

## 5. 低 TRAIN 拟合：信息、记忆、表示还是读出？

### 5.1 为什么 R40 和 R49 不能代替当前问题的答案

R40 A/B 分别有151,200/162,000总更新，约8,400/9,000次每题更新，使用旧数据与不同训练配置。终点高 TRAIN、低 VAL 是过拟合证据，且选中点早于终点。不能把其终点 TRAIN 与当前按成本选出的 checkpoint 相减，声称增加同样步数一定有同样效果。

R49 中2k样本在约1,800更新达到接近99%训练正确，内部留出仍约50%。同样的512k样本曝光，在8k样本上终点只能拟合到83.025%。这同时说明重复次数、独立样本数和可记忆难度都重要；“网络能记住2k”不是“81k/题的可迁移函数已经藏在最后一层”的证据。

Zhang 等的随机标签/随机像素拟合实验也只说明某些网络可插值这些有限样本，不能从可记忆推出泛化，亦不能从本项目低拟合倒推标签随机。[P12，§2.1、§3](https://arxiv.org/pdf/1611.03530v2)

反过来，当前 TRAIN≈VAL≈71% 也至少允许三种解释：能学到的判别不足、现有可学判别没有被优化出来、按固定推理表示看条件不确定性较高。现有观察不足以把三者分开，更不能给“Bayes上限71%”赋值。全18平均还可能遮蔽题间完全不同的难度结构。

### 5.2 四种常被混称的“梯度饥饿”

| 层次 | 确切问题 | 当前证据能否确认 |
| --- | --- | --- |
| 输出层竞争 | 正确候选是否因softmax而没有梯度？ | 标准CE的winner导数是 `p_y-1`；被强烈漏选时接近-1，不是0。R85 OVR/EQL也已改变此层压力。 |
| 参数层竞争 | 不同实例/任务的参数梯度是否抵消，或互相破坏优势？ | 有可能；R64测到冲突，但减冲突只带来极小Top1收益。当前模型没有因此被因果确诊。 |
| 特征形成不足 | 对优势相关方向的有效表示是否根本没有形成？ | 未确证。整体梯度范数或参数变化不能回答“哪类可迁移特征在形成”。 |
| 已有特征未被读出 | 优势信号已在隐藏状态，但原评分函数没有采用？ | 未充分排除。截距校准、固定概率成本头失败不等于任意合法隐藏表示读出都失败。 |

在普通梯度流近似下，令 `u(theta)` 为堆叠logits，`J` 为其参数Jacobian，

\[
\nabla_\theta L=J^\top(p-e_y),\qquad
\dot u=-JJ^\top(p-e_y).
\]

错误样本输出梯度很大，但如果其需要的区分方向在 `J` 中很弱，或与其它样本强耦合，依然可能学得慢。这是数学上的可能机制；实际 AdamW、裁剪、移动表示并不等于上述固定核模型，本文不把它当实测结论。

Pezeshki 等的 Gradient Starvation 在 §3.3 Theorem 2 证明了线性化网络、二分类、特定两特征耦合条件下，一个强特征的增强会降低另一特征的稳态响应。它支持“强梯度不等于所有判别方向都被充分学到”，并不直接证明当前节点几何特征被抹掉。其 §3.4 的 spectral decoupling 改变logit正则的动力学，也不是所有架构的通用修复。[P04](https://proceedings.neurips.cc/paper_files/paper/2021/file/0987b8b338d6c90bbedd8631bc499221-Paper.pdf)

需要同时读取反面文献。DFR 的实验证明，在若干 spurious-correlation 基准上，看起来依赖shortcut的网络仍学到了核心特征，重新训练最后层就能利用。因此，仅根据最终漏选便断言“必须重建特征”，证据不足。其方法使用已知组构造平衡读出数据，部分设置使用额外数据；本项目不得借此用VAL标签训练，更不能默认方法winner平衡等同正确的spurious-group平衡。[P06，§4.1-4.2、§5](https://arxiv.org/html/2204.02937)

### 5.3 Shortcut 不必是伪相关；lazy readout 也不是严格的 lazy training

规模、容量和题型可能是真实优势机制的一部分，而不只是坏shortcut。Feature Sieve 通过辅助分类器和阶段性向均匀目标的遗忘，抑制早期易学特征；其 §3.1 同样承认简单特征可以正确。若这里抹去合法规模/容量信息，可能先破坏已经有效的自然风险判别。[P07，§3.1-3.2、Algorithm 1](https://proceedings.mlr.press/v202/tiwari23a/tiwari23a.pdf)

严格 lazy training 指网络轨迹接近初始化处的线性化模型，受参数化尺度等影响；不是“最后层学得快”“训练准确率低”或“参数总范数改动小”的同义词。Chizat 等给出有限时间路径接近界及额外假设下的收敛分析；lazy模型也可以插值训练集，而泛化较差。当前没有相应线性化轨迹/核演化证据，所以只能说“读出优先/特征更新不足的假设”，不能给现模型贴已证实NTK-lazy标签。[P05，§1.1、§2.2-2.3](https://papers.neurips.cc/paper/8559-on-lazy-training-in-differentiable-programming.pdf)

R67 已经换新评分头做表征/读出分阶段，R101 已做hint预热，R108 已做先验加残差。再提“让模型不要只学先验、先学复杂特征”时，必须说明缺失的学习信号是什么，而不是仅换术语。

## 6. 共享、顺序、完整逻辑 batch 与学习时间尺度

### 6.1 自然风险相同，不等于优化轨迹相同

当前逐题等权自然经验目标可写为

\[
L_{\rm nat}(\theta)=\frac1{18}\sum_{t=1}^{18}
\frac1{N_t}\sum_{i=1}^{N_t}\ell(f_\theta(x_{ti}),y_{ti}).
\]

当前训练是一次更新取一个任务的1024例，任务按18题循环置换；不是同一参数点同时取18题梯度后平均。两者都可围绕上述风险，但动量、自适应预条件、梯度裁剪及非线性使其轨迹不同。

**本文局部推导，仅对光滑、普通GD且同样两次步长成立：**先A后B，令各梯度/Hessian在起点计算，

\[
\theta_{AB}=\theta-\alpha(g_A+g_B)+\alpha^2H_Bg_A+O(\alpha^3),
\]

交换顺序的差为 `alpha^2(H_B g_A-H_A g_B)+O(alpha^3)`。因此长期任务次数相同不是“已经完成同时优化”的证明。不过，这个二阶项既可能正迁移也可能负迁移，不能看到顺序训练就诊断干扰。

PCGrad 的理论正好提醒这一点：§2.2 的有害条件涉及梯度冲突、幅度不平衡与曲率，Theorem 2 是两任务一步改善的充分条件，不是负余弦就应投影的普适结论。[P08](https://papers.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf) R64 的真实小收益及 R69 完整私有bank没有联合改善，已经降低“共享干扰是主要瓶颈”的先验可信度；不能一面引用负梯度，一面忽略这些结果。

### 6.2 完整逻辑 batch 是学习目标的语义，不只是显存技术

对样本可加损失，正确逻辑梯度为

\[
g_B=\sum_k\frac{n_k}{B}g_k,
\]

随后一次裁剪、一次更新。若把每个microbatch分别裁剪/更新，就改变了优化动态。对跨实例 AUC、contrastive、CMR 等，损失本身还依赖逻辑batch内的配对；分别算小batch再平均一般不等价。

这里不把“可能microbatch错了”当科学解释：第二轮已读到 R81/R82 聚合完整1024个logits、在整批构造跨实例目标并反传logit梯度，再以RNG重放传播到模型；R85的有状态梯度统计也按成功逻辑更新提交。它们已认真实现相关语义。完整1024仍只是一题内的batch，不等于18题同一参数状态的联合batch；这是两个不同维度。

### 6.3 更新数、曝光次数、独立样本和活跃参数时间不是同一个时钟

| 对照 | 每题更新 | 样本曝光除以训练规模 | 可推论与不可推论 |
| --- | ---: | ---: | --- |
| R77 10k | 2760 | 282.624 | 更多重复，不是更多独立信息。 |
| R77 70,114 | 2760 | 40.309 | 同更新下更多独立信息；实际留出收益支持这种改变，但非无限扩容定律。 |
| 当前81k、57,600总更新终点 | 3200 | 40.454 | 不能简单说“90k每题只训了很少epoch”；这是终点曝光，选中点可能更早。 |
| R80 batch512 | 全局99,360 | 与其1024比较按曝光匹配 | 已改变更新数与梯度噪声，结果仍无联合突破。 |

以上曝光比是按记录计算的平均呈现次数，不表示每实例恰好均匀见过这么多次。R69 私有支路的活跃更新少于共享模型，inactive 参数不推进其优化器状态；“参数更多且同样训练总时长”不是相同的每参数学习时钟。

作为解释性模型，固定核平方损失中某方向误差可近似为

\[
e_k(s)\approx e_k(0)\exp[-\kappa_k\sum_{r\le s}\alpha_r].
\]

真正决定是否学会慢方向的是该方向的有效强度与累积学习时间，不只是日志中的epochs。这一式不是当前CE/AdamW的拟合规律，也不推出应增LR或长训。要支持慢学习解释，需要优势相关方向随学习持续形成并能迁移；R40式TRAIN上升、VAL下降只支持记忆。

Deep Double Descent 的 epoch-wise/sample-wise 非单调现象说明更长训练或更多样本不保证单调改善；但主要证据围绕插值附近及特定模型/噪声设置，不可用来承诺当前71%平台最终会“grok”。[P10，§6-7](https://arxiv.org/html/1912.02292)

### 6.4 一个应当明确写出的量级：WD 不宜被描述为巨大直接收缩

本节为协议数值的数学推导，不是新的训练或参数测量。AdamW 的显式衰减因子为

\[
D_S=\prod_{s=1}^{S}(1-\lambda\alpha_s).
\]

对每步都施加衰减的参数，使用当前 `lambda=1e-4`、`alpha_s<=1e-4`、`S<=57600`，在实数算术下

\[
1-D_S\le\sum_s\lambda\alpha_s\le0.000576=0.0576\%.
\]

真实warmup/cosine积分更小。这只衡量显式乘法衰减，不是完整参数轨迹或与WD=0的模型差异界。它不足以排除极小梯度方向上的相对影响、尺度耦合、早期路径依赖，也不涉及dropout或KL。但它确实排除了“该WD通过很大的直接收缩把权重整体压回原点”这一朴素数量级解释。

Lewkowycz-Gur-Ari 研究L2、LR、训练步数耦合，并在特定过参数化SGD实验中观察学习时间与 `1/(eta lambda)` 的关系；§4 Theorem 1 还依赖无限宽、齐次性及相关函数猜想。不能将其直接套到有残差/LayerNorm、自适应动量和解耦WD的当前网络，推成“减WD即可突破”。[P09](https://papers.nips.cc/paper_files/paper/2020/file/32fcc8cfe1fa4c77b5c58dafd36d1a98-Paper.pdf)

## 7. Hard negative 与条件监督：新学习路径还是新信息？

### 7.1 先区分三个不同对象

**同实例困难候选。** 全池CE已同时含所有合法候选，并通过softmax给高分错误候选更大的输出梯度。当前池通常只有3至10个候选，不能套用百万类别近似softmax的直觉，声称模型从未见过困难负类。R85 candidate subset改变候选竞争与条件任务，R81则不是这一类方法内负候选挖掘。

**跨实例同方法困难样本。** R81 用同方法的winner/strict nonwinner构造跨实例排序，R82更进一步匹配任务、规模分箱、记录的分布/容量条件，以相对regret差排序。它们利用已有标签中不同的比较结构，但AUC只保证某种排序，既不自动校准跨方法概率，也不保证最终完整池argmax。

**条件化后的判别问题。** 将总体标签分解为条件任务，能改变各参数接收的监督，却不必增加信息。R111属于这一层，R108则用条件先验残差改变概率估计路径。二者与“换一批hard sample”不应混为一谈。

### 7.2 R95 与难例重加权并不相同

R95 在warmup后按梯度代理构造抽样概率，形式为

\[
q_i=\frac{0.5}{N}+\frac{0.5h_i}{\sum_jh_j},\qquad
\widehat g=\frac{1}{Nq_i}\nabla_\theta\ell_i.
\]

条件于固定采样概率且求梯度时不对概率求导，有

\[
\mathbb E_{i\sim q}[\widehat g]
=\frac1N\sum_i\nabla_\theta\ell_i.
\]

所以它主要改变估计方差和样本到达路径，不改变裁剪前的经验目标梯度期望。非线性裁剪/Adam更新的期望并不因此无偏。若去掉逆概率校正，则一般是在改变目标，而不只是“更充分利用同一目标的数据”。[R95报告](/public/home/shiys/0selection2/code/V4/runs/R95_adaptive_example_sampling/RESULTS.md)

JTT 的两次训练把初始模型错例集合固定后上调权重，目标是worst-group robustness，不是R95式无偏采样。其方法仍需要验证阶段的信息帮助选择设置，平均准确率并无普遍改善保证。对于本项目，重新命名为hard-negative curriculum不能自动满足自然Top1与成本联合目标。[P11，§3-4、Eq.6-7](https://proceedings.mlr.press/v139/liu21f/liu21f.pdf)

### 7.3 R108 是一个尤其重要的已完成反例

冻结 TRAIN 条件先验 `pi_m(z)`，其中 `z` 为任务/规模上下文，残差logit为 `t_m(x)`，最终分数为

\[
s_m(x)=\log\pi_m(z)+t_m(x).
\]

R108 A 在 `s` 上做自然 CE；B 的目标为

\[
\ell_{\rm ratio}(x,y)
=\tfrac12\operatorname{softplus}(-t_y)
+\tfrac12\sum_{m\in\mathcal M(x)}\pi_m(z)\operatorname{softplus}(t_m).
\]

负项包含真实winner那一项。理想条件最优为 `t_m*=log eta_m-log pi_m`，故恢复后的 `s_m*=log eta_m`，不是一个故意按方法均衡的决策器。这是有明确统计目标的、保持自然决策目标的训练路径变化，**已经完成且未突破**。不能在新报告中再次以“正负分布密度比”推荐同一方向。[R108模型准备](/public/home/shiys/0selection2/code/V4/runs/R108_conditional_density_ratio/MODEL_PREP.md)、[目标实现](/public/home/shiys/0selection2/code/V4/r108_objective.py)

### 7.4 哪些监督只是在重排信息，哪些可能触及新的判别？

从既有TRAIN数据确定性计算的pair、runner-up、subset、density ratio、hardness都没有添加独立观测，但可能提供更合适的归纳偏置。不能把“没有新信息”误读为“绝不可能改善”。需要问它改变了哪条有限容量学习路径，以及是否有可迁移的收益证据。

同 winner 的 SupCon 可能将内部不同优势机制拉得过近；latent mixup 的软标签不保证对应一个物理可行实例；教师伪标签可能只增加旧决策的覆盖；hint虽含更细的教师行为，但未必含当前方法之间的相对优势。R67/R70/R73/R101各自触及这些方向，不能只依据论文名称再提一轮。

成本差监督也要防止夸大信息增量。若只是以平方误差拟合全部pair difference，令成本预测误差为 `e_m`，有

\[
\sum_{m<n}(e_m-e_n)^2
=K\sum_m(e_m-\bar e)^2.
\]

它本质上学习中心化成本误差，不是获得了 `K(K-1)/2` 份独立标签。真正有意义的改变应是更准确地学习输入如何决定优势差，而非对同一成本表换一种计数。

## 8. 何时改目标或数据构成，而不是继续改网络？

**当失败主要是已接受决策风险与训练风险不一致时，应改目标的定义与声明，而非寄望更大网络消除本来存在的取舍。** 例如若最终真正优先的是期望相对regret，应学 `E[C_m/C_*|X]` 或适合该决策的surrogate；不能只用winner后验再宣布“理论上也应成本最优”。但这里已做过多种成本目标，当前没有证据支持仅凭这一原则再开一支成本头；更不能借目标改名偷偷放弃自然Top1/真实恢复要求。

**当少选方法有稳定、输入可识别的优势条件，但TRAIN在这些条件中的独立覆盖不足时，应优先讨论数据构成。** 需要的是该条件中的覆盖，而不只是总winner数。更多重复采样不添加独立信息；更多教师伪标签不等价于更多真实winner。R77支持增加真实独立样本曾有收益，但不证明无差别扩大现有分布仍是最佳方向。

**当错误集中在当前目标可辨认、标签可信的特定条件内时，条件监督可能比增层更直接。** 它应明确条件由推理可得输入或固定候选关系定义，不能按当前查询真实winner路由前向；否则只是泄露答案。R82已有输入条件匹配，R111已有固定方法族条件概率链，所以必须承认最近历史，而不是宣称发现了条件学习。

**当只有“许多模型都错”“TRAIN没有拟合”“训练更久也许有用”时，既不应自动扩数据，也不应自动改网络。** 这三种观察没有识别缺的是独立信息、目标对齐、表示还是优化时间。保持问题未识别，是比连续提出同信息变体更有价值的结论。

以上是科学选择边界，不是新label、新split或采集指令。当前冻结TRAIN/VAL/TEST、候选池与历史TRAIN上下文排除规则均不变。

## 9. 最多两个方向，以及不能无限延后的停止边界

### 方向一：让正在运行的部分私有条件监督回答有限的问题

**优先级：只等待已有 R109/R111 的锁定完整结果，不另开同类实验。**

R109 A/B改变候选节点变换权重是共享还是按方法私有，保留原复合目标；R110是同类节点结构配纯CE的已完成近邻。R111进一步改变为固定方法族的条件概率链，而不只是把全池头拆成几段。机制依据为 [R109既定方案](/public/home/shiys/0selection2/code/V4/runs/R109_method_conditioned_nodes/PLAN.md)、[R111既定方案](/public/home/shiys/0selection2/code/V4/runs/R111_family_conditional_features/PLAN.md) 与实际模型；方案旧状态字段不覆盖本次运行快照。

\[
p_\theta(m\mid x)=\pi_\theta(g(m)\mid x)\,q_\theta(m\mid g(m),x),
\]

\[
L_{\rm cond}
=\mathbb E[-\log\pi_\theta(g(Y)\mid X)
-\log q_\theta(Y\mid g(Y),X)].
\]

学习信号是正确族的根损失和正确族内部的条件NLL；推理仍对所有合法方法的概率乘积做一次完整argmax，不以true winner选择运行分支，不做硬root路由。单成员族的条件概率为1。概率链在无限表达能力下只是联合分布分解，不增加可用信息或Bayes最优能力。

**必须准确描述哪些仍共享。** R111 的根读取shared post-joint信息：共享实例摘要、公共context、该族的shared joint solver token均值以及family embedding；不是完全独立的根，也不是不含候选/族信息的原始输入分类器。私有节点输出不反馈给根；但各族的私有节点计算之后仍使用共享pool和共享score，前面的encoder/joint表示、solver token与context也共享。族embedding参与根和私有输入，并非完全隔离的参数。[实际模型说明与路径](/public/home/shiys/0selection2/code/V4/r111_model.py:49)

在精确归一化概率链下，单样本非winner族的**私有节点参数直接NLL梯度为0**，并不意味着该族所有相关参数都无梯度，也不意味着Adam动量/WD下相关参数一定不动。根损失继续训练共享表示；正确族条件损失通过shared pool/score/context/encoder影响其它族将来看到的表示。R111因此只减少了一种直接的跨族条件监督，不是全部特征隔离，更不是消除了跨题共享。

**与最近历史的实质区别：**R110虽然私有节点参数，但所有候选仍在同一自然全池CE中竞争，非winner候选私有分支也接收直接分类信号；R111将私有条件分支的直接信号限制到winner族，同时新增族根与概率分解。R69早有完整task-private bank，但分割轴是问题，不是这里的方法族条件梯度。这些不同不保证R111优越，也不允许把R111称作“从未尝试过私有化”。

**最脆弱的假设：**实现来源族不一定等于真实可学习的优势机制；条件任务降低负类异质性，也减少每个私有分支的正向有效样本。根看不到私有优势，可能给正确族过低概率，即使该族内排序正确也无法夺回全池argmax。共享pool/score还可能成为瓶颈。以上是风险，不是结果出来前的免责条款。

可否决观察是自然完整池中仅TRAIN/族内条件正确率提高、测试仍交换错误；或只靠提高目标方法选择频率换来同量错误；或全局Top1略好而目标真实恢复和成本恶化。这样的结果不能支持“主要瓶颈已找到”。不用新增消融矩阵即可作这一有限判断。

### 方向二：学习时间与共享更新的研究问题，只在有定向证据时保留

**优先级：低于等待现有结果；当前不是可直接推荐的新训练。**

科学目标不是再换优化器，而是在保持 `L_nat` 与输入信息不变时，区分“优势表示形成太慢/被随后更新破坏”与“同一表示上的读出或标签信息限制”。设样本winner margin为

\[
m_i(\theta)=u_{y_i}(x_i)-\max_{j\ne y_i}u_j(x_i).
\]

在固定活跃竞争者的局部近似下，一次更新对其作用为

\[
\Delta m_i\approx\nabla_\theta m_i^\top\Delta\theta.
\]

这里真正相关的学习信号是优势margin的**定向变化及其跨样本可迁移性**，而非总体梯度norm、loss下降或负余弦比例。参数可概念性分为表示与读出两块，判断它们对同一margin变化的贡献；不要求每个正交方向都被保护，也不把所有任务间负变化视为错误。

决策链应是：如果已有证据显示某任务更新创造了能迁移的优势，而其它任务随后系统性消除了它，才提升共享顺序/学习时间假说；若只有训练点margin增大、留出不增大，则更接近R40/R49式记忆；若隐藏表示已有可迁移读出而原头未采用，优先解释为读出问题，而不是再扩大私有网络。当前报告没有足够现存证据把这条链走到前一分支，所以不把它包装成已诊断机制。

最近历史已经很近：R64同状态三任务PCGrad、R69完整私有化、R80同曝光更多更新、R89优化器几何、R67/R101阶段学习都部分触及此问题。唯一值得保留的区别是“可迁移的优势被创建又被破坏”的方向性证据，而不是又一种梯度手术名称。若没有这种证据，或延长学习只提高记忆，便否决它作为下一阶段主解释。R109/R111完整阴性后，不得自动用此方向续命；必须有不同于其最终错误交换的独立机制证据。

### 9.3 对 R109/R111 完整结果的预先解释边界

| 将来完整结果 | 应支持/降低到哪一层 | 不应再说什么 |
| --- | --- | --- |
| R109私有对共享、R111相对固定参照均没有自然Top1、真实恢复与成本的联合收益，主要只是交换错误 | 结合R64、R69、R93、R110，显著降低“晚期共享节点或非winner直接分类压力是当前主瓶颈”的优先级；停止同信息的共享/私有/族条件改名续训。 | 不能说所有共享干扰数学上都不存在；也不能据未完全隔离这一事实，永远转向“再私有一层”“再换分组”而不接受否决。 |
| R109私有在复合目标下有联合收益，而R110纯CE没有 | 支持该具体节点私有计算与训练路径组合可能改善有限学习；与目标交互的解释值得保留。 | 不证明复合损失普遍更好，不证明收益仅来自解除共享；参数量、计算、单/双view及跨版本比较仍有限制。 |
| R111有联合收益，R109/R110没有 | 支持“族条件概率链 + 部分私有节点 + 该共享读出”这个组合获得了可迁移优势。 | 不证明只有条件梯度隔离在起作用；新增根、分组、有效样本分配与路径变化未独立分离，更不证明全特征隔离有益。 |
| R109和R111都有联合收益 | 支持这个机制家族中存在比原固定表示更可用的判别路径，值得围绕实测收益作有限科学总结。 | 不证明所有任务、方法和数据分布均受同一干扰；不证明增加信息、达到Bayes界或多seed稳定。 |
| 只有族内准确率/目标选择频率/少量TRAIN救回上升，完整池和成本未联合改善 | 只能支持局部行为变化，不能升格为selector恢复成功。 | 不得通过重选TEST checkpoint、移动目标组或改指标名称把它变成成功。 |

这里的“停止”是停止将这一机制家族列为默认下一步，不是暂停健康运行任务，也不是判定所有未来架构都不可能有效。反转停止建议需要新的优势信息或可迁移的定向机制证据，不是另一个论文名称。

## 10. 一手论文核读及直接支持/推断界线

下面13篇均实际打开作者论文或官方会议/期刊全文，并阅读所列方法/理论关键节；并非只依据摘要。选它们是为交叉制约机制解释，不声称它们都是首次出现在本项目，也不声称这是截至2026年的穷尽检索。结构化书目信息、URL、阅读位置、历史近邻与限制见 [03_sources.csv](/public/home/shiys/0selection2/docs/research/comprehensive_direction_review_20261025/03_sources.csv)。

| ID / 论文 | 核读位置与直接支持 | 本文推断及不可移植条件 |
| --- | --- | --- |
| P01 Tewari & Bartlett, JMLR 2007, On the Consistency of Multiclass Classification Methods | §2 Definition 1、Theorem 2：多分类校准与理想风险一致性之间的联系。 | 用来限制proper-loss论证；不是当前神经网络有限步收敛保证。 |
| P02 Menon et al., ICLR 2021, Long-Tail Learning via Logit Adjustment | §3 Eq.7-8；§5.1 Eq.10；§5.2 Theorem 1：balanced error下的先验修正与一致性。 | 不把balanced error等同自然microaccuracy；论文经验收益不是当前候选池的保证。 |
| P03 Awasthi et al., NeurIPS 2022, Multi-Class H-Consistency Bounds | §2 minimizability gap；§3 Theorems 1-2：在假设类约束下连接条件与总体风险。 | 有限模型可发生目标错配；具体损失/假设类条件尚未在本项目满足。 |
| P04 Pezeshki et al., NeurIPS 2021, Gradient Starvation | §3.3 Theorem 2；§3.4 Eq.17-18：线性化两特征耦合与spectral decoupling。 | 支持可能的方向性竞争，不证明本项目缺失几何特征或应加logit penalty。 |
| P05 Chizat et al., NeurIPS 2019, On Lazy Training in Differentiable Programming | §1.1；§2.2 Theorem 2.2及§2.3：参数化尺度与线性化轨迹接近。 | 区分严格lazy与口语读出优先；当前缺相应测量及定理条件。 |
| P06 Kirichenko et al., ICLR 2023, Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations | §4.1-4.2、§5：核心特征可能已存在，平衡数据重训读出可恢复。 | 防止过度诊断feature starvation；已知spurious组/重加权集与当前自然winner任务不同。 |
| P07 Tiwari & Shenoy, ICML 2023, Overcoming Simplicity Bias in Deep Networks using a Feature Sieve | §3.1-3.2、Eq.1-3、Algorithm 1：辅助读出与阶段性遗忘易学特征。 | 不能盲删合法规模/容量特征；不是当前已证实适用的新路线。 |
| P08 Yu et al., NeurIPS 2020, Gradient Surgery for Multi-Task Learning | §2.2-2.4、Algorithm 1、Theorem 2：投影与两任务局部改善条件。 | 负余弦不足以证明有害；R64是真实近邻，不能绕过其小收益。 |
| P09 Lewkowycz & Gur-Ari, NeurIPS 2020, On the training dynamics of deep networks with L2 regularization | §2时间规律；§3调度局限；§4 Theorem 1：L2会改变学习动力学。 | 不是AdamW/非齐次有限Transformer定理；0.0576%是本文协议推导，不是论文报告。 |
| P10 Nakkiran et al., ICLR 2020, Deep Double Descent | §6 epoch-wise、§7 sample-wise double descent。 | 更多数据/更久训练并非单调；没有证据让当前模型等待插值后的第二下降。 |
| P11 Liu et al., ICML 2021, Just Train Twice | §3任务定义；§4 Eq.6-7：先训练、固定错例、再上调权重优化组鲁棒。 | 与R95保持自然经验风险的采样不同；不能保证自然平均准确率、也不授权使用VAL训练。 |
| P12 Zhang et al., ICLR 2017, Understanding deep learning requires rethinking generalization | §2.1随机标签/像素；§3正则实验：容量与记忆可很强。 | 支持memorization probe边界，不据此认定当前标签随机或全部81k可泛化。 |
| P13 Elmachtoub & Grigas, Management Science 2022, Smart “Predict, then Optimize” | arXiv v5 §3.2定义/次梯度、§4 Theorem 4.2：决策损失surrogate及附条件一致性。 | 有限候选代入是本文推导；分布条件未证实，原简单形缺环境空间内部，且目标不是winner Top1。 |

## 11. 最终科学判断

当前最稳妥的解释不是“正则太强”“优化器不行”“类不平衡”或“Bayes上限71%”中的任何一个。证据支持的是更窄的判断：**已有输入与监督下，多种学习路径形成了相近但不相同的决策；部分能改进局部成本或拟合，却尚未稳定形成少选方法的可迁移优势区域。**

相较旧调研，本镜头新增的主要约束是：把自然Top1/真实恢复的冲突写成可核对的区域风险；用有限假设类反例分开properness与实际最优；给显式WD收缩作数量级上界；把不同学习时钟与R77/R80历史对齐；将R108从“可尝试”移入“已完成”；并为R109/R111的部分共享条件路径设定不能无限后移的停止标准。

因此不推荐再提出一组同信息的focal/OVR/对比/NCE/成本头/优化器新名字。先让已经运行的科学问题获得完整回答；若仍只交换错误，应坦诚降低这一机制家族的优先级，而不是因其从未达到绝对全隔离就拒绝接受阴性证据。
