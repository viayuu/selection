# 给 Sol 的科学交接：24 题类型化完整候选选择器

日期：2026-10-29。版本：完整科学交接，与父任务已有四层编码、两层双流对齐。

职责：本文只给论文依据、科学机制和监督语义。实现、数据版本、标签预算、训练与结果核验由父任务负责。最新继续授权优先于旧准备文档的暂停措辞；这里不新增操作门槛。R119 最终评估仍未纳入科学证据，不提供新成绩。本文没有操作 GPU 或 Git。

## 1. 建议先实现什么

采用一个 **typed relational selector**：保留 routing 的原生几何/ATSP 前端，新增类型化原始输入适配器，共享原 128d、四层 self-attention 核心，再复用两层 node/solver 双流、mean/max、decoder 与共享 utility/maximin 头。候选 ID 和原 34 维已声明静态特征保留。核心变化是编码阶段加入关系类型、方向和逐边数值，而不是重建另一套选择器。它不生成解、不运行 query solver、不要求 89 个过程教师。GOAL 的共享主干与输入输出适配器、L2D 的析取图、DAN 的工序/机器关系、GraphGPS 的局部关系加全局交互分别提供依据；不是把任何一篇论文的求解器直接改名。

|输入|必须保留的内容|主干中的表示|
|---|---|---|
|18 routing|原坐标/完整 ATSP 有向矩阵、depot/start 角色、需求、容量、TW、服务时间、长度、B/O 等约束|customer/depot 节点，真实有向弧特征，资源/约束 token；不丢原矩阵或用伪 XY 替代|
|MIS|`n, edges`，包括孤立节点|完整无向冲突图，两条有向消息边；不把度分布当完整输入|
|JSSP|每个作业的全部有序工序，唯一机器与加工时间|operation、machine 节点，前驱/后继有向边，工序到机器关联边；同机互斥由关联结构恢复，不提供求解后的机器顺序|
|FJSP|全部兼容机器及逐对加工时间|同一 scheduling 前端；每条合法 operation-machine 边保留自己的时间；不兼容是 mask，不是零时长|
|BPP|每件大小、箱容量、数量|item 集合和容量 token，经非线性全局集合交互；不输入隐藏 triplet 分组，不用均值直方图替代物品|
|MKP|所有利润、完整资源消耗矩阵与容量|item-resource 二部图，边权为消耗；资源是可变长度节点，5/10/30 维均直接表达，不补零伪装 DeepACO 的 5 维接口|
|SMTWTP|每个作业的 `p,w,d`，保留 `d=0`|job 集合和 single-machine token；全局交互学习排序压力，不预填规则生成的顺序或真实完工时间|

共享层使用带关系类型、方向、边值的消息函数；operation-machine 等二部关系双向传播。逐边数值不仅进入 attention bias，也进入 value/message，例如 `sum_j alpha_ij (Vh_j + v_relation(e_ij))`；仅有线性 bias 时，同一行时间的公共平移可能被 softmax 抹去。每层保留节点残差和全局交互，不能先汇总成几个统计量再进主干。原双流 candidate query 继续读取全部最终节点，mean/max 只是补充。密集原关系按块精确计算，稀疏关系用边表消息；不持久化 `batch*N*N*128` 边状态，不为省显存静默丢边。若全节点注意力以后成为瓶颈，inducing-token 全局通路是唯一优先的结构替代，但不是本轮必须同时改变的组件。

候选 query 可表达为 `task + method_family + deployment_id + known_config`，接入原候选表示即可。L2D 四权重、DAN 八权重各是独立动作，共享 family embedding 但保留 deployment residual；禁止逐例免费取同族最好权重。原双流保留候选间通过节点的交互；候选集合 self-attention 是可选增强，不是接入前提。输出原 `u_m`，D1 另从同一候选表示输出中心化优势预测 `Ahat_m`。推理始终完整合法池 `argmax(u)`；若 maximin 仅作用于 `u_m-u_j`，其排序与此一致。一个 checkpoint 覆盖各题，不做 ReLD 二分类或独立 pair 投票。

## 2. 目标定义：先把符号和零值处理正确

令目标为 `f_m`：routing仍沿用原`aug_score`、native/FP64 `ind`、B问题候选过滤及单`reld_mt`规则；新增六题使用原生objective，不伪造`aug_score`。最小化任务 `sigma=+1`，MIS/MKP `sigma=-1`。定义：

$$
c_m=\sigma f_m,\quad c_* = \min_{m\in M(x)}c_m,\quad
\Delta_m=c_m-c_*\geq0,\quad B(x)=\{m:c_m=c_*\}.
$$

`c` 仅统一排序，不作为相对 gap 的分母；MIS 的真实损失是 `f_*-f_m`，不是 `(1+n-f_m)/(1+n-f_*)-1`。整数目标在原生整数上比较并列，再计算浮点监督；routing 的 native/FP64 `ind` 原样保留，不以 FP32 argmin 改写。

一个可固定的尺度方案：先在自然 TRAIN 拟合原生均值最优的 SBS，令 `s_t=mean_TRAIN(f_SBS)`。SBS须是覆盖全题声明域的固定部署，不能以仅适用5维的DeepACO充当全维MKP的SBS，也不能逐层取最优冒充单方法。若该尺度为零，改用明确的正输入尺度；SMTWTP可取 `mean_TRAIN[(sum p)(sum w)]`。若全域SBS尚未定义，也应另声明正输入尺度，SBS本身保持缺失。这是固定单位，不是 query oracle，也不是新 gap 定义。候选池和训练版本改变时重新记录尺度，不依赖 VAL/TEST。

$$
\rho_m=\Delta_m/s_t,\qquad
A_m=(c_m-\bar c)/s_t,\qquad
\bar c=|M|^{-1}\sum_{j\in M}c_j.
$$

`rho` 是 TRAIN 尺度归一化绝对损失；`A` 是全候选相对性能坐标。二者均支持 max/min、整数并列和零目标，不使用 log(0) 或人为 offset。家族响应必须在同一题、同一合法候选集合和同一固定尺度下计算。

简单接入用任务等权的 `L_select + beta L_soft-risk`，D1再加 `alpha L_adv`：routing 的 `L_select=-log p_native`；新六题为 `-log sum_{m in B} p_m`，全并列时恰为零；`L_adv` 是每例完整合法候选平均的 Huber(Ahat,A)；`L_soft-risk=sum_m p_m rho_m`。系数由父任务在训练前固定。soft-risk 是随机策略期望的可微代理，不保证确定性 argmax；优势回归也不等于 winner 概率。不要强迫并列候选均匀选择。D2 的硬动作反馈用于检验这个代理缺口。

注意三个掩码分开：事前可知的适用性 `legal_mask`、标签是否存在 `observed_mask`、事后证书是否有效 `valid_mask`。不能根据 query 某 solver 跑失败后再把它从前向候选中删掉。完整池尚缺标签时，只能诚实称部分池；缺失不是零成本或最差名次。

## 3. D1 与 D2 分别改变什么

**D1：合法结构族和完整优势响应。** 同一基础实例构造结构变化端点，全部测量该题已声明合法候选，而非全部预期/暂缓 solver。routing 改变空间与资源耦合；MIS 改变冲突拓扑；JSSP 改变机器竞争与工序链；FJSP 改变兼容结构/替代机时间；BPP 改变组合装填结构；MKP 改变资源紧度与收益消耗耦合；SMTWTP 改变交期压力与 `p,w,d` 耦合。保持整数、非负交期、原任务约束和候选适用性，生成域偏移单列。

$$
L_{\rm resp}=E_{b,z,z'}\,|M|^{-1}\sum_m
\operatorname{Huber}\left[(\hat A_m(X_{b,z'})-\hat A_m(X_{b,z}))
-(A_m(X_{b,z'})-A_m(X_{b,z}))\right].
$$

端点自身的全候选锚定和自然 TRAIN 选择损失保留。族 ID、扰动 ID、成本、winner 均不进入 query 前向。响应由端点标签导出，不增加独立信息；真实随机运行差也不自动等于期望性能差。R114 已有全池，R115 已有中心化响应且跨 base 失败；换成神经网络或多六题不构成新科学发现。

**D2：整模型的硬动作约束。** 以同架构参照 `theta0` 的 TRAIN 风险为右端，最小化等权任务硬选择错误，同时约束实际硬损失和固定漏选群体不恶化。routing 用 native 错误，新题用 `a not in B`；分别保护 routing18 和 nonrouting6，不能让新题全并列的轻松分数掩盖 routing 退化。历史 fixed36 保留原定义；新任务只有正的独有优势/替代代价才构成有意义的漏选，不给每个权重配置强制分配选择率。

具体保护量：routing18的native错误与原 `r=Delta/f_*` actual relative regret；nonrouting6使用零值安全的 `rho`，同时另报正oracle子集的relative regret。不能用新的rho替换旧routing regret后声称原约束也满足。参照须有同版本六题输出，不能直接把仅有18题的R119当作全24参照。

参数更新使用代理，乘子反馈使用 TRAIN 真实 argmax 的约束违反量；不通过 formal VAL 梯度或 TEST 数值更新。Cotter 的 proxy-Lagrangian 提供这种分工，论文的随机混合解保证不能移植到本项目单 checkpoint。若只做简单投影乘子更新，应明确这是启发式实现，不冒称复现其带 swap-regret 的理论算法。

集合CE不是硬错误上界：概率 `(0.4,0.3,0.3)`、后两类并列最好时，硬决策错误但 `-log(0.6)/log2<1`。D2可用 `h_B=softplus(max_bad u-max_best u)/log2`，全并列时定义0；硬风险代理可用 `sum_m v_m softplus(u_m-max_other u)/log2`，routing取v=r，新题取v=rho。被选动作的系数至少1，故上界成立；松紧与优化效果仍未知，完整推导见正文7.2节。

## 4. 两个有辨识力的有界比较

1. D1：同架构、同端点、同全候选标签、同自然数据和更新量，比较静态端点训练与保留真实族的响应训练。它回答“显式族监督是否比同量标签有益”，不单独证明端点采集优于不采集。若改用配对打散对照，必须保留端点标签、按实际新配对重算差分，控制每端点重复次数；不得制造错误标签。打散对照可能自身有害，须限制归因。
2. D2：同架构、同自然完整数据与相同初始化来源，比较固定系数全信息损失与硬动作反馈的整模型约束目标。若只有一个 A/B 而同时改变 D1 和 D2，只能评价组合，不能宣称识别各自机制。没有全24标签时，不把21题或已覆盖子集叫完整24题。

D1 反证：响应仅在同 base 有效、低于零/均值响应基线、或只改善辅助误差而无自然域硬选择收益。D2 反证：只改善 soft 风险/选择次数/TRAIN 约束，或真实恢复持续需要牺牲总体质量。两者都要报告纠正与误伤，不以“从零变成一次选择”判成功。

## 5. 报告接口

独立报告 `ROUTING18/native`、`NONROUTING6/tie-aware`、`ALL24/tie-aware`；另可有明确命名的 `ALL24/mixed`，不能混称同口径 Top1。所有 ALL 是先题内均值再题间等权，不混合分钟、箱数和利润的 raw cost。

逐题保留原生 mean_objective、direction、TRAIN-SBS、pool-oracle、绝对损失及可定义的比率。SMTWTP：oracle>0 子集报告普通相对 regret；oracle=0 子集报告数量、选出正拖期的比例和绝对损失；另报覆盖全体的固定尺度 `rho`。0/0 为不适用，不填零或加 epsilon。pool-oracle 不是证明过的全局最优。

权威总览的2026-10-28 23:52快照中，BPP两候选全部并列，TieTop1为100%但没有质量选择空间；箱子分组不同不改变此结论。该快照仅确认MIS/JSSP/FJSP正式train/val/test完整。父任务已开始继续补齐，本文不把历史缺项断言成实时状态，也不把最新授权当成已经完成。

## 原始依据入口

- [GOAL，3.3-3.5节](https://arxiv.org/html/2406.15079v3)
- [L2D，3-4节](https://papers.neurips.cc/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf)
- [DAN，II-A、III-B节](https://arxiv.org/pdf/2305.05119)
- [GraphGPS，3.3节](https://arxiv.org/pdf/2205.12454)
- [Set Transformer，3.1-3.2节](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf)
- [Cotter等，4.2、4.6节](https://www.jmlr.org/papers/volume20/18-616/18-616.pdf)

已实际打开12篇原论文并阅读相关方法/评价段落。完整推导见 [REPORT.md](/public/home/shiys/0selection2/docs/research/multidomain_selection_design_20261029/REPORT.md)，阅读范围与移植边界见 [sources.csv](/public/home/shiys/0selection2/docs/research/multidomain_selection_design_20261029/sources.csv)，优先方向及反证见 [directions.csv](/public/home/shiys/0selection2/docs/research/multidomain_selection_design_20261029/directions.csv)。PDF由父任务机械排版；本sidecar没有创建PDF、运行实验或完成结果认证。
