# 评分、损失与优化：实际数学与代码核对

研究日期：2026-10-23，跨日于 2026-10-24 收束。范围：full18，而非某一 ReLD 对。

## 结论先行

**已证明：maximin 不是 utility 的共同平移；它保持单次前向的完整候选排序，但额外放大领先者间隔，改变概率、CE 梯度及其与 pair BCE 的尺度兼容性。已实测历史：R91 改链接、R93 去掉复合损失后仍在约 71% 平台，因此该链接及比较/风险/R-Drop 不是平台的必要原因。未证明：当前模型已优化到类内最优、表示没有可学信号、或存在共同大 bug。**

最值得继续区分的是：已有表示上的条件混杂与特征学习可达性；有限共享参数下不同监督对象的折衷；单任务交替更新、预条件和有效步长对学习方向的影响。现有结果不足以把任何一个认定为共同主因。

主线程提出的 winner/runner-up joint 和 conditional density-ratio 均列为**低优先、证据不足**。前者严格包含条件 runner-up 辅助分类，后者与 CE 有相同理想冠军决策；都不能包装为已找到平台突破机制。本文不建议立即新增训练。

### 证据与操作边界

- 已先读[前版报告](../deep_direction_review_20261023/REPORT.md)、[R35-R79 history](../deep_direction_review_20261023/astra_history_r35_r79.csv)、[R80-R104 history](../deep_direction_review_20261023/astra_history_r80_r104.csv)，再读下列源码、锁定协议、小 CSV 和历史 JSON。历史 negative 只约束实际配置，不排除整个方法大类。
- 未训练、未调用模型前向/推理、未加载权重或预测数组、未访问 GPU、未运行 launcher、未检查或操作实验进程、未修改科学源码、未提交 Git。所有 shell 命令先激活 `easynco`。仅新增本文件与 `loss_math_checks.py`。
- 主线程已告知其核验本支线 session 配置为 `gpt-6-astra/max`；本支线不另作模型身份自查。
- 行号指本次读取的工作区源码。`pairwise_selector.py`、`pairwise_objective.py`、`r76_experiment.py`、`r77_experiment.py` 与 R84 保存哈希一致；R81 AUC、R82 CMR、R85 objectives、R89 optimizer、R91/R93/R95 objectives 与各自协议哈希一致。当前 `r84_experiment.py` 哈希 `834ac040...` 与 R85/R89/R91/R93/R95 协议一致，但与 R84A 保存的 `045c3eed...` 不同；故不把当前整个 runner 无条件认证为 R84 当时的逐字版本。R84 的具体超参另核对其保存协议。
- 以下历史 Top1 沿用 native/FP64 `ind`，ALL 按题等权；TEST 已参与研发。小张量证明不等于项目性能原因证明。

## 1. 八条最有价值的代码事实

| # | 实际事实 | 代码入口与含义 |
| --- | --- | --- |
| 1 | 原路径是共享 score MLP 得到每候选 utility，再构造差值矩阵，再排除自身做 maximin。没有第二个独立的比较头。 | [pairwise_selector.py:11](/public/home/shiys/0selection2/code/V4/pairwise_selector.py:11)、[同文件:27](/public/home/shiys/0selection2/code/V4/pairwise_selector.py:27)、[dual_stream.py:70](/public/home/shiys/0selection2/code/V4/dual_stream.py:70)。R84B/D 是例外：删除 score MLP，用平均多头点积直接作 logits，不经 maximin，[r84_attention_selector.py:44](/public/home/shiys/0selection2/code/V4/r84_attention_selector.py:44)。 |
| 2 | 原每视图目标为 `0.35 CE + 0.35 comparison + 0.02 risk`；risk 先将期望相对 regret 截到 1，再除以 0.01。两个 dropout 前向的目标取平均，再加对称 KL。 | [pairwise_objective.py:63](/public/home/shiys/0selection2/code/V4/pairwise_objective.py:63)、[r76_experiment.py:164](/public/home/shiys/0selection2/code/V4/r76_experiment.py:164)、[r42_experiment.py:59](/public/home/shiys/0selection2/code/V4/r42_experiment.py:59)。不是四个独立读出头；各项共享 utility 和编码器。 |
| 3 | pair 标签来自原 FP64 严格成本比较，精确相等的 pair 不监督。true3/pred3/all 的组均值权重为 0.50/0.25/0.25；空组重归一化。pred3 来自两次 maximin logits 的均值并 detach。 | [pairwise_objective.py:17](/public/home/shiys/0selection2/code/V4/pairwise_objective.py:17)、[同文件:39](/public/home/shiys/0selection2/code/V4/pairwise_objective.py:39)。不是仅监督 winner 对，也不是按所有 pair 个数直接累加。true3 成本边界遇 ties 可包含超过三个方法。 |
| 4 | R91A 仅 CE 改用 utility；B 进一步把 risk/KL 改用 utility。二者保留比较目标、原 averaged-maximin Top3、原推理。R93 才是一次前向、系数 1 的原生 winner utility CE；A dropout=0.1，B=0。 | [r91_objective.py:15](/public/home/shiys/0selection2/code/V4/r91_objective.py:15)、[同文件:48](/public/home/shiys/0selection2/code/V4/r91_objective.py:48)、[r93_objective.py:18](/public/home/shiys/0selection2/code/V4/r93_objective.py:18)。R93 仍保留 weight decay；其保存 eval CE 仍是 maximin CE，不是训练 CE，[r93_experiment.py:80](/public/home/shiys/0selection2/code/V4/r93_experiment.py:80)。 |
| 5 | 每个逻辑更新只含一个问题的 1024 行；18题交替。按实际 micro 行数除以逻辑行数累积全部目标，之后只 unscale/clip/step 一次。尾部数据跨 shuffle 周期保留。 | [r77_experiment.py:52](/public/home/shiys/0selection2/code/V4/r77_experiment.py:52)、[同文件:66](/public/home/shiys/0selection2/code/V4/r77_experiment.py:66)、[r84_experiment.py:83](/public/home/shiys/0selection2/code/V4/r84_experiment.py:83)。不是一次更新内对18题梯度求平均，也不是各 micro 等权。R77 所调用的旧 step 同样按真实行数缩放，[r76_experiment.py:222](/public/home/shiys/0selection2/code/V4/r76_experiment.py:222)。 |
| 6 | R81/R82 没有把跨实例目标拆成独立小 batch：先拼完整逻辑 batch 的两视图 utility，求整体 `dL/du`，再按保存 RNG 重放前向并链式回传。R85 的 EQL 权重在整次 attempt 固定，成功后才提交完整行均值状态。 | [method_auc_objective.py:195](/public/home/shiys/0selection2/code/V4/method_auc_objective.py:195)、[context_matched_objective.py:115](/public/home/shiys/0selection2/code/V4/context_matched_objective.py:115)、[r85_objectives.py:290](/public/home/shiys/0selection2/code/V4/r85_objectives.py:290)、[同文件:327](/public/home/shiys/0selection2/code/V4/r85_objectives.py:327)。重放的 utility 梯度已含全 batch 分母，不再乘一次 micro 比例。 |
| 7 | R84 及此后基线为 AdamW，LR 从成功更新数 warmup 到 1e-4 后 cosine 至 1e-6，WD=1e-4，整个逻辑梯度 norm clip=1。R89B 只把白名单隐藏 Linear 矩阵交给 Muon，其余包括最终单行读出留在 AdamW。 | [r84_experiment.py:45](/public/home/shiys/0selection2/code/V4/r84_experiment.py:45)、[同文件:359](/public/home/shiys/0selection2/code/V4/r84_experiment.py:359)、[r89_optim.py:61](/public/home/shiys/0selection2/code/V4/r89_optim.py:61)、[同文件:111](/public/home/shiys/0selection2/code/V4/r89_optim.py:111)。不是全模型 Muon；全参数 WD 未显式豁免 bias/LayerNorm。 |
| 8 | checkpoint 依据完整 VAL 的 TRAIN-fixed-SBS 成本比，严格更小才替换；不是训练 loss、Top1 或目标类召回。R93 也继承这一规则。 | [r77_experiment.py:360](/public/home/shiys/0selection2/code/V4/r77_experiment.py:360)、[r84_experiment.py:463](/public/home/shiys/0selection2/code/V4/r84_experiment.py:463)、[r93_experiment.py:96](/public/home/shiys/0selection2/code/V4/r93_experiment.py:96)。Top1 训练目标与成本选型存在对象差别，但下文已量化其在若干轨迹上的实际大小。 |

## 2. maximin 的精确数学，而非“argmax 一样所以训练一样”

### 2.1 共同平移之外，领先者还有额外加成

固定一个实例的合法候选集合，`K>=2`，记 utility 为 `u_i`。源码计算：

\[
d_{ij}=u_i-u_j,\qquad z_i=\min_{j\ne i}d_{ij}
=u_i-\max_{j\ne i}u_j.
\]

设唯一第一名为 `a`、唯一第二名为 `b`，`delta=u_a-u_b>0`，则

\[
z_a=\delta,\quad z_i=u_i-u_a\ (i\ne a),\qquad
z=u-u_a\mathbf1+\delta e_a.
\]

因此 `z_a-z_b=2 delta`；非领先者彼此的差仍是 `u_i-u_j`；领先者对其它每个方法的差多了 `delta`。完整排序及 stable global-ID tie-break 保持，但 softmax 不保持。若并列最高，`delta=0`，值上退回共同平移；并列边界的梯度由 `min` 的分支规则决定，上述唯一前二梯度公式不可直接套用。单合法候选在源码中单独返回分数0。

令 `q=softmax(u)`、`p=softmax(z)`、`D=1+(exp(delta)-1)q_a`，则

\[
p_a=\frac{e^\delta q_a}{D},\quad p_i=\frac{q_i}{D}\ (i\ne a),\qquad
\mathrm{CE}(z,y)-\mathrm{CE}(u,y)=\log D-\delta\,1[y=a].
\]

这不是 K>2 时的单一温度：只有领先者相对其它类的 odds 增强，非领先者之间 odds 不变。K=2 时恰为间隔乘2，即有效温度减半。

更重要的反向边界：**这个链接本身没有排除某些冠军概率向量。** 对任意严格正、唯一前二为 a/b 的目标概率 `p`，设 `u_a=0`，其余 `u_i=log p_i-(log p_a+log p_b)/2`，便有 `softmax(maximin(u))=p`。故其影响是参数化、平滑度/梯度和与其它目标的耦合，不是点态概率空间缺了一块。有限网络能否产生所需 utility 函数是另一问题。

### 2.2 CE 到底把梯度给谁

记 `g=p-e_y`。在前二顺序局部不变时，源码的链式梯度为

\[
\frac{\partial L}{\partial u_a}=2g_a,\qquad
\frac{\partial L}{\partial u_b}=g_b-g_a,\qquad
\frac{\partial L}{\partial u_i}=g_i\ (i\notin\{a,b\}).
\]

普通 utility CE 则为 `q-e_y`。所以“min 只给最强对手/前两名梯度”是错误的：每行都含本方法 utility，所有其它有限概率方法仍有自己的梯度。若错选 a、真实 winner y 不是 b，`g_b-g_a=p_b-p_a<0`，梯度下降还会**抬高错误的第二名 b**，同时抬高 y、降低 a。这是分段坐标变换的精确结果，不是实现 bug，也不证明长期零选择由此造成。

人工例 `u=(2,1,0), y=第三类`：

| 路径 | 概率 | CE | 对 utility 的梯度 |
| --- | --- | --- | --- |
| utility | (0.665241, 0.244728, 0.090031) | 2.407606 | (0.665241, 0.244728, -0.909969) |
| maximin，z=(1,-1,-2) | (0.843795, 0.114195, 0.042010) | 3.169846 | (1.687589, -0.729600, -0.957990) |

参数级不能只看 `dL/du`。末层是所有方法共享的 `u_i=w^T h_i+b`，故

\[
\nabla_wL=\sum_i v_i h_i,\quad \nabla_{h_i}L=v_iw,\quad
\nabla_\theta L=J_\theta(u)^T v,\qquad v=\nabla_u L.
\]

原差值目标及 utility softmax CE 都对共同平移不变，因此精确实数下 `sum_i v_i=0`，**共享末层标量 bias 的数据梯度为0**；AdamW 仍可对该参数衰减。不能把这个规范自由度误报为“候选 bias 都不学习”：solver ID embedding、方法相关 token、MLP 隐层以及共享 w 均可收到梯度；R85 的绝对 utility OVR 也不具有同一平移不变性。FP16 舍入下不要宣称逐位精确为0。

同样，大输出误差不保证有用的参数更新。若局部 `h_i` 相近，`sum_i v_i h_i` 可以抵消；整个编码器还有共享 Jacobian 的样本间和任务间抵消。**这是可能机制，不是已经测得的特征坍缩。**

### 2.3 两次前向平均与梯度作用域的两个易错点

单次排序等价不保证“先变换再平均”与“先平均再排序”等价。人工两视图 `(4,3,2,0)`、`(0,1,2,3.2)`：平均 utility 的 Top3 为 A/B/C；平均 maximin 的 Top3 为 A/D/B。R91 明确保留后者，所以该控制并未顺带更换 focus。

R85C 的 conditional nonwinner KL 对**被排除的 winner logit**直接导数为0，但不等于对 winner utility/表示永不反传。若 native winner 恰是当前第二名，它影响领先者的 maximin logit。人工 `u=(2,1,0), native y=第二类`、非 winner 目标概率 `softmax(-1,-2)`，该 KL 对 utility 的梯度为 `(0.443031,-0.221516,-0.221516)`。这是[原条件 KL 定义](/public/home/shiys/0selection2/code/V4/r85_objectives.py:77)与 maximin 链式法则的区别，不是给现有代码添加了新路径。

## 3. 复合监督、尺度兼容性与候选数

设两 dropout 视图为 v=1,2，`r_i=(C_i-C_*)/C_*`。原目标为

\[
L=\tfrac12\sum_{v=1}^2\left[
.35\,\mathrm{CE}(z^v,y)+.35\,B(d^v)
+2\min\{\sum_i p_i^v r_i,1\}\right]
+\lambda_t\tfrac12[KL(p^1\|p^2)+KL(p^2\|p^1)].
\]

`lambda_t` 在 warmup 内线性升至0.35。最后一项不是跨训练时间保持输出不变，也不是将另一视图 detach 当固定 teacher，两路都参与反传。

**pair 项的精确对象。** 对每个有效组 G，先取该实例组内 BCE 均值，再以有效的 `(.50,.25,.25)` 权重重归一化。单 pair 导数为 `sigma(u_i-u_j)-1[C_i<C_j]`，分别以正负号传给 `u_i,u_j`；group 和 Top3 选择本身不求导。它学习严格两两胜负，而 CE 学全池 native champion。K>2 时两者的条件概率不是同一统计对象。

**确有可证明的尺度张力，但不是固定 hinge margin。** 代码没有要求某个固定数值 margin，也没有固定 utility norm；它使用单位温度 logistic/softmax、固定风险单位0.01。二类、无 ties、同一表示区域中 `eta=P(Y=1|Z)=0.8` 时，maximin CE 希望间隔 `d=0.5 logit(eta)=0.693147`，pair BCE 希望 `d=logit(eta)=1.386294`。在前者处 CE 导数0、pair 导数-0.133333；R91 的 raw-utility CE 消除了这个二类尺度矛盾，但不能消除多类 champion 与 pair-probability 对象差异。

**不能夸大为监督不可兼容。** 对单个确定成本排序、足够表达力且忽略正则，winner 第一、其它方法按真实成本排序并拉大差距，可同时令 CE/pair/risk 很小。折衷主要出现在表示压缩后的条件混杂、有限共享函数类、随机正则和优化过程中，而非“真实标签逻辑互相矛盾”。

**risk 梯度不等于 Top1 梯度。** 未触发截断时，原 risk 加权项对 logits 的导数为 `2 p_i(r_i-E_p[r])`；当该实例预测期望 regret>1 时，截断分支梯度为0，不能把它误写成单方法 regret>1 就被裁剪。条件层面，若暂忽略 pair/KL/截断，最小化 `-alpha sum eta_i log p_i + beta sum p_i E[r_i|Z]` 给出 `p_i=alpha eta_i/(nu+beta E[r_i|Z])`，其 argmax 未必是 `argmax eta`。原始成本均值、逐实例 actual regret、软概率期望风险是不同对象。此推导不证明项目实际冲突强度。

**候选数依赖存在，但已部分归一化。** 冻结 NEW90k 池为 TSP8、CVRP10、ATSP5、七个非B routing题6、八个B题3。初始 CE 约 `log K`；all-pairs 先除以有效 pair 数，不会简单以 K平方放大总 loss。无成本 ties 的 K=3 时三种 focus 都是全池，比较目标退化成同一个均值；K=10 时非focus pair 仅有 `.25/45` 的权重，focus pair 另获组权重。故相同全局系数不意味着各方法/各任务梯度同等强，但不能仅由此宣称大池受损或小池主导。

**形式表达约束也要分清轴。** utility 差矩阵满足 `d_ij+d_jk=d_ik`，矩阵秩至多2，不能自由表示循环的条件 pair odds；然而每个实例的任意候选都能成为最大 utility，且该 KxK 差矩阵的低秩不是“跨实例x候选分数矩阵”低秩的证明。R43B 已试过更自由的显式 pair decoder，却在旧池单seed中没有战胜 utility 差版本，削弱简单的“解除差值约束就够了”，不排除其它表示。

## 4. batch、AMP 与优化：已核实什么，没核实什么

| 项目 | 实际执行口径与证据边界 |
| --- | --- |
| R77预算 | 49,680 成功更新、每题2,760；warmup3,726，VAL每1,242。A/B同预算，同1024逻辑行；各自 TRAIN 统计，不是同实例遍历次数。 |
| R84及后续主干预算 | 57,600 成功更新、每题3,200；warmup4,320，VAL每1,440；无基于VAL的早停/LR更改。固定预算完成不是收敛证明。 |
| 实际microcap | 保存协议中 R84A、R85A 的 TSP/CVRP 为384；R77B、R81B、R82B、R89B、R91A、R93A、R95B为352；其它题1024。不可把所有历史都写成352。 |
| 行均值 | 352/352/320三块应权重352/1024、352/1024、320/1024。人工验证完整梯度=5.1875，与正确累积一致；错误等块均值=5.333333。源码走前者。 |
| AMP | 主要模型前向在 CUDA autocast FP16，utility/margins和CE/BCE/risk/KL在FP32算。不是整个数学计算FP64，FP64是原始标签来源。原成本差可能很小而标签仍稳定，不能据此换成FP32 argmin。 |
| overflow | 恢复逻辑batch起点RNG，清梯度，降低scale并重放；成功计数不推进，不接受半个batch。R85失败 attempt 也不提交状态。保存 history 的最终累计 retries：R84A25、R89B26、R91B25、R93A/B26，不能把启动期0重试外推全程0。未在本轮重跑训练审计。 |
| optimizer | AdamW用实际Torch实现及默认betas/epsilon；调用点未覆盖它们，本轮未读取当时已安装Torch实现逐字认证其值。WD=1e-4作用于传入的全部参数。norm1裁剪发生在unscale后、预条件前；并不限制AdamW或Muon实际参数位移的norm为LR。 |
| Muon | [r89_optim.py:99](/public/home/shiys/0selection2/code/V4/r89_optim.py:99)：动量0.95、Nesterov混合、5次BF16 Newton-Schulz，方向乘`0.2 sqrt(max(rows,cols))`；参数/动量FP32，外层共同LR和WD。隐藏矩阵55个的更新记录不能代表全部参数换了优化器。 |
| 目标函数日志 | R84A最后区间优化loss0.382282、grad_norm均值0.229195；R93A/B纯CE最后0.685226/0.676295，grad_norm0.437629/0.531622，末LR均为1e-6。它们不是同一个loss，不能用数值大小比较优劣。grad_norm是裁剪前逻辑梯度的区间平均，不能据此恢复裁剪比例。 |

持续正常更新只排除“完全没有执行更新”的粗糙解释。非零梯度可能来自 minibatch 波动、其它任务/其它类别或同一已有特征的置信度变化；经过裁剪、动量、预条件、很小末期LR后，关键实例的分数差不一定越过0。反之，末期不越过边界也不证明全是优化问题。没有逐项梯度夹角、Jacobian有效方向、条件亚群学习轨迹及合适的类内最优对照，本轮保留未验证。

### R81 / R82 / R85 / R95 的具体排除边界

R81用 `a_m=u_m-logsumexp_{j!=m}u_j`，对同方法的 unique-winner 正例和严格非winner负例作跨实例 `softplus(a_negative-a_positive)`，按 TRAIN eligible 方法平均；系数0.35随warmup升起。[目标定义](/public/home/shiys/0selection2/code/V4/method_auc_objective.py:58)。它是同类跨样本排序，不是冠军概率校准或同实例跨类决策的一致性认证。

R82在相同 TRAIN context 组内，对每方法按连续 FP64 regret 差排序，pair 权重 `min(abs(delta_regret)/0.01,1)`；依次平均 pair、context、有效方法，系数0.10。[目标定义](/public/home/shiys/0selection2/code/V4/context_matched_objective.py:34)。它保留原复合目标；不等于把条件冠军概率完整拟合。

R85A/B仅把0.35 CE换成 utility OVR，保留pair/risk/KL。OVR按 `log(K)/(K log2)` 缩放各合法方法BCE；A自然权重，B按累计正负绝对梯度产生冻结的EQL式权重。C是条件非winner cost-profile KL；D加合法子池CE。都不是“所有不平衡学习/成本监督已被排除”。

R95在前7,200更新后，A均匀有放回、B按 `q_i=.5/N+.5 h_i/sum h` 抽样，以 `1/(N q_i)` 乘**整个逐行复合loss**并取行均值，不除以weight总和。[抽样](/public/home/shiys/0selection2/code/V4/r95_sampling.py:34)、[加权](/public/home/shiys/0selection2/code/V4/r95_objective.py:86)。`h_i`是旧TRAIN评估时最后共享affine层的完整确定性梯度norm，不是当前全网络/dropout梯度；它先按方法求和，保留内部抵消，[proxy](/public/home/shiys/0selection2/code/V4/r95_objective.py:126)。给定冻结q，校正保证预裁剪经验梯度期望不变；clip/AdamW等非线性操作后更新期望不必相同。这个有限收益不能排除所有曝光或优化机制。

## 5. 真实 winner 频率、平均概率和选择区域是三个量

定义 `f_m=mean 1[Y=m]`，`pbar_m=mean p_m(X)`，`a_m=mean 1[argmax_j p_j(X)=m]`。前两者接近不要求第三者接近；classwise calibration 也不保证每类有最大后验区域。

以下直接读已保存 R77B 34776步的[VAL校准表](../../../code/V4/runs/R77_sample_size_batch1024/train_val_selection_diagnosis/R77_B_full_bs1024/class_calibration_val.csv)，每题3,895例；概率是 maximin 的温度1 softmax，不是 utility softmax。不能搬作 NEW90k 的当前数值。

| 问题/方法 | true winner数 | winner比例% | 平均概率% | 实际选择数 | classwise ECE20，百分点 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSP/DIFUSCO | 330 | 8.4724 | 8.8075 | 0 | 0.4458 |
| TSP/LEHD | 641 | 16.4570 | 15.4223 | 0 | 1.5507 |
| ATSP/MATPOENET | 318 | 8.1643 | 5.9249 | 0 | 2.3191 |
| VRPL/MTPOMO | 137 | 3.5173 | 3.6945 | 0 | 0.1772 |
| OVRPB/MVMOE | 234 | 6.0077 | 3.3125 | 0 | 2.7526 |

同一模型TRAIN：TSP/DIFUSCO有6,084/70,114个winner、平均概率8.8432%、选择0；LEHD有11,881个winner、平均概率15.4138%、选择4。源为相邻 `class_calibration_train.csv`。说明零选择并非仅在留出集出现；不说明网络已经学到精确条件后验。

逻辑反例：每个可见表示区域都具有真实后验 `(0.45,0.35,0.20)`，模型精确给出该后验，则校准正确、平均概率等于类别频率，后两类仍永不argmax。**这仅证明频率与区域不能互推，不是对项目的分布估计。**

## 6. 历史对照能排除什么

NEW90k表中 actual regret 为百分数，沿用现有报告定义；斜线两值按所列臂顺序。固定36组来自R84A VAL，不能将其与R77目标组混合。

| 版本/实际变化 | VAL Top1 % | TEST Top1 % | TEST actual regret % | 可支持的有限结论 |
| --- | --- | --- | --- | --- |
| R84 A原始 / B直接attention | 71.0852 / 71.0000 | 71.2358 / 71.2235 | 0.394847 / 0.394592 | 该直接评分替换没突破；同时改变读出结构，非纯链接消融。 |
| R84 C加focal / D attention+focal | 71.0235 / 70.9543 | 71.2716 / 71.2173 | 0.397181 / 0.395155 | 测的是原CE之外再加focal，不是单独focal替换全部监督。 |
| R85 A自然OVR / B梯度平衡OVR | 70.9679 / 70.9358 | 71.2321 / 71.2111 | 0.397280 / 0.394872 | EQL式自然类梯度再平衡没有整体突破；不是自然梯度/所有不平衡方法的排除。 |
| R85 C条件cost-profile / D候选子池 | 71.0086 / 71.1272 | 71.3111 / 71.4370 | 0.389713 / 0.390487 | 存在成本/Top1取舍，不能简单全部称负面；未满足既定联合筛选。 |
| R89 A AdamW / B Muon hybrid | 71.0506 / 71.2481 | 71.2321 / 71.4198 | 0.394236 / 0.389646 | 匹配优化器对照有小正收益，反驳“优化器完全无关”；B固定36组仅45/4598命中、50误选。 |
| R91 A utility CE / B全部utility概率 | 71.0728 / 71.0914 | 71.2938 / 71.4086 | 0.395084 / 0.398966 | 两者仍有pair；B固定组26/4598命中、27误选。不能将链接修正宣称充分。 |
| R93 A纯CE dropout0.1 / B纯CE dropout0 | 71.1259 / 71.1099 | 71.3765 / 71.3062 | 0.402214 / 0.401964 | 无pair/risk/KL也有平台；各57/4598命中、73/75误选。去掉这些项不是足够方案；未排除共同编码器/优化/数据对象问题。 |
| R95 A均匀IID / B重要性抽样 | 71.0000 / 71.0988 | 71.2494 / 71.2543 | 0.392933 / 0.395211 | 此旧末层proxy的校正曝光没有整体突破；不是完整梯度方差机制的排除。 |

数值源为前述 history CSV 所逐行引用的现有 RESULTS/target-recovery 文件，未本轮重新推理。R89/R91/R93/R95在TSP沿用R88B physical-node0 role；R84A为较早参考。R91/R93没有新训的原loss匹配臂；R93两臂还不是逐位硬件匹配。R93将CE系数从0.35改为1、两次前向改为一次，也改变梯度尺度与随机数流；不能把它说成完全保持优化轨迹的逐项删除实验。不能从这些历史差值作单变量、跨seed因果定论。

旧 NEW70114：R77匹配10k/70,114 TRAIN的TEST Top1为70.5691/71.2252，支持该条件下增大独立样本的收益；R81 AUC的VAL/TEST为70.8558/71.1510，R82 CMR为70.9314/71.2780，二者无新训匹配A。R80 CE-only的TEST71.2266、batch512的71.3151、峰值LR3e-4的71.1753，分别有预算/参考边界，不能将这些读成“所有CE、batch与LR机制均已被排除”。旧LEGACY的R65 ScheduleFree有匹配正收益，R68 SAM没有；R69独立任务银行也无突破，均不是NEW90k全类别可学性的结论。

### 成本选型能解释多少平台？

本轮只从保存的40个VAL点比较最低成本所选步与轨迹最高Top1，不重新选模型，不评估其它checkpoint的TEST：

| 模型 | 原成本所选步 / VAL Top1% | 保存轨迹最高Top1步 / VAL Top1% | 差距pp |
| --- | --- | --- | ---: |
| R84A | 37440 / 71.085185 | 37440 / 71.085185 | 0 |
| R89B | 36000 / 71.248148 | 36000 / 71.248148 | 0 |
| R91B | 40320 / 71.091358 | 40320 / 71.091358 | 0 |
| R93A | 57600 / 71.125926 | 50400 / 71.167901 | 0.041975 |
| R93B | 37440 / 71.109877 | 44640 / 71.140741 | 0.030864 |

源为各run `history.json` 的 `.val.macro`。**选型对象不同是真事实，但这几条实际轨迹不支持“换Top1 checkpoint就能解释整个平台”。** 只观察保存步，不能排除未记录中间态；也不能把VAL最高值当盲测性能。

## 7. 仍值得区分的三个机制，不升级为根因

| 假设 | 现有支持与反驳 | 尚缺的区分证据 |
| --- | --- | --- |
| 条件混杂与特征形成不足，而非标签类别没有误差梯度 | TRAIN也零/近零选择；大输出误差可以在共享Jacobian抵消。R93说明复合loss不是必要条件。 | 当前表示是否已有可泛化的winner区分方向；若有，是否读出/优化没用上。小gap和非零梯度均不能回答。 |
| 固定共享函数下不同统计对象与尺度的优化折衷 | 二类CE/pair尺度矛盾可精确证明；成本目标与冠军目标也可不同。R91/R93明显限制其作为唯一主因。 | 各目标在同一参数状态、同一批次的有效更新方向和目标组分数差贡献；loss标量大小不是梯度冲突证据。 |
| 任务交替、预条件和有效步长的动力学限制 | 每步单题且共享动量，18题平均曝光不等于18题平均更新；R89有小正收益。 | 关键方向是否因有效步长/曲率而长期未改善；还是该函数类在当前表示上已经没有可泛化增益。没有这些证据，不再建议换一个优化器名字或扫系数。 |

共同平台还见于非本次selector SGD的冻结ICL，故不能用同一个当前AdamW故事覆盖所有历史模型。以上三个机制有交叉，不是互斥诊断结果。

## 8. 两个候选的审慎评价：均不主推

### 8.1 winner/runner-up joint：不等价于历史，但只是低优先的细分监督

令 `Y` 为保存的native winner，`R`为第二名。合法有序事件 `m!=n` 的logit为 `z_mn(X)`，统一softmax得 `pi_mn`，最终

\[
P_m=\sum_{n\ne m}\pi_{mn},\qquad \hat Y=\arg\max_mP_m,
\quad L_{joint}=-\log\pi_{Y,R}.
\]

**历史去重结果：四者均非同一个训练目标，但R71/R85C是实质近邻证据。**

| 历史 | 实际统计对象/代码 | 与joint-top2的关系 |
| --- | --- | --- |
| R43B explicit pair | 反对称pair margin，每对独立方向打分，再maximin；沿用champion CE、pair BCE、risk。见[pairwise_selector.py:37](/public/home/shiys/0selection2/code/V4/pairwise_selector.py:37)。 | `i胜j`不要求两者分别全池第一/第二；没有K(K-1)互斥事件的联合归一化，也没有按winner行求和。不是等价已试。 |
| R71 joint cost mixture | 八分量全协方差成本差密度，NLL/(K-1)训练；固定Sobol积分全池冠军频率。见[r71_model.py:121](/public/home/shiys/0selection2/code/V4/r71_model.py:121)、[同文件:130](/public/home/shiys/0selection2/code/V4/r71_model.py:130)。 | 理想完整成本分布当然足以推出top2联合分布，因此并非“历史从未联合建模/从未边缘化”。但实际训练的是连续密度和有限混合，不是离散top2 CE；其负结果不能直接否定新目标，也削弱“更细联合输出自然更好”。 |
| R85C cost profile | 排除native winner，在非winner上以`softmax(-regret/0.01)`监督条件KL；保留原复合目标。[r85_objectives.py:77](/public/home/shiys/0selection2/code/V4/r85_objectives.py:77)。 | 已利用runner附近和其它非winner相对优劣，不是全空白；但没有显式`P(Y=m,R=n|X)`或冠军边缘化。额外竞争信息有限收益，是必须面对的近邻证据。 |
| R100 pair context | 独立严格pair二分类上下文，包含第三方法冠军的实例；固定mean-win/Borda。[r100_model.py:75](/public/home/shiys/0selection2/code/V4/r100_model.py:75)、[同文件:161](/public/home/shiys/0selection2/code/V4/r100_model.py:161)。 | pair边际不确定联合冠军/第二名事件。不存在这里的joint likelihood；不等价。 |

可复算的非等价例：排名分布 `ABC/CBA` 各半与 `BAC/CAB` 各半，两两胜率全为0.5，冠军分布分别 `(0.5,0,0.5)` 与 `(0,0.5,0.5)`。因此pair+Borda不能凭定义还原joint-top2。

**关键恒等式并不会因“非因式化logits”失效：**

\[
q_{n|m}=\pi_{mn}/P_m,\qquad
-\log\pi_{y,r}=-\log P_y-\log q_{r|y}.
\]

对legal joint logits，

\[
\partial L_{joint}/\partial z_{mn}=\pi_{mn}-1[(m,n)=(y,r)],
\]
\[
\partial L_{winner}/\partial z_{mn}=\pi_{mn}-1[m=y]q_{n|y},
\]

差值恰为真winner行的 `q_{n|y}-1[n=r]`。数值例joint行概率分别 `(0,.30,.29)`、`(.40,0,.005)`、`(.003,.002,0)`：`Ljoint=1.203973 = Lwinner 0.527633 + Lconditional 0.676340`。最大joint格属于B，而正确winner边缘化选A，不能把joint MAP当winner MAP。

任何正joint在概率上都可以因式分解。若**参数化**为独立winner头`a_m`与条件头`b_mn`，并令 `z_mn=a_m+b_mn-logsumexp_n b_mn`，则边缘概率精确退回`softmax(a)`；新增内容只是共享表示上的runner-up辅助监督，属于R54式辅助任务思路而非新决策路径。无约束joint logits的winner logit则是`logsumexp_n z_mn`，改变了读出计算，但NLL依然是上述两项之和。

它不等于“提高softmax矩阵rank”：任意原winner概率P都能用`pi_mn=P_m/(K-1)`表示，边缘仍是原P；普通足够宽MLP也可表达多模态。低维或满秩都不保证候选覆盖。联合方案可能改变有限参数归纳偏置，**不保证增加实际可用表达力、Bayes Top1或优化可达性**。

**风险与标签约定：** y必须保留native `ind`，先核验其对应合法FP64最小成本，不以新的排序规则覆盖；runner从排除y后的FP64成本最小者取值，精确ties必须事前锁定规则。runner第二/第三名的微小差和exact ties可能制造与冠军判别无关的细分；不得按TEST调容差。K=2时runner由winner完全决定，无新增监督；K=3时top2已定义整个排列；本项目K=3/5/6/8/10对应6/20/30/56/90细类，罕见joint类只会更稀。额外条件熵、共享梯度和拟合预算可以反伤winner，不能因新增标签是“真实成本产生”就排除负迁移。

若未来仍研究，能明确的单一计算候选是：只以input-derived方法token对构造无反对称约束的pair MLP `z_mn=f(h_m,h_n,h_m-h_n,h_m*h_n)`，全合法有序pair归一化，训练自然joint CE，部署严格winner边缘化。最必要对照是**完全同一个joint头/容量/初始化**仅训练`-log sum_n pi_y,n`，以区分新增runner监督与logsumexp读出本身，而不只是拿旧R93作非匹配参考。本轮不实现、不安排训练，也不把该候选当论文主线。

Feature Averaging论文的一手页面说明其定理对象是特定多簇分布、两层ReLU和对抗鲁棒性；细粒度标签的正结果不等于本项目runner身份就是可辨认特征簇，也不保证自然Top1提高。本支线不扩展该定理。[原文](https://arxiv.org/html/2410.10322v2)

### 8.2 conditional density-ratio：与CE同理想决策，低优先的优化假设

对由允许输入确定的组g，正例来自`P(X,Y|g)`，负例来自`P(X|g) pi_train(m|g)`，正负比1:1。若把冻结TRAIN经验分布作为目标总体、且相关类有正prior，最优二分类logit为

\[
t^*(x,m,g)=\log\eta_m(x,g)-\log\pi_m(g),\qquad
s_m=\log\pi_m+t_m,\quad \hat y=\arg\max_m s_m.
\]

这是密度比恒等式，不是自然标签先验已准确估计的保证。实际有限TRAIN先验有误差，稀疏g或零计数需事前定义有支持的估计；不能删除合法候选、用TEST补prior或按TEST改阈值。

给定x,g，忽略共同因子，其总体BCE为

\[
L_{NCE}=-\tfrac12\sum_m[\eta_m\log\sigma(t_m)
+\pi_m\log\sigma(-t_m)],\quad
\partial L/\partial t_m=\tfrac12[(\eta_m+\pi_m)\sigma(t_m)-\eta_m].
\]

因此最优`s=log eta`，与自然winner CE有相同Bayes argmax，但有限样本梯度不是CE的`p-e_y`。**如果仅把CE的logits写成`log pi+t`，自由t下就是旧CE的prior-offset重参数化；只有实际使用上述product-noise BCE/抽样才改变训练目标。** 这种改变仍主要属于学习分工、梯度曲率和负采样，并未提供新的query信息。

与历史的关系：R81在同方法winner/nonwinner之间作AUC排序，不能识别类间校准offset；R82按context regret作排序，非联合分布对product的判别。R85 OVR的负项权重是`1-eta_m`，本NCE是`pi_m`，也非严格相同，但二者都属于逐方法sigmoid监督。R66学习已有模型F的输入相关残差，R78含冻结分数上bias拟合；TRAIN先验残差与它们不逐字等价，却不能仅凭“先验+残差”声称新机制。旧CE/OVR/校准/残差的有限结果要求更强机制证据，而非仅另换proper surrogate。

负采样必须允许负对恰好抽到当前真实y；排除它会改变product-noise分布，上式不再成立。按自然pi抽负类使同方法正负总频率在组内相等，但总曝光仍与pi成比例，**不是稀有winner自动获得更多训练机会**。`t=0`初始化只恢复prior选择；更容易学会输入相关密度比仍是假设。未收敛时`sum exp(s)`也不必为1，不应把判别器sigmoid直接当冠军概率。

评价：可精确定义且不等价于R81/R82；若只有offset则是旧方法重参数化，若加NCE则是不同surrogate，但没有证据优先于已试CE/OVR。与joint-top2一样保留为低优先，**本支线没有足够证据推荐一个新的高优先损失或优化方案**。

## 9. 本轮轻量验证与交付

[loss_math_checks.py](loss_math_checks.py)仅调用人工CPU张量和源码纯函数`maximin_logits`，没有模型对象/数据/权重/optimizer step。最终运行退出码0，通过：三种真实标签下maximin CE梯度；1,215个含ties/合法mask的排序案例；目标概率逆构造；两视图Top3反例；二类尺度冲突；共享affine bias消去；条件KL间接梯度；micro行权重；joint NLL及梯度分解；pair边际不确定冠军。

首次运行因验证脚本对当前Torch的`argsort`使用位置参数而停止，已仅在本脚本改为`dim=`并完整复跑通过；这不是项目训练源码错误。

复算命令：

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -B docs/research/deep_direction_review_v2_20261023/loss_math_checks.py
```

**供主报告引用的一句话：分类边界的点态等价，不等于训练概率、梯度或有限模型优化等价；但这些可证明差异也不是71%平台的因果诊断，R91/R93已否定“修一个链接或去掉复合项就足够”的强解释。**
