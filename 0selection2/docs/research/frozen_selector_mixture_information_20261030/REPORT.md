# 冻结完整选择器的条件集成：支持域、统计目标与三种科学表述

**日期：2026-10-30。历史资料基准为2026-10-28；当前工程事实仅采用本轮Sol给定信息。**

科学综合请求 `gpt-6-astra/max`；接口返回身份见 [ROUTING.json](/public/home/shiys/0selection2/docs/research/frozen_selector_mixture_information_20261030/ROUTING.json)，非独立后端认证。

本轮已读取指定的 [HANDOFF.md](/public/home/shiys/0selection2/docs/research/post_r119_global_reassessment_20261028/HANDOFF.md) 与 [历史REPORT.md](/public/home/shiys/0selection2/docs/research/post_r119_global_reassessment_20261028/REPORT.md)，并联网打开原论文核对引用；Astra依据这些历史全文、来源记录及原文摘录完成科学综合。未读取实验代码、结果或D1采集目录，未核验运行。新接口不是现有代码事实；旧快照仅作历史。下文的信息、支持域与风险关系均明确作为**本文推导**，不冒充文献或实验结论。

## 一、事实与问题定位

据给定记录，R123冻结完整R77B/R80A/R83，新建graph4layer128原图门，以三个独立二元正确性Brier目标训练；全对时目标均为1，全错时均为0。门不见专家输出，只执行选中的选择器并输出其全池排名。TRAIN来源记录未见交集每题10,886，属于条件队列，并非内容或同base新颖性认证。正式full18 TEST为Top1 **71.30%**、regret **0.39487%**；冻结R77B约为71.26%／0.39444%，变化可忽略。R124是独立full18 Top2专家，71.33%，纠正879、误伤823；R122硬成本／漏组惩罚更差。这些记录未经本轮导入复查，亦非匹配因果比较。

**答案是有条件地可以：冻结意见可能把难计算预先完成，但不会凭空增加当前查询信息，也不保证元学习更容易。** R123的近零变化约束的是“原图→专家正确率→硬选择”，尚未直接回答“完整专家意见是否提供较易解码的条件偏好”。

## 二、信息与目标：不能把三个问题混成一个

令 \(X\) 包含任务及合法输入，\(Z\) 为冻结完整选择器的全池意见，\(H\) 为可选冻结表示。候选ID和有限合法mask必须对齐；这里的分数、概率及表示只是研究接口。仅有排名不足以定义软混合；下文的概率向量是规范化预测接口，不把现有分数自动视为校准概率。

给定固定专家权重 \(\Theta\)，有：

\[
I(Y;Z,H\mid X,\Theta)=0.
\]

因此，意见可能相对压缩原图门特征增加可用信号，或降低重新计算结构的样本／计算负担，却没有增加超出完整 \(X\) 的查询信息。若元模型仍须从高维表示重新发现全部优势规律，原来的困难就会回来。

所有表述的前向仅允许合法输入、冻结输出／表示，以及**排除当前查询和同base拷贝的TRAIN上下文**；不读取当前真winner、成本或求解结果。执行多个选择器是额外推理，不是执行多个solver，也不免费；元计算后仍只部署一个solver。

目标语义分别为：

\[
\begin{aligned}
Y&=\text{native/FP64保存的ind，不以FP32 argmin替代},\\
r_a&=(C_a-C_*)/C_*,\qquad
\mathbb E_{18}=\frac1{18}\sum_{t=1}^{18}\mathbb E_t,\\
a_{\rm acc}(U)&=\arg\max_a P(Y=a\mid U),\\
a_{\rm reg}(U)&=\arg\min_a\mathbb E[r_a\mid U].
\end{aligned}
\]

其中 \(C_* = \min_{a\in\mathcal A_t} C_a>0\) 是同一实例合法候选池的Oracle成本；\(U\) 只含允许的前向信息，\(C\)、\(Y\) 只作监督或历史TRAIN上下文。

自然winner准确率与距离regret一般不同，不能混合任务raw cost。正winner频率也不要求正最优动作频率：若给定合法信息的后验恒为 \((0.6,0.3,0.1)\)，后两类可以永不成为最优Top1动作。这只是概念反例，**不是本项目的Bayes上限或随机标签结论**。

R123的独立Brier并非天然错误。对 \(b_k=1[a_k=Y]\)，平方损失的条件最优预测正是 \(P(b_k=1\mid U)\)，用于硬专家选择完全自洽；它不是互斥类别分布，也不是软混合贡献概率。全错目标相同丢失了错法之间的排序和风险差别，但硬门本来也救不了全部专家Top1都错的查询。严格proper所保证的是其指定预测对象，不是任何经验评分改善都会改变动作。[S07](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)

## 三、支持域：软混合既比硬路由宽，也有精确边界

以下均为**本文推导**。固定查询，令 \(z_{ka}\) 为专家 \(k\) 对候选 \(a\) 的分数，\(v_a=(z_{1a},\ldots,z_{Ma})\)。硬路由动作支持仅为各专家实际Top1的集合；随机抽专家也不会扩大它。共同凸池则为：

\[
s_a=\alpha^\top v_a,\qquad \alpha\in\Delta_M.
\]

**构造例一：从未成为专家Top1的候选可以成为混合Top1。**

\[
\begin{array}{c|ccc}
 &A&B&C\\\hline
p_1&0.50&0.10&0.40\\
p_2&0.10&0.50&0.40\\
(p_1+p_2)/2&0.30&0.30&0.40
\end{array}
\]

两位专家都不选C，平均概率却选C；若真winner为C，软池能救回硬路由不可能救回的查询。

**构造例二：没有单个对手统一支配，仍可能完全不可达。**

\[
p_1=(0.30,0.60,0.10),\qquad p_2=(0.30,0.10,0.60).
\]
\[
v_A=(0.30,0.30),\quad
\tfrac12v_B+\tfrac12v_C=(0.35,0.35)>v_A.
\]

B、C各自都没有在所有专家坐标上压过A，但它们的凸组合做到了。任何共同权重下，至少一个对手高于A；输入相关权重也绕不过这个逐查询限制。

令 \(b\prec a\) 表示同分时b优先。**实际输出a的精确可行判据**是存在同一组权重满足：

\[
\alpha\in\Delta_M,\qquad
\alpha^\top(v_a-v_b)
\begin{cases}
>0,&b\prec a,\\
\ge0,&a\prec b,
\end{cases}
\quad \forall b\ne a.
\]

只问能否进入argmax集合时全部用非严格不等式；要求唯一第一时全部严格。凸组合对手的完整障碍还可写成：

\[
\gamma_a=
\max_{\alpha\in\Delta_M}\min_{b\ne a}\alpha^\top(v_a-v_b)
=
\min_{\beta\in\Delta_{\mathcal A\setminus a}}
\max_k\left(z_{ka}-\sum_{b\ne a}\beta_bz_{kb}\right).
\]

这是有限线性极小极大关系：\(\gamma_a\ge0\) 当且仅当允许并列时可达；\(\gamma_a>0\) 当且仅当可唯一胜出；\(\gamma_a<0\) 当且仅当某个对手凸组合在每个专家坐标严格胜过a。仅弱支配时必须保留同分分析，不能一概删除候选。

这是闭单纯形的表达上限，受限权重模型可能更窄。利用 \(Y\) 选择可行权重的是不可部署oracle；**凸包内存在，不等于可从合法意见学得。**

## 四、三种实质不同的表述

### 1. 条件预测池：学习“何种意见形态下相信谁”

统计对象是条件预测分布。信息流为：

\[
X\rightarrow\{E_k^{\rm frozen}\}\rightarrow(t,Z)
\rightarrow\alpha_\theta(t,Z)\rightarrow q\rightarrow\arg\max q.
\]

TRAIN监督是native \(Y\) 上的类别CE或类别Brier，不是将专家正确率归一化。条件权重可以部分汇聚，表达稳定共享规律与局部偏好；相同专家意见使权重不可辨识，但预测分布仍可能有意义。层次stacking提供这种统计组织方式，不保证本项目收益。[S05](https://arxiv.org/pdf/2101.08954)

概率池与logit池必须分开：

\[
q_a^{\rm prob}=\sum_k\alpha_kp_{ka},
\qquad
q_a^{\rm logit}=\operatorname{softmax}_a\!\left(\sum_k\alpha_k\ell_{ka}\right)
=\frac{\prod_kp_{ka}^{\alpha_k}}{\sum_b\prod_kp_{kb}^{\alpha_k}},
\quad p_k=\operatorname{softmax}(\ell_k).
\]

前者在概率凸包内，后者是归一化几何池，一般不在同一个概率凸包；上一节分别适用于各自的分数坐标。混合不自动保持校准，但固定正权分布池的过离散定理也不是所有动态多类门失败的定理。[S06](https://arxiv.org/pdf/1106.1638) 总体、无约束预测空间中的严格proper一致性不能直接覆盖受限凸包和有限优化。[S08](https://jmlr.org/papers/volume17/14-294/14-294.pdf)

**反证含义：** 若关键winner几何上不可达，否定“共同凸池足够”；若可达却意见无法稳定对应所需权重，削弱“条件偏好容易学”的假设；只有log-score改善不支持Top1收益。S05的分离、已知区域及对数分数条件不能省略。

### 2. 广义stacking：重新判别候选，而非分配专家份额

计算类改为：

\[
X\rightarrow(Z,H)\rightarrow F_\theta(t,Z,H)\rightarrow q,
\qquad
s_a=\sum_k w_{ka}p_{ka},\quad q=\operatorname{softmax}(s)
\]

其中线性式只是类特定例子，通用 \(F\) 还可利用跨候选关系。\(w_{ka}\) 不再是所有候选共用的单纯形权重；分数本身也不自动是概率。原始stacking并未限制为共同凸池，Ting–Witten的多类回归同样使用类特定系数，因而这里不是全新广义技术。[S01](https://cafri-labs.github.io/lab-manual/papers/wolpert1992.pdf) [S03](https://arxiv.org/pdf/1105.5466)

TRAIN目标是 \(P(Y=a\mid t,Z,H)\)。同样的意见可经更自由的映射走出原凸包，扩大的是函数类，不是输入信息。\(H\) 只有包含尚未编码进 \(Z\) 的相关信号时才可能有增量；即便 \(I(Y;H\mid Z)>0\)，也不保证有益argmax发生变化，更不保证降低代价。

**反证含义：** 更自由的模型若只改善TRAIN或概率拟合，表示差异却不能稳定对应纠错，就不支持“预计算降低负担”。R105/109/110的门控及私有表示历史限制了“加表示必有效”的解释；这里的区别必须是完整历史决策意见的复用，而非又增加随机表示容量。

### 3. 意见条件的行动风险：直接学习每个动作的损失

统计对象改为逐动作条件均值，而非winner后验或联合成本密度。以意见profile连接合法TRAIN上下文：

\[
U=(t,Z,\mathcal C_{\rm TRAIN}(t,Z)),\qquad
X\rightarrow Z\rightarrow U\rightarrow \hat m(U)\rightarrow\arg\min_a\hat m_a(U),
\]
\[
m_a(U)=\mathbb E\!\left[\frac{C_a-C_*}{C_*}\middle|U\right].
\]

上下文绑定历史完整损失、winner与mask，排除当前／同base记录；当前查询仍没有成本观测。训练对象是完整损失向量的条件均值，也可等价写成相对合法基线 \(b(U)\) 的差：

\[
\Delta_a(U)=\mathbb E[r_a-r_{b(U)}\mid U],\qquad
\arg\min_a\Delta_a=\arg\min_a m_a.
\]

必须先逐实例归一化再取条件期望，不能换成均值比。固定类代价矩阵 \(L(a,Y)\) 不包含实例成本幅度，无法替代完整TRAIN成本；成本矩阵校准与0–1校准本来也不同。[S09](https://arxiv.org/pdf/1603.00126)

R47／R97–100相关历史已包含强检索、上下文与成对比较，不能称首次上下文学习；META-DES也使用历史局部正确性和相似输出profile，但其相关元样本不是额外独立样本。[S04](https://arxiv.org/pdf/1810.01270) 与R71的联合成本密度相比，此处只学习决策所需均值。SPO提供决策损失视角，其一致性条件不能直接移植。[S10](https://arxiv.org/html/1710.08005v5)

**反证含义：** 相近意见若没有可迁移的风险结构，或均值拟合改善却没有改变错误argmin，机制便受削弱。regret下降而native Top1下降是目标取舍，不是共同突破，也不能靠选择频数配额消除。

## 五、如何解释历史，而不是重新命名历史

三种表述分别定位于**凸包内条件权重、凸包外候选再判别、行动风险统计量**，不是三个门网络变体，也不要求叠加。R124的纠正／误伤记录没有证明这些信息能提前区分两者；R122也不能被解释成所有成本风险目标均无效。

这不同于把R78式校准再做一遍：校准不能补回丢失的信息，评分进步也未必推动正确边界。R115的响应迁移失败仍是约束；历史已经指出，响应预测成功未必转为选择收益，seed受控的一次实现差不等于条件期望响应。本报告不把这些警告改写成新发现。

文献中的排除样本或交叉验证level-one构造，不能直接等同于当前来源记录未见交集，更不能推出同base独立性。[S01](https://cafri-labs.github.io/lab-manual/papers/wolpert1992.pdf) [S02](https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf) 历史选型／开发的TEST暴露，以及条件队列向自然18任务总体的迁移，仍限制泛化解释；这里将其作为假设边界，不安排审计或重训专家。

科学上最合适的判断不是“混合必胜”，而是：**完整意见可能预计算可迁移的优势结构；几何可达性、条件可学习性与目标一致性必须分别成立。** 不满足其中一项，都不能由另两项补证。

## 六、来源与适用边界

以下编号对应本轮实际打开的原论文，完整记录见 [sources.csv](/public/home/shiys/0selection2/docs/research/frozen_selector_mixture_information_20261030/sources.csv)。S01为原刊扫描镜像；S07仅核对§1，后文定向抓取超时，未用其后文作证据。检索是主题性研究，不声称穷尽。

| 来源 | 精确位置 | 支持命题 | 限度 |
|---|---|---|---|
| [S01 Wolpert, *Stacked Generalization*](https://cafri-labs.github.io/lab-manual/papers/wolpert1992.pdf) | §§2.2–2.4；Fig.4；p.242脚注2 | 在排除实例预测上学习level-one；不限共同凸池 | 单模型也可能胜过stacking；无普适提升保证 |
| [S02 Breiman, *Stacked Regressions*](https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf) | §§1–2；定理1，p.52 | 加和一约束时风险为 \(\alpha^\top R\alpha\)；单一k最优 iff \(R_{kk}\le R_{ik}\ \forall i\) | 非负权不等于凸权；固定平方误差几何不是动态Top1定理 |
| [S03 Ting–Witten, *Issues in Stacked Generalization*](https://arxiv.org/pdf/1105.5466) | §2、§2.2；§3表3–4 | 完整概率意见；类特定线性系数 | 特定数据与学习器；输出分数非自动校准概率 |
| [S04 Cruz等, *META-DES*](https://arxiv.org/pdf/1810.01270) | §§3.2.2.1–3.2.3；算法1；表1、图2 | 正确性元目标；邻域、输出profile与置信度特征 | 元样本相关；硬投票不能选无人投票类别；不移植其阈值 |
| [S05 Yao等, *Bayesian Hierarchical Stacking*](https://arxiv.org/pdf/2101.08954) | §§2.3–2.4，式11–14；§3.2定理1–4 | 条件、部分汇聚权重及log-score理论 | 定理1是目标值近似，非正确率权重等式；定理4需强分离和精确已知区域；专家索引≠solver动作 |
| [S06 Gneiting–Ranjan, *Combining Predictive Distributions*](https://arxiv.org/pdf/1106.1638) | §2；§3.1定理3.1(c)、3.2 | 指定条件下固定正权线性池过离散、缺乏coherence | 需要相异性、正则性等条件；coherence指条件法则，不是所有动态多类池的否定 |
| [S07 Gneiting–Raftery, *Strictly Proper Scoring Rules*](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) | **仅§1定义** | 无约束预测空间中真实分布唯一优化期望评分 | 不推出经验评分改善必然改变动作；不引用后文 |
| [S08 Williamson等, *Composite Multiclass Losses*](https://jmlr.org/papers/volume17/14-294/14-294.pdf) | Definition 2；Proposition 3，pp.7–8 | 连续严格proper与所有内点c的c-calibration；均匀c对应argmax一致性 | 总体、无约束条件结论；不覆盖有限优化、受限凸包或实例成本幅度 |
| [S09 Duchi等, *Multiclass Classification, Information, Divergence, and Surrogate Risk*](https://arxiv.org/pdf/1603.00126) | §3.2定义3.1、式13；§§4.1–4.3，定理1–2 | 成本校准不同于0–1；跨全部先验的普适等价要求广义熵仿射关系 | 固定表示一致性不能认证联合压缩；类代价矩阵不能编码实例全成本 |
| [S10 Elmachtoub–Grigas, *Smart Predict, then Optimize*](https://arxiv.org/html/1710.08005v5) | §3.2定义3.5、命题3.6；§4假设4、定理4.2 | SPO决策损失；SPO+凸上界 | 一致性需唯一条件均值最优、中心对称、全空间连续密度和非空内点；单纯形在环境空间内点为空，保存单次成本也不满足自动套用条件 |

## 授权边界

本文仅作科学研究解释，不提供执行顺序或配方。D1结构族教师标签采集及当前四项主实验保持不动；不现在加入R125，也不在四项结束后自动加入。任何后续行动均须另行授权。
