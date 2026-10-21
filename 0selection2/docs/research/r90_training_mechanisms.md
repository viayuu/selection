# R90: 分类链接与任务混合的局部机制

路由记录：继承父线程已接受 native 请求 01a155fd-d788-7dd1-aeff-bee068378eca；requested gpt-6-astra，reasoning_effort=max。本轮无独立 runtime attestation，不据此自称已核验实际运行模型。

边界：项目信息均为用户给定前提；未读项目源码、数据或结果。独立 NSS 在同一数据上也零选择 LEHD 等，却不共享这套 18 任务 maximin 链路。因此，共同零选择不能作为下述任一 V4 专属机制的归因证据，更不能称 universal root cause；也不能反过来证明它们对 V4 毫无局部作用。

主线程后续事实补充（Sol，不属于上述论文调研）：源码梳理发现 R64/R65 已有 raw-utility CE 和同一步多任务梯度实现。其数据与目标并非当前 NEW90k 的严格单因素对照；实现存在也不证明运行完成或取得收益。下一步先读取这些历史结果，不将本文两种机制包装为项目首次提出，也未据本文启动 R90 训练。

## 1. Raw Utility CE: Conditional

[Vernet 等，Composite Multiclass Losses，NeurIPS 2011，§2-3](https://papers.neurips.cc/paper_files/paper/2011/file/2afe4567e1bf64d32a5527244d104cea-Paper.pdf) 区分概率损失的 properness 与分数空间的链接参数化；不能仅凭非线性链接就认定分类损失不 proper。以下是由给定 maximin 定义自行推导，不是该论文针对 V4 的结论。

在单实例合法候选内，设前二名无并列，领先者为 \(a\)、次名为 \(b\)，\(d=u_a-u_b>0\)，则

\[
z_a=d,\quad z_m=u_m-u_a\ (m\ne a),\qquad
z+u_a\mathbf1=u+d e_a.
\]

所以固定同一 \(u\) 时 argmax 完全相同，但 maximin CE 等价于在 raw utility 上额外增加“当前领先者的领先差”。设 \(p=\operatorname{softmax}(z)\)，\(e_j\) 为单位向量，固定前二排序区间内：

\[
\nabla_u\ell_z
=p-e_y+(p_a-\mathbf1_{\{y=a\}})(e_a-e_b),
\qquad
\nabla_u\ell_u=\operatorname{softmax}(u)-e_y.
\]

两式连概率也不同。若真实 winner 不在前二，\(\partial\ell_z/\partial u_b=p_b-p_a<0\)：utility 坐标的下降方向会同时抬高错误次名；真实 winner 仍有 \(p_y-1\) 的监督，绝非被 min 截断。网络参数还要经过共享 Jacobian，不能直接把坐标方向当成实际预测变化，更不知道当前漏选是否处在这种排序。

这不是共同 score-shift：平移消去后仍有 \(d e_a\)。也不是普通温度：[Guo 等，ICML 2017，§4.2 Eq.(9)](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) 的单一温度统一缩放所有差值；这里非领先者之间的差不变，领先者对其他方法的差额外增加 \(d\)。关键反例是仅两个合法候选时，\(z=2u-(u_1+u_2)\mathbf1\)，此时确实只是固定温度等价变换。并列处不可微，不能套用上述固定排序导数。Guo 讨论的是后处理校准，不是这里的训练收益证据。

也不存在自动的概率表达缺口：任给正概率向量 \(q\)，取其前二为 \(a,b\)，令 \(d=\frac12\log(q_a/q_b)\)、\(u_a=0\)、\(u_m=\log(q_m/q_a)+d\ (m\ne a)\)，即可得到 \(\operatorname{softmax}(z)=q\)。因此，无约束 utility 下 CE 在其自身预测概率上仍可保持 proper 的解释，不意味着 \(\operatorname{softmax}(u)\) 同时校准，也不保证有限共享网络或混合目标的性质。

**判断：conditional，局部机制较明确。** raw CE 去掉额外的领先者/次名梯度耦合，既非 OVR、固定 bias，也非增加容量，推理 argmax 无须改变。反面是 winner 恰为次名时，原额外项正帮助它；去掉未必更好。pair/risk/RDrop 的原链贡献仍在，不能由 CE 分析推出整个目标、natural Top1 或 cost 必然改善。

## 2. 跨任务混合: Conditional，证据更弱

给定等量自然数据与完整 18-cycle，目标可写为 \(\bar L=\frac1{18}\sum_t L_t\)。当前并非某任务整轮缺席；同一参数点上的任务平均梯度本就对应该目标。混合的不同之处是减少任务间逐步切换、改变顺序漂移和优化器状态，不是给题内漏选方法增加标签或强制名额。

[Zhao 与 Zhang，Accelerating Minibatch SGD using Stratified Sampling，2014，§3.3 Eq.(1)、§3.4](https://arxiv.org/pdf/1405.3080) 给出保持原目标的加权分层梯度与方差分析。固定参数、将 cycle 相位 \(T\) 视作统计索引，令 \(g_T=\mathbb E[\hat g\mid T]\)，有

\[
\mathcal V(\hat g)
=\mathbb E_T\mathcal V(\hat g\mid T)+\mathcal V_T(g_T),
\quad \mathcal V(X)=\mathbb E\|X-\mathbb EX\|^2.
\]

保持自然任务权重的分层混合有降低任务间逐步波动的依据；但此式不是把实际固定 cycle 假定成随机任务采样。论文依赖簇内梯度离散度，其强凸、光滑 SGD 分析不保证深网固定循环 AdamW 受益，routing 任务也未必是良好的梯度分层。完整 cycle 已覆盖任务，长期累计方差不能直接等同于单步方差。

按 AdamW 二阶矩定义，逐任务平方再积累与先混合再平方不同，\(\overline{g_t^{\odot2}}\ne\bar g^{\odot2}\)；所以即使平均目标相同，预条件和轨迹仍会变。这是机制推论，不是新优化器建议；若目标或归一化依赖 batch 组成，“仅改变梯度估计”的前提也不自动成立。

[Yu 等，Gradient Surgery for Multi-Task Learning，NeurIPS 2020，§2.2-2.4、Algorithm 1、Theorem 1](https://arxiv.org/pdf/2001.06782) 明说冲突本身不必有害，其困难假说还涉及幅度失衡与曲率。PCGrad 对冲突梯度先投影再求和，普通混合只求平均，不能借用其收益；其两任务凸情形定理甚至容许停在完全反向的次优点。

**判断：conditional，仅支持调度/波动假说，不支持“混合自动消除冲突”。** 若 \(g_1=v,g_2=-v\)，混合为零，未必有利任一任务；它也可能正是联合目标的合理驻点。若各任务均值梯度相同，则上述任务间方差收益不存在。降低波动也不保证泛化，更不能直接解释题内 LEHD 漏选。两方向均只是当前 V4 的可辨别机制，不是 NSS 与 V4 共同零选择的已证原因。
