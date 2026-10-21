# R87：训练内持续漏选的下一机制

**范围与路由**
2026-10-20。按 research-lit 的 web-only 模式，仅做文献与机制推演；未读取项目代码、数据、结果，未运行实验或调用 reviewer。主线程通过原生 `spawn_agent` 显式请求 `model=gpt-6-astra`、`reasoning_effort=max`，工具接受并返回研究 worker `01a155fd-d788-7dd1-aeff-bee068378eca`；此为父线程可核验的请求路由，worker 未独立核验底层运行时模型标识，不作额外自我认证。项目数字及版本状态均是主线程提供的假设，不是本次复核结论。

## 1. 问题边界

给定 R84A TRAIN/VAL 为 71.24%/71.09%，TRAIN 中 TSP LEHD 的 14279 个 winner 仅命中 7，TSP DIFUSCO 的 6806 个 winner 与 ATSP MATPOENET 的 6476 个 winner 对应方法均未被选择，不能只解释为过拟合。但这也不证明共享参数冲突：输入不可辨识、有限容量和多目标折衷同样可能产生该现象。R85 A/B/D 未完成，不判成败；C 未满足联合接收条件，不借 TEST 小涨背书。小样本 angle-only 诊断既未否定归一化训练，也不足以把它排到优先位。

关键区别是：**缺少有效参数更新，不等于缺少 winner 监督。** 令 \(C(x)\) 为实例的固定合法候选池，\(y\) 为既有口径的真实 winner。对直接以 utility 为 logits 的自然 CE，令 \(p=\mathrm{softmax}_{C(x)}(s)\)，有

\[
\frac{\partial\ell_{\rm CE}}{\partial s_m}=p_m-\mathbf1[y=m],
\qquad
\nabla_\theta\ell_{\rm CE}=\sum_{m\in C(x)}
(p_m-\mathbf1[y=m])\nabla_\theta s_m .
\]

错分 winner 的输出梯度并不天然微弱；问题可能在共享 Jacobian 中发生抵消或难以形成专属区分方向。上述公式是机制说明，不是对现有 CE 接线位置的断言。

## 2. 三篇原文

| 原始论文 | 已读关键位置与结论 | 对本题的证据边界 |
|---|---|---|
| Fang、He、Long、Su，**Exploring Deep Neural Networks via Layer-Peeled Model: Minority Collapse in Imbalanced Training**，PNAS 2021。[原文](https://arxiv.org/html/2101.12699v3)；[作者仓库](https://github.com/HornHehhf/LPM) | arXiv v3 §2 式(7)、§4 Theorem 5：范数受限、两组类别规模的模型中，\(R=n_A/n_B\to\infty\) 时，少数类 \(w_k^\star-w_{k'}^\star\to0\)；有限阈值现象来自数值研究。 | 训练内失败有理论先例，但不能把本题数千 winner 直接认作极端长尾坍缩；也不移植论文的重采样补救。 |
| Misra、Shrivastava、Gupta、Hebert，**Cross-stitch Networks for Multi-task Learning**，CVPR 2016。[原文](https://arxiv.org/pdf/1604.03539) | §3.1-3.3 式(1)-(2)：\(\tilde h=Ah\)，反传为 \(A^\top\nabla_{\tilde h}L\)；私有非线性流和跨流混合是不同自由度，非对角项为零时保留私有路径。 | 支持“共享多少”是可独立改变的参数化问题；原文是多任务视觉学习，不是 winner 分类。本文只迁移共享/私有原则，不建议复制整套网络。 |
| Pezeshki 等，**Gradient Starvation: A Learning Proclivity in Neural Networks**，NeurIPS 2021。[原文](https://proceedings.neurips.cc/paper_files/paper/2021/file/0987b8b338d6c90bbedd8631bc499221-Paper.pdf)；[作者仓库](https://github.com/mpezeshki/Gradient_Starvation) | §3.3 Theorem 2：在二分类线性化、特定特征耦合条件下，增强强特征会压制弱特征响应；§3.4 式(17)-(18)用 logit 的平方惩罚替代参数 L2，消去其对偶动力学中的谱耦合。 | 这是特征学习现象，不等于少数类梯度为零；原文部分案例已拟合 TRAIN，不能直接解释本题训练漏选。 |

以上均为正式发表论文。作者仓库仅核对归属，未审阅实现；PLE 全文入口获取失败，未把摘要或第三方复现作为机制证据。三者分别提供训练失败先例、共享方式的参数化依据与优化备选，不构成本项目的因果证明。

## 3. 优先：逐方法私有非线性残差

**保留统一实例编码和自然全候选竞争，给每个合法方法一条专属、依赖实例的非线性修正路径。** 设 \(h_m(x)\) 为已有方法条件实例表示，建议的参数化是

\[
s_m(x)=g_\theta(h_m(x))+r_{\phi_m}(h_m(x)),
\qquad
r_{\phi_m}(h)=a_m^\top\sigma(B_mh),\quad
\phi_m=(a_m,B_m).
\]

不加固定输出 bias；所有合法分支均参与同一次预测，不按 winner 分流，也不由学习路由决定谁能得到容量。私有分支不再回流混合到其它方法的末端分数，因此

\[
\frac{\partial s_j}{\partial\phi_m}=0\ (j\ne m),\qquad
\nabla_{\phi_m}\ell_{\rm CE}
=(p_m-\mathbf1[y=m])\nabla_{\phi_m}r_{\phi_m}.
\]

**新意在参数化，不在标签。** 仍是原来的自然 winner、同实例 pairwise 与成本监督；只让某方法的条件胜出边界能够独立弯曲。它不消除跨样本正负竞争，也不隔离共享 Encoder 的梯度；只减少末端跨方法共用参数导致的牵制。methodID 加共享 MLP 本来就能表达非线性，不能声称其表达能力必然不足；这里改变的是有限容量下的分配与优化耦合。

这是全 18 题同一模型内的分支，同一方法在适用问题间复用私有参数，不逐题拼接模型。残差写法也不意味着冻结或续训 R84；讨论的是联合学习的参数化，不是在已训练输出上事后加分。目标仍是自然 full18 Top1 与真实漏选及成本，某个方法召回上升本身不等于总体收益。

与旧路线的区别：OVR/EQLv2、focal、cost-profile KL 改监督或梯度权重，跨实例排序改比较对象；此处不重写自然先验。去掉共享 MLP 没有引入私有非线性，不能代替这个机制。它也不是 ensemble、源码 embedding 或路线辅助回归：所有分支联合形成同一组可比较的 utility，前向不读取成本或标签。与单 ReLD 专家存在真实重叠风险：其具体参数化未知；若旧方案已经采用相同的实例条件残差和全候选联合损失，本建议只剩扩展覆盖面，增量依据明显减弱，不能包装成全新机制。

若 score_difference 严格为 \(s_m-s_j\)，且 maximin 在同一至少有两个候选的合法池上计算，则

\[
\arg\max_m\min_{j\ne m}(s_m-s_j)=\arg\max_m s_m .
\]

因此仅把 maximin 改名为 argmax 不会产生新能力。这里不声称已核实项目实现，也不建议引入不传递的任意 pairwise 输出。

**反例与风险。** 若可见输入始终满足 \(P(y=A\mid x)=0.6,\ P(y=B\mid x)=0.4\)，自然 Top1 的最优规则就是永远选 A，即使 B 有数千 winner。若已有表示丢失了决定胜负的信息，私有末端也不能找回；若残差退化为方法常量，它只是先验修正；若缺少可泛化模式，增加参数可能只记忆 TRAIN。更多正确 winner 也可能伴随少量高成本误选，不能保证 Top1 与真实成本同时改善。

## 4. 唯一备选：谱解耦

低于私有参数化的优先级。受 Pezeshki 等启发，可考虑在自然目标上约束单实例、合法候选的中心化 logits：

\[
\mathcal L=\mathcal L_{\rm natural}
+\frac{\lambda}{2}\mathbb E_x
\frac{1}{|C(x)|}\sum_{m\in C(x)}(s_m-\bar s(x))^2,\qquad
\bar s(x)=\frac1{|C(x)|}\sum_{m\in C(x)}s_m .
\]

这是本次提出的多类迁移形式，不是论文原式或已证实保证。它改变训练中的输出梯度，而不是在固定模型上删除 feature norm；也不规定跨实例方法选择比例。原论文替代参数 L2 的推导，不能直接推广为与现有多项损失、weight decay 叠加后仍严格解耦。风险是仅缩小 margin、放大难例噪声或损伤原本可靠的自然优势；它对“TRAIN 仍漏选”的直接证据较弱。

结论：优先考虑**自然竞争下的逐方法私有非线性自由度**，不把选择均匀、balanced-accuracy 或固定偏置当收益；谱解耦仅作备选。两者都不依赖新 solver 标签或 TEST 调参，也都不承诺联合收益。

