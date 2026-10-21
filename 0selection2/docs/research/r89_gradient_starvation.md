# R89：是否推进 Gradient Starvation / Spectral Decoupling

范围：research-lit、web-only；项目状态与数字均按主线程假设理解，未读取项目代码、数据或结果，未调用 reviewer。路由沿用 accepted native 请求 `01a155fd-d788-7dd1-aeff-bee068378eca`：requested model `gpt-6-astra`、reasoning_effort `max`；无独立 runtime attestation。

**判断：本轮不推荐将 SD 升级为下一项训练机制。** 最关键理由是，尚未建立“TRAIN 持续漏选”与“可由输出幅度惩罚释放的特征饥饿”之间的联系；惩罚确定作用于分类间距，纠正排序属于间接假设。给定 R87 未联合接收，只削弱该次容量分配建议的依据，不能反推饥饿诊断；R88 尚无结论。以下说明它在数学上可以怎样迁移，避免把拒绝推荐写成已经证伪。

## 原文与作者实现

Pezeshki 等，**Gradient Starvation: A Learning Proclivity in Neural Networks**，NeurIPS 2021。§3.3 Theorem 2 在二分类、线性化网络和特定特征耦合条件下证明：强特征增强可压低另一特征的最优响应。§3.4 式(17)以输出平方惩罚替代式(5)的参数 L2，式(18)由此消去其对偶动力学的谱耦合。它没有证明每个被漏选类别都会恢复；§4.2 虽含多类 CIFAR 经验，附录 B.3 也明确过强惩罚会损伤 IID 表现。[原文](https://arxiv.org/pdf/2011.09468)

原文的两月牙例子已能取得近零训练损失，却遗漏有用方向；它与给定 TRAIN 71.24%、数千 winner 仍错分的现象并不相同。论文 CelebA 证据还出现平均准确率下降、最差群体准确率上升，不能把后者替代自然 Top1 收益。

作者的 Colored-MNIST 实现使用单个二分类 logit，核心为 `(logits ** 2).mean()`，并采用未设置 WD 的 Adam；这不是变长合法池的 centered softmax 实现。CelebA 的按类平移变体也不能作为本题自然频率目标的直接证据。[作者实现](https://github.com/mpezeshki/Gradient_Starvation/blob/main/Figure_4_and_table_1/cmnist.py)

## 合法迁移及 maximin 边界

令 \(C_i\) 为实例合法池，\(K_i=|C_i|\)，\(z_{im}\) 为**实际进入自然 CE 的 softmax 之前**的最终有限分类 logits，包含既定变换与缩放；非法 mask 填值不进入中心化。可定义

\[
\bar z_i=\frac1{K_i}\sum_{m\in C_i}z_{im},\qquad
R_i=\frac{\lambda}{2K_i}\sum_{m\in C_i}(z_{im}-\bar z_i)^2,
\quad
\mathcal L=\mathbb E_{\rm natural}[-\log p_{iy_i}+R_i].
\]

\[
\frac{\partial(\ell_{{\rm CE},i}+R_i)}{\partial z_{im}}
=p_{im}-\mathbf1[m=y_i]+\frac{\lambda}{K_i}(z_{im}-\bar z_i).
\]

中心化去掉 softmax 不可辨识的共同平移。惩罚等价于所有合法 logit 差的平方收缩：

\[
\frac1{K_i}\sum_m(z_{im}-\bar z_i)^2
=\frac1{2K_i^2}\sum_{m,n}(z_{im}-z_{in})^2 .
\]

因此可称“SD 启发的多类中心化 logit 正则”，属于项目迁移，未继承严格解耦。若 maximin 为 \(z_m=\min_j(s_m-s_j)\) 且包含自身，则它只是 \(s_m-\max_j s_j\)，中心化后与 utility 相同；若排除自身，领先者项还依赖第二名，即使 argmax 相同，惩罚和反传也不同。这是条件推导，未核查项目接线。

R86 的 posthoc 去 feature norm 不更新表示；此项在学习过程中反传，故 R86 阴性不构成 SD 的反证，也没有给 SD 增加正面证据。RDrop 约束两个 dropout 输出的一致性，两者仍可共同过度自信；focal 按预测难度重调 CE；label smoothing 改软目标，此处则加入与标签无关、随中心化 logits 线性增长的梯度。平滑目标为 \(q=(1-\epsilon)e_y+\epsilon\mathbf1/K\)，梯度为 \(p-q\)，与上式随 \(z\) 改变的收缩项不同。二者在接近均匀输出时有置信度收缩的局部重叠，不能包装成新 winner 监督。这里无类别频率权重，允许单例概率变平，却不规定跨实例 argmax 次数或均匀选择配额。

## WD、目标取舍与反例

保留 AdamW WD 是额外参数收缩，已经偏离原文“替代参数 L2”的推导。Loshchilov、Hutter 的 **Decoupled Weight Decay Regularization**（ICLR 2019）§2 Proposition 2、Algorithm 2 进一步说明，AdamW 的衰减步骤一般不等价于向损失添加参数 L2；不能把两者合写后声称复现式(18)。[必要澄清原文](https://arxiv.org/pdf/1711.05101)

就机制解释而言，纯自然 CE 加 \(R_i\) 比在 pairwise、risk、RDrop 上叠加更清楚：额外目标仍会牵引间距和表示。但删项本身会改变成本偏好与优化轨迹，任何变化均不能直接归因于 SD；这只是目标取舍，不是启动建议。

关键反例是 \(z(x)=a z_0(x),\,a>0\)：若模型只能调整正尺度，正则可以缩小置信度，却始终不改变 Top1 或漏选。即使网络能学习新方向，也可能削弱本来可靠的特征，并无自然 Top1、成本联合改善保证。错分 winner 的 CE 输出梯度原本就非零，非零输出梯度与内部特征饥饿仍可并存，缺的是其进入表示后受抑制的证据；接近 100% TRAIN 也可能来自记忆。当前保留这项可解释目标，但拒绝仅凭“尚未尝试”推进它，无需新 solver 或新标签。

