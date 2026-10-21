# R91 后的正则压力：条件性文献判断

2026-10-20。research-lit，web-primary-only，共2篇。请求路由 `gpt-6-astra/max`；无独立运行时身份核验。项目事实仅采用用户给定的 Sol 摘要，不作结果复核。

**结论与边界**
**Conditional：仅保留 V4 局部假设；Decline：共同必要根因、必然收益及自动追加变体。** 同数据 NSS 无 V4 显式输出 KL，却也零选部分方法，故该 KL 不是两者现象的共同必要原因；不排除它对 V4 有局部影响。零选不必然有害；平均概率近 winner 频率不保证逐例 argmax。

给定 NEW90k 全18题全 TRAIN 评估的小差距，不是旧8k记忆情形，也不能定位根因；容量、优化、信号复杂度仍可能。R80 同时去 comparison/risk/KL，R64/65 异 cohort，均非当前 KL-only 证据。R89/R91 无最终结果。

**原文与反证**
Liang 等，NeurIPS 2021，*R-Drop: Regularized Dropout for Neural Networks*：令两路为同一输入、同一当前参数的独立 dropout 输出 $p_1,p_2$，目标为

$$
L=-\log p_1(y)-\log p_2(y)+\frac{\alpha}{2}
[\mathrm{KL}(p_1\Vert p_2)+\mathrm{KL}(p_2\Vert p_1)].
$$

两路算完才更新参数，不约束跨训练时间预测不变，也不阻止两路共同改变。[第2.1节式(2)-(4)、第2.2节算法1](https://proceedings.neurips.cc/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf#page=3)

命题2.1仅对归一化线性 softmax，在平均 KL 不超过 `epsilon` 时，给出 `|L_full - E L_dropout| <= c sqrt(epsilon)`。这是同一训练集上的模型间经验 NLL 差，不是 TRAIN/VAL 差或 Top1 保证。[第2.3节式(5)-(6)、命题2.1](https://proceedings.neurips.cc/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf#page=4)；证明还使用逆映射 Lipschitz 条件，不能无条件推广到 V4。[附录B式(5)-(7)](https://proceedings.neurips.cc/paper_files/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Supplemental.pdf#page=4)

原文不证明移除 KL 必要：其 KL-only 消融保留双路 NLL，翻译 BLEU 从37.25降至34.93。[附录C.2](https://proceedings.neurips.cc/paper_files/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Supplemental.pdf#page=5)

Liu 等，ICML 2023，*Dropout Reduces Underfitting*：早期 dropout 在其视觉任务中反而降低最终训练损失，不能把随机正则等同于妨碍拟合；这也不是 V4 适用性证明。[第3节、第5.1节表2](https://proceedings.mlr.press/v202/liu23aq/liu23aq.pdf#page=4)

**仅限机制推断**
若有效区分依赖 mask 敏感表示，KL 可能与监督竞争；去掉它可能改变目标偏置及优化轨迹。多训仍优化原目标，不能自动解除这种竞争；若只是未收敛，多训仍可能有用。去 KL 不保证拟合、Top1 或 VAL 改善。

所问假设保留分类/比较/成本监督及其权重与归约、架构、自然分布、静态推理及 dropout。只去 KL 与减弱采样随机性是不同机制；改双路为单路还改变随机估计。dropout=0 若消除全部两路随机性，则 KL 也为0，但同时取消子网采样，绝非只去 KL。

文献只留下条件性可能，未补足当前配方的因果证据。R89/R91 保持不动，不自动新增变体；后续决策归 Sol。

