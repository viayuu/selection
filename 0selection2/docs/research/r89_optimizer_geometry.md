# R89：优化器矩阵更新几何

范围：research-lit、web-only；项目事实沿用主线程假设，未读取项目代码或结果，未安装环境、运行实验或调用 reviewer。路由沿用 accepted native 请求 `01a155fd-d788-7dd1-aeff-bee068378eca`，requested `gpt-6-astra / max`，无独立 runtime attestation。

**结论：recommend Muon，作为有条件的优化几何探索；SOAP 仅比较，不优先推荐。** 建议保留自然原 loss、原模型与候选池，研究已有矩阵梯度方向的相对步长。这里没有把更新各向异性诊断为漏选根因，也不保证自然 Top1、成本或 winner 召回改善。new90k、全18题、seed2、逻辑 batch1024 均按给定背景理解。R88 未有结论，不构成讨论此机制的理论障碍；Muon 同样无法补回缺失的物理起点信息。

## 原始依据与机制

Liu 等，**Muon is Scalable for LLM Training**（2025 technical report），§2.1 式(1)-(2)描述动量后 Newton–Schulz 正交化，§2.2 Lemma 1、式(3)-(4)给出更新尺度与 WD。§3.2 的主要证据是语言模型损失与训练计算效率，不是 routing 选择准确率。[论文](https://arxiv.org/pdf/2502.16982)

对隐藏权重 \(W\in\mathbb R^{a\times b}\)，令 \(B_t\) 为既有 loss 梯度的动量/Nesterov 矩阵。理想化地，对其非零奇异方向，

\[
B_t=U_r\Sigma_rV_r^\top,\qquad
O_t=U_rV_r^\top,\qquad
W_{t+1}=(1-\eta_t\lambda)W_t-\eta_t s_{a,b}O_t .
\]

这会相对提高较小奇异值方向的步长。实际作者实现用有限次多项式迭代近似，输出不等于精确的 \(U_rV_r^\top\)。正交化对象是更新，权重本身没有被约束为正交矩阵；也没有计算或求逆真正 Hessian。“conditioning”在此指更新谱的尺度重整，不代表已证明改善目标曲率的条件数。

Jordan 的[原始说明](https://kellerjordan.github.io/posts/muon/)明确把“弱方向可能重要”称为假说。它不能推出小奇异方向就是 LEHD/DIFUSCO 的 winner 特征。其可取之处在于：方向信息已经存在于自然梯度时，可改变其相对推进速度，而无需修改标签、损失或模型容量。这与输出置信度惩罚有实质区别，亦没有把类别频率改为平衡目标。

Vyas 等，**SOAP: Improving and Stabilizing Shampoo using Adam**（arXiv:2409.11321），§4 Algorithm 3 用梯度左右二阶矩的特征基构造 \(G'=Q_L^\top GQ_R\)，在该基下运行 Adam，再旋转回来。它更直接利用跨步统计预条件，但增加矩阵统计、基更新和状态成本；论文主要也是 Transformer 语言模型证据。这里选择 Muon 的理由是机制更集中于更新谱，不能声称 SOAP 在本题较差。[论文](https://arxiv.org/pdf/2409.11321)；[官方初步实现说明](https://github.com/nikhilvyas/SOAP)

## 参数分组、步长单位与 WD

原作者将 Muon 限于隐藏矩阵。迁移到给定结构，可考虑 attention、FFN 的内部矩阵；methodID embedding、输入表征映射、最终 scalar utility 输出及归一化增益、bias 保留 AdamW。连续坐标投影与词表 lookup 并不相同，这项输入分组只是谨慎迁移；不能把所有二维张量机械归入 Muon。Q/K/V 独立矩阵与拼接矩阵的正交化也不是相同几何。[作者分组与实现](https://github.com/KellerJordan/Muon)

论文的满秩理想更新满足

\[
\operatorname{RMS}(O_t)=\frac1{\sqrt{\max(a,b)}}.
\]

Moonshot 式(4)使用 \(s_{a,b}=0.2\sqrt{\max(a,b)}\)，将更新 RMS 对齐到其观察到的 AdamW 量级；0.2 是文献经验刻度，不是选择器已校准的常数。Keller 公开代码采用的 \(\sqrt{\max(1,a/b)}\) 是另一套缩放，不能与前者混用后直接沿袭同一个数值学习率。

WD 是正交化之外的参数衰减，原有 decay/no-decay 的参数语义应沿用。改变学习率单位会同时改变 \(\eta_t\lambda\) 的收缩；RMS 对齐也不等于有效训练轨迹或正则强度完全匹配。保留原 loss 仍会改变隐式偏好。另因正交化非线性，完整逻辑 batch 的梯度求和后变换，与逐 microbatch 变换再求和不等价；张量如何分块也属于算法定义。

## 数值证据与关键风险

[作者 muon.py](https://github.com/KellerJordan/Muon/blob/master/muon.py)明确将 Newton–Schulz 内部矩阵转换为 BF16，并展示单设备版本；这支持作者路线中的 BF16 使用，不能认证当前 AMP、参数/状态精度组合。FP16 与梯度缩放的完整组合：**unknown**。PyTorch 2.5.1 兼容性：**unknown**；[setup.py](https://github.com/KellerJordan/Muon/blob/master/setup.py)没有声明经验证的 torch 版本，Moonlight 的模型推理环境说明也不是优化器训练兼容证明。SOAP 官方初步 README 将进一步低精度支持列为后续工作，不能拿它作为现成兼容证据。

关键反例：\(B=\operatorname{diag}(1,\epsilon)\)，若 \(\epsilon\) 只是噪声，理想正交化会把噪声方向抬到与有效方向相当的尺度；有限迭代虽不必完全等幅，也有此风险。若方向已经精确相消为零，迭代不会创造信息。因此推荐只成立为“已有弱而有用方向可能推进不足”的研究假设。新 loss、强行均匀选择、新标签或 solver 都不是该假设的组成部分；LLM/vision 的效率记录也不能替代本项目证据。

