# 回到完整18题：科学交接

日期：2026-10-30。范围：原18个routing任务及其冻结合法候选池。本文提供机制、公式和适用边界，父任务负责具体实施。R120训练状态只采用本轮用户说明：尚无完整TEST；本文不等待它，也不解释其成绩。新增六题不进入下列ALL指标。

作者路由：本续接收到开发者的模型切换说明，为基于GPT-6的Codex；不能独立声称本轮仍是`gpt-6-astra/max`。当前可用工具没有显式选择该模型的委派入口，详见ROUTING.json。已有Astra报告作为冻结科学来源保留。

## 1. 四条路线应当分别回答什么

|用户方向|建议研究对象|不充分的替代|
|---|---|---|
|D1结构族优势、D2硬动作目标|D1改变训练人口与族关系监督；D2约束完整模型实际argmax的错误、regret和固定群体漏选|仅加优势回归；把R115的Ridge换MLP；再调固定CE/risk系数|
|历史Rxx互补，类似MoE|冻结完整历史selector作为专家，输入条件gate决定其组合或采用哪个专家动作|R105内部FFN路由、R109晚期私有层、每题事后拼最佳Rxx|
|惩罚真实漏选|按真实winner支持与误选成本定义漏选；约束反馈作用到最终硬动作|规定每个方法相同选择率；为selection=0增加一次错误选择|
|全18 Top2精排|固定第一阶段后，对其实际Top2做共享的候选对条件化结构计算|只在一个ReLD pair上训练；把Top2覆盖率当可达Top1|

## 2. 统一符号与D1/D2

对任务t，合法候选为M_t，原生aug_score为C_m>0，保存的native/FP64 winner为Y。定义C_*=min_m C_m、r_m=(C_m-C_*)/C_*，E_18为逐题等权。保留原B问题过滤和单reld_mt，不重建FP32 winner。

D1为基础实例b的合法结构族X_bz测量完整合法候选成本，使用：

$$
A_m(X)=\log C_m(X)-|M_t|^{-1}\sum_j\log C_j(X),
$$
$$
L_{\rm resp}=E_{b,z,z'}\operatorname{Huber}
\left([\hat A(X_{bz'})-\hat A(X_{bz})]-[A(X_{bz'})-A(X_{bz})]\right).
$$

自然TRAIN的native CE与端点完整优势锚定保留。族要改变簇间连接、有向循环竞争、空间与需求/容量/TW/L/B/O耦合，并包含无优势反例，不能只围绕一个方法或一种demand扰动。推理只读单个当前X。真实族与同端点静态训练的比较区分族监督收益；响应由端点成本导出，不是新的独立标签。R114已有全池，R115已有中心化全向量响应且跨base失败，仍是本方向必须面对的负证据。

D2以同口径参照theta0的TRAIN估计为右端：

$$
\min_\theta E_{18}1[a_\theta\ne Y],\quad
E_{18}r_{a_\theta}\le E_{18}r_{a_{\theta_0}},\quad
E_{18}1[(t,Y)\in F]1[a_\theta\ne Y]
\le E_{18}1[(t,Y)\in F]1[a_{\theta_0}\ne Y].
$$

F可保留历史fixed36，不能用动态零选择名单替代。参数梯度用代理，约束反馈用真实TRAIN argmax；反馈更新整条表示路径。对非负动作损失v_m，可使用代数上界：

$$
\widetilde v(u)=\sum_m v_m\frac{\operatorname{softplus}(u_m-\max_{j\ne m}u_j)}{\log2}
\ge v_{\arg\max u}.
$$

被选动作的系数至少1。此上界可能很松；Cotter的proxy-Lagrangian支持代理/硬反馈的分工，不保证本项目一个非凸checkpoint可行。简单投影乘子更新应称研究启发式，不能冒称论文的混合策略理论已经成立。

## 3. 历史完整模型的条件路由

专家e输出完整合法池概率p_e(m|X)，动作a_e=argmax p_e。gate读原图结构、任务约束，以及在实际部署会计算的专家预测，产生g_e(X)。可先研究概率混合：

$$
q(m|X)=\sum_e g_e(X)p_e(m|X),\quad a=\arg\max_m q(m|X).
$$

不要直接相加不同Rxx的原始logit：其尺度和分数含义可能不同。上述混合是预测组合，不等于校准概率；依赖所有专家输出的gate也不能声称只执行了一个专家。

概率混合可以选出没有任何专家放在Top1的方法，例如两专家`(.5,.1,.4)`与`(.1,.5,.4)`的均值选择第三方法。因此任一专家硬命中率只约束硬专家路由，不是所有soft mixture的统一准确率上限。两种模型须分别命名和解释。

若要真的按输入选择一个完整selector，用输入侧gate输出expert index，监督完整动作损失向量：

$$
\ell_e(X)=\eta\,1[a_e\ne Y]+\zeta\,r_{a_e},\quad
L_g=E\sum_e g_e(X)\ell_e(X).
$$

eta/zeta在此只标识Top1与风险两个不同偏好，不提供调参方案。部署取gate的确定性最大者；soft训练值不替代硬结果。多个专家同样正确时保留最优集合，不用任意expert ID制造唯一真标签。原始query成本和winner仅用于监督。该路线利用不同完整学习映射的条件互补，区别于R105/R109的内部模块化。

**metaTRAIN的科学条件。** 设D_e为专家训练及其它使用标签拟合的成员，T为当前81k TRAIN；共同未见集合U=T\(union_e D_e)才可直接支持该冻结bank的诚实gate训练。R77明确写每题70114，R80/R83索引也为NEW70114；在它们共享同一旧训练子集且它确实包含于新TRAIN时，差额为10886/题。这个数量是条件推导，成员及源身份尚由Sol核实。加入看过全部81k的专家会使此共同未见路径失效；应使用该专家真正未见的metaTRAIN或OOF版本。事后切一块TRAIN、冻结权重，不能抹去专家曾用这些标签训练的事实。

OOF只对生成该预测的折外专家诚实；之后换回全TRAIN专家会有分布变化，不能把OOF标签与全量专家输出混成同一个对象。formal VAL/TEST/LIB不作为gate梯度数据。历史TEST已参与研发，不能称盲测。

**已有反证。** R96是21个锁定模型的VAL答案依赖覆盖分析：相对R89B的23289个错误中17332个被全部21模型漏掉，fixed36的4625例中任一模型正确146例。这不是本次新bank的上限，但说明已有相近模型有大量共同失败。R77/R80/R83不在该catalog里，不能用它否定它们的额外互补；也不能把任一模型命中视为可部署gate收益。

## 4. 真实漏选损失

所有方法的漏选总和就是0-1错误：sum_m 1[Y=m]1[a!=m]=1[a!=Y]；把它改名不构成大改loss。真正变化应来自固定群体的条件保护、结构优势幅度与误选代价，而非逆频率或选择次数。

若要区分独有质量机会，定义Gamma_Y=(min_{j!=Y} C_j-C_*)/C_*。Gamma_Y>0才是不能被同成本候选替代的独有优势；native平局仍按Y报告，但Gamma可为0。可报告或约束E 1[(t,Y) in F] Gamma_Y 1[a!=Y]，同时保留总体r_a以计入误伤。加权漏选不是普通自然Top1目标，二者的取舍必须显式。

## 5. Top2精排

固定第一阶段f0及其全池分数，S2(X)=Top2(f0(X))={a,b}。共享比较器读当前原图/矩阵、约束、两个候选身份和第一阶段预测，输出反对称d(X,a,b)=-d(X,b,a)，pi=sigmoid(d)。它必须有独立的候选对条件化结构读出；只重用原utility差没有增加决策能力。整个18题共享比较机制，适用所有实际出现的合法pair。

监督保留两种对象：Y在S2内时，用native二分类目标；所有实例均可用原生C_a/C_b学习二者的成本差或pi*r_a+(1-pi)*r_b。Y不在S2时不得把pair内较好者冒称全池winner。精排的TRAIN候选应由诚实未见/OOF的第一阶段产生，避免只学习自包含训练的过易pair。

精确分解：

$$
\mathrm{Top1}=P(Y\in S_2)P(\hat a=Y\mid Y\in S_2),
$$
$$
R=E\frac{\min_{m\in S_2}C_m-C_*}{C_*}
+E\frac{C_{\hat a}-\min_{m\in S_2}C_m}{C_*}.
$$

前项是召回/候选集质量损失，后项才是精排能修复的损失。旧报告R94B的Top2覆盖90.8%、rank2错误贡献73.56% regret，只说明历史机会结构。成功须看所有实例上的纠正减误伤，不能只报原错子集的pair准确率。

## 6. 当前推荐

D1+D2仍是原18题主线；历史专家路由和Top2条件比较是两个实质不同的候选载体。真实漏选目标作为D2的明确科学目标，不再包装成第五个独立范式。优先区分“输入中有可泛化的专家/候选优势”与“只是答案依赖的离线覆盖”；总体Top1、actual regret、固定群体真命中和误伤共同决定结论。

完整推导已进入 [REPORT.md](/public/home/shiys/0selection2/docs/research/return18_major_routes_20261030/REPORT.md)，逐项阅读边界见 [sources.csv](/public/home/shiys/0selection2/docs/research/return18_major_routes_20261030/sources.csv)，四个机制与反证见 [directions.csv](/public/home/shiys/0selection2/docs/research/return18_major_routes_20261030/directions.csv)。本轮实读7篇原论文的相关方法/讨论段落，两个访问受限尝试单列、不作为方法证据。当前模型身份限制见 [ROUTING.json](/public/home/shiys/0selection2/docs/research/return18_major_routes_20261030/ROUTING.json)。本目录不改写旧冻结报告，也没有正式训练或TEST改善结论。
