- wandb: true
- wandb_project: selector
- wandb_entity: yjkds-southern-university-of-science-technology

GPU 环境
- 这台机器有直接 GPU 访问（不需要 SSH）
- GPU：2x RTX 3090 24GB，2xRTX 3090 24GB都可以使用
- 实验环境：`easynco`
- 任何命令都在 conda activate easynco 环境中执行
- 激活前任何 Python 命令：`conda activate easynco`
- 代码目录：`/public/home/zhoucl/shiys/0selection`

当前目录（0selection2文件夹）结构说明

你永远不用看、不要关注的文件/文件夹：无

文件夹的简单介绍
literature文件夹：是我自己想出来的，我的项目可以参考的论文和对应的代码
观测指标文件夹：在监控实验时，明确要监控的指标和需要绘制的图。对于每个问题，都至少应该关注观测指标.md中的指标
data文件夹：训练数据，对数据的详细介绍在 data/README.md 文件
tools文件夹：你可以调用的工具。尽量不要修改tools文件夹的代码。


你更改AGENTS.md的时候，不要更改以上的内容。

## 模型分工 2026-10-15

- `gpt-6-astra`，`reasoning_effort=max`：仅负责论文调研和下一步改进思路。
- `gpt-6.1-sol`，`reasoning_effort=max`：负责所有其它工作，包括实验规划、预算、协议、gate、实现、代码审查、结果检查分析、运行监控和交付。
- 本用户规则覆盖 skills 中 Astra reviewer、experiment planning 等旧默认；不得为结果复核调用 Astra。
- 指定模型不可用时必须如实报告，不得冒称使用该模型或静默替换。

## 长期报告与完整模型交付约定 2026-10-15

1. 报告优先交付可直接下载的 PDF，不以依赖外置图片的 HTML 作为主要交付。若附带 HTML，图片和数据均须内嵌，确保单文件可离线查看。
2. 每跑完一个新的完整模型，至少交付方法分布 PDF：TSP 与 CVRP 各包含四栏图，依次为 LIB 模型选择、LIB 真实 winner、TEST 模型选择、TEST 真实 winner；TSP 的 LIB 对应 TSPLIB，CVRP 的 LIB 对应 CVRPLIB。其它任务各包含两栏图，依次为 TEST 模型选择、TEST 真实 winner。候选按名称对齐并采用固定顺序，清晰标注实例数、各候选次数与比例、模型版本及数据版本；真实 winner 频数不得当作模型选择频数。不强制增加 TRAIN 面板。模型选择面板标题必须标注对应数据集的 `模型Top1=xx.xx%`，定义为模型预测与该数据集真实 winner 的逐实例一致率，不是最高方法选择频率；必须写清原生/native/FP64 ind、FP32 argmin 等 winner 标签口径，不得静默混用；真实 winner 面板不得虚构 100% 的模型准确率。
3. 对锁定的同一个当前完整模型，在 TSP TEST、CVRP TEST、TSPLIB、CVRPLIB 四个数据集输出 NSS 与当前模型的并列表，至少包含 Top1(%) 和 mean_cost，并明确新旧数据集、候选池及版本。同一个当前模型不得为不同数据集拼接不同 Rxx 的最佳成绩。已有同口径 NSS 结果可复用，不无故重训；缺项须明确标为缺失，不得冒充已有结果。
4. 当前模型整体 ALL 及每个问题均须报告 Top1、mean_cost、SBS cost、Oracle cost、vs_SBS、vs_Oracle、actual regret。Oracle 指逐实例合法候选池中的最低成本；SBS 须明确在哪个划分拟合，若采用事后口径须显式说明。沿用项目指标定义，vs_SBS、vs_Oracle 的均值比与逐实例 actual regret 不得混用；ALL 保留逐问题等权口径，不得靠跨问题混合 raw cost 掩盖表现。
5. 使用 TRAIN 计算训练损失并更新参数；validation 用于调参、学习率控制、早停和 checkpoint 选择，主要选型指标须在训练前固定。每个版本先锁定唯一 checkpoint 和推理规则，再完整评估 TEST 及 LIB，并与 VAL 并列报告。不允许依据 TEST/LIB 结果重新选择该版本的 checkpoint、温度、阈值或推理规则，也不得逐问题拼接不同模型、checkpoint 或 Rxx 的最优成绩。“一次完整 TEST”不是执行次数限制，允许复用固定预测重算指标、画图、生成 PDF 及必要的一致性复跑，但不得借此重新选型。

   补充说明：多轮研究若使用 TEST 结果指导后续方案，须如实说明 TEST 已部分参与研发，不能称为完全未参与决策的盲测；最终确认研究结论时，优先使用未参与研发的独立评估集。本说明不是现在新建划分的指令。

6. PDF 为主要交付，可机器读取的 CSV 为辅助交付；仅做必要的少量核对，完成一个版本后再集中提交 GitHub，不因本次记录约定自动提交。所有命令均须先激活 `easynco`（`conda activate easynco`），再在该环境中执行。
