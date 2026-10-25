# R112：保留条件的区域接口组合

## 本轮问题

依据 `docs/research/major_paradigm_review_20261025/REPORT.md` 主线一，以及上一轮综合研究的净决策收益准则，从头训练两个完整18题模型。不是ReLD局部专家，不使用旧selector分数旁路。

- A_collapsed：每次跨区域信息传递前后，将入口/出口/资源条件平均后广播。
- B_interfaces：持续保存有向入口、出口及资源条件的对应关系。
- 两组有完全相同的可训练参数、初值、张量shape、监督、批次顺序及更新预算；不是推理时临时破坏训练好的B。

## 端到端实现

输入与执行角色 → 两层原始节点stem（完整距离/有向矩阵bias仍参与）→ 输入确定的4/2/1区域层级 → 每区域3×3入口/出口、4个资源条件，共252个接口token → 两轮子区域联合组合与父区域条件下传 → 原双流候选交互与score-difference/maximin → 全合法池一个方法。

接口是判别潜状态，不能将其数值称作真实子路线成本或可行性。父边界通过最近子端口投影，这是明确近似；不生成路线，不求解query，不采用真实路线筛边。资源坐标不是经过认证的物理载荷；不混淆signed-load与经典backhaul。

## 数据与初始化

保持NEW90k冻结的81,000/4,500/4,500每题划分与所有候选删除/别名规则。复用R84输入缓存、TRAIN-only几何统计、seed2随机参数资产，绝不使用已训练checkpoint初始化。保留TSP推理前已知的物理起点角色。

两组模型各1,878,027参数；新接口参数使用seed112002，原节点stem和共享读出使用原seed2对应随机参数子集。完整初值hash相等已经通过CPU检查，不宣称新模型等于原R84的完整参数集合。

## 训练与选型

初步固定逻辑batch512，每组最多57,600次成功更新，约20.2次样本曝光；warmup4,320次，AdamW峰值1e-4、末值1e-6、weight decay1e-4、dropout0.1。保留原CE/比较/relative risk/R-Drop及自然采样；不同时改目标。每2,880次完整VAL，每14,400次完整TRAIN评估。microbatch与eval batch由一次真实TRAIN最长输入测量后写入protocol_lock.json，不因TEST或运行后OOM静默改变。

选型为完整18题VAL宏Top1严格最大；精确并列时宏actual regret最小，双重并列保留最早点。随后锁定唯一checkpoint、执行同版本完整TEST、LIB、全推理时间与PDF。TEST/LIB已参与历史研发，不声称全新盲测，不按问题拼接不同checkpoint。

## 分析与交付

逐题与ALL报告Top1、mean_cost、TRAIN拟合SBS、Oracle、vs_SBS、vs_Oracle、actual regret；比较B对A及固定历史完整参照的纠正、伤害、双方均错时的净成本。固定零/近零清单只作机制诊断，不以更多选择次数宣布成功。交付同轴并列方法分布PDF及TSP/CVRP TEST/LIB对NSS表，全部使用同一锁定权重。

## 资源与版本

使用现有1475 allocation的gpu05两张4090，各模型一张、独立进程；两张3090上的R109不受干预。复用原UUID互斥锁和容器热缓存，不新增allocation，不在普通SSH会话训练。W&B保留selector项目与既有offline模式并显式记录。

用户本轮已授权自行设计batch/控制以及新增求解/过程标注。R113过程监督和R114成对响应独立准备，不与R112架构对照混合，不覆盖原成本表。完成一个版本并生成结果后集中提交GitHub；失败记录保留，不自动反复重跑。
