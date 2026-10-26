# R114 真实成对响应：可执行准备

本次交付512-base pilot的持久化分组队列，不训练、不运行solver、不占用GPU，不改其它版本、冻结表或Git元数据。当前精确命令见 [生产交接](/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/production_execution/QUEUE_HANDOFF.md)。旧逐任务队列只保留为历史记录，不再作为启动入口。

## 现在有什么

| 产物 | 实际状态 |
| --- | --- |
| 独立数据版本 | `r114_train_paired_v1_seed2`，不能覆盖 NEW90k |
| 固定 base | 16 个适用问题各 32 个，共 512 个；只来自冻结 TRAIN；保留 local/source/base ID |
| 真实输入 CPU 转换 | 固定512个base已全部完成，仅TRAIN，admin01、最多2核、数值库单线程 |
| 转换结果 | 2,048端点、1,536对、9,600个全池label任务、0个构造拒绝；未求解；旧36端点cpu_smoke保留不变 |
| 源码/权重清单 | 89 个当前合法部署，适用干预的为 76 个；581 个唯一依赖及 598 个 overlay 文件已做哈希核对 |
| 静态缺失/变化 | 0 个部署缺文件，0 个部署与生产依赖哈希不符 |
| 运行证据 | 本轮56项有界CPU测试通过（7.960秒）；未实例化真实solver、未加载solver权重 |
| 路线证据 | 76/76只有源码hook；真实runtime验证为0/76，不能用源码检查代替 |
| 真实新成本/winner | 0；`labels_unrun.json` 全部缺失 mask，不是标注完成文件 |

`base_selection.json` 固定基础实例与候选池。完整清单在 `production_inputs/manifest.json`。保留的冷启动source锁是 `production_execution/deployments.production.lock.json`；当前plan与补充锁在 `grouped_execution/queue_v1/plan.json`、`grouped_execution/queue_v1/deployments.grouped.lock.json`。旧 `production_execution/label_queue*`、`cpu_smoke/manifest.json` 与 `dispatch_dry_run.json` 均保留，不覆盖。
`deployment_sources.json` 保存逐问题/候选的实际权重绝对路径、预算、profile/dependency 文件和哈希，不用泛称的默认参数代替。

## 生产来源能复用哪些

实际入口是 `/public/home/shiys/easynco_v3_bridge/scripts/train_100k_sampling.py` 的 `make_confirmed_backend(problem, source_method, device, legacy_root)`。
它先走采样确认与 alias-safe 加载，再到后端；不能绕过它直接使用默认 `make_backend`。
不直接启动旧 `label_train_100k.py`：其校准、旧分片、控制状态和写入目录属于旧生产任务；R114 只读复用 factory，并独立派发。

下表“可复用”仅指代码、权重、协议文件存在且匹配，**不表示本轮在 GPU 上跑通，更不表示严格复现旧标签**。
所有列使用独立新生产协议，两端重跑，不能拿旧 TRAIN 成本填原端点。

| 族 / 当前候选 | 实际 adapter / 权重 | 保留预算要点 | 静态缺口 |
| --- | --- | --- | --- |
| BQ，TSP/CVRP | `train_100k_backends.BQBackend`；`bq-nco/pretrained_models/{tsp,cvrp}.best` | beam=1，aug=1，greedy | 无 |
| ELG，TSP/CVRP | `OriginalELGBackend`；原仓库 `ELG/TSP/weights/ELG*.pt`、`CVRP/weights/ELG.pt` | 原 multiple_width，aug=8，greedy；星号是实际文件名 | 无 |
| LEHD | TSP `NativeARBackend` + `lehd_tsp100.ckpt`；CVRP `train_100k_nss_ar.OriginalLEHDCVRPBackend` + `checkpoint-40.pt` | TSP aug=8 但只选原方向；CVRP starts=1、aug=1、RRC=0 | 无 |
| OMNI | TSP 原 EasyNCO checkpoint；CVRP `OriginalOmniCVRPBackend` + 原 `maml-fomaml-checkpoint-250000-batch-no-track.pt` | CVRP min(N,100) starts、aug=8、无微调；TSP 原方向选择 | 无 |
| ICAM_CVRP | `train_100k_icam.OriginalICAMCVRPBackend`；`ICAM/pretrained/icam_cvrp.pt` | N starts、aug=8、greedy、iteration=0 | 无 |
| MVMOE_CVRP | `train_100k_nss_ar.NSSMVmoeCVRPBackend`；`mvmoe_mvrp100.ckpt` | min(N,100) starts、aug=8、greedy | 无 |
| MTPOMO/MVMOE 变体 | `HistoricalSamplingBackend`；`mtpomo_mvrp_100.ckpt` / `mvmoe_mvrp100.ckpt` | **sampling**、aug=1、原 N/B starts、FP32 medium；不是默认 greedy | 无；旧随机结果不保证逐值复现 |
| RouteFinder、MoSES_RF、MoSES_CaDA | `RouteFinderBackend` / `AliasSafeMosesBackend`；原仓库 50/100 checkpoint | N starts、aug=8、原 autocast；N<=75 取50，否则100 | 无；仅保留非 B 的当前合法列 |
| RELD_CVRP | `RELDBackend`；`CVRP/weights/ReLD/model_epoch_90.pt` | min(N,100) starts、aug=1、greedy | 无 |
| reld_mt | `RELDBackend`，source_method=`RELD_MOEL`；`Multi-Task/pretrained/reld_moe_light/epoch-5000.pt` | N starts、aug=1、sample_size=1、greedy | 无；不恢复 RELD_MTL、不取两种 ReLD 最小值 |
| DIFUSCO / DIFUSCO500 | `train_100k_diffusion.OriginalDiffusionBackend`；原 100/500 checkpoint | 50 denoising、100 次 2-opt，单采样 | 无；本干预不适用于 TSP |
| T2T / T2T500 | 同上；`T2TCO/ckpts/tsp{100,500}_categorical.ckpt` | 同上，另 3 次 rewrite、每次10步 | 无；不同 checkpoint 仍为不同候选，不称独立算法 |
| ATSP 五列 | `train_100k_atsp` 的 GLOP、ICAMATSP、NativeATSP、UnicoATSP 后端及原权重 | GLOP 原 insertion、无神经权重；ICAM N/1；MATNET aug128；MATPOENET/UniCO aug8；完整细节见 JSON | 无；两类干预均不适用，不伪造 ATSP 配对 |

## 干预与标签定义

需求：在同符号客户中选两个不相交的交换，将需求幅值与第一空间主轴的对应增强；几何、容量、需求多重集、B 节点角色均固定。
TW：只交换中心，按 depot 距离定向；每个节点窗宽、中心多重集、几何、需求、服务时间均固定；不裁剪窗口，不放宽截止时间。
两者均从固定 256 个输入限定 proposal 中选择，不能查看 solver 成本。随机对照来自同族合法 proposal，四个客户改变，L1 幅度差不超过25%，且输入与定向臂不同；不优化对照的几何得分。
这是具体的小幅对应干预，不声称已经识别 solver 内部机制；PCA 轴也不冒称真实簇标签。构造失败保留拒绝记录，不换 base，不自动补量。

合法性独立检查采用 singleton-route 存在性证书；B 仍是原 signed-load，先完成全部送货 singleton 再取货 singleton 可给出证书，**路线验算不额外强制 classical-B 先送后取约束**。
封闭 TW depot deadline=3、原 route_limit=3、服务 START 落窗；open route 不计返仓距离/时间。不复用 R58 的新语义检查器。
当前保留的 B 部署也不因此被宣称内部语义完全一致。

端点只接收带相同输入、部署锁、seed 与 profile 来源的 fresh solver receipt，并重新独立验算路线。
成本是原生 reported cost 转成 FP64；FP64 路线重算值只审计，不静默替换标签。精确同成本按稳定 global ID 取 winner；任何合法列缺失、超时、错误、路线不合法，整行全池 winner 都 mask。
`relative_response_m=C_m-mean(C)` 仅完整池定义；`candidate_change=C_after-C_before`；`Delta_mn=(C_after_m-C_after_n)-(C_before_m-C_before_n)`。
部分有效列可有局部 Delta 及 mask，但不产生全池冠军；不填0、不用伪标签、不虚构跨问题归一化尺度。端点 winner 锚点始终保留。
相关端点共用 base_id；未来支持/query 或响应验证必须按 base 排除关联副本，不能随机拆端点。

## 实际入口

在 `/public/home/shiys/0selection2` 执行，所有命令先激活easynco。精确单GPU命令只以生产交接中的 `r114_run_queue` 为准；资源worker提供UUID和CPU绑定。CPU测试已完成，不需要再运行转换、source审计、smoke或额外冷启动比较。

`python -m code.V4.r114_response_data --help` 列出 `inventory/freeze/prepare/lock/dispatch/assemble`。
已经生成的 JSON 不同内容禁止覆盖；改协议需另建 version 与输出目录，不改已有文件。
完整32/base的 `production_inputs` 已生成，不再重新挑base或覆盖；CPU执行与异常记录在 `production_execution`。
`lock --inventory "$P/deployment_sources.json" --manifest MANIFEST --output LOCK` 只锁源码、权重、运行环境和新生产契约，**不是启动授权、不是 GPU preflight**。
历史 `deployments.cpu_smoke.lock.json` 只对应小清单；当前 `production_execution/deployments.production.lock.json` 独立绑定完整清单，`runtime_verified=false`。

由资源worker持有外部GPU mutex，再启动 `r114_run_queue --execute --finalize`；仅使用一张显式UUID的GPU，不挑卡、不等待或抢占其它GPU。
当前76个runtime分组、76个label分组。100个精确部署桶由原/干预对覆盖；必要group probe/replay共352次，加52次eager冷启动参考，共404次runtime调用，另有9,600次label调用；均未执行。
每个问题/方法一个持久进程，单端点batch=1，两端重求解且每次重置Python/NumPy/Torch/CUDA seed。52个eager组复用backend；24个RF/MoSES组每端点冷重建lazy子runner以保持预算/RNG，不冒称总共只加载76次权重。
每题全合法池runtime通过后才标注。首个不可用路线/profile/等价性失败/超时停止该题，保留未运行尾部，继续其它独立问题；不自动重试、修native协议或恢复删除候选。
每次调度退出（含部分失败/中断）均在独立run目录export、assemble，再执行只读 `r114_response_analysis` CSV/PDF。失败问题整题winner mask，runtime成本不导出为label；部分结果不标为完整生产完成。

## 阻断与建议规模

不是缺权重，而是**本版本GPU端到端尚未验证、真实GPU labels=0，资源worker尚未分配空闲GPU**；不将历史R112资源状态当作当前事实。本轮没有时间/GPU小时的测量或承诺。
固定32 bases ×16题：2,048端点、1,536对、9,600个label调用（原端点每base/候选按同锁定新协议重算一次，用于相关两臂），另列404次必要runtime调用；未增加base，不对TSP/ATSP伪造干预，不宣称训练或泛化收益。
少量 CPU smoke 仅证转换通路，不批准 GPU 生产。构造拒绝或响应不支持机制时保留失败/负证据，不自动十倍扩量。

Git：基准 commit 与职责边界已保存在 `implement-stage/SPEC.md`；仅新增指定代码、测试与 preparation 内容；未 stage/commit/push/切分支，未修改仓库配置。
`implement-stage/SILENT_ASSUMPTION_SWEEP.json` 如实记录无可核验指定模型 reviewer；单元测试不冒充外部模型复核。
