# R106 Engineering Review, Round 01

- Score: **8/10**
- Verdict: **ready**
- MAX_ROUNDS: 1; render_html: false.
- review_independence: same-family; acceptance_status: provisional.
- Scope: engineering correctness only, not scientific acceptance, efficacy evidence, main-ready receipt, or GPU launch authorization.
- Reviewer routing requirement: Sol/max only. No Astra or secondary model was invoked. This tool interface does not independently expose/verify the session's exact model ID or effort; this report does not certify that routing.

## Overall Judgment

在本轮直接读取的实现中，未发现需要阻断的 P0/P1 错误，也未发现有证据成立、需要修改的 P2 正确性问题。原生适配、共享图、原损失、LR 倍率及完整权重恢复的关键路径相互一致；这里的 ready 仅指这次有限工程审查未发现阻断项，不等于效果成功或已经允许启动。

按最终科学锁审查：TSP/CVRP micro256、ATSP128、其它题1024、eval128、logical1024，seed2，57,600 次成功更新。`protocol_lock.json:21` 与同步后的 `PLAN.md:59` 一致；默认 tentative64/256 不是正式协议要求。

## Blocking Findings

**P0/P1: none found.** 无需为通过本轮审查新增 gate、超参、hash 框架、兼容层或试验方案。

## Verified Code Paths

文件名均相对于 `code/V4/`，行号为本轮读取版本。

| 重点 | 直接代码依据与判断 |
| --- | --- |
| Masked native 及确定性 eval 图 | `r106_donor.py:50`, `:59`, `:84`, `:122`, `:148`, `:208`, `:221`：attention 排除无效 key，InstanceNorm 只在有效节点求 biased variance 并保留 eps/affine；MatNet 保留 FROM/TO、距离 mixed score 与同时读取上一层双角色；ReLD 只对有效节点复用原 eval topk2/dispatcher/experts，不增加辅助损失或噪声。已对照真实 ELG、MatNet/backbone、ReLD 源码，而非仅采信准备摘要。 |
| A 冻结、B 保留梯度及本 micro 图复用 | `r106_donor.py:192`, `:221`；`r106_model.py:182`, `:205`；`r106_experiment.py:161`：A 不建 native 参数梯度图，B 不 detach；原两次 selector dropout 前向共用一个 native 图并对组合 loss backward。输入/状态版本校验、最多两次使用和 finally 清空，没有跨 micro/update 的 stale 图复用路径。 |
| 原损失、真实行数归一化、AMP/clip | `r106_experiment.py:167`, `:295` 接入原 `r76_experiment.py:164` 与 `r84_experiment.py:83`；CE/comparison/risk/R-Drop 未改，每 micro 按实际行数/logical1024 缩放；整 batch 一次 unscale、全局 clip1、step，非有限时恢复同一 batch RNG 后重放，不推进成功更新。 |
| Native LR0.1 不被旧 loop 覆盖 | `r106_experiment.py:191`, `:213`, `:244`, `:251`, `:278`；旧 `r84_experiment.py:417` 虽逐组重设公共 LR，私有 AdamW 在真正 step 内乘 core1/native0.1，finally 恢复公共值；resume 校验并恢复组角色、倍率和状态。A native 无 grad，不发生 AdamW decay 更新。 |
| Fresh R84、相同完整初态、恢复保存本体 | `r106_model.py:250`, `:265`；`r106_donor.py:173`；`r106_experiment.py:390`, `:493`：保留 fresh R84 core、固定 seeded projections 与同三 pinned R102B native；完整 state 包含 native 参数和 buffers。恢复用无外部权重 I/O 的类布局加严格 state 加载，resume 也走保存本体，不把外读预训练权重当成微调后的权重。 |
| Raw-only、同 pool、VAL 锁后线上评估 | `r106_model.py:42`, `:205`；`r106_experiment.py:101`, `:305`, `:407`, `:462`, `:474`, `:616`；旧 `r84_experiment.py:463`, `:472`：白名单隔离监督字段、拒绝外部 native 特征；沿用冻结 raw cache/pool。whole18 VAL cost 严格更小才换 checkpoint，exact tie 留首个；之后恢复同一 whole model 做 TEST/LIB/timing，无 TEST/LIB 重新选型分支，无 R58。 |
| 报告及行政完成状态 | `r106_reporting.py:71`, `:98`, `:119`, `:152`, `:473`, `:543`；`r106_goal.py:35`, `:171`, `:222`, `:301`：nullable capture SHA 有 exact state/protocol/update 的 parent 证明，不把空行政字段当作丢失 VAL。最终 NSS CSV 已处理 BOM/header 对接；pending_report、科学 stage 字符串和历史 error 描述不被递归当成当前失败。whole-version 要两臂、同锁 TRAIN/VAL/TEST、真实线上 timing、LIB、fixed36、NSS4 与全部 PDF，publication 仍须显式调用。 |
| 最终动态 cap 接口 | `r106_experiment.py:62`, `:71`, `:287`, `:462`, `:486`；`r106_launch.py:105`, `:131`；`r106_wave.py:9`：final spec 从科学锁生成 argv，TRAIN 使用传入最终 caps，TEST/timing 从所选模型嵌入锁恢复 caps；eval 使用独立 cap，不错误裁成 TRAIN cap。 |

MatNet CPU native/adapter 约0.0049 的已披露差异不是 bitwise 声明；本轮未发现其背后存在角色、mask 或权重替换。ReLD eval 真正不读取的3,072个 w_noise 参数无 grad 符合该固定图，不应伪造零连接来制造 grad。

## Nonblocking Boundaries

- Main 已完成硬件见证。本轮只读其 `preparation/kernel_witness/final256_1024/complete.json:24`, `:55` 及科学锁记录：complete=true、exit_status=0、optimizer_steps=0，无正式 TRAIN 或 TEST/LIB；未重跑 GPU，也不要求再做一遍硬件见证。它证明输入驱动的图/梯度/显存路径，不是实际 AdamW 更新、完整训练吞吐或效果证据。
- `r106_experiment.py:213` 的 first-step witness 是有限参数切片记录，不冒充全参数变化审计；实际 optimizer/GradScaler 状态分配、首次正式更新及 worker UUID/affinity 仍由 main 的正常启动负责，不因本轮评分自动视作已观测，也不据此新增启动 gate。
- runner/reporter 收尾测试及最终 source freeze 仍是 main 既定交付事项。本轮未重复模型10 tests、行政27 tests或整库测试，不把历史通过数冒称本轮复跑结果，也不把文档 Pending 当成当前失败。源码继续变化时，本结论不自动覆盖未读的新逻辑。

## Evidence And Version Boundary

先读本目录 REVIEWER_MEMORY.md、PLAN.md、MODEL_PREP.md、RUNNER_PREP.md、preparation/LAUNCH_PLAN.md；直接审查全部七个新模块及必要旧接口/原生定义，使用普通文件 diff 对照前版 runner/reporter，未运行 Git。读取了对应 CPU tests 的关键 fixture；未修改源码、测试、数据、协议、reviewer memory 或其它代理文件。

本轮唯一新执行的 CPU fixture：在 `easynco`、`CUDA_VISIBLE_DEVICES=''`、`python -B` 下，为 A/B 只在内存构造 final spec，解析六个 train/TEST/timing argv 并调用当前 `protocol_lock` 校验；全部 PASS，最终256/128/1024、eval128、logical1024一致，并确认私有 evaluator 不受 TRAIN cap 裁剪。未写 fixture 文件、未构造模型、未调用 stage、optimizer、GPU 或任务启动。

末次读取源码 mtime（2026-10-23 +0800）：donor16:19:55、model16:22:13、experiment16:41:20、reporting16:45:34、launch16:35:19、goal16:36:20、wave16:20:13。reporting16:45 的 NSS CSV 小修已纳入；科学锁为08:47:56 UTC记录。仅写本 round01.md，不产生 HTML、ready.json、source freeze 或发布记录。

## Memory Update

本版本没有延续未决旧怀疑；本轮对 masked native、B autograd、单 micro 双 dropout 共享图、原损失/归一化、step-time LR0.1、完整 native checkpoint 恢复及报告/行政 parent-lock 路径未发现阻断错误。最终 cap 是256/128/1024、eval128、logical1024，已通过只读 CPU 接口 fixture；main 的 input-driven GPU 见证已有 PASS，不应再标缺失。保留边界：本报告仅 same-family/provisional 工程意见，收尾 CPU 证据、真实 optimizer 启动行为和最终发布由 main 负责；不把图/梯度见证、review 分数或流水线 complete 当作效果成功，不把 nullable capture、pending 或历史失败说明误判为当前科学失败。
