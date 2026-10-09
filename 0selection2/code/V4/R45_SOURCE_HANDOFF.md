# R45 源码与历史求解配置补全：助手交接任务

整理日期：2026-10-08。项目根目录：`/public/home/shiys/0selection2`。

## 1. 本次任务与边界

请补全“当前全局 solver 身份 -> 生成当前成本标签的实际实现与配置”的对应关系，
并据此完善 R45 的源码语料清单。不是重新设计 selector，不是重新生成成本标签。

R45 已实现：保留 R43A 的 `ScoreDifferenceSelector`、双流主体、比较监督及 maximin
规则；以离线生成的四视图代码向量替代原 34 维手工方法特征，保留 solver ID。
本次只补来源证据与实际配置，不修改模型、loss、候选池或数据划分。

当前状态：

- 全部 22 个 solver 都有候选核心源码，所选符号可以在本机解析。
- 已选 62 个不同 Python 文件，生成 67 个去重片段，共 100794 个模型 token。
- 清单中 11 项部署标为 `partial`、11 项标为 `unverified`，尚无 `confirmed`。
- 草稿有 36 条问题：22 条部署对应关系待确认，另有 14 条 Config 视图内容为空。
- 上述数字是这次快照，不表示 36 个互不相关的故障，也不表示标签已经被证明错误。
- R45 的 39 项离线检查已经通过；未生成正式代码向量，未调用收费 API，未启动训练。

限制：不要为了通过检查而批量把状态改成 `confirmed`、清空所有 `gaps`，或使用
当前默认参数冒充历史有效配置。找不到的内容要保留为明确缺口。
本次不上传源码、不调用 embedding API、不安装或运行一整套 solver 复现实验；
如确需少量复跑作为证据，先列出范围和预算，由项目负责人另行决定。

## 2. 已整理好的文件在哪里

| 文件 | 已有什么 | 如何使用 |
| --- | --- | --- |
| `/public/home/shiys/0selection2/code/V4/solver_source_manifest.json` | 22 个 solver 的 ID、候选入口、源码白名单、AST 符号、四视图归属、已知配置、部分权重/哈希及缺口 | **唯一需要补全的主清单**；先备份再修改 |
| `/public/home/shiys/0selection2/code/V4/runs/R45_solver_code_embeddings/corpus/missing_sources.md` | 当前逐 solver、逐角色缺口 | 自动生成的待办摘要，不手工删条目 |
| `/public/home/shiys/0selection2/code/V4/runs/R45_solver_code_embeddings/corpus/source_snapshot.json` | 生成现有草稿时使用的完整清单快照 | 保留作修改前对照，不当作另一份主清单 |
| `/public/home/shiys/0selection2/code/V4/runs/R45_solver_code_embeddings/corpus/corpus.json` | 清洗后的真实源码片段、token 数、内容/源码哈希、来源符号与行号、四视图索引及缺口 | 用于审阅已选择内容；不是生成好的 embedding |
| `/public/home/shiys/0selection2/code/V4/runs/R41_signal_audit/original_solver_provenance.md` | TSP 与 OVRPTW 原标签来源、已有推理预算、权重候选、decoder 歧义与证据边界 | 优先阅读，避免重复查已确认信息 |
| `/public/home/shiys/0selection2/code/V4/runs/R41_signal_audit/solver_manifest.json` | R41 成本文件来源、状态、部分权重哈希和实测核验记录 | 历史证据，只读，不覆盖 |
| `/public/home/shiys/0selection2/code/unified_selector/registry.py` | 当前问题、候选池与全局方法编号的生成规则 | 身份和列顺序以它及当前数据为准 |
| `/public/home/shiys/0selection2/code/V4/R45_README.md` | R45 模型接入、API、缓存、命令、兼容性与当前限制 | 了解消费者接口，不需要重写代码 |
| `/public/home/shiys/0selection2/code/V4/solver_code_corpus.py` | 白名单路径解析、AST 抽取、清洗、分块与 ready 判定 | 阅读清单字段实际如何生效 |
| `/public/home/shiys/0selection2/code/V4/runs/R45_solver_code_embeddings/implementation_checks.json` | 已执行的离线检查记录 | 不是训练结果 |

当前 R45 新文件尚未提交/推送；不能假定只打开 GitHub 就能看到本机这些材料。
交接压缩包只装说明、清单、草稿和相关代码，不包含数据集、权重、API 密钥，
也不包含全部外部 solver 仓库。外部助手仍需能够读取下述源码目录，或单独获取源码。

`data/README.md` 的部分方法名单已经过时，不能据它重建候选池。

## 3. 源码根目录与路径记法

清单中的 `root:relative_path` 不是模型输入，而是本机来源定位：

| root | 本机绝对路径 |
| --- | --- |
| `easynco`，下文简称 E | `/public/home/shiys/EasyNCO` |
| `sources`，下文简称 S | `/public/home/shiys/methods论文_mineru_output/source_code` |
| `reld`，下文简称 R | `/public/home/shiys/reld-nco-main` |
| `bridge`，下文简称 B | `/public/home/shiys/easynco_v3_bridge/scripts` |

例如 `reld:CVRP/weights/ReLD/model_epoch_90.pt` 对应
`/public/home/shiys/reld-nco-main/CVRP/weights/ReLD/model_epoch_90.pt`。

其他重要来源：

- 旧 NSS TSP 结果：`/public/home/shiys/0selection/neural-solver-selection/datasets/TSPtrain/results/`
  与 `/public/home/shiys/0selection/neural-solver-selection/datasets/TSPval/results/`。
- NSS 结果导入入口：`/public/home/shiys/easynco_v3_bridge/scripts/merge_nss_and_paper_source_into_data.py`。
- OMNI TSP train 记录：`/public/home/shiys/EasyNCO/results/nss_eval/omni_tsp/TSPtrain/eval.log`。
- MTPOMO/MVMOE OVRPTW 记录：`/public/home/shiys/EasyNCO/results/label_v4/mvrp/` 下
  `train/`、`val/` 内的 `mtpomo_ovrptw/`、`mvmoe_ovrptw/`。
- RELD 历史事件：`/public/home/shiys/easynco_v3_bridge/exports/reld_nss_like_no_aug_v1/_bridge_state/reld_bridge_events.jsonl`。
- 当前标签在 `/public/home/shiys/0selection2/data/<Problem><split>/`。
  如需确认原始来源，只读核对；不得把实例、成本、winner 或排名放进源码语料。

## 4. 当前候选池与完整 22 项待办

TSP 8 项，CVRP 10 项，ATSP 5 项，15 种 MVRP 每种 7 项。
下表 ID 必须保持不变；完整文件/类/函数白名单已经在主清单的 `implementations` 中，
下表仅给入口和核心目录，不能用本表取代完整符号清单。

| ID | Solver / 问题 | 已有实现入口与信息 | 需要补全/确认 |
| ---: | --- | --- | --- |
| 0 | BQ / TSP、CVRP | S `bq-nco/test_tsp.py`、`test_cvrp.py`；model/encoder/attention 及两类 decoding 已选 | 生成成本列的实际命令与源码版本；TSP/CVRP 权重；greedy/beam 与 beam 大小、增强、输入归一化 |
| 1 | DIFUSCO / TSP | E `eval.py -> TSPDiffusionPolicy`；GNN、扩散、merge/2-opt 已选；有 tsp100 候选权重 | 原入口/权重对应；扩散步数与日程、采样数、稀疏图设置、增强、后处理及随机策略 |
| 2 | DIFUSCO500 / TSP | 与 DIFUSCO 共用候选实现；有 tsp500 候选权重 | 同上；确认与 DIFUSCO 的真实差别，不从后缀猜推理次数 |
| 3 | ELG / TSP、CVRP | E `eval.py -> ELGPolicy`；AM 主干、local policy、distance penalty 与初始化已选 | 各问题实际源码与权重、local-size、解码/推理预算、增强；不能用后来 paperlike 配置替代原标签配置 |
| 4 | GLOP / ATSP | S `GLOP/eval_atsp/test_glop.py`；ASHPP/ATSP 模型与重构 tester 已选 | 原权重及网络参数；分区与局部重构设置、迭代数、推理预算；不能换成 CVRP GLOP 实现 |
| 5 | ICAM / CVRP | E `eval.py -> CVRPICAMPolicy`；encoder/decoder/初始化已选 | 原入口、权重、实际 adaptation 设置、模型参数、多起点/采样/增强与推理预算 |
| 6 | ICAM_ATSP / ATSP | B `paper_source_atsp_runner.py::ICAMATSPRunner`；S `ICAM/ICAM_ATSP/ATSPModel_ICAM.py`；已记 neighbors=50、N 起点、aug1 | 各规模权重、最终有效 model_params；确认近邻行/列转换和实际运行配置对应 |
| 7 | LEHD / TSP、CVRP | E `eval.py -> LEHDPolicy`；encoder、decoder、两类 iteration 已选；有 TSP 候选权重 | 两类问题实际权重与网络参数、重构次数、增强、解码预算；TSP 原 NSS 配方缺失 |
| 8 | MATNET / ATSP | E `eval.py -> ATSPPolicy`；matnet encoder/decoder/initialization 已选 | 历史源码版本、权重、网络参数、多起点/采样/增强及预算 |
| 9 | MATPOENET / ATSP | E `eval.py -> MatPOENetPolicy`；独立登记，不是 UNICO bridge | 原生成入口、权重、位置编码配置、网络参数与推理预算 |
| 10 | MTPOMO / 15 MVRP | E `eval.py -> MTPOMOPolicy`；已记 N 客户起点、aug1、记录 seed1234 | **有效 greedy/sampling 尚未确认**；权重、模型参数、原约束 mask/环境依赖与输入转换 |
| 11 | MVMOE / CVRP、15 MVRP | E `eval.py -> MoEPolicy`；encoder/MoE/decoder 已选；已记 N 客户起点、aug1、记录 seed1234 | 同上；另确认 CVRP 与 MVRP 是否用不同权重、环境、模型参数或实现 |
| 12 | MoSES_CaDA / CVRP、15 MVRP | B `paper_source_routefinder_runner.py::MosesRunner`；CaDA/LoRA 配置已有实际值，见主清单 | 规模50/100权重对应；原 rl4co decoder/环境版本、有效解码/RNG；确认各问题推理配置 |
| 13 | MoSES_RF / CVRP、15 MVRP | 同一 bridge 的 RF 分支；LoRA 与 prenorm/postnorm 配置已有实际值 | 同上；不能把 RF 与 CaDA 分支混用 |
| 14 | OMNI / TSP、CVRP | E `eval.py -> OMNI_POMO_Policy`；**TSP train 有记录，val 来自 NSS**；有候选权重 | 分开确认 TSP train/val 是否同一部署；补 CVRP 配方；确认解码、起点、增强后到底取 no_aug 还是 best_aug |
| 15 | RELD_CVRP / CVRP | B `run_reld_cvrp_single_bridge.py::instantiate_single_cvrp_model`；R `CVRP/CVRPModel.py`；候选权重 model_epoch_90.pt 存在 | 原标签实际权重绑定与哈希；解析后的网络、输入转换和推理配置 |
| 16 | RELD_MOEL / 15 MVRP | B `reld_bridge_common.py::instantiate_reld_model(model_key=reld_moe_light)`；权重/哈希/模型参数已记录；OVRPTW 复跑成本匹配 | 补其他注册问题的调用、输入字段/约束环境与部署证据；不要把 OVRPTW 已确认内容全部重做 |
| 17 | RELD_MTL / 15 MVRP | 同 bridge，model_key=reld_mtl；权重/哈希/模型参数已记录；OVRPTW 复跑成本匹配 | 同上 |
| 18 | RouteFinder / CVRP、15 MVRP | B `paper_source_routefinder_runner.py::RouteFinderRunner`；OVRPTW 已确认8倍增强、N 起点、best_aug | 规模相关权重、原 rl4co decoder/环境版本与有效配置；确认其他问题部署 |
| 19 | T2T / TSP | E `eval.py -> TSPT2TPolicy`；引导去噪/merge/2-opt 已选；有 tsp100 候选权重 | 原入口/权重；扩散与梯度搜索预算、采样、增强、后处理、归一化与随机策略 |
| 20 | T2T500 / TSP | 与 T2T 共用候选实现；预期位置的 tsp500 权重未找到 | 原权重身份或文件、源码/配置及后缀实际含义；不能拿 tsp100 权重冒充 |
| 21 | UNICO_MatPOENet / ATSP | B `paper_source_atsp_runner.py::UnicoMatpoenetRunner`；S `UniCO/MatPOENet/ATSPModel.py` | 各规模权重与最终 encoder 层数/网络参数、位置编码与实际推理配置；不能与 MATPOENET 合并 |

## 5. 已经找到的权重与配置，不要丢失

### 5.1 权重文件

以下候选文件在本次交接时存在，但“文件存在”不等于“已证明用它生成了当前标签”。

| 方法 | 路径，相对第 3 节根目录 |
| --- | --- |
| BQ TSP | S `bq-nco/pretrained_models/tsp.best` |
| DIFUSCO | E `pretrained/DIFUSCO_pretrain/difusco_tsp100.ckpt` |
| DIFUSCO500 | E `pretrained/DIFUSCO_pretrain/difusco_tsp500.ckpt` |
| ELG TSP | S `ELG/TSP/weights/ELG.pt` |
| LEHD TSP | E `pretrained/lehd/lehd_tsp100.ckpt` |
| OMNI TSP | E `pretrained/omni/omni_tsp_maml_fomaml.ckpt` |
| T2T | E `pretrained/T2T_pretrain/t2t_tsp100.ckpt` |
| RELD_CVRP | R `CVRP/weights/ReLD/model_epoch_90.pt` |
| RELD_MOEL | R `Multi-Task/pretrained/reld_moe_light/epoch-5000.pt` |
| RELD_MTL | R `Multi-Task/pretrained/reld_mtl/epoch-5000.pt` |

E `pretrained/T2T_pretrain/t2t_tsp500.ckpt` 当前不存在；这是该路径未找到，不是已经
证明所有目录都没有对应权重。R45 不编码权重字节，但仍需知道实际权重/版本身份。
TSP/CVRP 权重不能因为方法同名就假定相同。

RELD 的两个已记录 SHA256：

```text
RELD_MOEL  2ce5fb49950c4c9f51d2015a125744be9960aa0c60c609cfe235e74d44ad4be3
RELD_MTL   2a67f0c1ada0a05563280fcd56d8375b1b0007978bad55c625c862fc3cc3b7ab
```

### 5.2 已确认信息的范围

- **OMNI TSP train**：记录为顶层 greedy、batch1、seed1234、N 起点、8-fold，
  无迭代/微调；当前标签取 `no_aug_score`，不是最佳增强结果。不能推广至 val/CVRP。
- **MTPOMO/MVMOE OVRPTW**：N 起点、aug1、batch64、seed1234；但实际 decoder 仍不确定。
- **RouteFinder/MoSES OVRPTW**：8-fold × N 起点，取最佳增强；权重规模分桶为
  `N<=75 -> 50`，否则100，不是选择最近规模。该事实先按 OVRPTW 已核范围记录。
- **OVRPTW 输入转换**：R41 已核对 train/val 的 depot、坐标、需求、service/TW；
  需求归一化且 capacity=1；paper-source depot TW 为 `[0,inf]`，RELD 为 `[0,3]`。
- **RELD 两个多任务模型的 OVRPTW 推理**：显式 argmax、N 客户起点、aug1、sample1，
  历史 batch128；两次独立运行成本及二者胜负与旧标签匹配。
- **RELD 共同模型参数**：embedding128、encoder6、decoder1、qkv16、heads8、
  ff_hidden512、logit_clipping10、norm=none、norm_loc=norm_last、problem=Train_ALL。
- **MOEL 额外参数**：4 experts、topk2、node routing、input_choice、ffidt=true，
  expert_loc 为 Enc0..Enc5 与 Dec。完整值已经在主清单，不要重新猜测。
- **MoSES 配置**：主清单保存完整 `resolved_config`，包括 rank=[32,32,32,32,32]、
  4 experts、top_k4、alpha1、temperature1、rms、silu，以及动态 topK/basis/linear 开关。
  RF 的 prenorm/postnorm 为 true、LoRA activation=softplus；CaDA 的这两个 norm 开关为
  false、activation=sigmoid，并有 sparse_ratio0.5、sparse_applied_to_score=true。

### 5.3 MTPOMO/MVMOE 解码歧义必须具体处理

R41 检查的 204 份旧 shard 配置写了顶层 `decoder_strategy=greedy`，但没有
`settings.module.decoder_strategy`。所检查历史版本
`7f50c94e9b4e312696d71014563bc7afa7ee0ff5` 的调用链显示：

- `eval.py` 没有把顶层 decoder_strategy 转交给 module。
- `phases/rl/ar_reinforce.py` 的 module 默认 sampling。
- 初始化只调用 eval，不会自动改成 greedy；sampling 分支使用 multinomial。

这使 sampling 有源码支持，但没有当次运行的版本/有效参数记录将其最终绑定。
应找原运行脚本、配置解析结果、源码快照或运行时证据；不能仅看顶层字段就认定 greedy，
也不能仅看当前默认值就认定历史一定 sampling。

## 6. 每个部署需要补充什么

这里的“部署”是实际不同的实现/配置组合，不是强制每个问题复制一份内容。
若源码和配置相同，可在 `problems` 中列出适用问题并去重；若确实不同，分别登记。

| 信息 | 需要记录的内容 | 放在哪里 |
| --- | --- | --- |
| 方法身份 | solver_name、solver_id、适用问题；必要时分 train/val、规模段 | `solvers[].deployments[]`，不得改 ID |
| 实际入口 | 生成成本列的脚本/命令、实际调用链，不只是同名 Model.py | implementation 的 entrypoint 与部署 evidence |
| 关键源码 | 输入转换、encoder/decoder、状态更新、推理/后处理及关键依赖的真实文件和符号 | `implementations.<key>.views.<role>.sources` |
| 网络配置 | 实际生效的层数、维度、heads、专家/LoRA/位置编码等，按方法适用填写 | 部署 `resolved_config` |
| 推理配置 | greedy/sampling/beam、起点/采样数、迭代/扩散/重构预算、增强、结果归约方式 | 部署 `resolved_config` |
| 输入与目标定义 | 归一化、depot、需求/TW/service、成本计算/缩放等合法实现机制 | 对应源码及 `resolved_config`；不得加入观测成本值 |
| 权重与版本 | 实际 checkpoint 路径/哈希、源码 revision；关键 decoder/环境依赖版本 | 部署溯源字段，不放到语义配置文本 |
| 证据 | 哪个文件/行/日志字段/配置快照证明了哪个值及部署对应 | 建议新增 `evidence` 列表 |
| 仍未知内容 | 精确到方法、问题/规模/分割、字段，区分未找到与尚未核查 | 保留 `gaps` 和 partial/unverified 状态 |

不要求补齐论文未记录的训练历史、每层参数量或宣传性的“适合聚簇/大规模”能力。
随机种子如果只用于来源核验，保存在证据；只有实际已知且影响运行方式的配置才写入
语义内容。无法恢复某个历史 digest 时可以明确记录限制，不伪造精度更高的证据。
不需要仅为了 R45 的源码表示就执行整个 R41 全候选复跑工程。

## 7. 如何修改主清单

1. 保留修改前快照，主清单路径保持不变，顺序严格与当前 `GLOBAL_SOLVERS` 对齐。
2. 先从标签生成入口反向确认调用，复用现有已确认参数，不把候选文件自动当成原实现。
3. 将有证据的实际参数填入对应部署的 `resolved_config`；路径/哈希/日志证据另存。
4. 确实不同的实现新增 `implementations` 条目，再由不同 `deployments` 引用；不能只在
   部署上写一个新文件名却仍抽取旧 implementation 的源码。
5. 四视图顺序固定为 `encoder / decision / inference / config`。
   `sources` 是明确的 `.py` 文件与 AST 符号白名单，不能把目录/日志/README 当源码送入。
6. Config 视图目前大部分 `sources=[]`：通常通过补全 `resolved_config` 生成它的内容。
   不要只粘配置模板或未解析的参数名；已有部分值可保留，未知项继续列入 gaps。
7. 缺失核心源码不能标成 absent。确实没有独立 inference/config 内容时，抽取器支持
   `status=absent` 加真实理由，但不能拿它绕过未确认的实际推理配置。
8. 每条 gap 确实解决后再删除；全部关键对应关系有证据时才标 `binding_status=confirmed`。
   若仍无法核实，交付 partial/unverified 和具体原因，不虚报全部完成。

建议部署记录格式如下，仅为字段示例，不是已确认事实，不直接用占位内容替换现有条目：

```json
{
  "implementation": "现有或新增实现键",
  "problems": ["实际适用问题"],
  "splits": ["实际核实的 split"],
  "binding_status": "partial",
  "resolved_config": {},
  "checkpoint": "根目录键:实际相对路径",
  "checkpoint_sha256": "实际计算得到的哈希",
  "source_revision": "实际核实的版本或明确说明未知",
  "evidence": [
    {
      "file": "证据文件的绝对路径",
      "locator": "函数、行号或配置字段",
      "supports": ["这份证据具体确认了什么"]
    }
  ],
  "gaps": ["尚未解决的具体问题"]
}
```

不要把 API key、checkpoint 路径/哈希、实测 cost、winner、gap、排名、用户实例、
测试结果或推测的能力描述填进 `resolved_config`。路径和哈希留在溯源字段。
合法的路线长度/成本计算函数可以保留，不是禁止一切名称里带 cost 的代码。

## 8. 本次助手的交付与验收

请交付：

- 更新后的主清单，保留22个方法及原顺序。
- 一份 `/public/home/shiys/0selection2/code/V4/runs/R45_solver_code_embeddings/solver_source_completion.md`，
  按方法列出原先已有、此次新增证据、实际配置、尚未解决项；注明对应问题/规模/split。
- 必要的配置/源码证据定位；能由当前文件计算的哈希自己计算，不要求用户逐个手工提供。
- 重新生成的语料与来源快照，先输出到独立 `corpus_reviewed/`，不要覆盖旧草稿。
- 完成与无法完成都如实记录；不要为了得到 ready=true 猜值、删 solver 或用随机向量代替。

环境和检查命令如下；这些命令不训练、不运行 solver、不调用 Voyage API：

```bash
cd /public/home/shiys/0selection2
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco

HF_HUB_OFFLINE=1 python -m code.V4.solver_code_corpus --draft \
  --output /public/home/shiys/0selection2/code/V4/runs/R45_solver_code_embeddings/corpus_reviewed

python -m code.V4.solver_code_embeddings --stage plan \
  --corpus /public/home/shiys/0selection2/code/V4/runs/R45_solver_code_embeddings/corpus_reviewed/corpus.json

python -m unittest code.V4.test_r45 -v
```

缺口全部解决后，用同一路径再运行一次不带 `--draft` 的 corpus 命令，确认能够通过
生产检查；`--stage plan` 只统计，没有费用。不要在此任务运行 embed、assemble、train、
test 或 `--stage all`。后续正式向量生成与训练由项目负责人另行启动。

除 ready 检查外，需人工审阅每个方法的语义内容是否来自正确实现：状态通过不等于事实
自动被证明。检查四视图覆盖关键输入/决策/推理机制、配置是实际值、来源可追踪、没有
观测结果泄漏。相同实现可得到相同代码向量，不能为了区分 ID 而虚构内容。

## 9. 压缩包和回传说明

本次交接包位于：
`/public/home/shiys/0selection2/code/V4/runs/R45_solver_code_embeddings/R45_source_handoff.zip`。

包内的主清单、source_snapshot 和 corpus 都是交接时的快照。应修改项目中的主清单，
而不是同时维护多个版本。外部助手若不能读取本机 E/S/R/B 目录，需要先获取相应源码
和历史运行记录；只有 GitHub 项目代码和草稿片段不足以恢复所有历史配置。

最终回传更新后的主清单、补全报告及新增证据即可。不回传密钥，不覆盖历史 R41/R43
结果，不宣称这次补文档已经证明源码语义能提高选择性能。
