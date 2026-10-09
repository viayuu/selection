# R45 求解器源码与配置补全报告

核对日期：2026-10-08。项目：`/public/home/shiys/0selection2`。

## 1. 交付与边界

已更新主清单 `code/V4/solver_source_manifest.json`，保留 22 个 solver 的名称、ID 和顺序。
本次只读核对原始标签、配置、权重元信息和源码；没有重新求解、生成标签、训练、上传源码或调用 API，
没有修改 EasyNCO、外部求解器、模型/loss、候选池、数据划分或正在运行的实验队列。

交付文件均位于 `code/V4/runs/R45_solver_code_embeddings/`，主清单除外：

| 文件 | 内容 |
| --- | --- |
| `solver_source_manifest.before_completion.json` | 修改前主清单原样备份 |
| `来源补全核对证据.json` | 384 组标签来源的精确指纹核对、6864 份配置/日志记录、34 个候选权重路径的核对信息 |
| `corpus_reviewed/corpus.json` | 按补全清单重新抽取的独立审阅版语料 |
| `corpus_reviewed/source_snapshot.json` | 本次清单快照 |
| `corpus_reviewed/missing_sources.md` | 自动生成的剩余缺口，不手工删改 |
| `补全验收.json` | 补充验收：身份、原信息保留、标签完整性、来源、配置、权重和语料一致性 |
| `离线校验日志.log`、`离线校验记录.json` | 本次草稿生成、费用计划、单元测试与补充验收的实际执行记录 |
| `collect_source_completion_evidence.py` | 本地来源核对脚本；不导入或执行 solver |
| `validate_source_completion.py` | 独立的只读补全验收脚本 |

原 `corpus/`、R41 材料、交接文档、交接 ZIP、R45 模型与测试代码保持原样。
权重 SHA256 只表示本次读到的本地文件字节；除已有 R41 证据外，不能冒充历史任务记录的权重摘要。
清单原有 `baseline_commit` 也不是本次找回的历史 solver 运行版本。

## 2. 证据强度

本次区分三类证据，不混用：

1. **标签来源确定**：按原始实例 ID 顺序比较数值的 float64 指纹，并核对原始导入/导出入口。
   384/384 组均存在完全一致的历史来源，覆盖 18 个问题、128 个问题—方法组合和三个 split。
   各组合 train=10000、val=1000、test=1000，共核对 1536000 个成本项。
2. **历史配置有记录**：原 `.hydra/config.yaml` 或 eval.log 中明确记录了模型、权重名称及参数。
   解析插值后保留原路径、摘要；eval.log 另保留字段行号，供逐项追查。
3. **当前源码还原调用方式**：bridge 的明确参数、环境状态与 mask、配置导入、权重选择和归约逻辑。
   这些内容注明 `config_evidence_scope`；没有当次源码快照时，不写成历史运行已经完全确认。

精确文件匹配不等于求解复跑，不证明原始 RNG、依赖版本、实际 checkpoint 字节或调用代码版本。
证据文件只输出标签指纹和数量，不输出观测成本值；它和日志/权重路径不进入源码语料的语义配置视图。

### 当前成本列的来源

| 问题 | 当前列对应的实际来源 |
| --- | --- |
| TSP | BQ、DIFUSCO、DIFUSCO500、ELG、LEHD、T2T、T2T500 三个 split 均为 NSS；OMNI train/test 为 EasyNCO，val 为 NSS |
| CVRP | BQ、ELG、LEHD、MVMOE、OMNI 均为 NSS；ICAM 为 EasyNCO；MoSES_CaDA、MoSES_RF、RouteFinder 为 paper-source；RELD_CVRP 为单任务 RELD 导出 |
| ATSP | GLOP、MATNET、MATPOENET 为 EasyNCO；ICAM_ATSP、UNICO_MatPOENet 为 paper-source |
| 15 种 MVRP | MTPOMO、MVMOE 为 EasyNCO label_v4；MoSES_CaDA、MoSES_RF、RouteFinder 为 paper-source；RELD_MOEL、RELD_MTL 为多任务 RELD 导出 |

EasyNCO 的这些现有成本列匹配的是 `no_aug_score`，不是 `aug_score`；paper-source 的 RF/MoSES/UniCO
则取最佳增强结果。本次忠实记录现有标签口径，没有按某个统一增强口径重写数据。

## 3. 逐方法补全

下表的“参数已补”不代表状态已经 confirmed；具体来源、适用 split/规模和未确认字段见主清单。
S 为外部 `source_code` 根目录，E 为 EasyNCO，R 为 reld-nco-main，B 为 bridge/scripts。

| ID / Solver | 本次找到或纠正的内容 | 仍缺什么 |
| --- | --- | --- |
| 0 BQ | TSP/CVRP 六份成本列均绑定 NSS；补 `S:bq-nco/pretrained_models/{tsp,cvrp}.best` 候选及各自 SHA256 | 原 NSS 命令、源码/权重身份、beam/greedy、beam 宽度、增强和归一化；不填当前默认参数 |
| 1 DIFUSCO | 三个 split 均为 NSS；补本地 tsp100 候选权重摘要，保留扩散/合并/2-opt 候选源码 | 原扩散日程/步数、采样、稀疏图、后处理预算及源码/权重绑定 |
| 2 DIFUSCO500 | 三个 split 均为 NSS；补独立 tsp500 候选权重摘要 | 原配置及与 DIFUSCO 的真实差别；不把后缀解释为推理次数 |
| 3 ELG | TSP/CVRP 都是 NSS；补两类候选权重；明确 CVRP EasyNCO 复跑结果不是当前标签 | 原 local-size、网络、decoder、增强、预算及来源版本 |
| 4 GLOP | 改用真实 ATSP initializer：CPU 随机排列 + random insertion，NoIteration；补实际 Python 绑定与输入准备 | 历史 native 核心/二进制和源码版本；当前不是论文完整局部重构版 GLOP |
| 5 ICAM | CVRP 三个 split 绑定 EasyNCO 原输出；补历史 12 层、FF512、instance norm、clip50、模块 greedy、N 起点、aug8、无迭代/微调和命名权重 | 历史源码、未日志化的 wrapper 默认值和运行时权重摘要 |
| 6 ICAM_ATSP | 补实际 `_get_tester/run_pt` 而非未调用的 tester rollout；补128维、每子编码器6层、FF512、clip50、greedy和固定 icam_atsp 权重；保留近邻50/N起点/aug1 | 原导入配置快照、源码/依赖版本及 RNG；当前代码还原不等于历史配置快照 |
| 7 LEHD | TSP/CVRP 都绑定 NSS，补两类候选权重；不再使用不同成本的 EasyNCO CVRP 复跑配方解释旧列 | 原网络/重构次数/decoder/增强/预算及实际权重、版本 |
| 8 MATNET | 243 份原规模配置：256维、5层、16头、qkv16、FF512、clip10、ms16、N起点、aug128、模块greedy、无迭代；全部用 matnet_atsp100 权重 | 原源码版本和实际加载权重字节；现有列只选增强索引0 |
| 9 MATPOENET | 243 份配置：512维/PE512、固定5层、16头、qkv16、FF512、clip10、ms16、N起点、aug8、greedy、NN初始化、mix权重；补PE/NN源码 | 历史源码/native NN 版本；cosh/scaler100 来自当前对应代码，不是原二进制快照 |
| 10 MTPOMO | 45 组标签和2295份相关配置覆盖全部15问题/3split；补128维、6层、8头、qkv16、FF512、clip10、instance norm、mode1、权重及真实 POMOInitialization 调用链 | 顶层greedy未传到module，有效decoder仍未确认；旧环境命名空间/实际mask版本，B类有效起点数 |
| 11 MVMOE | 将 NSS CVRP 与15种 EasyNCO MVRP分开；MVRP补完整128维/6层/instance norm、4专家/topk2/node/input_choice和权重，保留既有seed/aug信息 | CVRP整个原配方仍缺；MVRP decoder歧义、历史mask版本与实际起点数仍缺 |
| 12 MoSES_CaDA | 保留已确认 LoRA/CaDA 参数；补50/100权重及<=75分桶规则、实际bridge/环境mask/输入转换、8倍增强取best | 原rl4co/环境/import版本、有效decoder/RNG、base网络的历史解析值和运行时权重摘要 |
| 13 MoSES_RF | 保留 RF 的 prenorm/postnorm=true 和softplus分支；补独立RF权重、分桶、实际输入/环境/归约 | 同上；不与CaDA权重或分支混合 |
| 14 OMNI | 分TSP train/test、TSP val、CVRP三个部署；仅train/test填历史6层/8头/qkv16/FF512/batch_no_track/clip10、greedy、N起点、aug8、禁用微调和NoIteration | val及CVRP为NSS且原配方缺失；train/test历史源码和未记录默认值仍缺 |
| 15 RELD_CVRP | 修正错误的多任务推理白名单，改为单任务 `run_no_aug_scores`；补128维/6层/8头/qkv16/FF512/clip50、forcing_first_step=false、greedy、min(N,100)起点、aug1和权重摘要 | 原单任务bridge/source/dependency版本和运行时权重摘要 |
| 16 RELD_MOEL | 保留R41权重摘要、网络/MoE和OVRPTW复跑证据；新增全部45组标签来源、15种环境load/reset/step/距离计算、输入转换及dispatch源码 | 不把文件匹配扩大成其他问题的solver复跑；历史约束环境版本与逐问题完整运行元数据仍缺 |
| 17 RELD_MTL | 同样保留R41已确认信息；补全部15问题/3split原标签绑定和环境/输入/推理源码 | 同上；不补造历史复跑记录 |
| 18 RouteFinder | 保留aug8/best_aug，补50/100权重、<=75分桶、bridge和实际MTVRP状态/mask/输入转换 | 原checkpoint解析后的网络/decoder及rl4co/环境/RNG版本仍不明 |
| 19 T2T | 三份成本列绑定NSS；区分EasyNCO tsp100与T2TCO tsp100两个不同摘要的候选 | 原扩散/梯度引导/重写/采样/归一化/增强和后处理预算，真实源码与权重身份 |
| 20 T2T500 | EasyNCO预期路径仍不存在，但新找到 `S:T2TCO/ckpts/tsp500_categorical.ckpt` 候选及摘要 | 不能因此认定它就是NSS原权重；原部署和推理预算仍缺 |
| 21 UNICO_MatPOENet | 补实际bridge手动rollout、位置编码与NN输入，安全CPU读取checkpoint键：20/50为8层，100/mix为5层；分别登记；512维/16头/qkv16/FF512/clip10/ms16、softmax采样、N起点、aug8取best | 原bridge/import配置及native版本、实际batch/RNG历史和运行时权重摘要；不与EasyNCO MATPOENET合并 |

## 4. 四项会影响方法身份的重要纠正

### GLOP 当前列不是论文完整 GLOP 推理

原 shard 明确配置 NoIteration。实际 `GLOPInitialization.run` 的 ATSP eval 分支只调用
`insertion.atsp_random_insertion`，先生成一个 CPU 随机排列，执行一次插入初始化。
模型虽加载下层权重，但该分支不调用它，配置中的 revision_lens/iter 没有执行。
因此新增 `glop_atsp_init_only` 并由当前 GLOP 部署引用；原论文重构候选保留为未使用条目。
Python encoder 视图放真实输入准备，不伪造一个“已执行的神经编码器”；native C++ 缺口明确保留。

### RELD_CVRP 与多任务 RELD 的推理不是同一个函数

单任务使用 `run_reld_cvrp_single_bridge.py::run_no_aug_scores`，轨迹数 min(N,100)、clip50、aug1、显式greedy。
多任务使用 `reld_bridge_common.py::_run_no_aug_batch`，已有参数是N客户起点、clip10、aug1、显式argmax。
原清单误把单任务推理指向多任务函数，已修正；不能把 YAML 的测试模板 aug8 当作实际bridge参数。

### OMNI/MVMOE 同名列不代表相同部署

OMNI TSP train/test 匹配 EasyNCO，val 匹配 NSS；CVRP 也匹配 NSS。
MVMOE CVRP 匹配 NSS，15种MVRP匹配 EasyNCO label_v4。
这些来源分别登记。CVRP ELG/LEHD/OMNI 虽然有可读的 EasyNCO 日志，但数值核对不同，
不能使用这些日志“补齐”当前旧标签的原配方。

### MATPOENET 与 UniCO 的模型、解码和权重不同

EasyNCO MATPOENET 各规模固定5层、mix权重、greedy，并选择增强索引0。
UniCO bridge优先选exact-scale权重，否则mix，通过权重键覆盖层数；当前20/50为8层，其他规模5层，
本地test模块为softmax，多项式采样并取8份增强中的最好结果。
UniCO先把矩阵乘1e6、取整写TSPLIB，读取时除1e6；两者的输入路径也不能混为一谈。

## 5. 约束、起点与未知项

- MTPOMO/MVMOE 的2295份相关旧配置均只有顶层 greedy，没有 module.decoder_strategy。
  R41所检历史代码支持默认sampling假说，但缺少把当次job版本绑定到该代码的记录。
  主清单的semantic config没有填写一个虚假的有效decoder值；也没有用之后扩容的sampling方案补造历史。
- 旧配置写 pomo_size=N 是**配置值**。当前mode1环境在B类问题可能限制到floor(0.8N)个正需求起点。
  没有历史mask版本时不能把所有B类的实际轨迹数都宣称为N；原信息保留并加作用域说明。
- RF/MoSES按输入符号需求分linehaul/backhaul，容量为1，保留客户TW/service/route_limit，depot TW为[0,inf]。
  RELD OVRPTW已有核对的depot TW为[0,3]。本次没有消除这些实现差异或修改原实例。
- 没有在无历史依赖证据时，拿当前rl4co默认greedy/版本代替原部署；没有靠方法名猜训练规模或采样预算。

优先还需寻找的外部材料：原NSS逐solver运行脚本及源码/权重记录；MTPOMO/MVMOE当次源码版本与有效decoder；
paper-source原环境锁定/解析配置；历史native库/依赖与UniCO批量及RNG记录。
本次不以重新求解、付费embedding或重新训练来绕过这些材料缺失。

## 6. 离线验收与就绪状态

| 项目 | 修改前 | 本次审阅版 |
| --- | ---: | ---: |
| 全局solver数量 | 22 | 22 |
| 部署条目 | 22 | 26 |
| partial / unverified / confirmed | 11 / 11 / 0 | 16 / 10 / 0 |
| 选中Python文件 | 62 | 93 |
| 去重片段 | 67 | 85 |
| 真实模型token | 100794 | 154398 |
| 空Config视图 | 14 | 10 |
| 自动缺口条目 | 36 | 36 |
| ready | false | false |

缺口数没有直接减少，因为正确拆分了 OMNI、MVMOE 和 UniCO 的部署；新增作用域不能隐藏为一项“已完成”。
剩余10个空Config均为原配方未找回的NSS部署。所有源码白名单均可解析，无缺文件或缺AST符号；
仍未生产就绪的原因是历史绑定/配置缺口，而不是通过删空gaps来规避验证。

本次在conda easynco、CUDA不可见、Hugging Face离线、wandb禁用的环境中执行：

```bash
python -m code.V4.solver_code_corpus --draft --output code/V4/runs/R45_solver_code_embeddings/corpus_reviewed
python -m code.V4.solver_code_embeddings --stage plan --corpus code/V4/runs/R45_solver_code_embeddings/corpus_reviewed/corpus.json
python -m unittest code.V4.test_r45 -v
python -m code.V4.runs.R45_solver_code_embeddings.validate_source_completion
```

费用计划仅报告85片段/154398个未缓存token，不创建embedding或上传源码。
本次 `code.V4.test_r45` 实际发现并通过17项测试，不把交接材料中此前合并统计的“39项”当作本次测试数量。
额外17项补全验收全部通过；实际退出码见 `补全验收.json` 与 `离线校验记录.json`。
首次直接执行验收文件时遇到标准库code模块路径冲突，已改为上述项目模块启动方式；首次记录保留，不影响求解器。
原始ZIP内其余文件、旧语料和R41材料按字节对照；原已选solver源码也按旧语料摘要对照，
因此能检验本次没有动核心代码或覆盖原历史材料。

结论：已完成本机可证实的来源与配置补全，并把无法恢复的部分明确留下；
**没有虚报22个solver的历史部署已全部确认，暂不批准正式embedding上传。**
