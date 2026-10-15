# R81 当前标签部署清单

范围：当前 single-RELD 场景中的 VRPTW、VRPLTW，各六列；不涉及 R58 场景重建。

R58 第3点在此仅完成 CURRENT 成本列的部署条件记录，保留明确的运行环境缺口；不表示完成 R58 全量标签或训练。`code/V4/runs/R58_unified_scenario/task_contract.json` 是拟用于新 R58 场景的契约，不是这些现有成本列的生成契约，不能用它反推现有标签；R58 仍为 deferred。

## 数据集身份与划分

- CURRENT：`new_only_seed2_B_filtered_MOEL_as_reld_mt_v1`，目录 `/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014`。
- Parent：`new_only_seed2_B_without_RF_MoSES_v1`，目录 `/public/home/shiys/selection_data_new_only_B_filtered_seed2_20261014`。
- 冻结来源：`/public/home/shiys/训练数据_公共完整标签_20261013_1923`；生成记录：`/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/labels`。

| 问题 | TRAIN | VAL | TEST | 每题合计 | 每划分候选数 |
|---|---:|---:|---:|---:|---:|
| VRPTW | 70,114 | 3,895 | 3,895 | 77,904 | 6 |
| VRPLTW | 70,114 | 3,895 | 3,895 | 77,904 | 6 |

两题固定列序均为 `MTPOMO, MVMOE, MoSES_CaDA, MoSES_RF, reld_mt, RouteFinder`，global ID 为 `10, 11, 12, 13, 16, 18`。Parent 中这两题有七列；child 仅删除 `RELD_MTL`，将原 `RELD_MOEL` 原值命名为 `reld_mt`，保留 parent 零基列 `0,1,2,3,4,6`。没有 best-of-two、平均或成本混合，没有重跑 solver；parent 到 child 未改输入或实例划分。B 任务的 RF/MoSES 删除规则不适用于这两题，不能把全场景 `removed_methods` 的并集当作本题候选池。

CURRENT/parent 场景清单的 SHA256 分别为 `01713f40b5844a81d3e396e38a07c79e354135e542bbcf75ca3df84876b9e25b`、`210da080d9cfc122f95e40219ddc1e1f10dc35cf5515bcae4cbac44aa0539760`；冻结导出清单为 `cbfd06a760f19c037086081cafd7c24c83e53d0fff06ab3143c1891e120c74d6`。完整路径见 `manifest.json -> source_evidence`；两题上游 dataset/labels 哈希见 `dataset_contract.source_hashes_by_problem`。六个划分的元数据、targets、source index 与逐方法成本文件链接见各列 `alignment`，不将源文件哈希误写成新划分文件哈希。

## 已完成

- 12/12 列已从当前 targets 经 parent/source_index 绑定到 10/13 冻结结果，再绑定到生成 profile/complete；共 934,848 个 FP64 成本值完全一致，最大差 0。
- 每列 profile 配方哈希、完成记录文件哈希、merged 输出身份均与冻结导出清单吻合；12 列均对应 408 个已完成生成分片。
- 当前 train、validation、test 都是新增训练源数据的重新划分，共每题 77,904 例；不是原始旧 validation/test 部署。
- reld_mt 仅为 RELD_MOEL 的成本别名；没有使用 MTL 成本，也没有混回 B 筛除列。
- 复用八个固定 TRAIN 实例上的 48 条已有路线；48 条通过独立客户访问、容量、客户时间窗、含服务/等待的 depot 截止、L 距离与成本检查。最大重算成本绝对差 2.37147594e-06。
- 分片 JSON 保存原始浮点成本，冻结导出按 `.10f` 写成本，再解析为 FP64。48条分片成本经该原有格式化后与当前标签精确一致；不把十进制序列化尾差误判为路线错误。
- 新增 solver 运行 0 次、训练 0 次；性能标签、划分、solver 源码及 R58 deferred 状态均未修改。

## 当前部署字段

| 字段 | 当前场景结论 | 证据等级 |
|---|---|---|
| depot 时间窗 | 两任务六方法均为 [0,3]，闭合路线需在 3 前返仓 | 当前运行 profile/适配器记录；ReLD 为锁定源码推导；48条路线佐证 |
| speed | 1；行驶时间=距离/speed | RF/MoSES 运行适配器记录；其余为锁定环境源码推导 |
| 客户 service time | 原实例逐客户 service_time，depot=0；服务在等待后累加 | 运行适配器、固定输入及路线检查 |
| TW | 约束服务开始时间；等待和服务计入时间状态 | 锁定环境源码及路线检查 |
| VRPTW 的 L | 不启用 | 记录的输入与环境任务开关 |
| VRPLTW 的 L | 每车欧氏行驶距离上限3，包含返仓；不含等待/服务 | RF/MoSES显式传3；native/ReLD读取输入route_limit；路线检查 |
| cost | 闭合路线全部欧氏行驶距离之和，含出仓/返仓；不加服务/等待/车辆固定费 | 锁定计费源码及路线重算 |

## 逐列状态

| 问题 | 当前列 | 原列 | 标签绑定 | 已存路线核验 | 解码/预算 |
|---|---|---|---|---|---|
| VRPTW | MTPOMO | MTPOMO | 全部三划分 FP64 一致 | 4/4 | sampling；aug=1 |
| VRPTW | MVMOE | MVMOE | 全部三划分 FP64 一致 | 4/4 | sampling；aug=1 |
| VRPTW | MoSES_CaDA | MoSES_CaDA | 全部三划分 FP64 一致 | 4/4 | Original phase=test；aug=8 |
| VRPTW | MoSES_RF | MoSES_RF | 全部三划分 FP64 一致 | 4/4 | Original phase=test；aug=8 |
| VRPTW | reld_mt | RELD_MOEL | 全部三划分 FP64 一致 | 4/4 | greedy；aug=1 |
| VRPTW | RouteFinder | RouteFinder | 全部三划分 FP64 一致 | 4/4 | Original phase=test；aug=8 |
| VRPLTW | MTPOMO | MTPOMO | 全部三划分 FP64 一致 | 4/4 | sampling；aug=1 |
| VRPLTW | MVMOE | MVMOE | 全部三划分 FP64 一致 | 4/4 | sampling；aug=1 |
| VRPLTW | MoSES_CaDA | MoSES_CaDA | 全部三划分 FP64 一致 | 4/4 | Original phase=test；aug=8 |
| VRPLTW | MoSES_RF | MoSES_RF | 全部三划分 FP64 一致 | 4/4 | Original phase=test；aug=8 |
| VRPLTW | reld_mt | RELD_MOEL | 全部三划分 FP64 一致 | 4/4 | greedy；aug=1 |
| VRPLTW | RouteFinder | RouteFinder | 全部三划分 FP64 一致 | 4/4 | Original phase=test；aug=8 |

## 固定样本与证据范围

核验前按输入选择 TRAIN 中规模50、75、76、100各一个实例，同规模取最小 source index。六候选共用同一批物理实例，覆盖 <=75/ >75 权重切换边界。具体 local/source index 见 [route_check_plan.json](route_check_plan.json)；输出见 [route_checks.csv](route_checks.csv)。
已确认该 CSV 实际存在，不是生成中或 stale link：48条数据、16,416字节，SHA256 为 `721980b894e165596f3ab0ff97b74e84a37a6c42b688b76b19f0219a469cff5c`；绝对路径见 `manifest.json -> source_evidence.route_checks`。本次仅核对已有文件，未重新生成。
使用现有 r57_routes.py 的独立 check_route 函数，不加载原生 solver/mask；显式传入有当前部署证据的 speed=1、depot_tw_end=3。RF/MoSES 保存路线省略开头 depot 时补0，仅为序列表达归一化。
证据等级：documented=文档声明；source-derived=与运行哈希绑定的源码推导；current-run-recorded=生成当前标签时保存的运行记录；verified-on-route=本次对直接映射的已存路线检查；unknown=记录不足。路线通过不等于任意边界或所有实例已经逐条独立认证。

逐列字段中的 `profile.json#...` 均相对于该列 `run_records.profile` 的绝对路径和文件 SHA256；`profile_recipe_sha256` 另为配方身份，不是文件哈希。`run_records.completion` 与 `representative_logs` 保存 complete/log 的完整路径与 SHA256，冻结清单绑定保留于同一记录；日志只是代表样本。`source_references` 的 recorded/observed 哈希及 profile 内保存的 adapter 源码支持 source-derived 结论，不能把当前代码默认值冒充当年的运行记录。

RF/MoSES 的8倍增强、多起点测试及 start/augmentation 归约按各列锁定 profile/adapter/loader 记录：最终成本读取 `-max_aug_reward`，路线读取 `best_aug_actions`。这是已记录的输出选择规则，不证明第三方解码器实际使用 greedy；其 resolved decode 仍为 `null`、等级 `unknown`。
MoSES_RF 的 profile 保存的是委托给 `super().run` 的子类；其父类适配器证据来自已保存的 `train_100k_backends.py` 快照，SHA256 为 `46a1dd32b169366673218a6315aee1f44d256393b7bacb54968ec0122500d117`，完整路径与引用位置见 `evidence_resolution.rf_moses_output_evidence`，不按当前父类默认值补写。

## 保留的实际缺口

1. RF/MoSES 的 profile 锁定 phase=test、8倍增强及按规模权重，但没有单独捕获外部库最终解析的 decode type。清单按 unknown 保留，不用当前库的默认 greedy 补写。
2. 原始完整 shell argv 及完整外部库/设备环境不能由 profile 单独恢复；已保存分片有实际 seed、batch chunk、producer source 和权重哈希，可定位对应数值产物，但本轮不声称跨设备逐比特复跑。
3. 原生环境存在数值边界差异：native/ReLD 使用约1e-5容差，RF/MoSES有严格小于号。统一名义时间/距离参数不等于已证明所有浮点边界完全等价。
4. 本轮四个固定样本的 service_time 统计不外推为全体实例固定常数；规范仍是读取实例字段。
5. 不把旧前1万例 calibration 的历史成本差当作当前新标签错配；当前使用的 source index 均>=10000，且已直接核对对应冻结成本。
6. 已记录的源码/权重字节哈希不自动等于历史 Git revision 或完整库版本清单；未记录的历史 LIB、旧 validation/test 部署版本仍未获认证，不以当前 HEAD 或当前库默认值填补。

## 原有离线整理

以下是原有离线核验的生成入口，不是本次文档补充的必要步骤；本次未执行它，也未新增路线核验。该入口重建原始核验产物，不自动包含本次手工补充的 `r58_point_3`、`dataset_contract`、`evidence_resolution` 和报告说明。

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
python /public/home/shiys/0selection2/code/V4/runs/R81_current_label_deployment/build_manifest.py
```

本次交付 manifest.json 与 RESULTS.md；辅助 [alignment.csv](alignment.csv)、[route_checks.csv](route_checks.csv)、[route_check_plan.json](route_check_plan.json) 均沿用已有记录。已有 RESULTS.pdf 保留原离线核验内容，本次文档补充未重新生成 PDF。原 solver_source_manifest.json 只保留本场景引用及状态，原历史记录保持原样。
