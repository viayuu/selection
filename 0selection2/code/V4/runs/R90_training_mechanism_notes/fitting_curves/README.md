# NEW90k full18：TRAIN / VAL 拟合轨迹

主交付：`R90_fitting_report.pdf`（2页，嵌入 NotoSansCJK-Regular.ttc 字形）；辅助：`curve_points.csv`、`selected_and_last.csv`。

## 范围与口径

仅解析下列五组 `history.json`、`final_test_lock.json`，不打开权重、NPZ、缓存、TEST/LIB 或协议源码；不训练、不推理、不使用 GPU、不重选 checkpoint。
任务给定数据版本为 NEW90k：18题每题 TRAIN 81,000 / VAL 4,500；日志中的各题样本数、18题等权 macro、逻辑 batch 1,024（exposures/update）相符，预算均为 57,600 成功更新。
横轴使用 `successful_updates`，不是 epoch 或墙钟。VAL：1,440起每1,440步，共40点/模型；TRAIN：7,200起每7,200步，外加选中步补评。R87B选中36,000已在定期点中，其余四组选中点为补评；TRAIN共44点。未补造0步或缺测点。
自然 Top1 为与 native int64 `ind` winner 的逐例一致率（日志声明其与原始 FP64 argmin 核对，非 FP32 argmin），不使用 tie-aware Top1。未重新审计底层标签/缓存。
CE严格取 `history[i]['train'/'val']['macro']['macro_ce']`，两侧均为完整划分 eval/maximin CE；不是 `optimization['ce']`（区间、dropout训练态）。
Top1 gap = TRAIN - VAL，单位百分点(pp)；CE gap = VAL - TRAIN，单位为CE原值；只在同一成功步且模型状态哈希一致的实测点上计算。

## 已锁定选中步

各组选中规则为完整VAL `macro_vs_sbs_pct` 严格最小（共同的TRAIN拟合SBS、并列取首次），不是最高Top1或最低CE。本报告沿用锁定结果。

| 模型 | 更新步 | TRAIN Top1(%) | VAL Top1(%) | TRAIN-VAL(pp) | TRAIN CE | VAL CE | VAL-TRAIN CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| R84A | 37440 | 71.2439 | 71.0852 | 0.1587 | 0.696381 | 0.703577 | 0.007195 |
| R87A | 40320 | 71.3512 | 71.1481 | 0.2031 | 0.695477 | 0.703083 | 0.007606 |
| R87B | 36000 | 71.2594 | 71.0790 | 0.1804 | 0.693263 | 0.699840 | 0.006577 |
| R88A | 37440 | 71.2516 | 71.0580 | 0.1936 | 0.697184 | 0.704587 | 0.007403 |
| R88B | 38880 | 71.2973 | 71.0580 | 0.2393 | 0.695217 | 0.702205 | 0.006989 |

## 末步 57,600

| 模型 | 更新步 | TRAIN Top1(%) | VAL Top1(%) | TRAIN-VAL(pp) | TRAIN CE | VAL CE | VAL-TRAIN CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| R84A | 57600 | 71.2180 | 70.9395 | 0.2785 | 0.695697 | 0.704329 | 0.008633 |
| R87A | 57600 | 71.4158 | 71.0877 | 0.3282 | 0.693238 | 0.702689 | 0.009451 |
| R87B | 57600 | 71.3384 | 71.0852 | 0.2532 | 0.690113 | 0.698857 | 0.008744 |
| R88A | 57600 | 71.2663 | 70.9457 | 0.3206 | 0.695366 | 0.704108 | 0.008741 |
| R88B | 57600 | 71.2721 | 71.0123 | 0.2597 | 0.695087 | 0.704019 | 0.008932 |

## 可支持的结论与限制

末步TRAIN-VAL Top1差为 **0.2532–0.3282 pp**，VAL-TRAIN CE差为 **0.008633–0.009451**。选中步Top1差为 0.1587–0.2393 pp。全部44个配对实测点Top1差为 0.0325–0.3282 pp。
这些已完成NEW90k轨迹的full18等权宏平均未见巨大的TRAIN/VAL分离；不据此推断欠拟合根因、Bayes上限、没有可学习信号，或减弱正则一定有效，也不延伸为每题/每候选结论。聚合JSON不能定位具体漏掉的真实winner；无TEST结论或新实验计划。末步只作描述，不代替已锁定checkpoint。
候选池：五组全部已记录TRAIN/VAL点的逐题pool名称与顺序相同；这是日志级比较，未核验实例成员或底层成本。R87包含shared/private残差变体（B有私有参数），R88包含canonical/execution角色锚点变体；跨版本不是严格单因素对照，不能归因于某一参数或正则。
microcap是否恒定/跨模型相同、dropout及weight decay均未写入这两类JSON，故未核验，不假设相同。日志可核对的一致性权重轨迹五组相同：1,440步为0.1166667，2,880步为0.2333333，4,320步起为0.35；这不等于完整正则配置一致。

## 数据来源

- R84A：`/public/home/shiys/0selection2/code/V4/runs/R84_new90k_attention/A_baseline/history.json`；`/public/home/shiys/0selection2/code/V4/runs/R84_new90k_attention/A_baseline/final_test_lock.json`。
- R87A：`/public/home/shiys/0selection2/code/V4/runs/R87_private_method_residual/A_shared_residual/history.json`；`/public/home/shiys/0selection2/code/V4/runs/R87_private_method_residual/A_shared_residual/final_test_lock.json`。
- R87B：`/public/home/shiys/0selection2/code/V4/runs/R87_private_method_residual/B_private_residual/history.json`；`/public/home/shiys/0selection2/code/V4/runs/R87_private_method_residual/B_private_residual/final_test_lock.json`。
- R88A：`/public/home/shiys/0selection2/code/V4/runs/R88_execution_role_inputs/A_canonical_anchor/history.json`；`/public/home/shiys/0selection2/code/V4/runs/R88_execution_role_inputs/A_canonical_anchor/final_test_lock.json`。
- R88B：`/public/home/shiys/0selection2/code/V4/runs/R88_execution_role_inputs/B_execution_anchor/history.json`；`/public/home/shiys/0selection2/code/V4/runs/R88_execution_role_inputs/B_execution_anchor/final_test_lock.json`。

`curve_points.csv`仅含实测全量评估点（200个VAL、44个TRAIN）；未配对的VAL行差值留空。`train_point_origin`区分定期评估与选中步补评。`selected_and_last.csv`含10行选中/末步精确指标和锁定VAL vs_SBS。CSV保留浮点精度，PDF/本说明中Top1保留4位、CE保留6位。
