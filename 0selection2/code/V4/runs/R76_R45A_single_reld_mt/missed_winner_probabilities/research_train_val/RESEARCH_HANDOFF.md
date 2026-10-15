# R76 研究交接：仅 TRAIN/VAL 关键数字

接收研究代理：`01a13b22-453f-71f0-87e3-97680c2cb763`。这是研究数据交接，不是代码 review。

锁定 epoch 18；SHA256 `5ba5777e437235e22682c4d54cfd8cdefca6fb53c27204a8e1e3f753ea9043c9`。仅读取已有保存 NPZ，无推理、训练或规则改变。
此文件不含 TEST 统计；TEST 标签诊断单独隔离，不能用于研究规则、阈值或调参。以下是标签条件分析，不是任何新规则的验证。

| 关键指标 | TRAIN | VAL |
| --- | ---: | ---: |
| 实例数 | 1,262,052 | 70,110 |
| Top1 % | 69.3696 | 69.2369 |
| 宏平均 CE | 0.752703 | 0.755865 |
| 宏平均多类 Brier | 0.404732 | 0.406889 |
| pooled ECE20 | 0.033289 | 0.036925 |
| zero-count且winner>0的方法-问题对 | 55 | 54 |
| 上述方法的获胜样本数 | 255,370 | 14,252 |
| TSP/LEHD winner次数 | 11,881 | 641 |
| TSP/LEHD 模型选择次数 | 0 | 0 |
| TSP/LEHD P(LEHD) 条件均值 % | 18.7276 | 19.1577 |
| TSP/LEHD 条件平均分数排名 | 3.0581 | 2.9688 |
| TSP/LEHD 条件 CE | 1.712836 | 1.684035 |
| TSP/LEHD 本题成本损失贡献 % | 34.3980 | 33.5548 |
| CVRP/LEHD winner次数 | 10,116 | 558 |
| CVRP/LEHD 模型选择次数 | 3 | 1 |
| CVRP/LEHD P(LEHD) 条件均值 % | 15.1948 | 15.1976 |
| CVRP/LEHD 条件平均分数排名 | 3.7674 | 3.7849 |
| CVRP/LEHD 条件 CE | 1.923135 | 1.924398 |
| CVRP/LEHD 本题成本损失贡献 % | 28.1575 | 28.2324 |

TRAIN/VAL 共有 53 个方法-问题对同时选0次且真实获胜。
观察：零选择与 near-zero 在 TRAIN/VAL 都存在；即使有15–19%的条件概率质量，也可能分数排不到第一。平均概率不等于选择频数，也不是校准已证实的正确率。
ECE 固定20等宽bins，不拟合温度、不选择分箱；条件 CE 是本方法真实获胜样本上的稳定 FP32 NLL。不同问题的原始成本不跨题相加。

完整证据：[RESULTS.md](RESULTS.md)、[TRAIN方法表](method_summary_train.csv)、[VAL方法表](method_summary_val.csv)、[TRAIN/VAL名称对齐](train_val_comparison.csv)、[TRAIN校准](calibration_summary_train.csv)、[VAL校准](calibration_summary_val.csv)。
