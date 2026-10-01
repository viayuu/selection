# 缺失项

全部 56 行评估指标已齐全，没有需要补跑的模型或数据集。

NSS 沿用历史评估 JSON，未补造逐实例预测；native ind 口径的 NSS Top-k 因无逐实例预测留空。
所有 Ours 的 NSS 可比口径和 native ind 口径均已计算；R25 也补齐了逐实例预测。
R41 是冻结评估阶段，不存在独立 R41 selector checkpoint；其四个模型已按各自版本纳入。
R37/R38 不补 CVRP/CVRPLIB，是预定 TSP 专项范围，不属于缺失项。
