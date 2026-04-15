# 类似NSS的最终完整导出

本文件夹是选择器数据集的最终统一导出包。

## 包含的数据

- `TSP / CVRP`
  - `train / val / test / lib`
  - 复制自 `nss_like_export_v1`，并已对照 NSS 源数据集进行验证
  - `TSPtrain = 10000`
  - `CVRPtrain = 10000`

- `ATSP`
  - `ATSPtrain / ATSPval / ATSPtest`
  - 基于完整的 EasyNCO 数据及标签构建
  - `ATSPtrain = 10000`
  - `ATSPval = 1000`
  - `ATSPtest = 1000`
  - 方法：`GLOP / MATNET / MATPOENET`

- `MVRP` 多样分布版本
  - 15 种变体，每种包含 `train / val / test`
  - 训练集使用来自 `offline_init_v4/mvrp_diverse_v1` 的多样分布数据集
  - 验证集/测试集使用来自 `offline_init_v4/nss_style_splits_v1` 的多样分布划分
  - 每种变体包含：
    - `train = 10000`
    - `val = 1000`
    - `test = 1000`
  - 方法：`MTPOMO / MVMOE`

## MVRP 变体

- `OVRP`
- `VRPB`
- `VRPL`
- `VRPTW`
- `OVRPTW`
- `OVRPB`
- `OVRPL`
- `VRPBL`
- `VRPBTW`
- `VRPLTW`
- `OVRPBL`
- `OVRPBTW`
- `OVRPLTW`
- `VRPBLTW`
- `OVRPBLTW`

## 每个数据集的目录结构

每个数据集目录遵循相同的类 NSS 结构：

- `dataset.pkl`
  - 合并后的实例列表
- `results/result_*.txt`
  - 每个方法对应一个文件
  - 每行格式：`instance_id,no_aug_score`
- `raw_label.pkl`
  - 每个实例的标签汇总，包含各方法的成本及最佳方法索引
- `instance_meta.csv`
  - 为 `ATSP` 和 `MVRP` 提供
  - 记录分片/尺度对齐信息

## 清单

- `export_manifest.json`
  - 所有导出数据集的顶层摘要

## 备注

- `TSP / CVRP` 保留 NSS 的划分命名：`train / val / test / lib`
- `ATSP / MVRP` 的标签使用 `no_aug_score`
- 本包旨在作为下游选择器训练的最终整合包