# 类似NSS的最终完整导出

本文件夹是选择器数据集的最终统一导出包。

## 包含的数据

- `TSP / CVRP`
  - `train / val / test / lib`
  - `TSPtrain = 10000`
  - `CVRPtrain = 10000`
   - `TSPval = 10000`
  - `CVRPval = 1000`
   - `TSPtest = 1000`
  - `CVRPtest = 1000`
  - lib数据集是为了测试模型在ood的泛化能力

- `ATSP`
  - `ATSPtrain / ATSPval / ATSPtest`
  - `ATSPtrain = 10000`
  - `ATSPval = 1000`
  - `ATSPtest = 1000`
  - 方法：`GLOP / MATNET / MATPOENET`

- `MVRP` 
  - 15 种变体，每种包含 `train / val / test`
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

每个数据集目录遵循相同的结构：

- `dataset.pkl`
  - 合并后的实例列表
- `results/result_*.txt`
  - 每个方法对应一个文件
  - 每行格式：`instance_id,no_aug_score`
- `raw_label.pkl`
  - 每个实例的标签汇总，包含各方法的成本及最佳方法索引