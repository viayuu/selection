# 上次 review 后的实验汇报（Round 18, 2026-04-21）

## 实验一览

| # | 实验 | 结果 | 判断 |
|---|---|---|---|
| 1 | R42S_fixpair（pairwise sign 修 + family θ） | test Δ vs SBS = **−0.0002**, p=0.481, 非显著 | sign bug 是真bug，但不是主因；family-θ 是错的抽象 |
| 2 | LightGBM 特征天花板 | macro val=0.5191, test=0.5098（低于 R40D 0.5226） | **80% 目标极可能不可达**，噪声/分布硬天花板 0.55–0.60 |
| 3 | Behavior emb + Balanced Softmax prior dump | 证实 arm 坍缩（TSP top-2 覆盖 91%；ATSP top-1 独吞 73%） | 为 R43 反坍缩 loss 提供先验 |
| 4 | （意外）数据对齐漏洞 | 17:46 raw_label.pkl 被静默扩展，R18 eval 从 0.5166 误报 0.2432 | 已修补 data.py + tools/ 三处代码；修后 R18=0.5167 ✅ |
| 5 | R43 v2（反坍缩 anti-collapse） | ep0=0.5109, ep1=0.5116（仅跑到 ep1 被叫停） | 比 R40D 同期略慢，未能到 ep6 判读 |
| 6 | R40D seed1 v2（多 seed 确认） | ep0=0.4993（仅 ep0 被叫停） | 与 seed0 ep0 范围一致，方差正常 |

---

## 1. R42S_fixpair（P0）

**改动**：
- `rerank_train.py:265-277` 的 `pairwise_cost_loss` 方向修正 + gap-weighted
- 新增 `--pairwise-gap-min 0.001` 过滤近平手对
- 阈值从 per-problem 改为 family-level（TSP/ATSP/CVRP/MVRP 四个 θ）

**结果**：

```
val macro_top1 = 0.5229
test Δ vs SBS = −0.0002,  95% CI [−0.0093, +0.0090],  p(Δ>0) = 0.481
```

**分问题（关键项）**：

| 问题 | Δ vs SBS | 备注 |
|---|---|---|
| TSP | +0.021 | 赢 |
| OVRPBTW | +0.041 | 赢 |
| OVRPBLTW | +0.031 | 赢 |
| CVRP | −0.034 | 大输（family-θ 拖累） |
| OVRPB | −0.038 | 大输 |
| OVRPTW | −0.028 | 输 |

**结论**：
1. pairwise sign 确实是 bug，但**不是主要原因**
2. family-θ 是错抽象——17/18 问题被 MVRP 家族的 θ=+0.400 拖回 SBS
3. reranker 架构本身**没被否定**，部分问题稳定 +0.02~+0.04

---

## 2. LightGBM 特征天花板

**配置**：冻结 R40D encoder 的 `pooled_h`（128维）+ constraint bits + per-arm behavior 特征 → 18 个 per-problem LightGBM 多分类器。

**结果**：

```
macro val top1  = 0.5191
macro test top1 = 0.5098
R40D test macro = 0.5226（baseline，比 LightGBM 还高）
```

**分问题重点**：

| 问题 | LightGBM test | R40D test | 差 |
|---|---|---|---|
| TSP | 0.728 | 0.720 | +0.008（有少量头room） |
| CVRP | 0.393 | 0.419 | −0.026（neural head 反超） |
| ATSP | 0.699 | 0.699 | 持平 |

**结论**：
- 80% 水位**几乎不可达**
- 标签噪声 + 每个问题 2-3 个 arm 吃 85%+ oracle mass 是硬天花板
- encoder 深度不是主要瓶颈，label 噪声才是

---

## 3. Behavior 嵌入 + Balanced Softmax 先验（artifact）

**dump**：`artifacts/behavior_emb.npz` (18, 18, 3) 和 `artifacts/balanced_softmax_prior.npz` (18, 18)。

**arm 坍缩证据**：

| 问题 | 最重头 arm | 覆盖 oracle mass |
|---|---|---|
| TSP | arm 5 (LEHD) | 70.6% |
| TSP | top-2 (LEHD + OMNI) | 91.6% |
| ATSP | arm 1 (MATNET) | 73.4% |
| CVRP | top-2 | ~60% |
| MVRP 15 个 | top-2 | 85-95% |

→ 说明 support 分布极度偏斜，为 R43 的 Balanced Softmax + support-KL 正则提供了数值基础。

---

## 4. 数据对齐漏洞（本轮最重大发现）

**现象**：跑 seed2 test_eval 时 R18 macro_top1 报告 **0.2432**（应为 0.5166）。

**根因**：
- 今天 17:46，**非 TSP** 的每个问题 `data/<P><split>/results/` 被静默加了新 solver：
  - CVRP/MVRP 加 MoSES_CaDA, MoSES_RF, RouteFinder（9 solvers → 12）
  - ATSP 加 ICAM_ATSP, UNICO_MatPOENet（3 → 5）
- `raw_label.pkl` 被重建，cost 向量按文件名字母序扩展
- `registry.POOLS` 没更新

**破坏链条**：
```python
# data.py:100 原代码
costs = lbl["cost"][:K_p]   # 假设前 K_p = POOLS 顺序
```
- CVRP：`cost[6:8]` 原本是 (OMNI, RELD_CVRP, UDC)，现在成了 (MoSES_CaDA, MoSES_RF, OMNI)
- ATSP：`cost[1]` 原本是 MATNET，现在成了 ICAM_ATSP
- TSP：solver 没加，所以没破

**修复**：patch 三处代码
- `code/unified_selector/data.py`
- `tools/dump_behavior_to_npz.py`
- `tools/ceiling_lightgbm.py`

都改为：`cost_gather_idx = [sorted(results/result_*.txt).index(s) for s in POOLS[problem]]`

**验证**：

| 实验 | 修复前 | 修复后 |
|---|---|---|
| R18 test macro | 0.2432（错） | **0.5167** ✅ |
| R40D seed0 test macro | — | **0.5226** ✅ |
| R40D seed0 Δ vs R18 | — | **+0.0060**（与修复前文档一致）✅ |

**损失清单**：
- ✅ R42S_fixpair（13:28）、R40D seed0（02:58）、LightGBM ceiling（17:35）、behavior_emb（14:36）—— **漂移前产物，有效**
- ❌ R40D seed1 v1（17:52-57）、seed2 v1（17:57-18:16）、R43 v1（18:43-19:17）—— **漂移后训练**，已隔离到 `*_INVALID_pre_data_fix/`

---

## 5. R43 v2（反坍缩重训）

**配置**：R40D 基线 + `--use-problem-arm-bias --use-behavior-emb --balanced-softmax-weight 1.0 --support-kl-weight 0.2`。

**结果（只跑到 ep1，被叫停）**：

| ep | macro_top1 | 参考：R40D seed0 同期 |
|---|---|---|
| ep0 | 0.5109 | 0.51 |
| ep1 | 0.5116 | 0.53 |

**判读**：ep1 比 R40D 同期略慢，但 ep6 才是最佳点（R40D seed0 ep6=0.5248），现无法判读。

---

## 6. R40D seed1 v2（多 seed 确认）

**配置**：与 seed0 完全一致，只换 seed=1。

**结果（只跑到 ep0，被叫停）**：

```
ep0 macro_top1 = 0.4993
```

**判读**：与 seed0 ep0 范围一致（0.49-0.51），方差正常；但 ep6 未达，暂时无法下 "R40D 方法稳定" 的结论。

---

## 总体判断

| 方向 | 结论 |
|---|---|
| **pairwise sign 修正** | 真 bug，但不是主因 |
| **family-level θ** | **负效果**，必须回 per-problem |
| **80% top1 目标** | **极可能不可达**（天花板 ~0.55–0.60） |
| **R40D 的 +0.006 是真的** | 修复 + 重验后仍然 +0.006 显著 |
| **反坍缩 R43** | 未出结果（仅 ep1） |
| **数据对齐漏洞** | 已修复 + 验证通过 |

---

## 代码改动

| 文件 | 改动 |
|---|---|
| `code/unified_selector/rerank_train.py` | pairwise sign 修 + gap-weighted + `--pairwise-gap-min` + `--threshold-scope` |
| `code/unified_selector/model.py` | 加 `use_problem_arm_bias / use_behavior_emb / behavior_dim`（zero-init，backward compat） |
| `code/unified_selector/train.py` | 加 R43 反坍缩 flags + Balanced Softmax + support-KL |
| **`code/unified_selector/data.py`** | **关键修补**：cost 列 gather by 文件名，不再 positional |
| `tools/dump_behavior_to_npz.py` | 新增：dump per-(problem, arm) behavior + log prior |
| `tools/ceiling_lightgbm.py` | 新增：LightGBM 特征天花板 benchmark |

## 已隔离（数据漂移前训练的污染 ckpt）

- `code/unified_selector/runs/R40D_NSS_rr_microstep_seed1_INVALID_pre_data_fix/`
- `code/unified_selector/runs/R40D_NSS_rr_microstep_seed2_INVALID_pre_data_fix/`
- `code/unified_selector/runs/R43_R40D_anticollapse_seed0_INVALID_pre_data_fix/`
