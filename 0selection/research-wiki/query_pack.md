# Research Wiki Query Pack

_Auto-generated. Do not edit._

## Project Direction
# 研究简报

> **用于 `/idea-discovery` 或 `/research-pipeline` 的文档化输入模板。**

## 问题陈述
当前研究问题是：能否训练一个**监督学习的 neural solver selector**，让**一个模型**同时支持多个 routing problem，并在实例级别从多个已有 solver 中选出最合适的方法。通俗的讲，就是，输入一个实例，模型根据实例的特征选择一种方法来解这个实例

这项工作的重点不是设计一个新的 `TSP` 或 `CVRP` solver，而是把已有 solver 看成候选方法池，对每个实例做 solver sele
## Open Gaps
# Gap Map (v2, post oracle-pro review)

- **G1** — No supervised neural *selector* with variable/overlapping solver pools across ≥3 routing problems, including constraint-compositional zero-shot. (NSS: TSP+CVRP separately. CTS/GINES: TSP only. Classical AS for CVRP/VRPTW exists — Asín-Achá 2024, Gutiérrez-Rodríguez 2019 — but is not neural / not multi-variant.) [unresolved]
- **G2** — NCO-specific imbalance of overlapping solver IDs across 18 problems. Classical AS masking exists, but interaction with neural solver pools + long-tail-per-problem is unstudied. [unresolved]
- **G3'** — Constraint-compositional zero-shot solver selection: known solvers, unseen constraint combinations. Stronger framing than "disambiguate from identical features" (which a one-hot ID trivializes). [unresolved]
- **G4** — Solver cold-start in neural solver selection (add a new solver without retraining). Not addressed by NSS / CTS / GINES. [unresolved]
- **G5** — Cost-aware per-instance selection under runtime budget across heterogeneous VRP variants. FrugalML / RouterBench exist in ML-systems; no NCO analogue. [unresolved]

## Failed Ideas (avoid repeating)
- **Set-conditioned solver cold-start (ELIMINATED)**: 
- **PLE hierarchical experts per problem-family (ELIMINATED)**: 
- **Cost-aware cascade selector (ELIMINATED)**: 
## Key Papers (22 total)
- [paper:2024_prompt_vrp] Prompt Learning for Generalized Vehicle Routing: Problem-specific learnable prompts steer a shared VRP solver without full fine-tuning.
- [paper:berto2024_routefinder] RouteFinder: Towards Foundation Models for Vehicle Routing Problems: Mixed-batch training + global instance embedding scales a single RL model to 48 VRP variants.
- [paper:collins2020_rec_pias] Per-Instance Algorithm Selection for Recommender Systems via Instance Clustering: Cluster instances first, then predict best algo from 14 CF/CB; oracle gap monotonically shrinks as pool grows.
- [paper:drakulic2024_goal] GOAL: A Generalist Combinatorial Optimization Agent Learner: Shared backbone + per-problem I/O adapter, SL-trained on 10 problems; needs fine-tuning for new ones (no zero-shot).
- [paper:gao2025_nss] Neural Solver Selection for Combinatorial Optimization: First per-instance neural-solver-selector framework for COPs: extract instance features → classification/ranking head → robust selection strategy.
- [paper:gines2023_selector] Revisit the Algorithm Selection Problem for TSP with Spatial Information Enhanced GNNs (GINES): GNN-based TSP solver selector using coords + distances; outperforms CNN/feature-based baselines.
- [paper:kwon2021_matnet] MatNet: Matrix Encoding Networks for Neural Combinatorial Optimization: Encoder for distance-matrix-only problems (ATSP, Flexible FJSP) using dual-stream attention.
- [paper:li2024_cada] CaDA: Cross-Problem Routing Solver with Constraint-Aware Dual-Attention: Dual-attention explicitly encodes constraints — counter-evidence to "mask-only" designs.
- [paper:liu2024_mtnco] Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization (MTNCO / POMO-MTL): Treats 16 VRP variants as constraint combinations, shared POMO 
## Recent Relationships (2 total)
  exp:R22_hardfam_falsification --supports--> claim:wall_is_structural
  exp:R22_hardfam_falsification --supports--> claim:two_regime_decomposition
