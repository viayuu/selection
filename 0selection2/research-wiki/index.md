# Research Wiki Index

_Auto-generated. Updated 2026-04-18 after /idea-creator._

## Papers — Core
- [NSS](papers/gao2025_nss.md) — per-problem selector baseline (TSP+CVRP)
- [CTS](papers/zhao2020_cts.md) — first deep TSP selector (CNN, 2020)
- [GINES](papers/gines2023_selector.md) — GNN TSP selector (coords+matrix)
- [URS](papers/zhou2025_urs.md) — unified data repr + param generator
- [CoEKS](papers/yu2026_coeks.md) — constraint-experts + MDis

## Papers — Cross-problem VRP solvers
- [MTNCO / POMO-MTL](papers/liu2024_mtnco.md) · [MVMoE](papers/zhou2024_mvmoe.md) · [RouteFinder](papers/berto2024_routefinder.md) · [MTL-KD](papers/zheng2025_mtl_kd.md) · [GOAL](papers/drakulic2024_goal.md) · [CaDA](papers/li2024_cada.md) · [ICAM](papers/zhou2024_icam.md) · [PromptVRP](papers/2024_prompt_vrp.md) · [MatNet](papers/kwon2021_matnet.md)

## Papers — Algorithm Selection
- [Collins 2020 RecSys PIAS](papers/collins2020_rec_pias.md) · [Tornede 2021 Meta-AS](papers/tornede2021_meta_as.md) · [MetaOD](papers/zhao2020_metaod.md)

## Papers — Recsys / ML-systems structural
- [STAR](papers/sheng2021_star.md) · [MMoE](papers/ma2018_mmoe.md) · [PLE](papers/tang2020_ple.md) · [RouteLLM](papers/routellm2024.md) · [PLM — Partial Label Masking](papers/plm2021_partial_label_masking.md)

## Ideas — Active (proposed)
- [**Idea 1** (MAIN) — Masked selector + regret-soft labels](ideas/idea1_masked_soft_selector.md)
- [**Idea 2** — MVRP solver×variant factorized scoring](ideas/idea2_mvrp_factorized.md)
- [**Idea 3** (HEADLINE GENERALIZATION) — Compositional zero-shot MVRP](ideas/idea3_compositional_zeroshot.md)
- [Idea 4 — Loss-function diagnostic](ideas/idea4_loss_study.md)
- [Idea 5 — BS-mask (availability-conditional Balanced Softmax)](ideas/idea5_bs_mask.md)
- [Idea 6 — Problem-ID dropout sanity ablation](ideas/idea6_problem_id_dropout.md)

## Ideas — Legacy proposals from /research-lit (kept for context)
- [Angle A — Set-conditioned solver scoring](ideas/angle_A_set_conditioned.md) (subsumed by Idea 1 + elim1)
- [Angle B — Constraint-compositional zero-shot](ideas/angle_B_compositional.md) (→ Idea 3)
- [Angle C — Cost-aware cascade](ideas/angle_C_cost_aware.md) (→ eliminated)

## Ideas — Eliminated (negative memory)
- [Set-conditioned solver cold-start](ideas/elim1_cold_start_solver.md) — unprovable without new solver
- [PLE hierarchical experts](ideas/elim2_ple_hierarchical.md) — crowded turf, selector too small to need
- [Cost-aware cascade](ideas/elim3_cost_aware.md) — user excluded time from supervision

## Gaps
See `gap_map.md`. Active: G1, G2, G3', G4, G5.

## Experiments / Claims
_(none yet — Idea 1 pilot flagged as `needs manual pilot`)_
