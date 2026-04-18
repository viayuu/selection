# Literature Review — Multi-Problem Neural Solver Selector

_Generated 2026-04-18 via /research-lit. Downloads under `literature/downloads/`._

## Scope

Target: **one** supervised selector that, given a routing instance, picks the best solver from a method pool — spanning **TSP, CVRP, ATSP, and 15 MVRP variants (18 problems total)**. Challenges per the brief: (1) unbalanced / overlapping per-problem method sets, (2) problem-type disambiguation from raw instance features (e.g. OVRP vs CVRP), (3) unified-yet-discriminative instance representation. Advisor suggested URS / CoEKS and said to borrow from recommender systems.

## Literature Table

| # | Paper | Venue/Year | Method | Relevance to us | Source |
|---|-------|------------|--------|-----------------|--------|
| 1 | **NSS** (Gao et al., 2410.09693) | ICML 2025 | Per-problem selector: GAT + hierarchical pool encoder, classification/ranking loss, rejection top-k | **Direct baseline**. We extend from per-problem to cross-problem. Copy repo and modify. | local |
| 2 | **URS** (Zhou et al., 2509.23413) | ICML 2026 (under review) | Unified Data Representation (UDR) — decouple instance features from constraints, problem-conditioned parameter generator, LLM-driven masking | **Core reference for challenge 2/3.** UDR idea = same node feature schema for all variants; constraints as masks. | local |
| 3 | **CoEKS** (Yu et al., ICLR 2026) | ICLR 2026 | Combination-of-Experts: one FFN expert per *basic constraint* (C/O/B/L/TW), combiners + mutual-distillation knowledge sharing | **Core reference for challenge 1.** Structured sparsity = natural answer to "different variants need different method sets". Adaptable to "expert per solver". | local |
| 4 | **MVMoE** (Zhou et al., 2405.01029) | ICML 2024 | Node-level MoE in encoder; 16 VRP variants with one RL model | Baseline MoE approach. CoEKS shows its expert-vision is too local. | downloaded |
| 5 | **RouteFinder** (Berto et al., 2406.15007) | NeurIPS 2024 | Global embeddings + mixed-batch + efficient adapter layers for 48 VRP variants | Shows scalability of cross-problem MTL; useful instance-feature design. | downloaded |
| 6 | **MTL-KD** (Zheng et al., 2506.02935) | 2025 | Distill multiple specialist RL solvers into one heavy-decoder MTL model; 6 seen + 10 unseen tasks | Mirrors our setup: pool of specialists → one unified model. Distillation instead of selection. | downloaded |
| 7 | **GOAL** (Drakulic et al., 2406.15079) | 2024 | Generalist CO agent: backbone + per-task input/output adapter, SL on optimal tours, 10 problems | Adapter-based alternative to selection; constrained by fine-tuning. | downloaded |
| 8 | **CaDA** (Li et al., 2412.00346) | 2024 | Constraint-aware dual-attention; cross-task RL | Shows explicit constraint encoding helps; counter-evidence for "all mask-based" design. | downloaded |
| 9 | **ICAM** (Zhou et al., 2405.01906) | 2024 | Instance-conditioned adaptation for NCO scale generalization | Template for **instance → parameter modulation** (answer to challenge 3). | metadata |
| 10 | **Chain-of-Context** (Gui et al., 2603.01667) | 2026 | Stepwise constraint-dynamics context, 48 variants (16 ID + 32 OOD) | Recent SOTA OOD cross-task baseline. | metadata |
| 11 | **ReLD** (2503.00753) | 2025 | Heavy FFN + identity mapping decoder; SOTA backbone cited by CoEKS | Candidate backbone to pool into selector's solver set. | metadata |
| 12 | **Per-Instance AS for RecSys** (Collins et al., 2012.15151) | 2020 | Cluster instances → predict best algo from 14 CF/CB algos | **Cross-domain bridge**: shows clustering latent space before selector helps. | metadata |
| 13 | **Meta-Learned PIAS** (Collins & Beel, 1912.08694) | 2019 | Random-forest meta-learner; A/B on Mr.DLib | Reminds us: "average oracle gap → perfect selector" monotonically decreases with pool size. | metadata |
| 14 | **Algorithm Selection on a Meta Level** (Tornede et al., 2107.09414) | 2021 | Ensembles of algorithm selectors, beats any single selector | Suggests ensembling selection strategies (classification, ranking, regression). | metadata |
| 15 | **SATzilla / ASlib / Kerschke 2019** | JAIR | Classical AS portfolios with hand-crafted features | Historical grounding; shows feature-quality is usually bottleneck. | cited |
| 16 | **MMoE** (Ma et al., KDD 2018) | — | Multi-gate mixture-of-experts for multi-task | Canonical ref for multi-task shared backbone with task-specific gates. | cited |
| 17 | **PLE** (Tang et al., RecSys 2020 best) | — | Progressive Layered Extraction, separates shared vs task-specific experts | Better than MMoE when tasks conflict — matches our TSP-vs-MVRP asymmetry. | cited |
| 18 | **STAR** (Sheng et al., CIKM 2021) | — | Star-topology CTR: shared center + per-domain towers via weight addition | **Directly transferable**: one shared backbone + per-problem lightweight head/mask; elegant fit for 18 problems. | cited |
| 19 | **XSMoE** (Qu et al., 2508.05993) | 2025 | Expandable side MoE; new experts plug in for new modalities/drift | Matches CoEKS idea and extends to "add a new solver without retraining". | metadata |
| 20 | **PLM — Partial Label Masking** (2105.10782) | 2021 | Dynamic positive/negative ratio masking for imbalanced multi-label | **Attack on challenge 1** (label imbalance across problems sharing methods). | metadata |
| 21 | **Long-tail XMC** (2207.13186, 2311.05081) | 2022–23 | Propensity-scored loss for extreme multi-label with rare labels | Reweighting for rare solvers (e.g. lehd-on-MVRP) appearing in few problems. | metadata |
| 22 | **GraphMETRO** (Wu et al., 2312.04693) | NeurIPS 2023 | MoE on graphs aligning expert reps to handle distribution shifts | Shifts ≈ problem-variant heterogeneity; suggests an alignment loss. | metadata |
| 23 | **Poppy** (Grinsztajn et al., 2023) | NeurIPS 2023 | Population of diverse solvers with shared encoder | Pool-generation cousin (runs all). Motivates selection-vs-ensemble trade-off. | cited |
| 24 | **Kanda et al. — Meta-learning MHs for TSP** | ES 2016 | Label-ranking meta-learning on TSP meta-features | Closest classical analog to NSS; meta-feature design ideas. | via exa |

## Synthesis — Four Angles

**1) Cross-problem unified representation.** URS's *UDR* (a fixed node-feature schema + mask function for constraints) is the cleanest unification mechanism in the NCO literature. It cleanly handles OVRP-vs-CVRP ambiguity by letting the mask, not the input, carry constraint identity. For a *selector* (not a solver) this is even safer: the mask is only needed for the candidate-solver axis, and the encoder can still receive an **explicit problem-id / constraint-set token** the way CoEKS activates its constraint experts. The cleanest design is therefore: UDR-style node features + a compact **constraint bitvector** concatenated as an instance-level token (this resolves challenge 2 without adding domain bias, and disambiguates OVRP vs CVRP even when node features are identical).

**2) Handling unbalanced, overlapping method pools.** CoEKS is the structural inspiration: *one expert per basic element* (there: constraint; for us: *one "head" per candidate solver*, or a grouped head). Because not every solver is valid on every problem, the forward pass activates only a subset — mirroring CoEKS's `CS ⊆ E`. Output masking (your plan) is therefore not just a tactical fix but a principled choice. Beyond masking, the RecSys + long-tail literature offers two orthogonal tools worth trying: **partial-label masking / propensity-weighted loss** (PLM, long-tail XMC) to prevent dominant problems (TSP) from drowning rare ones, and **STAR-style per-domain heads on a shared trunk**, where each routing problem keeps a light problem-specific head that is weight-added to the shared one.

**3) Selection vs multi-task integration.** Three competing architectures are in play: (a) **NSS-style explicit selector** (our target), (b) **Poppy/MTL-KD ensembling** (runs a pool then picks best), and (c) **unified solver** (URS/RouteFinder/CoEKS). Our work plants a flag between (a) and (c): a selector that is *shared across problems*. The closest existing work is Collins 2020 (RecSys) where a selector over 14 algorithms is learned with clustering; and Tornede 2021 which shows **ensembling selectors** (classification+ranking+regression heads) beats any single selection strategy — worth replicating as an ablation.

**4) Instance-level conditioning.** For challenge 3, two mature recipes exist: ICAM's instance-conditioned adapter and URS's problem-conditioned parameter generator. Both are essentially hypernetwork-style modulation. Borrowing from RecSys: **STAR's partitioned weights** and **PLE's shared/task-specific expert split** give tighter inductive biases when the "domains" (= problems) are known-but-related. For 18 problems, PLE's design (a few shared experts + a small private expert per problem group: symmetric VRP / asymmetric VRP / pickup-delivery / MVRP-variants) is arguably cleaner than flat per-problem adapters.

## Gaps the project can claim

- **G1 (new)** No neural solver *selector* has been trained across > 2 routing problems. NSS is per-problem; URS/CoEKS are unified solvers, not selectors.
- **G2 (new)** Method-pool imbalance (some problems have 8 solvers, others 2) is unstudied in NCO — analogue to long-tail multi-label classification in RecSys.
- **G3 (new)** Problem-disambiguation when node features are identical across variants is an open setting for *selection*, though URS's mask-first design suggests a path.

## Recommended next reads (prioritised)

1. **URS** + **CoEKS** (re-read methods sections with a selector lens).
2. **RouteFinder** §3 (global instance embedding design).
3. **Collins 2020** (RecSys PIAS): the only per-instance selector with > 10 algorithms; transferable evaluation protocol.
4. **MTL-KD**: distillation from specialists = useful ablation against "selector" paradigm.
5. **STAR / PLE** (industrial RecSys): cheapest structural fix for 18-problem output heads.

## Files on disk
- Already local: `literature/Neural Solver Selection for Combinatorial Optimization .md`, `literature/urs.md`, `literature/CoEKS.md` (+ code).
- Downloaded to `literature/downloads/`: 2406.15007 (RouteFinder), 2405.01029 (MVMoE), 2412.00346 (CaDA), 2506.02935 (MTL-KD), 2406.15079 (GOAL).
