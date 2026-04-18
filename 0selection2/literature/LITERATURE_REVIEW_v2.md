# Literature Review v2 — Deeper Pass + Oracle Review

_Generated 2026-04-18 via /research-lit round 2, reviewed by GPT-5.4 Pro (oracle-pro, browser). Supersedes v1 for defense strategy._

## New papers added this round

| # | Paper | arXiv | Why it matters |
|---|-------|-------|----------------|
| 25 | **CTS — Feature-free TSP solver selection** (Zhao, Shi, Guo 2020) | 2006.00715 | **First deep TSP solver selector** (CNN-based end-to-end). Pre-dates NSS by 4 years; a must-cite prior work. |
| 26 | **GINES** (2023) | 2302.04035 | **GNN TSP solver selector** that takes coords + distance matrix — partially addresses coord/matrix bridge; direct baseline. |
| 27 | **MatNet** | 2106.11113 | Matrix-form NCO encoder designed for ATSP/matrix-only problems — canonical TSP↔ATSP bridge for encoders. |
| 28 | **MTNCO** (Liu et al. 2024) | 2402.16891 | The "constraint-combination" MTL baseline (often cited as POMO-MTL). Base for 16-VRP cross-task. |
| 29 | **Prompt Learning for Generalized VRP** | 2405.12262 | Constraint-as-prompt conditioning; close to our problem-ID/constraint-token design. |
| 30 | **EFormer** | 2506.16428 | Recent unified VRP transformer (coord+matrix). |
| 31 | **MetaOD** (2020) | 2009.10606 | Dataset-level model selection — analogy only, but reviewers may ask. |
| 32 | **FrugalML / RouterBench / RouteLLM / UniRoute** | 2006.07512 / 2403.12031 / 2406.18665 / 2502.08773 | Per-instance *model* routing in ML systems — the "RecSys-free" lineage for cost-aware routing. |
| 33 | **Logit adjustment** | 2007.07314 | Canonical long-tail loss; partially applicable with availability-conditional prior (see §Imbalance below). |
| 34 | **Balanced Meta-Softmax** | 2007.10740 | Prior-shift softmax; applicable per problem. |
| 35 | **Seesaw loss** | 2008.10032 | Head-vs-tail pairwise scaling; applicable only if reformulated within masked candidate set. |
| 36 | **Balanced Group Softmax** | 2006.10408 | Group-wise softmax — maps nicely onto per-problem solver groups. |
| 37 | **SATzilla / ASlib / AutoFolio** | 1111.2249 / 1506.02465 | Unavoidable classical AS baselines reviewers will ask for. |
| 38 | **Asín-Achá et al. 2024 (Networks)** / Gutiérrez-Rodríguez et al. 2019 (ESWA) | — | CVRP / VRPTW algorithm-selection papers outside NCO; shows classical AS has already touched VRP variants — tightens G1 phrasing. |

_(Downloaded PDFs: 2006.00715, 2302.04035, 2106.11113, 2402.16891, 2405.12262, 2009.10606, 2406.18665, 2007.07314, 2007.10740, 2008.10032 in `literature/downloads/`.)_

## Oracle-pro Review — Verdicts

**On the three gaps.**
- **G1** (no selector across >2 routing problems) — *defensible only if phrased narrowly*: "one supervised *selector* over 18 routing problems with variable, overlapping solver pools." Do **not** over-claim "no unified routing model"; MTNCO/MVMoE/RouteFinder/CoEKS/CaDA/URS already solve that. And even AS-for-VRP has precedents (Asín-Achá 2024, Gutiérrez-Rodríguez 2019) — classical not neural.
- **G2** (method-pool imbalance in NCO) — *partly defensible*. Variable/masked portfolios are standard in AutoML/AS; the NCO-specific angle (overlapping neural-solver IDs, long-tail across problems) is still a plausible contribution but must be framed as *empirical* rather than *conceptual* novelty.
- **G3** (problem-disambiguation from identical features) — **weak as stated**. Raw feature identity is a *definitional* ambiguity; a one-hot problem ID solves it trivially. The contribution survives only when reframed as **constraint-conditioned selection** (see §5 below).

**On the one-vs-18 question.** Gains come from **transfer** via shared latent factors (size, clustering, depot geometry, capacity tightness, asymmetry, constraint tightness). Reviewers will expect:
- 18 per-problem NSS baselines,
- shared-backbone + per-problem heads ablation,
- problem-ID token ablation,
- **leave-one-problem-out** zero-shot test,
- **negative-transfer diagnostics** (per-problem regret).

A monolithic masked softmax alone may underperform the 18 specialists — must demonstrate a *why* (e.g. low-data variant lift).

**On imbalance losses.** The clean recipe is **masked softmax / listwise ranking over the available solver set**. Global long-tail losses are *dangerous* because unavailable solvers are not true negatives.
- Logit adjustment / Balanced Softmax: apply only with *availability-conditional* prior p(s | s ∈ M(x)).
- Seesaw / BAGS: require reformulation inside the candidate set.
- PLM is conceptually closest but still differs (our labels are fully observed within a changing candidate set, not ambiguous partial labels).

**On OVRP-vs-CVRP novelty.** A one-hot problem ID is a fair reviewer attack. Frame the contribution as **constraint-conditioned selection**: selector sees a formal constraint signature / DSL prompt and learns how constraints shift solver rankings. Ablate: no-token (impossibility baseline) vs problem-ID vs constraint-bitvector vs natural-language/DSL prompt.

## Three ICML-grade angles that survive review

**Angle A — Set-conditioned solver scoring.** Learn `f(instance, constraint_signature, solver_id)` rather than a fixed classifier over a global solver vocabulary. Claim: handles variable pools, overlap, and **solver cold-start** (a *new* solver is added without retraining). Experiment: leave-one-solver-out & leave-one-problem-out, measure regret to Virtual Best Solver (VBS).

**Angle B — Constraint-compositional zero-shot selector.** Train on some MVRP constraint combinations, test unseen combinations. Claim: *selection generalizes compositionally even when underlying solvers do not*. Experiment: ablate problem-ID vs constraint-bitvector vs prompt; report zero-shot regret vs compositional baselines (CoEKS / RouteFinder).

**Angle C — Cost-aware cascade selector.** Predict Pareto distribution over (quality, runtime); choose one solver or top-k under budget. Claim: beats SBS, per-problem NSS, and AutoFolio-style baselines in **VBS-gap per second**, not just accuracy. Experiment: budget sweeps; compare against running all solvers (Poppy-style upper bound).

## Revised claim plan

1. **Primary claim**: A shared-backbone, masked, constraint-conditioned selector matches 18 NSS specialists on in-distribution problems while **beating them on low-data / unseen-combination MVRP variants** (transfer win). Must include negative-transfer diagnostics.
2. **Secondary claim**: Set-conditioned scoring enables solver cold-start (add LEHD-v2 without retraining) — untouched by any prior NCO work.
3. **Tertiary claim**: A simple prior-shift loss (Balanced Softmax restricted to `M(x)`) closes the low-data gap without degrading head-problem performance — empirical bridge from long-tail to per-instance AS.

## Updated gap-map (for wiki)
- **G1** → narrowed: "no supervised selector with variable/overlapping solver pools across ≥3 routing problems, including constraint-compositional zero-shot."
- **G2** → scoped: "NCO-specific imbalance of overlapping solver IDs across 18 problems."
- **G3** → replaced by **G3'**: "constraint-compositional zero-shot solver selection — known solvers, unseen constraint combinations."
- **G4 (new)**: "solver cold-start in neural solver selection" (inspired by set-conditioned scoring).
- **G5 (new)**: "cost-aware per-instance selection under a runtime budget across heterogeneous VRP variants" (inspired by FrugalML/RouterBench analogue).

## Files
- PDFs added to `literature/downloads/` (15 total now). v1 review still valid; this file supersedes it for **framing and claim strategy**.
