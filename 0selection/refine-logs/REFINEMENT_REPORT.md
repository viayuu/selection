# Refinement Report — What Survived, What Was Cut, and Why

## Survived (kept in the final method)
| Component | Source / justification |
|-----------|------------------------|
| Masked softmax over M(x) with listwise CE | Correct handling of variable/overlapping pools; non-pool logits set to `-inf` before softmax |
| **Regret-soft labels** `p*(s|x) ∝ exp(−r_s/τ)` with relative regret | Aligns loss with `mean cost` metric; robust to cost-scale heterogeneity across problems; handles numerical ties |
| Shared encoder (GAT+pool) + ATSP MatNet-style branch, shared projection | Bridges coord vs matrix instances without separate models |
| Problem-ID embedding + constraint-bitvector + coord-dist code | Disambiguates OVRP-vs-CVRP (identical node fields) and enables MVRP compositional generalization |
| Solver embeddings + pair-feature `[h,e,h⊙e,|h−e|]` | Interaction modeling; compatible with MVRP factorized add-on |
| MVRP factorized scoring (solver × constraint × coord-dist) | Exploits 15×4 MVRP label structure; enables C4 zero-shot claim |
| **Availability-conditional Balanced Softmax (BS-mask)** | Correct way to apply prior-shift when pool varies; attacks label imbalance across problems |
| Problem-ID dropout (0.25) | Prevents selector from degenerating to problem-ID lookup; defends against reviewer "lookup-table" attack |

## Cut (intentionally rejected complexity)
| Rejected | Why |
|----------|-----|
| Cost-aware Pareto / runtime cascade | User excluded `time` from supervision |
| PLE-style hierarchical per-family experts | Crowded turf (CoEKS/MVMoE); selector is too tiny to need capacity; evaluation story muddied |
| Set-conditioned solver cold-start | No genuinely new solver to add → unprovable; dot-product is standard |
| Hypernetwork per-problem parameter generator (URS-style) | Overkill at this compute scale; conditioning tokens suffice |
| Global long-tail losses (vanilla logit adjustment, Seesaw, BAGS) | Unavailable solvers aren't true negatives → unsafe; BS-mask is the correct restriction |

## What changed between rounds
- **G3 → G3'**: the "infer problem type from identical input features" framing was weak (one-hot ID trivializes it). Replaced by constraint-compositional zero-shot.
- **Soft labels added** after round 2 when oracle pointed out hard `ind` is noisy when solvers tie.
- **MVRP factorization injected** after round 2 as a separate novelty lever exploiting 15×4 label structure.
- **Metadata-only baseline added** after round 4 as the pre-emptive defense against "problem-ID lookup" criticism.
- **Compute target tightened** from "≤4h per idea" to **58 GPU-h total** across 5 claim groups (round 5).

## Remaining risks (honest)
1. **R1 (medium)**: TSP / some MVRP variants have dominant SBS → selector will correctly collapse; must frame as "matches SBS where oracle-gap is small". Mitigate by pre-computing per-problem oracle-SBS gap and reporting it.
2. **R2 (low)**: Regret-soft margin vs CE may be < 0.3% → C3 claim softens to "more stable/competitive".
3. **R3 (medium)**: MVRP zero-shot generalization may fail on some combos → report per-combo not just macro; factorized model must beat flat zero-shot baseline by ≥ 0.7%.
4. **R4 (low)**: ATSP 3-arm pool may collapse → expected behavior; report top-2 + regret, not top-3.
