# Final Proposal — Unified Supervised Neural Solver Selector

**Date**: 2026-04-18. **Venue target**: ICML. **Compute**: 2× RTX 3090.

## Problem Anchor
Given a per-instance cost vector over a **variable** candidate solver pool (size 3–10 depending on the routing problem) across **18 routing problems** (TSP, CVRP, ATSP, 15 MVRP variants), learn **one** supervised selector that picks the best solver per instance — beating the Single-Best-Solver (SBS-on-val) baseline on as many of the 18 problems as possible under macro mean cost.

## Dominant Contribution
A **unified masked-softmax selector with regret-soft listwise labels**, conditioned on a compact constraint-bitvector + coord-distribution code, so a single model handles heterogeneous input formats (coord / matrix), overlapping-and-variable method pools, and MVRP constraint compositionality.

## Intentionally Rejected Complexity
- Cost-aware / time-aware selection (user excluded `time` from supervision).
- Flat MoE per problem-family (PLE-style) — crowded turf vs CoEKS/MVMoE; selector is too tiny to need the capacity.
- Set-conditioned solver cold-start claim (no genuinely new solver to add → unprovable).
- Hypernetwork-style per-problem parameter generators — unnecessary at this compute scale.

## Final Method — 10-line forward

Global: solver vocab `M`; width `d`; MVRP constraint-bit `K`; coord-dist code `D`.
Batch inputs: `x_b, p_b, v_b∈{0,1}^K, q_b∈{0,1}^D, A∈{0,1}^{B×M}, C∈R^{B×M}`.

1. `T_b = Adapter_mod(x_b)` (coord-GAT for TSP/CVRP/MVRP; MatNet-style matrix for ATSP).
2. `g_b = Pool(Encoder_θ(T_b))` — **LayerNorm/GraphNorm only, never BatchNorm**.
3. `h_b = LN(W_g g_b + E_prob[p_b] + W_v v_b + W_q q_b)`.
4. `e_s = E_sol[s]`; `z_{b,s} = [h_b, e_s, h_b⊙e_s, |h_b−e_s|]`.
5. `ℓ^main_{b,s} = MLP_head(z_{b,s})`.
6. MVRP add-on: `ℓ_{b,s} = ℓ^main_{b,s} + 1[p_b∈MVRP]·MLP_fact([e_s⊙W_c v_b, e_s⊙W_d q_b, e_s⊙W_c v_b⊙W_d q_b])`.
7. `π_b = softmax_s(ℓ_{b,s} + log A_{b,s})`, `log 0 ≡ −1e9`.
8. `c^*_b = min_{s:A_{b,s}=1} C_{b,s}`.
9. `r_{b,s} = (C_{b,s} − c^*_b)/(|c^*_b|+ε)`; `y_b = softmax_s(−r_{b,s}/τ + log A_{b,s})`.
10. `L = −(1/B)Σ_b Σ_s y_{b,s} log π_{b,s}`; prediction = `argmax_s{A}π_{b,s}`.

Availability mask used **only** on lines 7 & 9. `ind` is sanity-check only.

## Key Claims

1. **C1 (main)**: The unified selector matches or beats SBS on macro mean cost across 18 problems.
2. **C2 (anti-leakage)**: Gains come from instance features, not problem-id / availability-mask priors.
3. **C3 (loss)**: Regret-soft listwise loss beats CE / pairwise / gap-regression.
4. **C4 (compositional zero-shot)**: Factorized MVRP scoring generalizes to 4 held-out constraint combos.
5. **C5 (imbalance)**: Availability-conditional Balanced Softmax (BS-mask) lifts tail without hurting head.

## Remaining Risks
- TSP / highly-dominant-SBS variants may leave no room for the selector → correct behavior is collapse to SBS; report honestly.
- Reviewer attack: "selector only learns problem-id + mask prior" → pre-empted by C2 experiment.
- Regret-soft loss margin vs CE may be small → keep C3 claim narrow ("more stable or competitive") if margin < 0.3%.
