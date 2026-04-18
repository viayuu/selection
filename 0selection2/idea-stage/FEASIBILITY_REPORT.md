# Feasibility Check — Unified Supervised Solver Selector (18 problems)

**Reviewed by**: GPT-5.4 Pro via Oracle MCP (browser), session `selector-feasibilit-review-1-2`.
**Question**: Not "is it novel?" — "will the proposed basic approach actually work, hit the goal, and give not-terrible performance?"

**Verdict**: ✅ **PROCEED** — basic approach is sound. Realistic expectation: **beat SBS on ~10–13 of 18 problems**, assuming oracle-SBS gap is non-trivial on each. Biggest risks are silent implementation bugs (masking, cost-scale normalization, BatchNorm across mixed problems), not the approach itself.

---

## 1. Is the basic approach sound? (Yes, with 5 landmines)

| # | Silent failure mode | Fix |
|---|---------------------|-----|
| 1 | Masked-softmax bug: unmasked logits leak into softmax; absent solvers receive gradients through padded labels | Set non-pool logits to `-inf` BEFORE softmax; compute loss only over valid IDs; assert `prob[~mask]==0` |
| 2 | Cost-scale bug: `c_s − c_*` is translation-invariant but **not** scale-invariant → TSP499 vs MVRP100 vs ATSP impose different label sharpness | Use **relative regret**: `r_s = clip((c_s − c_*)/(|c_*|+ε), 0, r_max)`, then `p_s = softmax(−r_s/τ)` |
| 3 | Numerical ties dominate hard `ind`: solvers within 1e-4 differ randomly across instances | Soft labels with tie tolerance (treat `r_s < 1e-3` as tied) |
| 4 | Feature normalization inconsistent across problems | Coords ∈[0,1]; demand/capacity normalized; route_limit / TW scaled by instance horizon or max distance; ATSP matrix normalized per instance (÷max or ÷mean). Missing fields = 0 **plus constraint-bits** (never arbitrary constants) |
| 5 | **Never use BatchNorm across mixed-problem batches** — it will catastrophically fail | Use LayerNorm / GraphNorm / per-instance normalization throughout |
| 6 | Metric leakage: SBS picked from test | Fix SBS from val, freeze for test |

## 2. Expected per-family outcome vs SBS

Compute these diagnostics **before training**: per-problem SBS cost, oracle cost, oracle-SBS gap, SBS win-rate, winner entropy. If oracle-SBS gap < 0.1–0.3% or SBS wins > 85–90% of instances → the selector has almost no room; that problem will **correctly** collapse to SBS.

| Family | Methods | Prediction |
|--------|---------|------------|
| TSP (10) | Likely small positive gain on top-1, larger on top-2/3. NSS itself beat the best single TSP solver. |
| CVRP (9) | **Strongest** chance of clear SBS-beating; NSS saw bigger gains here than on TSP. |
| ATSP (3) | Depends entirely on winner diversity. If one arm dominates > 90% → collapse is correct behavior. Top-2 and mean-regret are the honest metrics (top-3 is trivially 100%). |
| MVRP (4 × 15) | Mixed. Expect gains on **8–11 of 15**, mostly where constraint/coord-distribution flips the winner. Don't promise all 15. |

**Realistic headline**: selector beats SBS on **10–13 / 18** problems. The real predictor is oracle-SBS gap, not model capacity.

## 3. ATSP: will the selector collapse?

3 arms is fine statistically (10k labels). Collapse happens **only if one arm dominates > 90%** — and that collapse is the *correct* behavior. Report top-2 accuracy and mean-regret, not top-3.

## 4. Normalization — residual warnings

Per-instance relative regret (above) is the right default but watch for:
- Near-zero or negative costs (shouldn't happen for routing, but guard with ε and sign checks).
- Heterogeneous solver noise across problems — log **per-problem label entropy** to detect over-sharp or over-flat labels.
- Size-dependent regret distributions (TSP n=499 vs n=50) — consider **per-family τ** if global τ gives pathological behavior.

## 5. Mixed-problem sampling

Use **uniform-per-problem sampling** for the pilot (each of 18 problems equal batch quota; stratify MVRP's 4 coord-distributions inside each variant). **Do not** use curriculum tomorrow. Proportional-to-data-size only matters if you care about instance-weighted global average, which you don't.

## 6. Minimum-viable code path (by tomorrow)

1. **Precompute diagnostics**: per-problem SBS, oracle, SBS win rate, winner entropy, per-method mean cost. Save to `diagnostics.json`.
2. **Unified Dataset**: each sample returns `{problem_id, constraint_bits, node_tensor_or_matrix, valid_solver_mask, global_solver_ids, cost_vector, ind, relative_regret_labels}`.
3. **Reuse NSS coord encoder** (GAT + hierarchical pooling) for TSP/CVRP/MVRP — *survives intact*.
4. **Add minimal ATSP encoder**: MatNet-style row/col attention. If time is tight: matrix summary features (symmetry ratio, row/col entropy, skew, TI-violation count) + MLP — this is enough for pilot.
5. **Shared projection**: coord encoder and ATSP encoder → same `d_model`.
6. **Rewrite solver head**: global per-solver embeddings; masked dot-product logits; masked softmax.
7. **Rewrite loss**: soft-label listwise CE over valid pool only.
8. **Reuse NSS training loop**: optimizer, checkpointing, loss curve. *Survives*.
9. **Rewrite metrics**: per-problem top-1/2/3, mean cost, SBS comparison, per-method beaten/not-beaten, arm histogram — match `观测指标.md` exactly.
10. **Run overfit test on 128 mixed instances before full training** (see §7).

## 7. Three early-warning checks during training

1. **Micro-overfit test**: on 128–512 mixed instances, training loss should collapse and top-1 should go high. Failure → masking, labels, or encoder wiring is broken. **Always run first**.
2. **Mini-val macro regret vs SBS, every epoch**: should approach-or-beat SBS within a few epochs on problems with non-trivial oracle gap.
3. **Arm-distribution sanity**: predicted arm histogram must be (a) zero outside the valid mask and (b) roughly correlated with empirical winner distribution per problem. Immediate global collapse to one arm → conditioning, masks, or label-scale is wrong.

## TL;DR

- Approach is feasible. ✅ PROCEED.
- Main risk is **silent bugs** in masking + normalization, not the idea.
- Realistic outcome: **10–13 of 18 problems beat SBS**; ATSP and dominant-SBS MVRPs may not.
- Implement in the order §6 lists; fire the §7 sanity checks in order.
- Oracle session: `selector-feasibilit-review-1-2`.
