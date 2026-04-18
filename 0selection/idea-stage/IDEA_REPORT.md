# Research Idea Report — Unified Supervised Neural Solver Selector

**Direction**: One supervised neural selector across 18 routing problems (TSP / CVRP / ATSP / 15 MVRP variants); per-instance cost labels; metrics = top-1/2/3 accuracy, mean cost, per-method win/lose lists, arm distribution, loss curve.
**Generated**: 2026-04-18
**Reviewer**: GPT-5.4 Pro via Oracle MCP (browser), session `idea-creator-selector-review-1`.
**Ideas**: 8 candidates + 2 injected by reviewer → 7 kept → 1 main method + 4 ablations → no pilots run yet (flagged below).

---

## Landscape Summary (compressed)

Cross-problem neural VRP **solvers** are crowded: URS (2509.23413), CoEKS (ICLR 2026), MVMoE (2405.01029), RouteFinder (2406.15007), MTL-KD (2506.02935), MTNCO (2402.16891), CaDA (2412.00346), ICAM (2405.01906), PromptVRP (2405.12262), GOAL (2406.15079). Cross-problem **selection** is essentially empty: NSS (2410.09693) does TSP+CVRP separately; CTS (2006.00715) and GINES (2302.04035) are TSP-only. ATSP matrix encoding is owned by MatNet (2106.11113).

The novelty of this project must live in: **selection** × **availability masking with variable pools** × **cost-regret objective** × **MVRP solver×variant factor structure**. Unified VRP learning alone is not a sellable contribution.

---

## Ranked Ideas (post oracle-pro triage)

### Idea 1 (MAIN METHOD) — Unified masked selector with regret-soft labels
- **Hypothesis**: A single encoder + problem-ID + constraint-bitvector conditioning, with masked listwise loss **over cost-gap-weighted soft labels** (not hard `ind`), matches 18 per-problem NSS specialists on mean cost while needing one model.
- **Core architecture**: UDR-style node features; problem-ID + constraint-bitvector tokens; instance embedding `h`; per-solver embedding `e_s`; score `h · e_s + b_s`; masked softmax over `M(x)`. Supervised signal: `p*(s|x) ∝ exp(-(c_s−c_*)/τ)` using the full cost vector, not one-hot `ind`. τ swept on val.
- **Key metrics** (per problem): selector top-1/2/3 accuracy, mean cost, regret = (selector_mean_cost − VBS_mean_cost) / VBS, arm distribution, per-method win/lose lists. Add **macro-averaged across 18 problems** as the single headline.
- **Baselines**: per-problem NSS, per-problem single-best-on-val, RouteFinder/URS/CoEKS as solver-side baselines (they are unified *solvers*, not selectors; we compare our selected-solver cost to their unified-solver cost).
- **Novelty (oracle)**: 6.5/10. Risk: reviewer says "NSS + problem-ID + mask". Soft-label regret loss is the biggest novelty lever and the strongest answer to "mean-cost, not accuracy, is what matters".
- **Feasibility**: selector = encoder + MLP, tiny. Cost-vector labels already exist in `raw_label.pkl`. Full training ≈ 8–15 GPU-h on 2× 3090. **Pilot candidate.**
- **Pilot design** (≤4h): reproduce NSS on TSP+CVRP; add masked softmax + soft labels + problem-ID; train 1 seed, small epochs, verify top-1 ≥ NSS single-problem number on TSP & CVRP; check MVRP top-1 > random (1/4 = 25%).

### Idea 2 (ABLATION) — MVRP solver×variant factorized scoring
_Injected by reviewer_. 15 MVRP variants share exactly 4 solvers; exploit the 2-D structure.
- **Hypothesis**: Score = `h(x) · e_s + b_s + u_{s,constraint_bit} + u_{s,coord_dist}` beats a generic problem-ID model on MVRP mean cost and makes the zero-shot-combo story (Idea 3) interpretable ("which constraints make solver S preferred").
- **Metric fit**: native — arm-distribution becomes grounded by constraint-bit; per-method win/lose lists become per-constraint-bit.
- **Novelty**: 6/10. Directly addresses "method-pool imbalance/overlap" (G2) and supports MVRP zero-shot.
- **Pilot**: ≤3h (MVRP-only subset). **Pilot candidate.**

### Idea 3 (HEADLINE GENERALIZATION) — Compositional zero-shot MVRP
- **Hypothesis**: Hold out 4 MVRP constraint combinations (e.g. OVRPBL, VRPBLTW, OVRPBTW, OVRPLTW); train on the other 11. Selector generalizes compositionally even when solver ranks change.
- **Metric fit**: top-1/2/3 on held-out variants; macro mean-cost regret; arm distribution shift vs in-distribution.
- **Novelty**: 7/10. Distinct from URS-style solver zero-shot — *we test selection*, not solution construction.
- **Pilot**: ≤4h (train on 11, eval on 4). **Pilot candidate.**
- **Reviewer objection**: only 4 solvers; mitigated by reporting regret (cost), not just top-1.

### Idea 4 (ABLATION) — Loss study (CE vs pairwise vs listwise vs gap-regression)
- **Hypothesis**: Regret-weighted listwise loss > CE on hard `ind` under mean-cost metric.
- **Add metrics**: pairwise win-AUC, Kendall τ vs oracle ranking, regret at margin bins.
- **Feasibility**: same training harness × 4 losses. ≤2h each. **Needed as main-paper ablation**.

### Idea 5 (ABLATION) — BS-mask: availability-conditional Balanced Softmax
- **Hypothesis**: Adding `−log p(s|M(x))` logit shift fixes TSP-domination and lifts ATSP / MVRP top-1 without hurting TSP.
- **Add metrics**: macro-balanced top-1, arm entropy, ECE calibration.
- **Feasibility**: 1h. **Needed ablation**.

### Idea 6 (ABLATION) — Problem-ID dropout (anti-leakage sanity)
- **Hypothesis**: With problem-ID randomly dropped at 25/50/100%, top-1 on OVRP stays above random even on same-field-schema cases — shows the model learns from instance features, not just ID lookup.
- **Metric**: OVRP vs CVRP confusion matrix; per-problem top-1 with/without ID.
- **Feasibility**: 1h. **Needed sanity ablation**.

### Idea 7 (ENGINEERING, NOT A CLAIM) — Dual coord+matrix encoder for ATSP
- **Why**: ATSP has no coords. Use MatNet-style matrix encoder + shared projection head. Fold into Idea 1 quietly — not a headline contribution (MatNet already exists).
- **Extra ATSP features** (oracle suggestion): symmetry ratio, row/col entropy, triangle-inequality violation count, cost skew.

---

## Eliminated Ideas

| Idea | Reason |
|------|--------|
| Set-conditioned solver cold-start (dot-product new-solver embedding) | Cold-start claim not provable without actually adding *new* solvers; dot-product head is standard; risks review rejection on weak evidence |
| PLE-style hierarchical experts per problem-family | Competes with CoEKS/MVMoE on their turf; selector is too small to need expert capacity; evaluation story muddles the main contribution |
| Cost-aware cascade / Pareto-predicting selector | User explicitly excluded `time` from supervision |

---

## Pilot Experiments — Status: FLAGGED (needs manual pilot)

Per skill convention, pilots are flagged rather than launched because:
(a) base code (selector training loop) is not yet adapted from NSS to 18 problems — this is implementation, not a one-command launch;
(b) user has not authorized running training; this call is for ideation, not execution.

| Idea | Pilot design | Est. GPU-h | Status |
|------|-------------|-----------:|--------|
| 1 — masked + soft-label selector | Extend NSS repo, train on all 18, 10 epochs, 1 seed; check top-1 ≥ NSS on TSP/CVRP and > random on MVRP/ATSP | 3–4 | needs manual pilot |
| 2 — MVRP factorized scoring | MVRP-only run, 1 seed, compare to Idea 1 ablated | 2–3 | needs manual pilot |
| 3 — compositional zero-shot | Train on 11 MVRP, eval on 4 held-out; 1 seed | 3–4 | needs manual pilot |

Total pilot budget: ~10 GPU-h (within the 8-h skill cap is tight; user's override to **≤4h per idea** permits this).

---

## Suggested Execution Order
1. Implement base harness (extend NSS → 18 problems, masked softmax, soft-label option). Run Pilot-1.
2. If Pilot-1 passes thresholds (top-1 ≥ NSS on TSP/CVRP; >25% on MVRP), add Idea 2 factorization, re-run on MVRP subset.
3. Run Pilot-3 (compositional zero-shot) on the Idea-1+2 model.
4. Add ablations 4, 5, 6 as single-factor swaps on the fixed base.

## Key Reviewer Defenses (Pre-empted)
- **"One-hot ID trivializes disambiguation"** → frame as *constraint-conditioned selection*, include Problem-ID-dropout ablation.
- **"Why not train 18 NSS specialists?"** → compare macro mean-cost regret; report low-data lift on ATSP and per-MVRP-variant. Compositional zero-shot (Idea 3) is the real differentiator.
- **"Soft labels are just label smoothing"** → τ-sweep; show regret-weighted loss Pareto-dominates CE at fixed parameter count.
- **"ATSP pool is only 3 solvers"** → report ATSP separately, not in macro headline.

## Next Steps
- [ ] Set up `nss-code/` branch extended to 18 problems + masked softmax.
- [ ] Cache cost-vectors in a single `labels.npz` to avoid re-reading `raw_label.pkl` per step.
- [ ] Run Pilot-1 (Idea 1) — 4h budget.
- [ ] Decide on τ for soft labels on val.
- [ ] If positive, call `/auto-review-loop` after the base + 3 ablations are done.
