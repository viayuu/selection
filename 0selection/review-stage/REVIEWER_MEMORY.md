# Reviewer Memory

_Persistent across rounds. Loaded at the start of each round._

## Round 1 — Score: 7.2/10 (from prior session, oracle-pro)

- **Verdict**: Almost — "basic approach is sound but three concrete gaps to address before re-assessment"
- **Three prescribed fixes**:
  1. **SBS gate calibration** — confidence-gate the selector's pick against SBS so low-confidence cases collapse to SBS. Reason: selector occasionally ties or slightly under-performs SBS when uncertain; the calibrated gate closes the gap from +0.16% to near 0.
  2. **Combined cost loss** — weighted sum of CE (regret-soft) + expected-regret + switch-hinge. Reason: regret-soft CE alone under-penalises when selector chooses a dominated solver far from the oracle; the switch-hinge explicitly rewards picking the true winner *over* SBS whenever the instance-level VBS-SBS gap exceeds a threshold (2e-3 rel).
  3. **Per-problem per-solver learned bias + oversampling** — add a `problem_solver_bias ∈ R^{P×M}` table (init 0) and oversample the 10 under-performing problems (CVRP×4, VRPB×3, OVRPB×3, OVRP×2, rest ×1.5). Reason: the head-MLP shares weights across all 18 problems; some problem-solver-priors are hard to learn from pooled gradients; a tiny bias table + balanced sampling gives each problem a dedicated knob.
- **Suspicions flagged**:
  - Selector might be learning problem-ID priors rather than instance structure (already falsified by R2 MetaOnly baseline = +75.23% vs R1 +0.16%).
  - 2-epoch pilot may not generalize — oracle wants multi-seed + extended training.
  - ATSP pool has only 3 arms; risk that selector collapses to SBS there. This is OK if macro beats SBS elsewhere, but should be reported honestly.
- **Unresolved going into Round 2**:
  - Does the combined loss actually help vs regret-soft alone? (isolation run needed)
  - Does oversampling hurt under-sampled problems or help them? (check per-problem deltas)
  - Are all 观测指标.md metrics computed? (top1/2/3, per-method win/lose lists, arm distribution, loss curve, and the 3 bar-chart figures)
  - Gate calibration alone brought macro from +0.160% to +0.014% on the 2-epoch pilot; need to confirm this stacks with the other fixes.

## Round 2 — Score: TBD

## Round 2 — Score: 6.8/10

- **Verdict**: not ready yet. The project moved from "likely broken" to "technically plausible", but the current package still does **not** clear the bar for a strong NeurIPS/ICML accept.
- **What improved**:
  - **Fix A (SBS-gate)** is substantively validated on validation: `R4 soft_bias` goes from `+0.091%` to `-0.002%`, and `R4 combined_bias` from `+0.441%` to `-0.004%`.
  - **Bias-only part of Fix C** helped: `R4 soft_bias` improved over `R1` (`+0.156% -> +0.091%` ungated).
  - **No obvious overfitting** on the ungated main run: val `+0.094%` vs test `+0.088%`.
  - **Problem-ID-only shortcut concern** is more strongly falsified by the `MetaOnly = +75.23%` result.
- **What failed / weakened the story**:
  - **Combined loss (Fix B)**, at least in the current form, did not help and is not publishable as a main-method component.
  - **Oversampling part of Fix C** clearly hurt and should remain an ablation / negative result only.
  - The current headline is still weak: ungated model is slightly **worse** than SBS; gated model is only reported on val, with a very small effect size.

- **New critical concerns introduced in Round 2**:
  1. **Missing specialist-selector baseline**: there is still no comparison against training one selector per problem. Without this, the "one unified selector" story is under-justified.
  2. **Gate is not yet a complete result**: the calibrated gate is only shown on val. The paper-critical number is gated performance on **test with frozen thresholds**.
  3. **Effect size is tiny relative to likely variance**: a `0.09%` ungated gap and `~0%` gated gain cannot be trusted without multi-seed and confidence intervals.
  4. **Main novelty experiment is still missing**: compositional zero-shot MVRP generalization remains the most credible route to a top-venue story, and it has not been run yet.
  5. **Residual errors are concentrated in the large-arm problems** (`TSP`, `CVRP`): this suggests global calibration / temperature mismatch across heterogeneous problem families.

- **Status of prior suspicions / prescriptions**:
  - **Selector learning problem-ID priors?** `falsified` more strongly.
  - **2-epoch pilot / need longer training?** `partially addressed`: 4-epoch val/test match is good, but multi-seed evidence is still missing.
  - **ATSP only 3 arms — OK if collapses to SBS?** `sidestepped but acceptable`: not a current blocker, but also not a source of evidence for instance-level intelligence.
  - **Fix A (SBS-gate)** `addressed on val only`; still needs frozen-threshold test reporting.
  - **Fix B (combined loss)** `falsified in current implementation`.
  - **Fix C (bias + oversampling)** `split`: bias helped, oversampling hurt.

- **What would most increase score next round**:
  1. Add a **specialist-selector baseline**.
  2. Report **gated test** with **frozen val-calibrated thresholds**, plus **5-seed** uncertainty.
  3. Run the **compositional zero-shot MVRP** experiment; if it works, that can become the paper's real headline.

## Round 3 — Score: 7.9/10

- **Verdict**: almost ready, but not a clean 8/10 yet under a strict top-venue bar.
- **Major progress**:
  1. **Specialist baseline added**: per-problem specialist selector achieves `test_vs_sbs = -0.014%` (1 seed), establishing the in-distribution upper baseline.
  2. **Frozen-γ gated test with uncertainty added**: `R5 zero-shot` (train on 13, test on 18) reaches `test_gated = -0.002% [-0.005, +0.000]` over 5 seeds. This is much stronger evidence than Round 2.
  3. **Compositional zero-shot added**: held-out unseen composites achieve `+0.002% [-0.003, +0.006]`, very close to seen-problem performance.
  4. **Large-arm residuals improved** in the new regime: `TSP` goes substantially below SBS; `CVRP` is near-zero.
- **Why score still stays below 8**:
  1. **The zero-shot headline is still vulnerable to a trivial explanation**: all 15 MVRP variants share the same SBS (`RELD_MTL`), so near-SBS held-out performance could be explained by the gate frequently collapsing to the shared SBS rather than true compositional solver selection.
  2. **Test CI is still borderline**: `[-0.005, +0.000]` is much better, but not a clean strictly-negative upper bound under a conservative reading.
  3. **Specialist comparison is still underpowered**: only 1 seed for specialists, so "matches specialist" is not yet statistically secure.
  4. **Optimization instability remains visible**: severe epoch-to-epoch swings weaken confidence in robustness.

- **Status of Round 2 prescribed fixes**:
  - **Specialist baseline**: `addressed`, but only partially because it is still 1-seed.
  - **Frozen-γ gated test + 5 seeds + CI**: `addressed`.
  - **Compositional zero-shot**: `addressed`, but interpretation remains `partially secure` until trivial-SBS fallback is ruled out.
  - **Per-family τ / large-arm residual repair**: `sidestepped acceptably`; the residual improved through the R5 training regime, so this is no longer first-order.

- **New main suspicion introduced in Round 3**:
  - **"Zero-shot" may be mostly abstention-to-shared-SBS on unseen MVRP composites.** This is now the key thing to falsify.

- **What would most likely push score above 8 next round**:
  1. Report **held-out gate usage / fallback rate / non-SBS arm distribution** and show the model is not simply defaulting to `RELD_MTL`.
  2. Add **2-3 seeds for the specialist baseline** so the unified-vs-specialist gap can be interpreted fairly.
  3. Add a **held-out accepted-subset analysis**: on instances where the gate accepts a non-SBS arm, show a real gain vs SBS.

## Round 4 — Score: 8.3/10

- **Verdict**: weak accept / ready if the paper is framed carefully and honestly.
- **Why the score crosses 8**:
  1. **Specialist baseline is now statistically grounded**: `-0.010% [-0.014, -0.002]` over 3 seeds. This confirms a small but real in-distribution sharing penalty.
  2. **The shared-SBS loophole is no longer the main blocker**: the held-out analysis shows the unified model is not merely degenerating into trivial SBS abstention. In particular, ungated R5 beats **every non-SBS single method in every seed on every held-out composite**, which is strong evidence of non-trivial compositional ranking.
  3. **The final story is now coherent**: one unified model, near-specialist in-distribution performance, and unique zero-shot applicability to unseen composites with SBS-safe gating.
- **Why the score is not higher**:
  1. **The zero-shot paper number still leans on the gate**. Held-out gated performance is `+0.002% [-0.003, +0.006]`, i.e. matching SBS rather than clearly beating it.
  2. **Non-SBS accepted mass on held-out is still modest** (`8.6%`), so the compositional behavior is real but still sparse in the final deployed decision rule.
  3. **A severe over-confident failure remains** on `OVRPBTW` for non-SBS picks (`-11.389%` on that subset). This is now the main empirical risk.
  4. **Only one held-out compositional split** has been tested; the generalization claim is therefore credible but not yet maximally robust.
  5. **Optimization instability** remains visible.

- **Status of prior blockers**:
  - **Specialist baseline missing / underpowered**: `addressed`.
  - **Frozen-γ test + 5 seeds + CI**: `addressed`.
  - **Shared-SBS fallback loophole**: `mostly addressed`; no longer the key blocker, but still a residual caveat because the final gated decision uses non-SBS arms on only `8.6%` of held-out instances.
  - **Large-arm residuals (`TSP`, `CVRP`)**: `addressed in practice` by the R5 regime.

- **New main blocker after Round 4**:
  - **Risk calibration on unseen composites**: the model can still be sharply wrong when it confidently takes a non-SBS action on certain held-out variants (`OVRPBTW`). The next step is no longer "prove zero-shot exists", but "prove zero-shot decisions are trustworthy when the gate lets them through."
  - **Protocol caveat to clarify in the paper**: if held-out composites use their own validation labels to calibrate per-problem `γ_p`, then the claim is best described as **zero-shot model transfer + lightweight per-problem calibration**, not pure parameter-free zero-shot deployment.

- **What would most increase score next round**:
  1. Add a **second held-out compositional split** (or leave-one-3-bit-family-out split) to show the zero-shot story is not split-specific.
  2. Add a **risk-targeted gate refinement** focused on the held-out failure mode, e.g. calibrate with a higher abstention threshold on high-risk variants or with a global two-threshold rule.
  3. Show a **reliability analysis** on held-out non-SBS decisions: accepted-subset regret vs confidence bins / margin bins.

## Round 5 — Score: 8.7/10

- **Verdict**: solid weak accept, approaching clear accept territory, but not yet a 9.
- **Why the score goes up**:
  1. **Second held-out split added**: Split-B reproduces the key held-out result (`+0.002%` held-out gated, CI overlapping Split-A), which substantially weakens the "single lucky split" concern.
  2. **OVRPBTW failure is now mechanistically understood and appears fixable**: tightening the gate by `λ=1.25` removes the catastrophic non-SBS failure and improves held-out gated macro from `+0.002%` to `-0.002%` on Split-A.
  3. The overall story is now more robust: the method's zero-shot behavior has been shown on **two different held-out compositional splits**, and the main failure mode is no longer mysterious.
- **Why the score does not reach 9**:
  1. **The λ=1.25 result is test-tuned as presented**. This is strong diagnostic evidence but not fully paper-clean as a headline number unless `λ` is selected on held-out validation and frozen before test.
  2. **Split-B still has only 3 seeds and a fairly wide CI**.
  3. **Training instability remains**.
  4. The unified model still **loses slightly to specialists in-distribution**.

- **Status of Round 4 concerns**:
  - **OVRPBTW catastrophic non-SBS failure**: `mostly addressed diagnostically`; it appears fixable by mild gate tightening, but not fully closed for the paper until the λ-protocol is cleaned.
  - **Single held-out split concern**: `mostly addressed`.
  - **Shared-SBS fallback loophole**: `addressed`.
  - **Need stronger robustness evidence**: `partially addressed`.

- **Main remaining blockers after Round 5**:
  1. **Protocol cleanliness of the refined gate**: if `λ=1.25` is to be reported, it must be calibrated on validation rather than on held-out test.
  2. **Generalization robustness beyond two splits**: a third split or a leave-one-family-out study would make the claim much harder to dismiss.
  3. **Optimization robustness**: large epoch spikes still make the method feel brittle.

- **What would most increase score next round**:
  1. Re-run the **λ refinement with a clean validation-only protocol**, then evaluate once on test.
  2. Add **one more held-out split** or a small leave-one-family-out matrix.
  3. Add a **confidence / reliability plot** for accepted non-SBS decisions on held-out composites.

## Round 6 — Score: 8.8/10

- **Verdict**: clear weak accept, but still not a 9 under a strict top-venue standard.
- **What improved**:
  1. **The λ protocol caveat is largely closed**. The refined gate is now selected on held-out validation and frozen before test, which makes the lightweight-calibration story paper-clean.
  2. **Two held-out splits now replicate under the clean protocol**: both Split-A and Split-B achieve held-out `macro_vs_sbs = -0.001%`, which is strong evidence that the zero-shot transfer story is not split-specific.
  3. The correct headline is now clearer: the method **reliably matches SBS-level cost on unseen composites under lightweight calibration**, rather than beating SBS decisively.
- **Why the score does not reach 9**:
  1. **The OVRPBTW reliability failure is not actually closed by the clean protocol**. Validation-based λ selection only rarely chooses the safer `1.25`, so the catastrophic `-11.4%` non-SBS-subset failure persists in most seeds.
  2. **The held-out headline remains a tie-level result, not a win-level result**: both splits sit at `-0.001%` with CIs crossing 0.
  3. **Split-B still has only 3 seeds** and wide uncertainty.
  4. **Optimization instability remains unresolved**.
  5. The unified model still **underperforms specialists slightly in-distribution**.

- **Status of prior blockers after Round 6**:
  - **Protocol cleanliness of λ-refined gate**: `addressed`.
  - **Second split / split-specificity concern**: `mostly addressed`.
  - **Shared-SBS fallback loophole**: `addressed`.
  - **OVRPBTW catastrophic non-SBS failure**: `partially addressed only diagnostically`, not solved in the clean deployment protocol.
  - **General robustness beyond two splits**: `partially addressed`.

- **New best claim after Round 6**:
  - A unified selector can be transferred zero-shot to unseen composite routing variants and, with lightweight validation calibration, **match SBS-level cost across multiple held-out splits**. This is weaker but cleaner than claiming a zero-shot win.

- **What would most increase score next round**:
  1. Add a **protocol-clean conservative gate rule** that fixes OVRPBTW in deployment, e.g. a globally chosen `λ_min > 1` selected on pooled validation only.
  2. Add a **third held-out split** or leave-one-family-out matrix.
  3. Add a **risk-coverage / confidence calibration analysis** for accepted non-SBS actions.

## Round 7 — Score: 8.9/10

- **Verdict**: strong weak accept / borderline clear accept, but still not a 9 under a strict NeurIPS/ICML standard.
- **What improved**:
  1. **The deployment rule is now cleaner and simpler**: a single global `λ_min` is chosen from pooled held-out validation and then frozen for test. This is more publishable than per-problem ad hoc tuning.
  2. **The clean global rule replicates across both held-out splits**: Split-A and Split-B both achieve held-out `macro_vs_sbs = -0.001%`, which stabilizes the "safe matching" story.
  3. The paper's best claim is now quite coherent: under a clean lightweight-calibration protocol, the unified selector **matches SBS-level cost on unseen composites** while remaining deployable as a single model.
- **Why the score still does not reach 9**:
  1. **The result is still a tie-level result, not a superiority result**. `-0.001%` with CIs crossing 0 is respectable, but not a strong empirical win.
  2. **OVRPBTW remains the main unresolved reliability blemish**. The global rule does not remove the catastrophic non-SBS-subset failure; it merely keeps the macro average near SBS.
  3. **Only two held-out splits** have been studied.
  4. **Training instability** remains.
  5. Specialists still retain a **small but real in-distribution edge**.

- **Status of prior blockers after Round 7**:
  - **Clean λ protocol caveat**: `addressed`.
  - **Need for a clean conservative deployment rule**: `addressed`.
  - **Single-split concern**: `mostly addressed`.
  - **OVRPBTW catastrophic residual**: `still present`; now an honest limitation rather than a hidden protocol flaw.
  - **Need stronger generalization breadth**: `partially addressed`.

- **Best current claim after Round 7**:
  - A unified selector can be deployed zero-shot to unseen composite routing variants and, with a pooled-validation-calibrated global safety gate, **match SBS-level cost across multiple held-out splits**. This is a clean and defensible claim, but still not a decisive win claim.

- **What matters most for a 10/10 next round**:
  1. A **leave-one-family-out** or similarly structured generalization matrix, which would be stronger than yet another random split.
  2. A **risk-coverage / reliability analysis** that shows accepted non-SBS actions are trustworthy.
  3. A **third held-out split** as additional robustness evidence.
  4. **Training stability** improvements, which matter but are lower priority than the two items above.

## Round 8 — Score: 9.0/10

- **Verdict**: clear accept under a strict top-venue standard, but still short of a 10.
- **Why the score crosses 9**:
  1. **Structured generalization is now demonstrated**, not merely suggested. The leave-out-TW experiment is the first evidence that the model can transfer to an entire withheld constraint family rather than only to random unseen composites.
  2. **The zero-shot story now replicates across three qualitatively different holdout settings**: two random composite splits plus one structured leave-family-out split.
  3. **Risk-coverage evidence upgrades the deployment story**: the remaining anomaly is now localized and interpretable rather than an uncharacterized failure mode.
  4. The best current claim is scientifically coherent and sufficiently strong for a top venue: with lightweight validation calibration, the unified selector **matches SBS-level cost on unseen problem families and unseen composites** while specialists remain non-transferable.
- **Why the score is still not higher**:
  1. **The headline is still an equivalence / matching result, not a superiority result**. Held-out performance clusters around `+0.002%` / `0.000%`, not a decisive win.
  2. **The full structured generalization story is still incomplete**: only leave-out-TW has been run, not a fuller leave-family-out matrix.
  3. **Training instability** remains.
  4. Specialists still retain a **small but real in-distribution advantage**.

- **Status of prior blockers after Round 8**:
  - **Leave-one-family-out / structured generalization evidence**: `addressed`.
  - **Risk-coverage / reliability plot**: `addressed`.
  - **OVRPBTW catastrophic residual**: `mostly addressed`; it now looks like a split-specific local anomaly rather than a general deployment failure.
  - **Need stronger generalization breadth**: `partially addressed`; one family-out split is strong evidence, but not a full matrix.
  - **Need decisive zero-shot win**: `not addressed`.

- **Best current claim after Round 8**:
  - A unified selector can be transferred zero-shot to unseen composite variants and to an unseen routing-family slice (TW variants) and, under lightweight validation-calibrated gating, **match SBS-level cost across all three holdout settings tested**.

- **What now matters most for a 10/10 next round**:
  1. A **leave-family-out matrix** beyond TW alone (e.g. withhold B, L, TW, O separately or in combinations).
  2. A **stronger deployment / non-inferiority story**, ideally formalized as equivalence or risk-controlled deployment rather than raw mean comparison alone.
  3. **Training stability cleanup**.

## Round 9 — Score: 6.4/10 (nightmare / codex xhigh / GPT read repo)

- **Focus shift**: user asked the loop to maximize *in-distribution* effect (↑top1, ↓mean_cost, ↓gap vs SBS). Previous 9.0/10 was for the *zero-shot* story, which is a different axis.
- **Reviewer's independently verified numbers (test, like-for-like)**:
  - unified R4 3-seed mean: `macro_top1 = 0.5131`, `macro_vs_sbs = +0.045%`
  - 3-seed logit ensemble: `0.5157 / +0.0188%`
  - specialists 3-seed: `0.5197 / -0.0100%`
  - so: ensemble trims some of the gap but does NOT close it. Residual deficit is structural, concentrated in TSP / CVRP.
- **Verified bugs / red flags (raised in this round)**:
  1. `analyze.py:84` — top2/top3 computed incorrectly (compares row indices from `np.where`, not whether prediction lies in top-k set). Any table using those numbers is untrustworthy.
  2. `multiseed.py:99/115` — gate calibration is labeled "val" but actually uses a train subset via `calibrate_gating`. Protocol label ≠ code path.
  3. `model.py:85` — ATSP branch samples random triplets inside `forward` even in `eval()`. Reproducibility bug; re-running the same ckpt moves ATSP result by a visible amount.
  4. `data.py:24` — `coord_dist` is guessed by index slicing, not read from the dataset; at risk if data layout changes.
  5. `train.py:248` — checkpoint selection is single-objective on `macro_vs_sbs` only, misaligned with the new joint top1+cost goal.
- **Top-5 proposals for effect uplift** (reviewer's ranking; GPU-h all within the 10-h budget):
  1. Problem-conditioned head/adapters (FiLM/LoRA or 4 family heads) — biggest lever, 2-4 GPU-h, may weaken zero-shot.
  2. Reduce / sweep `problem_dropout` (0.25 → 0/0.05/0.1) — cheap, 1-2 GPU-h, weakens transfer regularization.
  3. Specialist → unified distillation — 2-3 GPU-h, best structural uplift that preserves unified model.
  4. 3-seed logit ensemble or EMA/SWA — 0-0.5 GPU-h; already verified empirically (+0.26 top1 pts, -0.027 vs_sbs pts).
  5. Short low-LR tail fine-tune on TSP/CVRP/ATSP only — 0.5-1.5 GPU-h; risk of regressing MVRP.
- **Unresolved going into Round 10**:
  - Does specialist distillation actually beat direct head/adapter conditioning?
  - Can problem_dropout be reduced without killing the zero-shot story in future rounds?
  - Are the 5 red-flag bugs actually affecting any paper-reportable number? (ATSP repro bug is the strongest candidate.)

## Round 10 — pre-review state (effect-only framing)

Actions completed under Round 9's five-fix plan (S0–S5):

- **S0** (red flags): all 5 fixed. `analyze.py`/`train.py` top2/3 now pick_rank<k; `multiseed.py` calibrates γ on val; `model.py` ATSP triplets deterministic in eval(); checkpoint selection adds `best_top1`/`best_joint`.
- **S1** (3-seed ensemble): test top1 0.5124→0.5157, test vs_sbs +0.087%→+0.019%. Cheap win.
- **S2** (problem_dropout sweep): pd=0.05 chosen (test top1=0.5147, test vs_sbs=+0.022%).
- **S3** (FiLM): 3-seed frozen-γ test_gated = +0.001% [-0.001, +0.003], test_ungated = +0.033% [+0.001, +0.068].
- **S4** (specialist distillation ∥ FiLM): 3-seed frozen-γ **test_gated = -0.003% [-0.005, -0.002]** (95% CI strictly below 0), test_ungated = +0.032% [+0.002, +0.074]. Ensemble ungated macro top1=0.5144, top2=0.8697, top3=0.9792, vs_sbs=+0.015%.
- **S5** (tail tune TSP+CVRP+ATSP, lr=1e-5, 1 epoch): 3-seed ensemble on test (ungated) **macro top1=0.5162, top2=0.8751, top3=0.9842, vs_sbs=-0.010%**. Single best seed: top1=0.5179, top2=0.874, top3=0.985, vs_sbs=-0.0147%. CVRP fixed: ungated +0.234%→+0.001%. ATSP unchanged (+0.066% — specialist teacher saturated at SBS).

**Status of Round 9 red flags**: all 5 addressed (S0).

**Effect-axis open questions for Round 10**:
- Is "−0.010% ensemble ungated on test" a statistically meaningful win, or within seed noise? The multiseed frozen-γ gated CI for S5 widened to [-0.012, +0.007]; the cleanest signed result is S4's gated [-0.005, -0.002].
- ATSP residual +0.066% persists — the ATSP specialist is itself at the SBS on test, so distillation cannot improve it. Room for a proper matrix-encoder upgrade.
- Unified vs specialist gap: top1 0.5162 vs 0.5197 (Δ=-0.35 pts), vs_sbs both ≈ −0.010%. Tie on vs_sbs, small top1 gap.
- Are any protocol caveats from Round 9 still present in the post-fix numbers? (expected: no).

## Round 10 — Score: 8.1/10 (nightmare / codex exec xhigh / GPT re-read repo)

- **Verdict**: "Almost" — real upgrade from 6.4 at R9, but still short of a clean effect-only win under a NeurIPS bar.
- **What improved**:
  1. Unified S5 3-seed ensemble ungated test: `macro_top1=0.5162, macro_top2=0.8751, macro_top3=0.9842, vs_sbs=-0.010%` (verified against `runs/S5_ensemble_top1_test/summary.json`). vs_sbs is now tied with specialists mean (`0.5197 / -0.010%`).
  2. S4 3-seed gated test CI is `-0.0032% [-0.0053, -0.0017]` — strictly below 0 (verified `runs/multiseed_S4_top1/multiseed_report.json`).
  3. All 5 Round-9 red flags fixed in code: `analyze.py:84` (pick_rank), `multiseed.py:113` (val split), `model.py:83` (deterministic triplets), `train.py:343` (`best_top1`/`best_joint` ckpts), `eval_full.py:86` (pick_rank semantics for top-k tables).
  4. CVRP residual repaired from `+0.234%` → `+0.001%` on test by S5 tail-tune.
  5. Anti-cherry-picking passes: S4 `best.pt` vs `best_top1.pt` give numerically identical `multiseed_report.json`.
- **Why the score does not cross 9**:
  1. **Top1 gap to specialists still real**: 0.5162 vs 0.5197 = Δ 0.35 pts. vs_sbs is tied, ranking accuracy is not.
  2. **S5's ungated `-0.010%` headline lacks CI**. The CI-backed signed result remains S4 gated.
  3. **ATSP is stuck at `+0.066%`** — the specialist teacher is saturated at SBS there; distillation cannot fix it without encoder upgrades.
  4. **Writeup mismatch**: `AUTO_REVIEW.md:582` claims `best_joint = top1 − 0.5·vs_sbs_pct`; `train.py:351` actually uses `vs_sbs − 100·top1`. Non-load-bearing here but sloppy.
  5. **`coord_dist` still size-bin from index quartiles** (`data.py:24`) — relabeled, not audited.
- **Status of R9 red flags**: `addressed` (all 5, verified against code).
- **Main unresolved concerns going into Round 11**:
  1. **CI-backed S5 ungated number**: 5-seed effect sweep or instance-paired bootstrap on the ensemble prediction to attach a CI to `-0.010%`.
  2. **Ranking (top1) deficit**: direct head-to-head top1/top2/top3 comparison with specialists; what in the unified model is still sub-specialist on rank correctness?
  3. **ATSP encoder ceiling**: either upgrade or explicitly concede as saturated in the paper's limitations.
  4. **Writeup↔code consistency**: fix `best_joint` text or code; audit `coord_dist` provenance (rename `size_bin` if no real data-side coordinate info).
- **Likely score path**: +0.4–0.6 per one of the above if executed cleanly; reaching 9/10 on effect-axis likely requires (a) CI on S5 ungated, AND (b) closing or explaining the top1 gap.

## Round 11 — pre-review state (focused on in-distribution effect, TOP1 goal ≥ 80%)

User-imposed new constraints for this round (explicit):
1. Push top1 from 51% → ≥80% via architecture / training redesign (not seed/HP noise).
2. Fix arm distribution collapse (many solvers never picked).
3. Implement 8-fold D4 augmentation (Kwon et al. 2020 / NSS).
4. Maximize GPU0 utilization (not GPU1 — shared with colleague on 0selection2).
5. Longer training runs (not just short validation).
6. Reference architectural ideas from `literature/CoEKS.md`, `literature/urs.md`, `literature/Neural Solver Selection*.md`; also look at recommender-system ideas, not just CO.

### Pre-round-11 diagnostics
- **Label ceiling analysis** (`runs/diag_r11/label_ceiling.md`): at 0.1% relative margin, 94.6% of test instances have a uniquely-best arm; at 1% margin only 57.9%. User's 80% top1 target is structurally feasible but requires distinguishing arms within 0.1–1% cost.
- **Arm collapse diagnosis** on S5 best model (test, 18 problems): severe — **TSP 8/10 arms never picked, CVRP 5/9 never picked, 6 MVRP problems have 100% mass on RELD_MTL**. This is a first-order problem: the selector ≈ "pick SBS by default".
- **Current ensemble top1** = 0.5162 (S5 best); ceiling at 0.1% margin ≈ 0.946; headroom ≈ 43 pts.
- **Worst in-distribution problems** by top1 gap vs ceiling: CVRP (Δ+0.51), VRPL (Δ+0.43), OVRPBTW (Δ+0.51), VRPTW (Δ+0.44).

### Implemented changes (phase S0–S2 of Round 11)
- `data.py` — 8-fold D4 coord aug (`_augment_xy`, `UnifiedProblemDataset(..., aug_8fold=True)`); applied on train split for coord problems (skipped for ATSP — no 2D geometry).
- `model.py` — added `plackett_luce_loss()` (NSS loss.py:7-19 port) + `diversity_entropy_penalty()`.
- `train.py` — new flags: `--aug-8fold`, `--plackett-luce`, `--pl-weight`, `--pl-only`, `--pl-topk`, `--diversity-weight`, `--amp`.

### Phase S3–S4 experiments (launched)
Multi-configuration sweep on GPU0, all with FiLM + pd=0.05 + spec0 distill + 8-fold aug:
- R11_mainv2: PL(0.3)+CE+distill+aug+FiLM, d=128 depth=4, 10 epochs — ep0 top1=0.5095 vs_sbs=+0.14% (process killed mid-ep1, no saved ckpt past ep0).
- R11_bigger: PL+CE, d=192 depth=6 — **diverged** at ep1 (top1=0.48, vs_sbs=+57.6%) — killed.
- R11_plonly: PL-only, d=128 depth=4 — weak (ep1 top1=0.47, vs_sbs=+0.85%) — killed.
- R11_plsmall: PL(0.1)+CE+distill+aug+FiLM — slow convergence — killed.
- **R11_final seed{0,1}** (current): PL(0.3)+CE+distill+aug+FiLM, d=128 **depth=6**, 8 epochs — running.

### Open questions for Round 11 review
1. Did aug + PL actually improve top1 beyond R4 baseline? (need ep8 result).
2. Did arm distribution de-collapse? (need per-epoch arm histogram).
3. Is the 80% top1 target realistic in this constraint budget (≤20 GPU-h)? Likely NO without a much larger encoder overhaul (hierarchical GAT + ReZero + MBM three-way attention for ATSP).

## Round 11 — Score: 5.8/10 (nightmare codex exec review, 2026-04-19)

- **Verdict**: not ready. Central objective (80% exact top1) failed.
- **Main finding**: R11 code changes (aug/PL/diversity) are real, but the new recipe does not beat the S5 baseline. R11 ensemble ties S5 single on exact top1 (0.5179 vs 0.51794) and is worse on `vs_sbs` (−0.009% vs −0.015%).

### Previous suspicions addressed?
- **R10 S5 ungated CI concern** (not R11's focus) → unaddressed; R11 did not produce a better single number anyway.
- **R10 ATSP ceiling concern** → unaddressed; R11 did not touch ATSP matrix encoder (MBM/MatNet would have been needed).
- **R9 arm-collapse concern** (carried from R9 → R10 → R11) → **NOT genuinely fixed**. R11 ensemble still has 52 zero-pick arms (S5 single: 48; S5 ensemble: 56). Codex flagged the author's "deeply tied to label distribution" framing as only half-true.

### New suspicions / unresolved (codex's own memory update)
- **`--amp` claim is loose**: parser-only flag, no `autocast`/`GradScaler` in training loop. Any future AMP claim should be verified by grep, not taken on assurance.
- **Arm-collapse framing was spun**: 37/52 zero-pick arms in R11 ensemble have non-zero oracle wins. That is genuine selector collapse, not benign SBS dominance.
- **Numeric counts drift under comparator ambiguity**: "≈54 zero-pick arms" was codex-verified as wrong: 48 for S5 single, 56 for S5 ensemble. Require explicit comparator specification going forward.
- **"6 MVRPs 100% RELD_MTL"** was wrong for the file author cited: S5 single has 4 (VRPBL, VRPBTW, VRPLTW, VRPBLTW).

### Verified claims (codex, this round)
- 8-fold D4 aug: data.py:24,76,161 — ✓
- PL loss + diversity penalty: model.py:264,285; train.py:308,316 — ✓
- CLI flags: train.py:159 — ✓
- Label ceiling 0.9461 @ 0.1%, 0.5793 @ 1% — ✓ (runs/diag_r11/label_ceiling.{md,json})
- Fuzzy top1@1% = 0.7104 — ✓ (reproduced from S5 ensemble summary)
- R11_final_seed0_test / seed1_test / R11_ensemble_test numbers match summary.json — ✓

### Recurring patterns (carry forward to R12)
1. **"Implemented all user-requested changes" overstatements**: wire-up should be grep-verified before claiming completion.
2. **Numeric claims need file-specific citations**: author tends to cite round-estimate numbers that don't match the specific JSON file. Require format "X = Y per FILE_PATH".
3. **Arm-collapse rationalizations**: author tends to blame label distribution rather than fix the model; the 37/52 genuine-collapse number should be the benchmark, not the 15/52 "benign-SBS-dominance" subset.

### Memory status going into R12
- **Primary concern**: 80% exact top1 is not reachable on ≤20 GPU-h with current architecture. Either reframe metric (codex's direct recommendation) or commit to a real encoder redesign.
- **Codex's Round-12 warning**: "Round-12 should not spend more GPU on PL/entropy tinkering unless the metric is reframed or the encoder is redesigned."
- **Score trajectory risk**: dropping to 5.8 from 8.1 signals that chasing a misframed headline metric burned authorial credibility. A Round 12 that does encoder redesign without closing the 80% target will be scored on whether the reframed metric bundle moved; PL/diversity tuning alone will not recover the score.

## Round 12 — Score: 6.0/10

- **Previous suspicions addressed?**
  - R11 `--amp` parser-only: FIXED (R12 wires autocast + GradScaler in train.py:319-367, codex spot-checked)
  - R11 arm collapse: Persists. 52 zero-pick arms with 37/52 having non-zero oracle wins in S5 ensemble — unchanged by R12 arch overhaul.
  - R11 "fuzzy top1 @1% ≈ 0.71 as a bundle metric": REFRAMED now explicit: 0.7110 for R18 (0.0004 behind S5's 0.7114).

- **New suspicions**
  - Author's new-arch variants R14/R15 hit val top1 = 0.5179 (matching S5 val) but test regresses to 0.5094-0.5096 — classic val overfit pattern. Val strict top1 should NOT drive checkpoint selection.
  - Author's reranker is structurally too weak: it only has shared-with-base features, so it cannot add new discriminative signal on near-ties. True pairwise with raw node tokens is the missing piece.
  - Ensembles consistently hurt strict top1 (S5×3: 0.5162; R12×2: 0.5144; R18×3: 0.5143). Pattern: soft averaging muddles argmax on 0.1%-margin ties. Argmin-of-mean-cost beats softmax-of-mean-logit for this task.

- **Unresolved (carried forward + new)**
  - 80% strict top1 target: INFEASIBLE. Label ceiling @ 1% margin = 0.5793, all models plateau at ~0.52 test. Unless sub-percent cost features are introduced, this is the wall.
  - Arm coverage for MVRP: 37/52 zero-pick arms still have oracle wins — pairwise classifier with raw node tokens is the best chance to recover them.
  - R18's +0.0011 gain over S5 is nominally real (single-seed single-run) but needs paired bootstrap CIs to claim it's significant.

- **Patterns to track next round**
  - Watch for val-test instability in any new checkpoint (R14/R15/R19 all failed this).
  - If pairwise classifier also shows val-only gains, treat as overfit and reject.
  - Track oracle-in-top2 and oracle-in-top3 rates by problem family (macro top2=0.8735, macro top3=0.9843 — retrieval is strong).


## Round 13 — Score: 4.0/10 (Codex MCP, threadId 019da79e-3f6a-7212-9a33-9bc9f789bff1)

- **Verdict**: "STOP AND WRITE". The author is at the wall, not near the wall.
- **Decisive evidence**: R21 pairwise comparator (raw node tokens + both arm embeddings, codex's own R12 recommendation) **fails on test** — val gain of +0.0024 at α=0.5 regresses to -0.0049 on test. This is the experiment that "should have broken the bottleneck if the bottleneck were merely feature access"; its failure reframes the remaining errors as "not robustly learnable from the current signal", not "awaiting the right architecture".
- **Bootstrap CI evidence**: macro top1 delta R18 − S5 = +0.0012, 95% CI [−0.0025, +0.0049]; macro mean-cost delta = −0.0027%, 95% CI [−0.0117%, +0.0065%]. Both include zero; R18 statistically equivalent to S5.

### Previous suspicions addressed?
- **R12 "true pairwise with raw tokens" recommendation**: EXECUTED faithfully (codex verified pairwise_train.py / pairwise_eval.py). Result: FAILED. Val-test overfit exactly as codex warned in R12 memory ("if pairwise classifier also shows val-only gains, treat as overfit and reject").
- **R12 "paired bootstrap CIs for R18-vs-S5"**: EXECUTED. Confirmed R18 ≈ S5.
- **R12 "argmin-of-mean-cost vs softmax-of-mean-logit"**: NOT YET TESTED (pending R13 ensemble eval).

### New suspicions / codex-sanctioned options
- **Option (A) — STOP**: codex's primary recommendation. Accept 0.519 as ceiling. Reframe around (vs_sbs_pct, fuzzy@1%) with ceiling analysis.
- **Option (D) — Preregistered falsification**: if one more run, oversample CVRP/VRPL/VRPTW/OVRPTW 2x from R18_alltail, 3 epochs, LR=1e-5. Success bar: +0.008 val macro top1. If not hit, the wall claim is confirmed.
- **Codex labelled (D) as "cheap falsification run, not a genuine hope run".**

### Quotable codex R13 framing (for paper narrative)
> "unified selector across 18 routing families is competitive with SBS on cost and near-optimality; strict exact top1 saturates far below 80% because much of the remaining mass lives in an underdetermined low-margin regime; increasingly expressive tie-breakers improve validation but do not transfer, which is itself evidence that the hard cases are not robustly learnable from the current signal."

### Patterns carried to R14
- **Val-test instability is THE pattern of this project**, confirmed across R14/R15/R19/R21. Any new val gain < +0.01 should be treated as noise. R22 (Option D) preregisters +0.008 as the success bar specifically to enforce this.
- **Ensembles blur tie-break signal**: R18 ×3 = 0.5143 worse than single R18 best. Do NOT claim ensemble wins without per-instance argmin-of-mean-cost aggregation.
- **80% strict top1 is structurally infeasible** on current data regime. Label ceiling @ 1% margin = 0.5793 is the hard wall; 80% target pre-dated our ceiling analysis.

### Memory status going into R14
- If R22 (Option D) fails the +0.008 val bar → score stays 4/10 or drops. Time to write.
- If R22 somehow hits +0.008 → remains suspicious of val-test transfer; demand test-set bootstrap CI before taking the gain seriously.
- Claude's next-best independent experiment is argmin-of-mean-cost ensemble of {S5, R18, R22} (if R22 finishes). If test top1 > 0.525 with 95% CI excluding 0, THAT would be a novel claim. Otherwise, pivot to write.


## Round 14 — Score: 5.5/10

- **Verdict**: Wall empirically confirmed. No more experiments required unless new signal. Paper ready to write with 2 additional analyses on existing runs.
- **Codex R14 requests (analyses only, not experiments)**:
  1. (MUST-HAVE) Margin-bucket decomposition: test split by oracle-gap buckets (<0.1%, 0.1-0.5%, 0.5-1%, >1%); show top1/top2/top3/fuzzy@1%/regret per bucket for all methods. Connects to label_ceiling.md.
  2. (should-have) Risk-coverage for R18 and R21: show pairwise/rerank gains live in low-confidence regions on val but do not transfer on test.

### Both R14 analyses completed

- **Margin-bucket** (`runs/margin_bucket_test/`): bucket mass = 53.7% / 8.1% / 9.0% / 29.2%. In the "easy" <0.1% bucket (53.7% of mass), top1 ≈ 0.81 across all methods. In non-trivial buckets (46.3% of mass), top1 drops to ~0.17 uniformly across S5/R18/R22 — THE WALL made visible. Fuzzy@1% stays at 0.97 in 0.1-1% buckets but collapses to 0.18 in the >1% bucket (29.2% of mass).
- **Risk-coverage** (`runs/rc_R18_R21/`): R18+R21 blend reduces mean regret on val (1.001% → 0.991%) but INCREASES it on test (1.007% → 1.020%). Val-test overfit pattern visually clear.

### Previous suspicions addressed (this round)

- **R12 "argmin-of-mean-cost vs softmax-of-mean-logit"**: COMPLETED via ensemble_analysis.py. Reviewer-memory claim REFUTED — `min_cost_proxy` = `mean_logit` = 0.5144. None of 8 aggregation modes beats single best R18 (0.5193).
- **R12 "bootstrap CIs for R18-vs-S5"**: CONFIRMED not significant. Now also R22 vs R18 and R22 vs S5 not significant.

### New codex-tracked patterns

- **The wall is a STRUCTURAL property of the data/signal**, not a failure of architecture or training. Codex's own words: "The only experiment that would genuinely challenge the wall now would require new signal... That's a new project, not a missing ablation."
- **Reframed headline is mandatory**: "A unified selector can match a strong baseline across 18 routing families, but exact oracle identification is structurally limited by low-margin ambiguity." NOT "71% within 1%" (which is matched by SBS baseline).
- **Paper angle**: ceiling/ambiguity analysis across 18 routing families, NOT "better selector".

### Memory status going into R15

- Both R14 must/should analyses are completed. Expected R15 score movement: +0.5-1.0 if analyses are well-documented.
- If R15 ≥ 6, move to termination phase (method description, /result-to-claim, status=completed).
- R15 must-watch: codex will verify bucket decomposition numbers against per_problem JSONs. Any claim-vs-file mismatch will undo the score gain.


## Round 15 — Score: 6.4/10 — "paper-ready"

- **Verdict**: Both required R14 analyses verified. Score crosses default 6.0 threshold. Explicit "paper-ready" for analysis/diagnosis paper (not "positive method beats S5" paper).
- **Codex's stronger headline**: "A unified selector matches a strong baseline across 18 routing families, and its exact-top1 errors split into two regimes: benign low-margin ambiguity and a smaller regret-dominant >1% failure subset."
- **Optional ask**: show each bucket's contribution to total regret. Codex predicted >1% bucket (29.2% mass) dominates.

## Round 16 — Score: 6.8/10 — "paper-ready" confirmed, TERMINATE LOOP

- **Verdict**: 6.8/10, "paper-ready" for analysis/diagnosis framing. Codex explicitly said: "I would terminate the autonomous review loop and transition to writing."
- **Verified numbers**: bucket-regret table (>1% bucket = 76-78% of total regret, confirming codex's prediction) and arm-distribution full JSON.
- **IMPORTANT correction codex flagged**: my claim "~8% of test mass affected by genuine mis-picks" was WRONG. Correct number is 17.2% (S5) / 20.0% (R18) / 20.3% (R22) — oracle wins on zero-pick arms. Fixed in AUTO_REVIEW.md Round 15 section.
- **Top-2 oracle concentration**: 67.4% (CVRP) to 97.6% (ATSP) per problem, mean ≈ 90%.
- **Framing for arm-distribution**: "Most apparent collapse reflects true oracle skew, but a nontrivial residual coverage gap remains." Not pure-positive or pure-negative.

### Blocking 7.0+

Not analysis; it's contribution ceiling. Codex's exact words:
- "there is no significant method improvement over the strong baseline"
- "the contribution is primarily negative/diagnostic rather than algorithmically constructive"
- "the empirical scope is deep on this benchmark family, but still narrow relative to what usually pushes a diagnosis paper into obvious accept territory"

Codex's scale:
- 6.5-7.0: strong borderline / respectable analysis paper
- 7.5+: clean accept territory for ICLR/NeurIPS/ICML
- 6.8 current = strong borderline, clearly paper-ready.

### Termination decision

- User's POSITIVE_THRESHOLD is 10 (unreachable); MAX_ROUNDS is 30 (budget remaining).
- Codex-provided verdict "paper-ready" explicitly meets the skill's default verdict-based stop condition.
- Codex explicitly recommended termination twice (R14 first, R16 reconfirmed).
- To push from 6.8 → 7.5+ requires NEW SIGNAL sources (solver-internal traces, runtime features, different supervision) — outside scope per codex and user's constraint.
- Further iteration under current setup would produce diminishing returns.
**Final call**: TERMINATE. Proceed to Method Description + /result-to-claim.

## Final summary — score trajectory across 16 rounds

R1 7.2, R2 6.8, R3 7.9, R4 8.3, R5 8.7, R6 8.8, R7 8.9, R8 9.0, R9 6.4 (nightmare introduced), R10 8.1, R11 5.8 (new goal: 80% top1), R12 6.0, R13 4.0 (wall hypothesis), R14 5.5 (wall confirmed), R15 6.4 (analyses added), R16 6.8 (paper-ready).

Final state: paper-ready analysis/diagnosis paper at 6.8/10. Main finding: unified selector across 18 routing families matches S5 baseline (statistically equivalent via bootstrap CI); exact-top1 errors decompose into benign 0.1-1% ambiguity (low-margin, high fuzzy@1%) and regret-dominant >1% bucket (29.2% mass, 77% of total regret). Attempted improvements (deep encoder overhaul, pairwise reranker, hard-family oversampling, ensembles, SWA) all fail to break the wall, constituting strong empirical falsification of "architecture expressivity is the bottleneck" within the current supervision signal.

---

## Fresh Loop (user re-invocation, 2026-04-20)

### Round 1 — Score 7.2/10 — Oracle-pro-browser

**Your suspicions from prior loop**: Wall structural. Small effects = DOF leakage. Need pre-registration.

**Prior suspicions addressed?**: Partially. Gated retrieval gave +0.0056 nominal but CI barely includes 0. Zero-pick subset rescue +12.7% is clean mechanistic contribution.

**New suspicions**:
- +0.0056 may be cherry-picked (test-grid sweep). Need val-lock pre-registration.
- R28 learned gate collapsed (α→0.05 everywhere): continuous α prediction is ill-conditioned.
- R30 per-problem per-HP: val +0.013, test +0.002 (classic overfit).

**Patterns**: All retrieval-family methods give +0.001-0.005 nominal, CI always includes 0. Effect size ≤ bootstrap noise.

### Round 2 — Score 7.8/10 — Oracle-pro-browser

**Your suspicions from R1 addressed?**:
- Validation-locked pre-registration protocol: YES (R36, R37)
- Binary learned gate: YES (R33) — revealed wr/rw ≈ 1.01 coin-flip (features don't transfer)
- Prospective blind-spot: YES (R34) — val blind-spots DON'T transfer to test
- Dual retrieval agreement: YES (R35) — no improvement over single prior

**Damning line**: "R33 wr=2045, rw=2022 on test — coin-flip discrimination. Available test-time observables do not contain enough transferable signal for 'will retrieval help this instance?'"

**Scientific execution up (9/10), method-success down (5.5/10)**: Better diagnostics, worse improvement story.

**Final recommendation**: ONE more experiment — train-time fusion head (R38). Pre-register pass/fail: Δtop1 ≥ +0.004, wr/rw ≥ 1.08, zp retained while outside harm halved.

**Patterns**: Val gains +0.003-0.008 → test gains +0.000-0.003 consistently. Val-test gap is the fundamental overfit signature.

### Round 3 — Score 8.5/10 **TERMINAL** — Oracle-pro-browser

**Your R2 final experiment**: R38 fusion head missed ALL pre-registered thresholds.
- Test Δtop1 = +0.0029 (threshold +0.004): MISSED
- wr/rw ratio 1.036 (threshold 1.08): MISSED
- Global method direction CONFIRMED EXHAUSTED per your R2 prediction.

**Key new element**: R39 Gross Rescue/Harm decomposition — your suggested CENTRAL FIGURE. Shows:
- ZP mass: rescue 12-17%, harm 0 (by definition)
- Non-ZP mass: harm 9-14%, rescue 6-9%, net HARM
- Total net = bounded by bootstrap noise ±108 instances

**Your verdict**: STOP loop, write diagnostic paper.

**Your final abstract** (to be used verbatim): [see AUTO_REVIEW.md Fresh Loop Round 3 final section]

**Score breakdown**:
- Experimental discipline: 9/10- Evidence for global improvement: 2/10
- Evidence for support-collapse diagnosis: **9/10**
- Evidence for retrieval-local-rescue mechanism: 8.5/10
- Evidence for stopping: 9.5/10
- **Paper readiness: 8.5/10**

**Remaining 1.5 points**: "definition cleanup and presentation, not more experiments."

**Fresh loop score trajectory**: R1 7.2 → R2 7.8 → R3 **8.5**. Net +1.3 within loop, +1.7 from prior loop's 6.8.

**Paper repositioned**: diagnostic paper on support collapse mechanism, NOT method-improvement paper. Clean three-way narrative: (1) wall + regret buckets, (2) support collapse + retrieval exposes it, (3) gross rescue/harm decomposition explains why global repair fails.

