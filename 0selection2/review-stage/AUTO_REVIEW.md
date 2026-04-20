# Auto Review Loop — Solver Selector

- **Topic**: Unified supervised neural solver selector across 18 routing problems (TSP / CVRP / ATSP / 15 MVRP)
- **Difficulty**: hard (Reviewer Memory + Debate Protocol, oracle-pro browser backend)
- **MAX_ROUNDS**: 10 · **POSITIVE_THRESHOLD**: 10/10 · skip >10 GPU-h · WANDB: on
- **Reviewer**: oracle-pro (GPT-5.4 Pro, browser) — per project convention in `~/.claude/CLAUDE.md`
- **Started**: 2026-04-18 (continuation of prior session's Round 1 review)

## Round 1 — Score: 7.2/10  (from prior session; reconstructed from conversation log)

### Assessment (Summary)
- **Score**: 7.2/10
- **Verdict**: almost — "basic approach is sound, 3 concrete fixes will close the gap"
- **Key criticisms**:
  1. Selector sometimes under-performs SBS → needs a confidence gate
  2. Regret-soft CE alone is not cost-aware enough → needs combined loss (CE + regret + switch-hinge)
  3. Head MLP can't capture all per-problem-per-solver priors → needs `problem_solver_bias` + targeted oversampling
- **Baseline state at Round 1 end**: R1-seed0 (2 epochs, soft-label CE only) → `macro_vs_sbs_pct = +0.156%`, beats SBS on 8/18 problems, CVRP badly behind (+0.94%).

### Actions taken (before Round 2)
- Implemented `gate.py` (SBS-gate calibration): confirmed +0.160% → +0.014% on R1 ckpt.
- Implemented `combined_cost_loss` in `model.py`: CE (regret-soft) + expected-regret + switch-hinge.
- Added `problem_solver_bias: nn.Parameter[P × M]` to `UnifiedSelector`.
- Added `--oversample` flag with `PROBLEM_WEIGHT` (CVRP×4, VRPB/OVRPB×3, OVRP×2, {VRPBL,VRPBTW,OVRPBL,OVRPBTW,VRPBLTW,OVRPBLTW}×1.5) in `train.py`.
- Added `--loss {soft,combined}` and `--no-problem-solver-bias` flags.

### Status
- Continuing to Round 2 after launching new runs with the three fixes.

## Round 2 (2026-04-18T23:55:10+0800)

### Assessment (Summary)
- **Score**: 6.4/10
- **Verdict**: not ready
- **Recommended path**: promote gated inference to the main method (`A`), not raw-selector-first (`B`) and not a full pivot (`C`) yet.
- **Scope of this round**: verify the reviewer-prescribed fixes, correct broken analysis paths, and decide whether the next loop should center on gating, raw-training changes, or both.
- **Important note**: the oracle response below was generated from a slightly stale Round 2 snapshot before the corrected `meta-only = +7.861%` analysis and before the gated test analysis was materialized. Those later fixes strengthen the same conclusion rather than overturning it.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

* **Score: 6.4/10**
* **Verdict: NOT READY**

## Top weaknesses ranked

### 1. The method still does not beat SBS as a selector; it mostly survives by abstaining to SBS

The best ungated model is still **R1_seed0**, with `macro_vs_sbs=+0.156%` and **negative VBS-gap closure**. That is not a win; it is a controlled failure. The gate improves validation macro from about `+0.160%` to `+0.014%`, which is valuable, but this is basically “do no harm relative to SBS,” not yet “learned selector beats SBS.” The strongest current result is a safety wrapper, not the raw supervised method.

Severity: **critical**.

### 2. CVRP and B-style VRP variants remain the empirical credibility sink

CVRP is not a small blemish: `+0.94%` vs SBS is large relative to the oracle-SBS gap of only about `0.85%`, so the selector is worse than SBS by more than the entire available oracle improvement on that problem. VRPB and OVRPB are also bad: `+0.67%` and `+0.89%`. A reviewer will immediately ask why the unified selector fails exactly where selection should matter.

Severity: **critical**.

### 3. The prescribed training fixes are not just unhelpful; they look actively damaging

The combined-loss / bias / oversampling direction is currently a dead end. `R3_combined_bias_over` gives `+4.89%`; `R3_soft_bias_over` gives `+8.37%`; both are much worse than the simple R1. The failures are too large to hand-wave as “needs more epochs.” The catastrophic TSP / ATSP degradation suggests either a bug, badly mis-scaled loss, or destructive sampling/objective interaction.

Severity: **critical**.

### 4. Longer training hurts, so the training protocol is not stable

`R1_seed0_ext` and `R1_nofact` worsening with more training means the 2-epoch checkpoint may be an early-stopping accident. For top-venue empirical readiness, this is a serious problem: the model is not converging toward better selection quality. It is drifting away from SBS safety.

Severity: **high**.

### 5. Current gate calibration is promising but too ad hoc to be the final claim

The gate is the only robust-looking improvement, but it currently collapses several weak problems entirely to SBS. That is acceptable only if framed honestly as **selective switching over SBS**, not as a free-standing neural selector. The current gate also needs held-out calibration/test separation, bootstrap confidence intervals, and per-problem switch-benefit diagnostics.

Severity: **high**.

### 6. Evidence is still one-seed, validation-heavy, and lacks statistical protection

A top reviewer will not accept “+0.014% vs SBS on validation” as a meaningful win. It is too close to zero, probably not statistically distinguishable, and there is no evidence yet that the thresholded policy generalizes to test.

Severity: **high**.

---

## Minimum fixes ranked

### 1. Make gated inference the official method and evaluate it properly

Type: **inference-time gating / calibration changes**

Stop presenting the raw selector as the main deployed method. The current best method is:

> SBS by default; switch only when the learned selector is confidently better.

Minimum next fix under 10 GPU-hours:

Train nothing. Reuse `R1_seed0` and do a rigorous calibration sweep.

Use a split like:

* train: fit neural selector only;
* calibration subset: fit per-problem gate thresholds;
* validation/test: report final frozen gated policy.

For each problem, report:

* ungated vs SBS;
* gated vs SBS;
* switch rate;
* switch precision;
* average benefit conditional on switch;
* average harm conditional on wrong switch;
* bootstrap CI for `vs_sbs`.

This is the highest-value next step.

### 2. Add per-problem fallback routing with a conservative rule

Type: **per-problem fallback routing**

Do not force every problem to use the selector. For problems where calibrated switching fails to show positive benefit on calibration, hard-route to SBS.

Minimum rule:

> For each problem, enable switching only if calibration-set mean cost improves over SBS by at least one bootstrap standard error; otherwise use SBS for that problem.

This would likely hard-fallback CVRP, ATSP, OVRP, OVRPB, VRPBTW, and maybe several near-zero variants, while preserving useful switching on TSP and the few MVRP variants with decent precision.

This is not aesthetically exciting, but it is empirically sane.

### 3. Replace margin-only gate with cost-aware switch calibration

Type: **inference-time gating / calibration changes**

The current gate uses log-prob margin against SBS. That is simple, but not necessarily aligned with cost. The minimum worthwhile improvement is not another neural training run; it is a better threshold statistic.

Try three cheap gate scores on existing checkpoints:

1. `logit_best - logit_SBS`
2. `prob_best - prob_SBS`
3. entropy-adjusted margin: `(logit_best - logit_SBS) / entropy`

Then select the best **per problem** by calibration-set mean cost, with bootstrap protection. This costs almost no GPU.

Do **not** train a separate gate classifier yet unless these simple scores fail; the data is only 1k val examples per problem and overfitting is likely.

### 4. Diagnose the R3 failure before launching any more objective variants

Type: **better diagnostics/analysis rather than more training**

The combined-loss / oversampling path should be paused. The losses are not mildly worse; they are explosively worse on TSP and ATSP. Before spending GPU, run CPU/GPU-light checks:

* Compare initial logits and selected arms for R1 vs R3 at step 0, 100, 500, 1000.
* Log per-problem gradient norms under oversampling.
* Log per-loss-term magnitudes for CE, expected regret, and hinge.
* Check whether the problem-solver bias rapidly becomes a dominant prior.
* Check whether oversampling changes the effective epoch definition enough to undertrain TSP/ATSP.
* Verify combined-loss `tau=0.07`, `r_cap=0.20`, and hinge margin are not flattening useful cost differences.

Minimum conclusion expected: either “bug found” or “objective is rejected.” Do not keep tuning it blindly.

### 5. Run a seed-0/1/2 replication only for the winning simple recipe

Type: **scheduler/training protocol changes**

Under 10 GPU-hours, do not run a large ablation matrix. Run only the recipe that currently works:

* soft CE;
* no oversampling;
* no combined loss;
* probably no problem-solver bias unless separately proven harmless;
* 2-epoch or early-stopped schedule;
* gated inference.

Run 3 seeds if cheap. Report mean and CI for gated macro `vs_sbs`. If the result is `0.00% ± 0.05%`, the honest claim is “safe matching of SBS with occasional improvements,” not “beats SBS.”

### 6. Add an early-stopping and checkpoint-selection analysis

Type: **scheduler/training protocol changes**

Since longer training hurts, the minimum fix is to make early stopping explicit and defensible.

Use calibration/validation macro cost, not training loss, to choose checkpoints. Show curves for:

* macro `vs_sbs`;
* per-problem `vs_sbs`;
* switch precision after gating;
* arm distribution.

The paper cannot rely on “epoch 1 happened to be best” without documenting that selection criterion.

### 7. Keep the metadata-only baseline, but stop overclaiming from it

Type: **better diagnostics/analysis rather than more training**

The meta-only baseline is useful because it proves instance features matter. But it does not prove the selector is good. It only rules out the most embarrassing leakage/prior-only explanation. Keep it as a sanity check, not as a central empirical victory.

---

## Recommended path: **A — make gated inference the main method**

The path should center on **A**, not B. The raw selector is not currently strong enough, and the attempted objective/oversampling fixes made it much worse. The gate is the only intervention that reliably improves final selection quality. The right framing is:

> a unified neural selector with conservative SBS fallback, optimized for safe cost-aware deployment across heterogeneous routing problems.

I would not choose C yet, because there is still a coherent method here: gated selective switching. But the framing must become much more conservative. The main result should not be “neural selector beats SBS everywhere.” It should be “the learned model identifies a subset of instances/problems where switching away from SBS is safe, while defaulting to SBS otherwise.”

---

## One-paragraph brutal summary

Right now this is **not top-venue ready**. The core raw selector still loses to SBS on macro cost, closes negative VBS gap, and fails badly on CVRP/VRPB/OVRPB. The combined-loss, bias, and oversampling direction should be treated as a dead end unless a concrete bug is found; the numbers are too bad to justify more blind tuning. The only empirically credible path is to promote the SBS gate from a patch to the actual method, add conservative per-problem fallback, calibrate it on a held-out split, and report switch precision/benefit with bootstrap confidence intervals. If that produces a stable non-positive `vs_sbs` on test over multiple seeds, the work becomes a defensible “safe selective solver selector.” Without that, it is an interesting pilot that mostly rediscovers SBS.

</details>

### Actions Taken
- Fixed a real evaluation bug in [train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/train.py) and [analyze.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/analyze.py): `meta-only` now zeroes instance features in both training and evaluation, instead of training-only.
- Fixed gate-analysis compatibility in [analyze.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/analyze.py): it now reads `gate_eval.json`'s top-level `gammas` block correctly.
- Fixed oversampling weight handling in [train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/train.py): fractional weights are no longer silently truncated by `int(...)`.
- Added an experimental `--schedule interleaved` path in [train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/train.py) to reduce cross-problem forgetting, but kept the default schedule as `sequential` because the interleaved path is not fully validated yet.
- Re-ran analysis to materialize the metrics/figures requested by `观测指标.md` for the current best checkpoint and its gated variant.

### Results
- **Ungated R1_seed0 (corrected full analysis)**:
  - Val: `macro_vs_sbs_pct = +0.159%`, `macro_top1 = 0.5104`, `macro_top2 = 0.8631`, `macro_top3 = 0.9728`.
  - Test: `macro_vs_sbs_pct = +0.228%`, `macro_top1 = 0.5051`, `macro_top2 = 0.8587`, `macro_top3 = 0.9717`.
- **Gated R1_seed0 (analysis now actually applies gamma calibration)**:
  - Val: `macro_vs_sbs_pct = +0.0141%`, `macro_top1 = 0.5129`, `macro_top2 = 0.8737`, `macro_top3 = 0.9837`.
  - Test: `macro_vs_sbs_pct = +0.0516%`, `macro_top1 = 0.5133`, `macro_top2 = 0.8721`, `macro_top3 = 0.9833`.
- **Meta-only baseline after bug fix**:
  - Val: `macro_vs_sbs_pct = +7.861%`, `macro_top1 = 0.4150`, `macro_top2 = 0.7624`, `macro_top3 = 0.8801`.
  - This remains a strong anti-leakage defense, but it is much less catastrophic than the previously mis-measured `+75.23%`.
- **Operational finding**:
  - A parallel attempt to launch two interleaved training jobs caused a container OOM kill. This is an infrastructure / memory-pressure issue, not a validated negative result on selector quality.

### Status
- Round 2 is complete and points to **continuing with gated SBS-fallback as the main method**.
- Current empirical story is strongest around the **gated R1 baseline**, not the later combined-loss / oversampling runs.
- Round 3 should prioritize cheap calibration / fallback / CI work before any more objective tuning.

## Round 3 (2026-04-18T23:59:00+0800)

### Assessment (Summary)
- **External reviewer**: pending. Oracle browser session `selector-round3-review-browser2` is running against the new calibrated/fallback test result.
- **Goal of this round**: implement the oracle-requested cheap gating fixes before spending any more GPU on training.

### Reviewer Raw Response

<details>
<summary>Pending external oracle response</summary>

Browser session `selector-round3-review-browser2` has not produced a raw response yet in this turn.

</details>

### Actions Taken
- Upgraded [gate.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/gate.py) from a single margin threshold into a configurable calibration/evaluation tool.
- Added per-problem score-family search across `margin`, `prob_gap`, and `margin_over_entropy`.
- Added conservative fallback logic: if calibration-set improvement does not clear one bootstrap standard error, disable switching and route that problem to SBS.
- Added per-problem diagnostics required by the reviewer: `switch_rate`, `switch_precision`, `switch_recall`, `avg_benefit_if_correct_switch`, `avg_harm_if_wrong_switch`, and bootstrap CIs.
- Added macro bootstrap CI for gated `vs_sbs`.

### Results
- Using the existing `R1_seed0` checkpoint only:
  - calibration split: 50% of `val` per problem;
  - final evaluation split: `test`;
  - per-problem score family chosen automatically from `{margin, prob_gap, margin_over_entropy}`.
- **Ungated test**: `macro_vs_sbs = +0.228%`.
- **New calibrated/fallback gated test**: `macro_vs_sbs = +0.011%`.
- **Bootstrap CI**: gated test `macro_vs_sbs` 95% CI = `[-0.004%, +0.026%]`.
- **Representative routing decisions**:
  - `CVRP`: fallback to SBS, test `vs_sbs = -0.000%`.
  - `ATSP`: fallback to SBS, test `vs_sbs = +0.000%`.
  - `OVRPB`: fallback to SBS, test `vs_sbs = -0.000%`.
  - `TSP`: switching enabled, test `vs_sbs = -0.062%`.
  - `OVRPBL`: switching enabled, test `vs_sbs = -0.058%`.
  - `OVRPLTW`: switching enabled, test `vs_sbs = -0.017%`.

### Status
- Round 3 evidence now supports the reviewer-requested **safe selective selector** framing much more directly than Round 2.
- Awaiting oracle reassessment on whether this is enough to move from `NOT READY` to `ALMOST`.

## Round 4 (2026-04-19T00:53:21+0800)

### Assessment (Summary)
- **Score**: 7.1/10
- **Verdict**: ALMOST
- **Key upgrade over Round 3**: the gated SBS-fallback story is now backed by 3 seeds instead of 1, and the reviewer explicitly agrees that this should be the main method rather than the raw selector.
- **Stopping rationale**: this satisfies the loop stop condition (`score >= 6` and verdict contains `ALMOST`), so the loop terminates here instead of spending more compute on low-yield raw-selector tuning.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

1. **Score: 7.1/10**

The project has moved from **“not ready because the method is unclear”** to **“modest but coherent, still not fully submission-ready.”** The new evidence directly addresses the prior review request: multi-seed simple recipe, held-out calibration, conservative fallback, and bootstrap reporting. The key aggregate is now: raw selector `+0.105%` vs SBS over 3 seeds, while gated fallback is only `+0.005%` vs SBS, with all gated seeds essentially at SBS parity. 

2. **Verdict: ALMOST**

Not **READY** yet, because the main result is still **near-zero parity with SBS**, not a convincing improvement over SBS. But it is no longer **NOT READY**. The story is now internally consistent:

**SBS is the default. The learned model is allowed to switch only under conservative calibration. The result is a safe selective selector that nearly eliminates raw-selector harm.**

That is a real, coherent paper direction. It is just a modest one.

3. **Remaining critical weaknesses ranked**

**1. The main effect size is essentially zero.**
The gated policy is not beating SBS in a practically meaningful way: mean `macro_vs_sbs = +0.005%`, with seed results `+0.011%`, `+0.004%`, and `+0.000%`. If positive means higher cost than SBS, the method is still infinitesimally worse on average, even if within noise. This cannot be sold as “learned selector beats SBS.”

**2. The raw selector is not a publishable main method.**
The raw selector is consistently slightly worse than SBS: `+0.227%`, `+0.071%`, `+0.015%`. The shrinkage across seeds is actually damaging to the “raw selector improves over SBS” narrative. Raw selection should be demoted to an ablation/failure mode, not presented as the central contribution.

**3. The gate is still too close to “mostly SBS with occasional tiny switches.”**
The gated policy matches 18/18 problems in the seed2 full analysis, but beats only 2/18, with macro gain effectively zero. That is safe, but reviewers will ask: “What did the learned selector add beyond SBS?” You need switch-rate, switch-precision, conditional benefit, and conditional harm to prove that there is a nontrivial selective-switching behavior.

**4. Some enabled switching remains low-confidence / low-precision.**
The aggregate per-problem table shows very low switch precision on several enabled problems, for example OVRPBL around `0.075`, OVRPL around `0.162`, OVRPLTW around `0.163`, and OVRPBLTW around `0.185`. Even when macro harm is tiny, this weakens the claim that the gate has learned a reliable “when to switch” rule. 

**5. The paper framing is still at risk of overclaiming.**
The old thesis “unified selector beats SBS across 18 routing problems” is not supported. The supported thesis is narrower: **a unified selector plus calibrated SBS fallback can safely match SBS while exposing small pockets of beneficial switching.** That can be publishable as a modest empirical methods paper, but only if the text is honest.

4. **For each weakness, the MINIMUM fix needed**

**1. Near-zero main effect size → minimum fix:**
Change the main claim from **“beats SBS”** to **“SBS-safe selective switching.”** Define success as **statistical parity with SBS plus nontrivial calibrated switch diagnostics**, not macro improvement. In tables, report SBS, raw selector, gated selector, and VBS gap, but phrase the result as “matches SBS within ≤0.01% macro cost while allowing calibrated switches.”

**2. Raw selector weakness → minimum fix:**
Move raw selector to an ablation section titled something like **“Why fallback is necessary.”** Use the raw selector’s `+0.105%` mean degradation as evidence that unconstrained learned selection is unsafe. The main method should be the gated policy only.

**3. “Mostly SBS” concern → minimum fix:**
Add one compact diagnostics table per seed and averaged over seeds:

`switch_rate`, `switch_precision`, `mean benefit when correct`, `mean harm when wrong`, `net switch benefit`, and `enabled problems`.

This is more important than another training run. Without this table, the method looks like SBS with decoration.

**4. Low-precision enabled switching → minimum fix:**
Tighten the enable rule one more notch using only calibration data. For example:

Enable a problem only if calibration improvement exceeds **2 bootstrap SE**, or if both conditions hold: positive calibration benefit and switch precision above a small predeclared floor.

Then rerun only gate evaluation on existing checkpoints. No selector retraining. The goal is to make the final gated policy either exactly SBS on weak problems or clearly beneficial where it switches.

**5. Overclaiming risk → minimum fix:**
Rewrite the abstract/introduction contribution as:

> “We study unified neural solver selection under a conservative deployment constraint: never switch away from SBS unless held-out calibration supports the switch. The resulting SBS-fallback selector matches SBS across 18 routing problems and identifies a small set of safe switching opportunities.”

Do **not** say “outperforms SBS” unless the final strict gate produces non-positive macro cost with a confidence interval that supports that wording.

5. **Is this now enough to center the paper on gated SBS-fallback as the main method?**

**Yes.**

This is now enough to center the paper on **gated SBS-fallback**, because that is the only method supported by the evidence. The prior review asked you to stop trying to rescue the raw selector and instead make the deployed method: **SBS by default, learned switch only when calibrated benefit is supported**. The new work does exactly that: three seeds, held-out calibration, conservative fallback, and a gated result that is stable and near-SBS.

But the paper must be centered on it as a **safe selective selector**, not as a strong performance-improvement method. The honest version is:

**“Raw learned selection is slightly unsafe; calibrated SBS fallback makes it safe and yields near-perfect SBS parity, with small selective gains on a few problem families.”**

That is modest, but coherent.

6. **What should we explicitly NOT spend compute on next?**

Do **not** spend compute on:

* combined loss / expected-regret / switch-hinge variants;
* oversampling or reweighted training;
* longer training;
* interleaved schedule training;
* new raw-selector seeds beyond the current 3, unless a reviewer specifically demands 5 seeds;
* larger models, deeper heads, or architectural tweaks;
* trying to make the raw selector the hero result.

The smallest next step that would change my verdict to **READY** is not more training. It is a **strict final gate pass on the existing 3 checkpoints**, with predeclared calibration-only enable rules, plus a clear switch-diagnostics table. If that produces exact SBS parity or slightly non-positive macro cost while preserving some nonzero switching, I would call this **READY as a modest, conservative paper**.

</details>

### Actions Taken
- Reproduced the simple winning recipe in the isolated `0selection2` worktree for two additional seeds:
  - `R1_seed1_debug`
  - `R1_seed2_serial`
- Kept the recipe fixed to the reviewer-approved low-compute setting:
  - soft regret-listwise loss only
  - no combined loss
  - no oversampling
  - no long training
  - early stopping by validation macro cost
- Reused the held-out gate pipeline for each seed:
  - calibration = 50% of `val` per problem
  - evaluation = `test`
  - per-problem score family search over `{margin, prob_gap, margin_over_entropy}`
  - conservative fallback-to-SBS when calibration gain failed the bootstrap rule
- Aggregated the 3-seed gated results in [2026-04-19-round4-seed-aggregate.md](/public/home/zhoucl/shiys/0selection2/review-stage/2026-04-19-round4-seed-aggregate.md).
- Generated full `观测指标.md`-style test analysis for the best new seed:
  - [analysis_test_fixed](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R1_seed2_serial/analysis_test_fixed/analysis_test.json)
  - [analysis_test_gated_fixed](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R1_seed2_serial/analysis_test_gated_fixed/analysis_test.json)

### Results
- **Multi-seed raw selector test macro_vs_sbs**:
  - seed0: `+0.227%`
  - seed1: `+0.071%`
  - seed2: `+0.015%`
  - mean: `+0.105%`, seed-bootstrap CI: `[+0.015%, +0.227%]`
- **Multi-seed gated selective-switch policy test macro_vs_sbs**:
  - seed0: `+0.011%`
  - seed1: `+0.004%`
  - seed2: `+0.000%`
  - mean: `+0.005%`, seed-bootstrap CI: `[+0.000%, +0.011%]`
- **Best new seed (`R1_seed2_serial`) detailed test metrics**:
  - raw:
    - `macro_top1 = 0.5141`
    - `macro_top2 = 0.8728`
    - `macro_top3 = 0.9827`
    - `macro_vs_sbs = +0.008%`
    - `beat = 3/18`, `match = 17/18`
  - gated:
    - `macro_top1 = 0.5143`
    - `macro_top2 = 0.8731`
    - `macro_top3 = 0.9827`
    - `macro_vs_sbs = +0.0003%`
    - `beat = 2/18`, `match = 18/18`
- **Operational note**:
  - Parallel two-seed training in one container again hit the memory boundary, so the final two extra seeds were run serially.
  - Direct foreground launch preserved the correct `WANDB_API_KEY` environment; `tmux` launch paths were less reliable in this setup.

### Status
- **Stopping.**
- Final reviewer verdict is `ALMOST`, which satisfies the loop stop rule.
- The supported main story is now:
  - raw selector is slightly unsafe;
  - calibrated SBS fallback removes most of that harm;
  - the deployed method should be framed as a **safe selective solver selector**, not as a broad SBS-improvement method.
- The smallest remaining paper-prep step is **not** more training. It is a stricter final gate sweep plus a compact switch-diagnostics table.

## Method Description

The final method is a two-stage unified solver-selection pipeline over 18 routing problems. A single shared neural selector consumes either coordinate-based routing inputs or ATSP cost matrices, then combines the instance representation with problem identity, routing-constraint bits, and simple distribution metadata. It scores a global solver vocabulary through a masked-softmax head, so each problem only competes among its valid candidate solvers. Training uses regret-soft listwise supervision: instead of hard winner labels, the selector is taught a soft target distribution derived from relative cost regret within each problem’s available solver pool.

At deployment time, the raw selector is **not** trusted directly. We calibrate a conservative per-problem gate on held-out validation data, choosing a switch score and threshold that decides whether the model is allowed to deviate from SBS. If the calibration evidence is weak, that problem falls back completely to SBS. The deployed policy is therefore: **SBS by default, learned switching only when held-out calibration supports it**. Empirically, this converts a slightly unsafe raw selector into a near-SBS-parity safe selective selector with small pockets of beneficial switching.

## Round 5 (2026-04-19T01:06:00+0800)

### Assessment (Summary)
- **Score**: 7.5/10
- **Verdict**: READY
- **Key change from Round 4**: a stricter final gate (`2 bootstrap SE` + calibration precision floor) plus the compact switch-diagnostics table that the reviewer explicitly requested.
- **Loop policy note**: even though the reviewer now says `READY`, this run does **not** terminate here because the user explicitly requested that the auto-review loop keep going instead of stopping at the first positive verdict.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

1. **Score: 7.5/10**

2. **Verdict: READY**

3. **Remaining critical weaknesses ranked**

   **1. The empirical gain is essentially zero.**
   The strict-gated method is at SBS parity, not meaningfully above SBS: 3-seed strict gated test macro_vs_sbs is **+0.001%**, with seed-bootstrap CI **[+0.000%, +0.003%]**. This is safe, but not a performance-improvement story. 

   **2. The method is now mostly SBS.**
   The gate enables only **1.0 problem per seed on average**, and seed 2 enables **zero** problems. That means the deployed policy is largely “use SBS unless a very narrow calibrated exception fires.” 

   **3. Switching remains real but fragile.**
   On enabled problems, mean switch rate is **0.112** and mean switch precision is **0.348**. That is enough to support a conservative selective-switching claim, but not enough to claim broad learned selector reliability. 

   **4. Per-problem evidence is sparse and uneven.**
   Most problems are exact SBS fallback. The positive visible movement is concentrated in a tiny number of enabled problem/seed cases, while TSP still shows a small negative mean under the strict aggregate. 

   **5. Only 3 seeds.**
   For a modest paper, 3 seeds is acceptable if the claim is conservative parity-plus-safe-switching. It is not enough for a strong claim about robust improvement.

4. **For each weakness, the MINIMUM fix needed**

   **1. Near-zero gain**
   Do **not** frame the contribution as beating SBS. Frame it as:
   **“A calibrated SBS-fallback selector that preserves SBS-level performance while allowing rare, auditable switches.”**
   The main table should say “parity with SBS,” not “improvement over SBS.”

   **2. Mostly SBS behavior**
   Make this explicit in the method and abstract. Say the policy is intentionally conservative and that abstention is the central mechanism, not a failure. Include the enabled-problems-per-seed and switch-rate diagnostics in the main paper, not just appendix.

   **3. Fragile switching**
   Include the compact switch-diagnostics table exactly because it makes the safety case: switch precision, switch rate, benefit when correct, harm when wrong. Do not hide that precision is only moderate.

   **4. Sparse per-problem support**
   Add one per-problem table showing which problems are enabled and which collapse to SBS. State that the method discovers only a small calibrated opportunity set under the strict gate.

   **5. 3-seed limitation**
   Add a limitation paragraph. No extra training is required. The minimum fix is honest scope control: “Results are 3-seed and support a modest conservative deployment claim, not broad superiority.”

5. **Is this now READY as a modest conservative paper centered on strict gated SBS-fallback? Answer yes or no, and why.**

   **Yes.**

   This is now READY **as a modest conservative paper**, provided the paper is centered on **strict gated SBS-fallback**, not on raw selector improvement.

   The reason is that the final blocker from Round 4 has been addressed: the gate was tightened to a stricter enable rule, and compact switch diagnostics were added. The resulting method does what the conservative story needs it to do: it stays at SBS parity under held-out calibration and test-only evaluation, with a nonnegative 3-seed aggregate CI and rare calibrated switching. The result is small, but internally coherent.

   The paper should make a narrow claim:

   **“Learned selectors can be made safe by treating SBS as the default and enabling switches only when held-out calibration shows sufficient evidence of benefit.”**

   It should not claim:

   **“We outperform SBS.”**

6. **If not ready, what is the smallest next step that could realistically change your verdict?**

   Not applicable, because my verdict is **READY**.

   The smallest next step is not more experimentation; it is paper assembly around the conservative claim. The main text should include:

   * strict gate protocol;
   * raw selector as an ablation/failure mode;
   * strict gated result as the deployed method;
   * switch-diagnostics table;
   * limitation that the method is mostly SBS and only selectively switches.

7. **What should we explicitly NOT spend compute on next?**

   Do **not** spend compute on new selector training.

   Specifically, do not spend compute on:

   * combined loss tuning;
   * oversampling variants;
   * longer training;
   * interleaved schedules;
   * larger raw selector models;
   * trying to force the raw selector to beat SBS;
   * additional score-family searches beyond the current strict gate;
   * relaxing the gate to make the result look larger.

   The next work should be writing and packaging, not training. The empirical contribution is now a conservative deployment protocol, and more compute is more likely to blur that story than strengthen it.

</details>

### Actions Taken
- Added a stricter deployment gate in [gate.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/gate.py):
  - enable only if calibration benefit clears `2 bootstrap SE`
  - plus a calibration switch-precision floor
- Extended [aggregate_gated_runs.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/scripts/aggregate_gated_runs.py) to report:
  - enabled problems per seed
  - mean switch rate
  - mean switch precision
  - mean relative benefit when correct
  - mean relative harm when wrong
- Re-ran strict gate evaluation on all 3 best checkpoints:
  - `R1_seed0/gate_eval_valhalf_test_strict.json`
  - `R1_seed1_debug/gate_eval_valhalf_test_strict.json`
  - `R1_seed2_serial/gate_eval_valhalf_test_strict.json`
- Aggregated the strict-gate result in [2026-04-19-round5-strict-gate-aggregate.md](/public/home/zhoucl/shiys/0selection2/review-stage/2026-04-19-round5-strict-gate-aggregate.md).

### Results
- **Strict-gated 3-seed aggregate**:
  - `macro_vs_sbs = +0.001%`
  - seed-bootstrap CI: `[+0.000%, +0.003%]`
- **Per-seed strict-gated test macro_vs_sbs**:
  - seed0: `+0.001%`
  - seed1: `+0.003%`
  - seed2: `+0.000%`
- **Compact switch diagnostics**:
  - enabled problems per seed: `1.0`
  - mean switch rate on enabled problems: `0.112`
  - mean switch precision on enabled problems: `0.348`
  - mean relative benefit when correct: `+1.798%`
  - mean relative harm when wrong: `+1.322%`

### Status
- Continuing to Round 6.
- No more training is warranted under the reviewer’s current guidance.
- The next round should package the evidence more explicitly:
  - per-problem enable/fallback table
  - conservative claim wording
  - raw selector demoted to ablation/failure mode

## Round 6 (2026-04-19T01:37:12+0800)

### Assessment (Summary)
- **Score**: 7.6/10
- **Verdict**: ALMOST
- **Key change from Round 5**: the evidence pack was reframed around a conservative SBS-fallback story, with a paper-facing per-problem enable/fallback table and a direct statement that the deployed policy is mostly SBS.
- **Main blocker**: the new table introduced a trust-breaking ambiguity: `Enabled seeds = 0` was read as zero enabled seeds, while the intent was `seed id = 0`.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

## 1. Score: 7.6/10

The conservative SBS-fallback framing is now basically the right paper, but the evidence pack still has a few paper-facing coherence issues that must be fixed before I would call it fully ready. 

## 2. Verdict: ALMOST

This is very close to ready as a modest, conservative paper. The main claim is now appropriately scoped: **strict held-out calibrated SBS fallback, mostly SBS behavior, rare auditable switching, no broad superiority claim**.

I would not mark it READY yet because there are internal inconsistencies in the final framing table and some wording still risks confusing “parity/safety” with “improvement.”

## 3. Remaining critical weaknesses ranked by severity

1. **Internal inconsistency in the strict per-problem enable/fallback table.**
   The Round 6 table labels TSP and OVRPBTW as `selective-switch`, but shows `Enabled seeds = 0` for both. The same document then says problems with at least one strict-gated switch are TSP, OVRPTW, and OVRPBTW. Round 5’s aggregate table also lists enabled seeds as 1 for TSP, OVRPTW, and OVRPBTW. This is the most serious issue because it directly affects reader trust.

2. **The sign convention for `vs_sbs` is still too easy to misread.**
   The evidence says raw selector is “slightly unsafe,” while reporting raw `vs_sbs = +0.104%` and strict gated `vs_sbs = +0.001%`. That is coherent only if positive means **worse/higher cost than SBS**, but the paper-facing language must state this explicitly every time the headline number appears. Otherwise readers may interpret `+0.104%` as an improvement.

3. **The strict gate is mostly an SBS identity policy, and this needs to be impossible to miss.**
   Enabled problems per seed are only about 1.0, and most problems are fallback-only. That is acceptable for the proposed contribution, but only if the paper says plainly that the method usually abstains and that this is the intended safety mechanism, not a hidden weakness.

4. **The empirical claim is still thin: 3 seeds, tiny effect size, and near-zero margin.**
   Strict-gated macro `vs_sbs = +0.001%` with CI `[+0.000%, +0.003%]` is a parity/safety result, not evidence of meaningful improvement. The correct claim is “does not materially degrade SBS under strict calibration,” not “beats SBS.”

5. **Switch precision is modest and should not be oversold.**
   Mean switch precision on enabled problems is about 0.348. The policy works because switches are rare and filtered, not because the selector is highly accurate. This should be framed as calibrated abstention over a noisy selector.

6. **The raw selector’s role must remain demoted.**
   The raw selector is useful diagnostically and as motivation for gating, but it should not be presented as the deployed method or as the main result.

## 4. For each weakness, the MINIMUM fix needed

1. **Table inconsistency:**
   Audit the strict per-problem table against the source aggregate and make `Strict status`, `Enabled seeds`, `Mean switch rate`, and the prose reading agree. Minimum fix: either set TSP and OVRPBTW enabled seeds to 1 if they truly switched, or relabel them as fallback-only if they did not. Do not leave `selective-switch` with `Enabled seeds = 0`.

2. **`vs_sbs` sign convention:**
   Add one sentence before the headline table:
   **“Throughout, positive `vs_sbs` denotes higher cost than SBS, i.e. degradation; zero denotes SBS parity.”**
   Then phrase the headline as: “strict gating reduces mean degradation from +0.104% to +0.001%.”

3. **Mostly-SBS behavior:**
   Add a short paragraph in the main results section:
   **“The deployed policy is intentionally conservative: on most problems and seeds it exactly returns SBS. Its contribution is not broad switching, but a held-out calibrated rule for allowing rare switches only where validation evidence supports them.”**

4. **Thin empirical margin:**
   In the abstract/conclusion, avoid “improves over SBS.” Use:
   **“matches SBS within measurement resolution while preserving rare calibrated switching opportunities.”**
   Also keep the 3-seed limitation explicit.

5. **Switch precision:**
   Include the switch diagnostics table, but interpret it cautiously:
   **“Precision is modest; the safety property comes from abstention and fallback, not from universally reliable instance-level classification.”**

6. **Raw selector demotion:**
   Move raw selector results to an ablation/failure-mode subsection. The main method should be named something like **Strict Calibrated SBS-Fallback** rather than “unified selector.”

## 5. Is this now READY as a conservative paper centered on strict gated SBS-fallback? Answer yes or no and explain briefly.

**No — not fully ready yet.**

Conceptually, yes: the paper now has the right conservative story. But paper-facing readiness requires the evidence pack to be internally consistent, and the current strict enable/fallback table has a visible contradiction. That is exactly the kind of thing a reviewer will notice and use to question the whole calibration story.

After the table/sign-convention cleanup, I would likely change this to READY.

## 6. If not fully ready, what is the smallest next step that could realistically change your verdict?

Do a **source-of-truth audit pass** over the Round 6 framing document only:

Check the strict per-problem table against the Round 5 aggregate/source logs, fix the TSP/OVRPTW/OVRPBTW enabled-seed/status mismatch, define the `vs_sbs` sign convention, and rewrite the headline claim as **“near-SBS parity via strict calibrated fallback”** rather than any form of improvement claim.

No new training or new experiments are needed for this verdict to change.

## 7. What should we explicitly NOT spend effort on next?

Do **not** spend effort on new selector training, larger models, longer training, combined loss, oversampling, interleaved schedules, more score-family tuning, or relaxing the gate to make the numbers look better.

Also do not try to turn this into a broad “beats SBS” paper. The viable paper is narrower and cleaner: **a strict held-out calibrated fallback policy that makes a slightly unsafe unified selector deployable by mostly abstaining to SBS.**

</details>

### Actions Taken
- Added a reusable report generator in [build_conservative_story_report.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/scripts/build_conservative_story_report.py) so the conservative framing can be regenerated from source JSON instead of hand-edited.
- Generated [2026-04-19-round6-conservative-framing.md](/public/home/zhoucl/shiys/0selection2/review-stage/2026-04-19-round6-conservative-framing.md) with:
  - strict-gate headline metrics
  - a paper-facing per-problem enable/fallback table
  - a limitations section centered on mostly-SBS deployment
- Updated [CLAIMS_FROM_RESULTS.md](/public/home/zhoucl/shiys/0selection2/CLAIMS_FROM_RESULTS.md) so result-to-claim reflects the strict-gate pass rather than the earlier loose-gate state.

### Results
- The conservative framing direction was validated.
- The only remaining blocker was a source-of-truth presentation bug in the per-problem table plus missing sign-convention wording.
- The reviewer again said **no new training** is needed.

### Status
- Continuing to Round 7.
- Round 7 is restricted to the minimum paper-facing fixes:
  - separate enabled seed count from enabled seed ids
  - define the `vs_sbs` sign convention explicitly
  - restate the method as a mostly-SBS abstention policy

## Round 7 (2026-04-19T01:42:35+0800)

### Assessment (Summary)
- **Score**: 7.6/10
- **Verdict**: ALMOST
- **Key change from Round 6**: the per-problem table was audited against the source aggregate, `Enabled seed count` was separated from `Enabled seed ids`, and the sign convention for `vs_sbs` was made explicit.
- **Main blocker**: the switch diagnostics still mixed denominator conventions; the report also still used slightly too-strong `SBS-safe` wording.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

## 1. Score: 7.6/10

## 2. Verdict: ALMOST

The core evidence pack is now much more coherent: the sign convention is explicit, the deployed method is correctly framed as mostly SBS, and the strict per-problem macro table reconciles with the strict aggregate. The remaining issues are small but still paper-facing, and a skeptical reviewer could catch them. 

## 3. Remaining critical weaknesses ranked by severity

1. **Switch-diagnostic aggregation is still ambiguously labeled.**
   The headline says “mean switch rate/precision on enabled problems,” but the reported values appear to be seed-averaged with the no-enabled seed counted as zero. For example, the enabled problem rows have switch precisions 0.760, 0.407, and 0.511, whose enabled-only average is not 0.348. The 0.348 value matches averaging seed-level values including the seed with zero enabled problems. This is exactly the kind of source-of-truth ambiguity that could trigger reviewer distrust.

2. **“SBS-safe” is still slightly too strong.**
   Under the now-explicit sign convention, strict gated `vs_sbs = +0.001%` is a tiny degradation, not an improvement. The result supports **near-SBS parity via strict abstention**, not a literal guarantee of SBS safety. The “nonpositive-seed fraction = 0.000” also means no seed achieved nonpositive degradation in unrounded arithmetic, so the paper must not imply empirical dominance over SBS.

3. **The “selective-switch” label can be misread as test-improving.**
   TSP is test-beneficial on average under strict gating, but OVRPTW and OVRPBTW are still positive-degradation rows: `+0.017%` and `+0.026%`. They are “calibration-enabled” or “validation-enabled” switch cases, not necessarily test-improving problem classes.

4. **The empirical support is intentionally thin and sparse.**
   Only 3 seeds support the final claim; enabled switching appears in only three problem/seed opportunities; the deployed policy mostly returns SBS; switch precision is moderate. This is acceptable for a conservative paper, but not for a strong solver-selection superiority claim.

5. **Rounded zeros and sign language remain easy to confuse.**
   The table contains `+0.000%` and `-0.000%`, while benefit/harm metrics are positive magnitudes. That is manageable, but it needs a table note so readers do not apply the `vs_sbs` sign convention to every diagnostic column.

## 4. For each weakness, the MINIMUM fix needed

1. **Aggregation-label fix:**
   Add one canonical definition block for switch diagnostics. Either rename the headline metrics to something like “seed-averaged diagnostics, with disabled seeds contributing zero,” or recompute and report truly enabled-only averages. Do not leave the current phrase “on enabled problems” attached to the 0.112 / 0.348 / 1.798 / 1.322 values unless that is mathematically true.

2. **Claim-strength fix:**
   Replace “SBS-safe deployment policy” with “near-SBS-parity deployment policy” or “strict SBS-fallback policy with negligible measured degradation.” Say “reduces raw-selector degradation from +0.104% to +0.001%,” not “beats SBS” or “guarantees safety.”

3. **Selective-switch label fix:**
   Rename `selective-switch` to `calibration-enabled switch` or add a footnote: “Enabled means the strict validation gate allowed switching in at least one seed; it does not imply positive held-out test gain for that problem.”

4. **Scope fix:**
   Keep the current limitations paragraph, but make it impossible to miss: “The final claim is based on 3 seeds and sparse switching; the contribution is calibrated abstention, not broad autonomous improvement.”

5. **Rounding/sign fix:**
   Add a note under the main table: “`vs_sbs` columns use signed cost difference where positive is worse than SBS. Benefit/harm columns are positive magnitudes. `±0.000%` values are rounded.”

## 5. Is this now READY as a conservative paper centered on strict gated SBS-fallback? Answer yes or no and explain briefly.

**No — ALMOST.**

Substantively, the paper is now very close: the macro evidence is coherent, the framing is mostly honest, and the strict gated SBS-fallback story is defensible. But I would not call it fully ready until the switch-diagnostic denominator and “SBS-safe” wording are cleaned up. Those are small fixes, but they are precisely paper-facing consistency issues.

## 6. If not fully ready, what is the smallest next step that could realistically change your verdict?

Make a **single source-of-truth diagnostics patch**:

Define each aggregate diagnostic’s denominator, update the labels to match, rename `selective-switch` if needed, and replace “SBS-safe” with “near-SBS-parity strict SBS-fallback.” No new experiment is needed. That patch would likely move my verdict to **READY**.

## 7. What should we explicitly NOT spend effort on next?

Do **not** spend effort on new selector training, larger models, objective tuning, oversampling, interleaved schedules, more score-family search, or relaxing the gate to manufacture a larger gain. Also do not try to rescue the raw selector as the main method. The remaining work is not compute; it is claim hygiene and table/caption consistency.

</details>

### Actions Taken
- Re-generated the framing report from source JSON as [2026-04-19-round7-source-of-truth-audit.md](/public/home/zhoucl/shiys/0selection2/review-stage/2026-04-19-round7-source-of-truth-audit.md).
- Split `Enabled seed count` from `Enabled seed ids` in the paper-facing table.
- Added an explicit sign convention: positive `vs_sbs` means higher cost than SBS and is therefore degradation.
- Added the first clear statement that the deployed method is mostly SBS and abstains by design.

### Results
- The previous trust-breaking mismatch in the per-problem table was resolved.
- The only remaining blockers are paper-facing consistency details:
  - switch-diagnostic denominators
  - `SBS-safe` wording
  - a clearer label for calibration-enabled problems

### Status
- Continuing to Round 8.
- Round 8 is a pure diagnostics-and-wording patch:
  - clarify denominator conventions in the source aggregate
  - use explicitly defined enabled-case statistics in the paper-facing report
  - replace `SBS-safe` with near-SBS-parity strict fallback wording

## Round 8 (2026-04-19T11:41:16+0800)

### Assessment (Summary)
- **Mode change**: the loop was explicitly redirected from paper framing to **effect-first model improvement**.
- **GPU policy**: all new experiments in this round were restricted to `gpu1`.
- **Main empirical finding**: a new raw-selector training recipe, `soft_sbs_risk + interleaved`, substantially improved over the old raw baseline and reached **near-SBS parity** in quick 3-seed held-out tests.
- **Main remaining blocker**: the new raw selector now tends to collapse to an almost-exact SBS policy, so we improved safety/performance but did not yet unlock consistent positive gain over SBS.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

## 1. Diagnosis: why does the new risk-aware raw selector improve to parity but still collapse to near-SBS behavior?

The new line is doing exactly what the current objective makes easiest: **avoid being worse than SBS**. In `soft_sbs_risk_loss`, SBS has normalized delta `0`, solvers worse than SBS contribute positive loss, and solvers better than SBS contribute negative loss, but the reward for improvement is clipped by `risk_cap` and averaged under the model’s softmax probabilities. Combined with `sbs_bias_init = 0.5`, the per-problem solver bias table gives SBS an initial logit advantage, so the model can satisfy the risk objective by keeping SBS as the argmax and placing plausible challengers in top-2/top-3 without actually switching. The reported profile is consistent with that: parity on held-out test, `match = 18/18`, and strong top-k but no positive gain. 

The current objective also has a **soft-probability / hard-argmax mismatch**. Training minimizes expected risk under `p = softmax(logits_pool)`, but evaluation takes `pred_rank[:, 0]`. A challenger can receive enough probability mass to improve CE/risk but still not cross the SBS logit. That is especially likely when SBS starts with a fixed +0.5 bias and the loss does not explicitly demand a logit margin over SBS on instances where the oracle beats SBS.

There is also a **small-gain suppression problem**. `regret_soft_targets` treats regrets below `tie_tol = 1e-3` as ties, and with `tau = 0.02`, very small SBS-beating differences do not create a strong CE preference for the challenger. That is rational for stability, but it makes the selector conservative exactly in the regime where you are trying to harvest slight positive gains.

Interleaving fixed the catastrophic forgetting mode because it prevents long same-problem blocks from dragging the shared selector away from other problems. Sequential full-data training blowing up TSP to `+111.25%` is not a subtle hyperparameter issue; it is a scheduling-induced forgetting/instability failure. So the current failure is no longer “model cannot match SBS.” It is: **the safest local optimum is SBS-as-default, and the loss does not force rare confident switches strongly enough.**

## 2. Highest-ROI next experiment: what single next experiment would you run to try to move from near-parity to slight positive gain, while keeping the raw selector stable?

Run a **validation-calibrated SBS-anchored selective switch experiment** on top of the current `interleaved + soft_sbs_risk + sbs_bias_init=0.5` checkpoints.

The selector should still default to SBS. At inference/eval time, switch away from SBS only when the best non-SBS challenger is sufficiently close to or above SBS according to the model score. Concretely, for each problem `p`, compute:

```python
sbs_logit = logits_pool[:, sbs_pool_idx]
challenger_idx = argmax over non-SBS pool logits
challenger_logit = logits_pool[:, challenger_idx]

switch if challenger_logit - sbs_logit >= threshold[p]
else choose SBS
```

But tune `threshold[p]` on validation, not test. Importantly, allow thresholds to be **positive, zero, or slightly negative**. A slightly negative threshold means: “switch to the best non-SBS solver when the model ranks it just below SBS but close enough.” That directly tests whether the current high top-2/top-3 signal contains useful near-miss information.

I would sweep a tiny grid per problem, for example:

```text
threshold[p] ∈ {+0.20, +0.10, +0.05, 0.00, -0.05, -0.10, -0.20, -0.35}
```

Then lock the selected threshold per problem using validation macro-vs-SBS with a conservative guardrail:

```text
accept nonzero threshold for problem p only if val vs-SBS improves by at least ε
and does not exceed a small damage cap on that problem.
```

Suggested guardrails:

```text
ε = 0.01% absolute per-problem improvement
damage cap = +0.03% vs SBS per problem
macro selection objective = minimize val macro_vs_sbs
```

This is the highest-ROI next experiment because it needs no retraining, directly targets the observed collapse, and can tell you whether the current model has usable switch information hidden below the SBS argmax. If it produces even slight positive validation and test gains, then the next training objective should imitate this decision boundary. If it fails, that is strong evidence that the model’s non-SBS scores are not calibrated enough to harvest gains without changing the loss.

## 3. One backup experiment: if the first fails, what is the next-best fallback experiment?

Add an explicit **SBS challenger margin term** to `soft_sbs_risk_loss`, while keeping `interleaved` scheduling and the existing SBS bias.

The loss should remain mostly risk-aware, but add a small hinge that only activates on instances where the best solver beats SBS by a meaningful validation-scale gap. For example:

```python
def soft_sbs_risk_margin_loss(
    logits_global,
    pool_ids,
    costs,
    sbs_pool_idx,
    tau=0.02,
    w_ce=0.5,
    w_risk=2.0,
    risk_cap=0.05,
    switch_gap=0.002,
    switch_margin=0.15,
    w_switch=0.25,
):
    base = soft_sbs_risk_loss(
        logits_global, pool_ids, costs,
        sbs_pool_idx=sbs_pool_idx,
        tau=tau,
        w_ce=w_ce,
        w_risk=w_risk,
        risk_cap=risk_cap,
    )

    logits_pool = logits_global[:, pool_ids]
    sbs_cost = costs[:, sbs_pool_idx]
    best_idx = costs.argmin(dim=1)
    best_cost = costs.gather(1, best_idx[:, None]).squeeze(1)

    gap_vs_sbs = (sbs_cost - best_cost) / (sbs_cost.abs() + 1e-9)
    switch_mask = gap_vs_sbs > switch_gap

    if switch_mask.any():
        best_logit = logits_pool.gather(1, best_idx[:, None]).squeeze(1)
        sbs_logit = logits_pool[:, sbs_pool_idx]

        hinge = torch.relu(switch_margin - (best_logit - sbs_logit))
        switch_loss = (gap_vs_sbs[switch_mask] * hinge[switch_mask]).mean()
    else:
        switch_loss = logits_pool.sum() * 0.0

    return base + w_switch * switch_loss
```

I would start with:

```text
schedule = interleaved
loss = soft_risk_margin
sbs_bias_init = 0.5
risk_weight = 2.0
risk_cap = 0.05
soft_ce_weight = 0.5
switch_gap = 0.002
switch_margin = 0.10 or 0.15
w_switch = 0.15, 0.25, 0.40 sweep
max_batches_per_problem = same as quick line first
3 seeds only after one promising seed
```

This fallback is more invasive than threshold calibration, but it targets the true training failure: the model needs an explicit reason to move a challenger **above** SBS on clear switch opportunities, not merely give it probability mass.

## 4. For each, specify:

### Highest-ROI experiment: validation-calibrated SBS-anchored selective switch

**Why it targets the current failure mode**

The current model appears to know useful rankings but refuses to cross the SBS argmax boundary. The top-k numbers suggest that the best solver is often nearby even when the selected cost matches SBS. A calibrated switch rule directly probes whether “nearby challenger” means “profitable challenger” or just noise.

**Minimal implementation scope in this codebase**

Add one evaluation/prediction path, not a new model. In `evaluate`, after computing `log_p_pool` or `logits_pool`, replace:

```python
sel = pred_rank[:, 0]
```

with an optional SBS-gated selection rule:

```python
sbs_idx = audit[p]["sbs_pool_idx"]
logits_pool = logits[:, batch["pool_ids"]]

sbs_score = logits_pool[:, sbs_idx]
masked = logits_pool.clone()
masked[:, sbs_idx] = -1e30
challenger = masked.argmax(dim=1)
challenger_score = masked.gather(1, challenger[:, None]).squeeze(1)

threshold = threshold_by_problem[p]
sel = torch.where(
    challenger_score - sbs_score >= threshold,
    challenger,
    torch.full_like(challenger, sbs_idx),
)
```

Then write a small validation sweep script that evaluates fixed checkpoints over a threshold grid per problem, stores selected thresholds, and runs test once with locked thresholds.

**Expected risk**

Low. The model weights do not change, and the SBS fallback is always available. The main risk is validation overfitting through per-problem thresholds, especially with negative thresholds. Keep the grid tiny, use conservative damage caps, and report both validation-selected and zero-threshold variants.

**2x3090 cost**

Cheap. This is evaluation-only. It should be much cheaper than one training run.

### Backup experiment: add SBS challenger margin to `soft_sbs_risk_loss`

**Why it targets the current failure mode**

The current objective optimizes expected soft risk but does not force a hard argmax switch. The margin term explicitly says: when the oracle beats SBS by a meaningful gap, the best solver’s logit must exceed the SBS logit by a margin. That addresses the probability/argmax mismatch while preserving the risk-aware SBS anchor.

**Minimal implementation scope in this codebase**

Add a new loss option in `train.py`, for example:

```python
ap.add_argument("--loss", choices=["soft", "combined", "soft_risk", "soft_risk_margin"], ...)
ap.add_argument("--switch-gap", type=float, default=0.002)
ap.add_argument("--switch-margin", type=float, default=0.15)
ap.add_argument("--switch-weight", type=float, default=0.25)
```

Then implement `soft_sbs_risk_margin_loss` in `model.py` as a wrapper around `soft_sbs_risk_loss`. No architecture changes, no data changes, no sequential training, no full-data long run until the capped pilot works.

**Expected risk**

Medium. It could reintroduce unsafe switching if `w_switch` or negative switch examples are too aggressive. The risk is controlled by activating the margin only when `gap_vs_sbs > switch_gap`, keeping `risk_weight = 2.0`, retaining `sbs_bias_init = 0.5`, and using the same interleaved quick-pilot protocol first.

**2x3090 cost**

Medium. One capped single-seed pilot is cheap-to-medium; a 3-seed confirmation is medium. Do not run long full-data versions until a capped run shows positive validation movement without per-problem blowups.

## 5. What should we explicitly NOT waste time on next, if the goal is beating or slightly improving on SBS rather than merely matching it?

Do **not** spend the next cycle on longer `sequential` training. The TSP blow-up is a decisive enough negative signal. Sequential blocks are solving the wrong problem: they may increase exposure, but they destabilize the shared selector.

Do **not** add more representation capacity right now: richer pooling, size features, problem FiLM, larger heads, or a more complex ATSP encoder. The update already shows that these pilots degraded validation, with the richer/FiLM/capacity line especially hurting ATSP. The current bottleneck is not basic representation power; it is conservative decision behavior around SBS.

Do **not** just run more seeds of the current collapsed configuration. More seeds may tighten the parity estimate, but they are unlikely to produce a systematic positive gain because the objective and bias structure favor SBS as the stable attractor.

Do **not** optimize macro top-1 as the next target. The useful metric is selected cost vs SBS. A selector can have decent top-k/top-1 diagnostics and still fail to beat SBS if the decision boundary is too conservative or if it switches on the wrong instances.

Do **not** remove the SBS anchor wholesale. Lowering or removing `sbs_bias_init`, dropping the risk term, or switching back to raw listwise CE may create apparent movement, but it is likely to recover the old failure mode: unsafe switches on CVRP/OVRPB/VRPB-style problems.

Do **not** start paper-style ablation grids yet. The next question is binary and experimental: **does the current model contain exploitable near-SBS switch signal?** First answer that with the validation-calibrated switch rule. If yes, train the margin behavior into the raw selector. If no, use the margin-loss fallback to create that signal deliberately.

</details>

### Actions Taken
- Redirected the loop to **effect-first** experimentation and limited all newly launched runs to `gpu1`.
- Added `soft_sbs_risk_loss` to [model.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/model.py) and the corresponding `train.py` knobs:
  - `risk_weight`
  - `risk_cap`
  - `soft_ce_weight`
  - `sbs_bias_init`
- Added `max_batches_per_problem` to [train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/train.py) so quick capped pilots can cover all 18 problems in a few minutes.
- Added several new model/training options and evaluated them:
  - `rich_pool`
  - `size_feature`
  - `problem_film`
  - `global_stats`
  - `soft_sbs_risk` with and without `switch_hinge`
  - `interleaved` versus `sequential`
- Extended [gate.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/gate.py) with a `--challenger-only` path so the best non-SBS challenger can be calibrated directly against SBS.

### Results
- **Best new raw-selector line**:
  - `soft_sbs_risk + interleaved + sbs_bias_init=0.5`
  - quick 3-seed test:
    - seed0: `macro_vs_sbs = +0.000%`, `match = 18/18`
    - seed1: `macro_vs_sbs = +0.000%`, `match = 18/18`
    - seed2: `macro_vs_sbs = +0.000%`, `match = 18/18`
  - representative test macro:
    - `top1 = 0.5144`
    - `top2 = 0.8732`
    - `top3 = 0.9827`
- **Strong negative result**:
  - full-data `sequential` training with the new risk loss was unstable and produced `val macro_vs_sbs = +6.562%`, including `TSP = +111.25%`.
- **No evidence yet for moving beyond parity**:
  - challenger-only SBS calibration on the current checkpoint gave test `macro_vs_sbs = +0.010%`
  - adding `switch_hinge` did not beat the stable near-parity line
  - adding `global_stats` alone did not beat the stable near-parity line
  - lowering `risk_weight` / reducing SBS bias / removing problem-solver bias also stayed at near-parity rather than opening positive gain
- **Soft-only alternative**:
  - `soft + interleaved + global_stats` produced a small tradeoff:
    - better `TSP` and `ATSP`
    - worse `CVRP`
    - overall `val macro_vs_sbs = +0.006%`

### Status
- Continuing effect-focused improvement work rather than stopping at a single round.
- Current working conclusion:
  - `interleaved` is necessary for stability;
  - `soft_sbs_risk` is the first change that reliably repairs the raw selector;
  - the current low-cost search space mostly converges to **near-SBS parity**, not clear positive gain;
  - the next credible direction is stronger switch-worthiness signal / representation, not more minor loss nudges on the same collapse regime.

## Round 9 (2026-04-19T12:15:00+0800)

### Assessment (Summary)
- **External reviewer**: not re-polled yet in this sub-round.
- **Scope of this round**: continue the user-requested effect-first loop, using the previous oracle feedback as guidance while prioritizing actual model improvement over narrative work.
- **Working hypothesis**:
  - the raw selector was collapsing because `soft_risk` from scratch made SBS too attractive too early;
  - a stronger discriminative warm start might preserve challenger signal before the SBS-relative risk term is introduced.

### Reviewer Raw Response

<details>
<summary>No new external reviewer response in this sub-round</summary>

This round executed directly on the prior effect-focused reviewer guidance instead of pausing for another oracle pass first.

</details>

### Actions Taken
- Tried a new two-stage training regime instead of more small loss nudges:
  1. Stage 1: `soft + interleaved + global_stats`
  2. Stage 2: initialize from Stage 1 and fine-tune with balanced `soft_risk`
- Added `--init-ckpt` support in [train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/train.py) to enable warm-start fine-tuning.
- Fixed checkpoint-aware evaluation for feature-augmented models:
  - [analyze.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/analyze.py) now restores `problem_film`, `size_feature`, `rich_pool`, and `global_stats`
  - [gate.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/gate.py) now restores `global_stats` too
- Kept all new GPU training bound to `gpu1` only.
- Queued continuing multi-seed follow-up in `tmux`:
  - `selector_r6_seed1`
  - `selector_r6_seed2`

### Results
- **Negative warm-start control**:
  - `soft + interleaved` from scratch without the stats warm start did not help:
  - [R6_soft_pretrain_seed0/eval_epoch0.json](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R6_soft_pretrain_seed0/eval_epoch0.json)
  - val `macro_top1 = 0.5049`
  - val `macro_vs_sbs = +0.0467%`
- **New best seed 0 raw-selector line**:
  - [R6_init_softstats_soft_risk_bal_seed0/eval_epoch0.json](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R6_init_softstats_soft_risk_bal_seed0/eval_epoch0.json)
  - val `macro_top1 = 0.5187`
  - val `macro_vs_sbs = -0.086%`
  - val `macro_vbs_gap_closed = +3.77%`
  - [test analysis](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R6_init_softstats_soft_risk_bal_seed0/analysis_test/analysis_test.json)
  - test `macro_top1 = 0.5190`
  - test `macro_top2 = 0.8724`
  - test `macro_top3 = 0.9831`
  - test `macro_vs_sbs = -0.070%`
  - test `macro_vbs_gap_closed = +2.59%`
  - strongest gain: `ATSP = -1.351%`
- **Seed 1 partial replication**:
  - [R6_init_softstats_soft_risk_bal_seed1/eval_epoch0.json](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R6_init_softstats_soft_risk_bal_seed1/eval_epoch0.json)
  - val `macro_top1 = 0.5183`
  - val `macro_vs_sbs = -0.073%`
  - val `macro_vbs_gap_closed = +2.79%`
- **Seed 2**:
  - still running in `tmux` at the time of this round log

### Status
- This is the first raw-selector recipe in the loop that clearly improves both:
  - held-out `top1`
  - held-out `mean cost` relative to SBS
- Current best direction is now:
  - `soft + global_stats` discriminative pretrain
  - then balanced `soft_risk` fine-tune
- The loop remains **in progress**:
  - `seed2` is still running
  - seed-1 held-out analysis is still being materialized
  - cheap gate calibration on the new best checkpoint is still pending completion

## Round 10 (2026-04-19T13:00:00+0800)

### Assessment (Summary)
- **External reviewer**: not re-polled yet.
- **Scope of this round**: stress-test the new best two-stage line against stronger alternatives before allocating a longer training budget.
- **Decision rule**: only variants that beat the new held-out 3-seed aggregate should receive long-budget continuation.

### Reviewer Raw Response

<details>
<summary>No new external reviewer response in this sub-round</summary>

This round continued the user-requested effect-first optimization loop without pausing for another oracle pass first.

</details>

### Actions Taken
- Materialized held-out `test` analysis for the best line on seed1 and seed2.
- Computed the new 3-seed held-out benchmark:
  - `macro_top1 = 0.5221 ± 0.0022`
  - `macro_vs_sbs = -0.081% ± 0.012%`
- Added and tested train-time 8-fold coordinate augmentation:
  - `--coord-augment 8`
- Added and tested a lightweight CoEKS/URS-inspired constraint-expert residual head:
  - `--constraint-experts`
- Added a more robust long-run script:
  - [run_r8_long_seed2.sh](/public/home/zhoucl/shiys/0selection2/code/unified_selector/scripts/run_r8_long_seed2.sh)
- Started minute-level monitoring logs:
  - [selector_monitor_r7.log](/public/home/zhoucl/shiys/0selection2/review-stage/selector_monitor_r7.log)
  - [selector_monitor_r8.log](/public/home/zhoucl/shiys/0selection2/review-stage/selector_monitor_r8.log)

### Results
- **Current champion, now fully confirmed on held-out test**:
  - seed0: `macro_top1 = 0.5190`, `macro_vs_sbs = -0.070%`
  - seed1: `macro_top1 = 0.5233`, `macro_vs_sbs = -0.076%`
  - seed2: `macro_top1 = 0.5240`, `macro_vs_sbs = -0.097%`
  - 3-seed aggregate:
    - `macro_top1 = 0.5221 ± 0.0022`
    - `macro_top2 = 0.8734 ± 0.0007`
    - `macro_top3 = 0.9833 ± 0.0003`
    - `macro_vs_sbs = -0.081% ± 0.012%`
    - `macro_vbs_gap_closed = +3.60% ± 0.94%`
- **Negative result: 8-fold augmentation**
  - stage-2 val `macro_top1 = 0.5159`
  - stage-2 val `macro_vs_sbs = -0.068%`
  - did not beat the current champion
- **Negative result: constraint-expert residual head**
  - stage-2 val `macro_top1 = 0.5157`
  - stage-2 val `macro_vs_sbs = -0.052%`
  - did not beat the current champion
- **Long-budget continuation promoted**
  - active mainline:
    - `R8_softstats_cont_full_seed2`
    - then `R8_init_softstats_soft_risk_cont_full_seed2`
  - full-data continuation uses:
    - `batch_per_problem = 16`
    - `num_workers = 0` on the active rerun
    - `epochs = 2 + 2`
    - `lr = 1e-4`
  - note:
    - the first `num_workers = 8` attempt was OOM-killed because the code creates a loader per problem
    - `num_workers = 2` was still not robust enough under the container memory cgroup
    - the current `tmux`-launched rerun with `num_workers = 0` is alive, and its training PID is confirmed on physical `gpu1`

### Status
- Two literature-inspired quick additions were tested and rejected for now.
- The strongest confirmed raw-selector direction remains the simple two-stage `soft+stats -> balanced soft_risk` line.
- The loop remains **in progress** with a longer full-data continuation now running for the best seed-2 branch.
- Follow-up queue is already staged:
  - after `R8` finishes, `R9_hier_softstats_pretrain_seed2 -> R9_init_hier_softstats_soft_risk_bal_seed2` will launch automatically
  - launcher log: [launch_r9_after_r8.log](/public/home/zhoucl/shiys/0selection2/review-stage/launch_r9_after_r8.log)
  - monitor log: [selector_monitor_r9.log](/public/home/zhoucl/shiys/0selection2/review-stage/selector_monitor_r9.log)
  - after `R9` finishes, `R10_rank_hier_softstats_pretrain_seed2 -> R10_init_rank_hier_softstats_soft_risk_bal_seed2` will launch automatically
  - launcher log: [launch_r10_after_r9.log](/public/home/zhoucl/shiys/0selection2/review-stage/launch_r10_after_r9.log)
  - monitor log: [selector_monitor_r10.log](/public/home/zhoucl/shiys/0selection2/review-stage/selector_monitor_r10.log)
  - after `R10` finishes, `R11_problemhead_softstats_pretrain_seed2 -> R11_init_problemhead_softstats_soft_risk_bal_seed2` will launch automatically
  - launcher log: [launch_r11_after_r10.log](/public/home/zhoucl/shiys/0selection2/review-stage/launch_r11_after_r10.log)
  - monitor log: [selector_monitor_r11.log](/public/home/zhoucl/shiys/0selection2/review-stage/selector_monitor_r11.log)

## Round 11 (2026-04-19T13:45:00+0800)

### Assessment (Summary)
- **External reviewer**: Codex subagent `Gauss`
- **Score**: `5/10`
- **Verdict**: `not ready`
- **Key criticisms**:
  - current supervision is still too class-like; gains are real but still SBS-safe rather than a strong best-solver predictor
  - generic `global_stats` are not enough; the selector likely still lacks problem-structured descriptors
  - `R8/R9/R10/R11` alone are unlikely to close the gap to the user's much higher top1 target
  - the highest-ROI next jump is to change the supervision target itself: `cost-gap regression + pairwise ranking`

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

1. Score current effect progress from 1-10.

`5/10`

You found the first reproducible raw-selector line that is actually better than SBS on held-out test, and the gain is stable across 3 seeds. That is real progress. But `macro_top1 = 0.5221` is still far from the stated target, the gains are concentrated in a few problems like ATSP, and the current recipe still behaves like “mostly SBS-safe” rather than a strong true best-solver predictor. Bluntly: `80%+` macro top1 is not a plausible next-step outcome from R8/R9/R10/R11 alone.

2. Top 5 most plausible next effect-improving changes, ranked.

1. Replace the current soft-label classifier objective with `cost-gap regression + pairwise ranking`.
2. Add a strong problem-specific manual feature branch, not just generic `global_stats`.
3. Upgrade to real problem-conditioned adapters/calibrators on top of the shared backbone.
4. Change the sampler to a switch-worthiness / hard-case curriculum.
5. Use symmetry-consistency and test-time augmentation, not raw 8x dataset duplication.

3. For each, explain why it is more promising than the already-failed lines.

1. `cost-gap regression + pairwise ranking`
The current model still compresses a full solver-cost vector into a soft target over classes. That is lossy. Your best line improved only after making the loss more cost-aware, which is the biggest clue that the supervision target is the bottleneck now, not raw capacity. This is more promising than `soft_risk from scratch`, `switch hinge`, and `challenger-only gating` because it changes the main training signal itself and uses all solver supervision, not just SBS-relative behavior.

2. `problem-specific manual feature branch`
`global_stats`, `rich_pool`, and FiLM underperformed because they are generic. Routing algorithm selection usually benefits from explicit instance descriptors: asymmetry, demand tightness, spatial dispersion, depot centrality, time-window width, prize sparsity, etc. This is more promising than “larger encoder” because it injects missing problem structure rather than hoping the trunk rediscovers it from pooled coordinates.

3. `problem-conditioned adapters/calibrators`
Your failures suggest the shared model learns a useful common trunk but cannot separate fine-grained decision boundaries inside each problem family. A tiny constraint-expert residual head was too weak and too coarse. This is more promising than the failed expert head because the right unit of specialization here is likely `problem` or `problem cluster`, not a lightweight constraint tag.

4. `switch-worthiness / hard-case curriculum`
The current training stream is still dominated by easy SBS-win cases. That pushes the model toward safe parity. This is more promising than adding another loss term like `switch_hinge`, because it changes what the optimizer sees often, not just how a dominated batch is scored. If you want more true switches, you need to train on more true switch opportunities.

5. `symmetry-consistency + TTA`
The failed 8-fold augmentation pilot does not prove augmentation is useless. It only shows naive duplication was weak. Plain duplication can over-repeat easy cases and distort optimization. Consistency regularization and test-time averaging are more promising because they directly enforce invariance and reduce prediction variance without pretending the data distribution changed.

4. For each, give the minimum concrete implementation sketch in this codebase style.

1. `cost-gap regression + pairwise ranking`
In `data.py`, expose per-instance `target_gap[j] = clip((cost_j - sbs_cost) / max(abs(sbs_cost), eps), -cap, cap)` for every valid solver, plus valid pair labels for `(i, j)` where both solvers exist. In `model.py`, keep the masked solver-score head, but interpret output as predicted gap. In `train.py`, add `--loss gap_rank --reg-weight --pairwise-weight --pair-sample topk`. Loss:
`Huber(pred_gap, target_gap) + lambda * pairwise_logistic(pred_gap_i - pred_gap_j, sign(target_gap_i - target_gap_j))`
weighted by `abs(target_gap_i - target_gap_j)`. Inference: choose the minimum predicted gap. Start from the current best Stage-1 checkpoint, not from scratch.

2. `problem-specific manual feature branch`
In `data.py`, compute per-problem dense features:
`n`, mean/std/quantiles of pairwise distances, nearest-neighbor distance stats, depot-to-node radius stats, asymmetry ratio for ATSP-like tasks, demand mean/std/load ratio for CVRP-like tasks, time-window width/tightness if present, prize/penalty stats if present, pickup-delivery pairing density if present. In `model.py`, add `ManualFeatureEncoder(mlp)` and concatenate its output with the current pooled embedding before the final scorer. Add `--manual-feats` and save the feature dimension in the checkpoint config. This is cheap and high-ROI.

3. `problem-conditioned adapters/calibrators`
In `model.py`, add a per-problem adapter after the shared hidden:
`h = h + A_p(ReLU(B_p(h)))`
with small rank, or a per-problem residual logit head:
`logits = shared_head(h) + residual_head_p(h) + bias_p`.
Expose flags like `--problem-adapter --adapter-rank 16 --adapter-hidden 64`. Do not make this a pure bias-only head; it needs feature-dependent corrections. Train from the current best checkpoint. This is the strong version of R11.

4. `switch-worthiness / hard-case curriculum`
In `data.py` or the sampler, precompute:
`switch_gain = max(0, sbs_cost - best_non_sbs_cost) / abs(sbs_cost)`
and
`ambiguity = 1[(second_best - best) / abs(best) < eps]`.
In `train.py`, add sampler weights:
`w = 1 + a * 1[best != sbs] + b * switch_gain + c * ambiguity`.
Use this inside the interleaved schedule, not as a post-hoc oversample hack. Add logs for:
`fraction_non_sbs_batches`, `mean_switch_gain_seen`, `hard_case_fraction`. This is cheap and directly attacks the current parity trap.

5. `symmetry-consistency + TTA`
In `data.py`, add coordinate transforms from the routing symmetry group where valid. In `train.py`, sample two transforms of the same instance and add:
`L_cons = KL(p1 || p2) + KL(p2 || p1)` or hidden-state MSE.
Expose `--aug-consistency-weight`. In `analyze.py`, add `--tta 4` or `--tta 8` that averages logits over transformed copies at evaluation. Do not expand the dataset 8x on disk; apply transforms online. Try TTA first on an already-trained checkpoint because it is nearly free.

5. Explicitly say whether R8 / R9 / R10 / R11 are likely useful or dead ends.

- `R8`: mildly useful, not a dead end, but low upside. It is a ceiling check for the current recipe, not a likely breakthrough. If it does not beat the current champion by the first solid validation checkpoint, stop spending time on longer continuation.
- `R9`: somewhat useful, but only as an encoder upgrade. By itself I do not expect it to move macro metrics much. It is not dead, but it is unlikely to solve the main bottleneck alone.
- `R10`: the most useful of the queued runs. Ranking-style supervision is directionally correct and much better aligned with mean-cost improvement than more CE variants. Keep this.
- `R11`: conditionally useful. If it is only a small residual head, it is close to a dead end. If you turn it into real problem-conditioned adapters/calibrators as above, it becomes one of the best next bets.

If I were cutting the queue for actual gain, I would keep `R8` only until it proves or disproves “more training helps,” keep `R10`, and upgrade `R11` before running it. The highest-ROI new work is still `gap-regression + ranking` plus a real manual-feature branch.

</details>

### Actions Taken
- Materialized held-out `test` analysis for the longer `R8` stage-1 checkpoint:
  - [R8 test](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R8_softstats_cont_full_seed2/analysis_test/analysis_test.json)
- Implemented three new effect-focused upgrades:
  - `--loss gap_rank` with Huber gap regression + pairwise logistic ranking
  - `--manual-features` richer instance descriptor branch
  - `--problem-adapter` stronger problem-conditioned hidden adapter
- Launched a new warm-started run on physical `gpu1`:
  - `tmux` session: `selector_r12_gaprank`
  - stage1: `R12_gaprank_manual_adapter_seed2`
  - stage2: `R12_init_gaprank_manual_adapter_soft_risk_seed2`
  - background monitor: [selector_monitor_r12.log](/public/home/zhoucl/shiys/0selection2/review-stage/selector_monitor_r12.log)

### Results
- **New strongest single-seed held-out test result so far**:
  - [R8 stage1 test](/public/home/zhoucl/shiys/0selection2/code/unified_selector/runs/R8_softstats_cont_full_seed2/analysis_test/analysis_test.json)
  - `macro_top1 = 0.5276`
  - `macro_top2 = 0.8766`
  - `macro_top3 = 0.9862`
  - `macro_vs_sbs = -0.118%`
  - `macro_vbs_gap_closed = +6.76%`
  - `beat = 12/18`, `match = 10/18`
- **R8 stage2 status**:
  - `R8_init_softstats_soft_risk_cont_full_seed2` was memory-cgroup OOM-killed around `ep0 step4100`
  - no summary was produced, so the loop is pivoting away from more budget on this continuation recipe
- **R12 early training health**:
  - warm-start from `R8_softstats_cont_full_seed2/best.pt` succeeded under `strict=False`
  - missing keys are exactly the new modules (`manual_proj`, `problem_adapters`, `adapter_ln`)
  - early loss is numerically healthy:
    - `step50 loss=0.1934`
    - `step100 loss=0.3023`
    - `step150 loss=0.2442`
    - `step350 loss=0.1810`
  - physical `gpu1` confirmed via `nvidia-smi` UUID mapping
- **WandB note**:
  - the run again fell back to local logging because the environment resolves to the wrong `~/.netrc` account
  - this does not block training or result collection, so the run was intentionally not interrupted mid-flight

### Status
- The effect-first loop is now explicitly pivoted from “more soft-risk polishing” to “change the supervision target.”
- The current active bet is:
  - warm-start from the best longer `soft+stats` checkpoint
  - then train with `gap_rank + manual_features + problem_adapter`
- The loop remains **in progress** with continuous background monitoring and no manual handoff required.

## Round 13 (2026-04-19 18:33)

### Assessment (Summary)
- Score: pending
- Verdict: pending
- Key criticisms being addressed:
  - top-1 remains far below target despite strong top-3
  - hidden / rare winners still need better top-1 treatment
  - augmentation had not yet been integrated into the strongest live lines

### Actions Taken
- Implemented a new gated/blended reranker in [rerank_train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/rerank_train.py):
  - richer candidate token features
  - train-time `coord_augment = 8`
  - val-time sweep over blend strength and uncertainty gates
- Added and launched a durable watchdog-backed reranker run:
  - [run_r21_rerank_gated_aug8_seed2.sh](/public/home/zhoucl/shiys/0selection2/code/unified_selector/scripts/run_r21_rerank_gated_aug8_seed2.sh)
  - [R21 status](/public/home/zhoucl/shiys/0selection2/review-stage/training-watchdog/status/r21-rerank-gated-aug8-seed2.json)
  - [R21 log](/public/home/zhoucl/shiys/0selection2/review-stage/training-watchdog/logs/r21-rerank-gated-aug8-seed2.log)
- Prepared a safer augmented base continuation:
  - [run_r20b_aug8_full_seed2.sh](/public/home/zhoucl/shiys/0selection2/code/unified_selector/scripts/run_r20b_aug8_full_seed2.sh)
  - queued by [launch_r20b_after_r21.sh](/public/home/zhoucl/shiys/0selection2/code/unified_selector/scripts/launch_r20b_after_r21.sh)

### Results
- `R21` is actively training on physical `gpu1`.
- Latest observed live progress:
  - `ep0 step50 loss=0.7810`
  - `ep0 step100 loss=0.7909`
  - `ep0 step150 loss=0.8224`
  - `ep0 step200 loss=0.8354`
  - `ep0 step250 loss=0.6961`
  - `ep0 step300 loss=0.5685`
- Watchdog now has a durable status/log pair instead of only transient pane capture.
- The first direct `R20` launch failed too early to count as experimental evidence, so it has been replaced by the queued `R20b` line.

### Status
- Continuing.
- Active:
  - `R21` gated reranker with `coord_augment = 8`
- Queued next:
  - `R20b` augmented base-model continuation, auto-launched after `R21`

## Round 13 (2026-04-20 01:12 CST)

### Assessment (Summary)
- Score: `4/10`
- Verdict: `not ready`
- Key criticisms:
  - the first deep-encoder B-line had a correctness bug in hierarchical pooling because downsampled nodes were not remapped when slicing MBM bias matrices
  - the first deep-encoder B-line also mixed too many new components at once while only partially inheriting old checkpoints
  - the first clean reranker C-line still let missing support residual branches act as random candidate generators
  - reranker token features were missing support-origin signals
  - `fixable_only` remained too aggressive for the reranker

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

- Score: 4/10
- Verdict: not ready

- Ranked weaknesses
1. `B` 线核心实现现在就有 correctness bug，不是单纯“还没调好”。在 [model.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/model.py:294) 里层级池化会选出任意节点子集，但后续 MBM block 在 [model.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/model.py:327) 只是拿 `bias[:, :x.shape[1], :x.shape[1]]` 前缀切片，没按保留下来的节点索引去 gather。第二层开始 attention bias 就错位了。Minimum fix: `downsample()` 返回保留索引，并按索引重建/抽取 bias 子矩阵；修完前，R22 结果都不该当证据。
2. `B` 线现在也不是“强基座继续训练”，而是“大量新 backbone 随机初始化 + 很小 lr + 很重 loss cocktail”。R22 live log 里几乎整套新 encoder 都是 missing/unexpected key，而且第一枪在 step 50 就被 kill 了。Minimum fix: 把 B 当新 backbone 训练，不要伪装 continuation；先只保留 `gap_rank + global_stats/manual_features + new encoder`，batch `16/24`，`max_batches_per_problem=256`，先跑完一个干净 epoch0，再谈 stage2。
3. `C` 线当前 candidate generator 可能被“半随机 support 分支”污染了。`build_base_model()` 会按 args 构出 `support_problem_heads` 和 `support_constraint_expert_heads`，但 R23 启动日志明确说这些权重 missing；冻结 base 后，这些分支就是随机的，还参与 support candidate 提议。Minimum fix: load base 时，如果这些 support residual 权重缺失，就把相关分支显式关掉或零初始化成 no-op，别拿随机 support logits 去救候选。
4. `C` 线 reranker token feature 还不够。现在 [rerank_train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/rerank_train.py:172) 只喂了 `cand_base/cand_prob` 这些 base 特征，没有 `support_prob`、`is_support_candidate`、`is_rescued`。于是被 support 救回来的稀有赢家，在 reranker 看来只是“base 分数低的垃圾候选”。Minimum fix: 给 token 追加这几个 support 来源/置信度特征，并同步改 `token_dim`。
5. `C` 线还在重复 R21 最大的训练错误：`fixable_only`。这会让模型只学“能救的样本”，完全不学“别乱动”的样本，所以最后只能靠 gate 收尸。R19/R21 都已经证明这会出现 `top1_given_fixable≈0.8` 但整体 `macro_top1` 崩到 `0.39~0.40`、`vs_sbs` 转正。Minimum fix: 下一版至少跑一个 `--fixable_only` 关闭的对照，或者给 non-fixable / base-correct 样本加 `delta≈0` 锚定损失。
6. 你们还在给 shortlist/rescue 投太多 loss 预算，但现有证据已经说明 shortlist recall 不是主瓶颈。R15 的 `macro_support_top1_recall≈1.0`、`macro_support_arm_coverage≈0.988`，最终 `macro_top1` 还是只有 `0.5309`。Minimum fix: 下一次干净 B run 先把 `support_loss_weight` 降到 0 或很小，先看 raw final scorer 能不能真把最后 1-2 个 competitor 拉开。

- Pivot or continue?
继续，但不是按当前实现原样继续。`R23` 可以让它跑完，因为成本已经花出去了；`R22` 我会直接视为无效试验，先修 bug 再说。我的建议很硬：只给这条路再一次“干净 B”加一次“干净 C”的机会；如果修完上面这些问题后，仍然打不穿 `R12/R15`，就立刻 pivot，不要再堆更复杂的 encoder/reranker 变体了。

</details>

### Actions Taken
- Fixed the hierarchical pooling bias-index bug in [model.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/model.py) by returning downsampled node indices and gathering the MBM bias submatrices with those indices.
- Added legacy checkpoint remapping `support_head.* -> support_heads.0.*` so older shortlist checkpoints load correctly into the current model layout.
- Updated [rerank_train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/rerank_train.py):
  - zero-initialize missing `support_problem_heads` / `support_constraint_expert_heads` in frozen base models so missing residual branches become no-op instead of random
  - added reranker token features for `support_prob`, `is_support_candidate`, and `is_rescued`
  - removed `fixable_only` from the new clean reranker script
- Re-launched corrected runs under new watchdog names:
  - `r22g-deep-overhaul-seed2`
  - `r23d-top23-rerank-seed2`

### Results
- Invalidated earlier `R22c/R22d/R22e` evidence after the reviewer identified the bias-index correctness bug.
- The corrected `R22g` deep-encoder line is now training stably past the early failure region:
  - observed live loss: `1.0689 -> 0.7349 -> 0.8270 -> 0.6609`
- The corrected `R23d` reranker line is also live:
  - observed live loss around `0.92`
- `gpu1` is back near `90%+` during corrected concurrent runs.

### Status
- Continuing.
- Active corrected runs:
  - [r22g status](/public/home/zhoucl/shiys/0selection2/review-stage/training-watchdog/status/r22g-deep-overhaul-seed2.json)
  - [r23d status](/public/home/zhoucl/shiys/0selection2/review-stage/training-watchdog/status/r23d-top23-rerank-seed2.json)

## Round 13 follow-up (2026-04-20 08:42 CST)

### Assessment (Summary)
- Score: `4/10`
- Verdict: `not ready`
- Key follow-up criticism:
  - `r23d` confirms that a plain top-3 reranker still harms the base selector more than it helps
  - if the reranker branch continues, it should become explicitly asymmetric: fix only truly fixable cases and stay near-zero elsewhere

### Reviewer Raw Response

<details>
<summary>Click to expand follow-up reviewer response</summary>

- Score update: 4/10

- Whether current direction should continue or pivot: continue only until `r22g` gives its first real eval; if `r23e` also fails, pivot away from the `C` reranker branch immediately. `C` is no longer missing candidates; it is missing the right objective.

- One highest-ROI next experiment after `r23e`:
  Implement an asymmetric `do-no-harm + fixable-only correction` residual scorer on top of `R20b`, not another standalone reranker variant.
  Use `R20b_init_gaprank_manual_adapter_soft_risk_aug8_seed2/best.pt` as the frozen teacher/base, not `R15`, since `r23e` already removed support rescue.
  In [rerank_train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/rerank_train.py:228), keep the current CE + pairwise term only on `fixable = oracle_in_topk & base_wrong`, but add an anchor loss on `safe = ~fixable`:
  `loss_anchor = smooth_l1_loss(delta_scores[safe], 0)` or KL between `softmax(cand_base + delta)` and `softmax(cand_base)`.
  Train with `base_topk=3`, `support_candidate_topk=0`, `coord_augment=8`, `anchor_weight=1.0`, `pair_weight=1.0`, `ce_weight=0.5`, and extra weight on fixable examples by switch gain.
  Reason: `R21` showed the reranker can fix cases but destroys the base; `R23d` showed that removing `fixable_only` collapses fixable recovery (`top1_fixable=0.1323`). The missing piece is not more candidate plumbing; it is an explicit asymmetric objective that says “fix when fixable, otherwise stay near zero.”

- One thing to stop doing immediately:
  Stop launching more `C` runs that only change candidate-source or gate knobs (`base_topk`, `support_candidate_topk`, `blend_alpha`, `base_gap_max`, `rerank_margin_min`). After the fixes in [rerank_train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/rerank_train.py:118) and the negative `r23d` result, that space is no longer the bottleneck.

</details>

### Actions Taken
- Extended [rerank_train.py](/public/home/zhoucl/shiys/0selection2/code/unified_selector/rerank_train.py) with:
  - `switch_gain` extraction from the frozen base selector
  - `anchor_weight` and `anchor_mode` for a do-no-harm objective on non-fixable examples
  - switch-gain-weighted fixable supervision while keeping older reranker scripts compatible
- Added the next full reranker experiment:
  - [run_r24_r20b_fixable_anchor_seed2.sh](/public/home/zhoucl/shiys/0selection2/code/unified_selector/scripts/run_r24_r20b_fixable_anchor_seed2.sh)
  - base checkpoint: `R20b_init_gaprank_manual_adapter_soft_risk_aug8_seed2/best.pt`
  - `coord_augment = 8`, `base_topk = 3`, `support_candidate_topk = 0`, `anchor_weight = 1.0`
- Queued durable auto-launch after `r23e`:
  - [launch_r24_after_r23e.sh](/public/home/zhoucl/shiys/0selection2/code/unified_selector/scripts/launch_r24_after_r23e.sh)

### Results
- `r22g` remains live after the correctness fix and has progressed to `ep0 step21750`.
- `r23e` remains live and has advanced into `ATSP`, with latest observed `loss=0.2761` at `ep0 step850`.
- `r22h` is still queued behind `r22g`, and `r24` is now queued behind `r23e`, so the GPU1 pipeline has continuous handoff on both branches.

### Status
- Continuing.
- Active:
  - [r22g status](/public/home/zhoucl/shiys/0selection2/review-stage/training-watchdog/status/r22g-deep-overhaul-seed2.json)
  - [r23e status](/public/home/zhoucl/shiys/0selection2/review-stage/training-watchdog/status/r23e-top3-clean-rerank-seed2.json)
- Queued:
  - `r22h-deep-overhaul-clean-seed2`
  - `r24-r20b-fixable-anchor-seed2`
