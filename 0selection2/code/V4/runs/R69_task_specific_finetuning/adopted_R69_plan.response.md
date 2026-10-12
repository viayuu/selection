`review_independence=same-family; acceptance_status=provisional`

**R68 performance: 4/10, not ready for 70%. TRAIN/validation integrity: 9/10, ready within the inspected scope.** No actionable implementation defect found. Both arms completed successfully; new R68 TEST and delivery-analysis artifacts were excluded.

SAM loses **28 validation decisions, −0.155556 percentage points**, against matched AdamW. Validation selects **A28**, ahead of B35. A remains **1.338889pp below R60B**; B remains **1.494444pp below**. This bounded SAM recipe does not advance the incumbent.

| Validation record | Native Top1 | CE | Actual regret |
|---|---:|---:|---:|
| A28, selected: 8644/18000 | 48.022222% | 1.119215 | 0.974169% |
| B35, selected: 8616/18000 | 47.866667% | 1.113930 | 1.013203% |
| A40, final | 47.050000% | 1.148645 | 1.064938% |
| B40, final | 47.177778% | 1.120747 | 1.090791% |
| A, last-five mean | 47.431111% | 1.156541 | 1.030852% |
| B, last-five mean | 47.516667% | 1.118958 | 1.028748% |

Last-five means summarize epochs36–40; they are not averaged checkpoints. Sources: completed [A result](/public/home/shiys/0selection2/code/V4/runs/R68_sharpness_aware_finetuning/A_plain_adamw_seed2/result.json), [B result](/public/home/shiys/0selection2/code/V4/runs/R68_sharpness_aware_finetuning/B_sam_adamw_seed2/result.json) and their complete histories.

At the selected checkpoints, each cell below is **Top1% / CE / regret%**:

| Family | A28 | B35 | Fixed R60B39 |
|---|---:|---:|---:|
| ALL | 48.022 / 1.1192 / 0.9742 | 47.867 / 1.1139 / 1.0132 | 49.361 / 1.0924 / 0.9228 |
| TSP | 36.000 / 1.7170 / 0.6571 | 36.300 / 1.7272 / 0.7564 | 37.600 / 1.6335 / 0.6242 |
| CVRP | 48.800 / 1.3788 / 0.6983 | 48.400 / 1.3603 / 0.6522 | 49.300 / 1.3349 / 0.6544 |
| ATSP | 77.600 / 0.6016 / 0.3600 | 74.100 / 0.6513 / 0.7708 | 78.400 / 0.5904 / 0.3063 |
| MVRP | 46.800 / 1.0966 / 1.0546 | 46.853 / 1.0875 / 1.0705 | 48.213 / 1.0737 / 1.0017 |

B improves **nine tasks and regresses on nine** against A. Its largest gains are OVRPL+2.3pp and OVRPTW+2.0pp; largest losses are VRPBTW−4.1pp and ATSP−3.5pp. Against R60B, A improves3/regresses14/ties1; B improves4/regresses14. Neither improves any family aggregate over the [fixed reference](/public/home/shiys/0selection2/code/V4/runs/R60_expanded_partial_supervision/B_expanded_seed2/result.json).

A shows clear late fitting/generalization divergence: selected-to-final TRAIN Top1 rises **49.5844%→53.5683%**, TRAIN CE falls1.06507→0.99185, while validation Top1 falls48.0222%→47.0500%. B fits less strongly: selected-to-final TRAIN49.6978%→50.1900%, validation47.8667%→47.1778%. SAM lowers late validation CE and slightly improves the last-five accuracy mean, but does not improve the predeclared selected-checkpoint target. These observations establish neither sharp minima nor label noise or an accuracy ceiling.

The inspected [initialization](/public/home/shiys/0selection2/code/V4/r68_model.py:30), [gradient update](/public/home/shiys/0selection2/code/V4/r68_gradient.py:87) and [training/selection path](/public/home/shiys/0selection2/code/V4/r68_experiment.py:241) correctly implement the shared terminal trunk/fresh head, ordinary AdamW weights, unclipped global SAM perturbation, repeated dropout masks, exact restoration and one clipped optimizer step. Boundary, tail, selected-epoch and retry records agree. All three B retries occurred in the perturbed pass; the whole update was retried. Both selected validation recounts agree exactly. No corrective replay is warranted.

Each arm completed **9,600 updates/7.2M selected rows**. Successful dropout-view rows were14.4M for A and28.8M for B; attempted totals14,405,632/28,807,680 include5/3 retries. Arm wall times were**3,002.02s/5,677.56s**, making B1.891× as costly; recorded training-epoch time was1.973×. Combined arm wall was8,679.58s, within the8,729s terminal job. Peak allocated memory was15.311/15.325GiB. The earlier3s preparation failure and inherited pretraining remain separate expenditure.

**Recommend R69: complete task-specific fine-tuning versus shared fine-tuning. Score: 8/10, ready, PLAN ONLY.**

The remaining hypothesis is that one shared parameter vector compromises task-specific winner boundaries. R64 changed gradients within shared parameters; R68 changed neighborhood optimization. Neither tested removing the sharing constraint throughout the complete selector. R49–56’s pair specialists and R59–67’s transfer, auxiliary supervision, recurrence and additive correction do not establish this full18 comparison. R64’s small gain and R68’s overfitting are counter-evidence to a simple interference explanation.

[Standley et al., Sections4–5](https://proceedings.mlr.press/v119/standley20a/standley20a.pdf) show that task-sharing outcomes depend on capacity and data volume. [Fifty et al., Section4.1](https://arxiv.org/pdf/2109.04617) distinguish task grouping from modifying updates within shared parameters. Their evidence concerns other domains. Our adaptation compares `mean_t L_t(theta)` with `mean_t L_t(theta_t)` using one fixed assignment: every task gets its own complete parameter copy. No affinity-measurement or grouping search is proposed.

1. **Architecture and initialization.** `A_shared_seed2` contains one R64 `NativeUtilitySelector`; `B_task_specific_seed2` contains18 complete copies indexed by known problem identity. Retain d128, four encoder/two joint layers, four heads, FFN512, readout256, dropout0.1, handcrafted34+ID, directed ATSP and all original inputs. Initialize every copy identically from R67B’s terminal stage1 averaged-x trunk plus the entire fresh head670003, using the existing R68 initialization identity. All parameters train; no trained R68 state transfers.

2. **Isolated implementation.** Add `r69_model.py`, `r69_gradients.py`, `r69_experiment.py`, one focused CPU-check entry and launcher. Reuse the existing input whitelist, native objective and full18 evaluation. The task-indexed wrapper chooses one branch from `problem_id`; labels and costs remain targets. Existing R68 sources and artifacts stay frozen.

3. **One contrast and fixed optimizer.** Both arms use R68A’s ordinary AdamW configuration: LR`3e-4*min(u/720,1)`, betas`(.9,.999)`, epsilon`1e-8`, weight decay`1e-4`, `foreach=False`; no SAM or parameter averaging. Keep two-forward native CE plus symmetric legal-pool RDrop with coefficient`0.35*min(u/720,1)`. A is the matched shared control; this choice holds the optimizer fixed while testing parameter sharing.

4. **Exact update semantics.** Compute the same three task losses at unchanged weights. A averages their FP32 gradients in shared coordinates. B assigns each active branch its own FP32 task gradient divided by3, equivalent to differentiating the same mean loss in disjoint coordinates. Clip the complete active gradient once at1, then call one AdamW step. All inactive branches retain `grad=None`, with no moment, parameter or weight-decay advancement. Use the common global successful-update LR clock; per-parameter Adam counters advance only when active.

5. **Data and budgets.** Reuse the exact R67 global41–80 task/row/dropout schedule, seed2, real256/task, three tasks/update, tail16/task. Fixed40 epochs give9,600 logical updates and7.2M selected rows per arm;14.4M dropout-view rows each. Each B branch receives1,600 active updates and400,000 rows. These are matched example/forward budgets, **not matched parameter capacity or per-parameter update counts**. Count inherited stage1’s9,600 updates/7.2M rows separately once.

6. **Recovery and necessary check.** Preserve whole-triple AMP recovery, restored RNG, no optimizer/cursor advancement before all gradients are finite, and20-attempt limit. One focused synthetic CPU check covers identical initialization, correct task routing/target isolation, inactive-branch immutability, the divided-by3 gradient, injected whole-update retry and complete-bundle save/load. Then formal training; no sizing or diagnostic campaign.

7. **Selection and deployment.** Evaluate all18,000 validation rows each epoch, FP32 batch128. Select maximum integer native-correct count, earliest exact tie; official arm by count, tieA. B’s checkpoint contains **all18 branches from one common epoch**. At inference, known task identity activates one selector, full-pool utility argmax uses global-ID ties, and one selected solver completes. After the parent freezes both complete models/rule/official arm, run one original full18 TEST each; reuse historical reference results. Cost remains disclosure.

8. **Feasibility and interpretation.** B has33,173,154 parameters versus1,842,953 for A. FP32 weights, gradients and two Adam moments total approximately0.49GiB for the bank, before temporary buffers; sequential task forwards preserve the measured batch256 activation shape. One selector forward remains active per query. Allow roughly2–3 hours for the pair as a planning estimate, including larger checkpoint overhead; measure ordinary runtime on the assigned UUID only. One completed-version commit follows delivery.

The strongest risk is fitting18 models from only10,000 fine-tuning examples each. A gain would support this unsharing recipe, not isolate harmful gradient interference from increased capacity and changed optimizer histories. Failure to beat matched A leaves the hypothesis unsupported; remaining below R60B means no incumbent improvement. No evidence establishes that70% is attainable. Historical TEST exposure remains disclosed; new R68 TEST did not inform this recommendation.

**Memory update:** R68 completed matched40-epoch arms. Plain AdamW A28 achieved8644/18000 validation correct,48.022222%; SAM B35 achieved8616,47.866667%,−0.155556pp against A and−1.494444pp versus R60B. SAM reduced late fitting and validation CE without improving selected native Top1; task effects split nine gains/nine regressions. Inspected initialization, SAM restoration/retry, accounting and integer-selection paths are correct; no corrective replay is required. Recommend one R69 full18 shared-versus-task-specific complete-selector fine-tuning comparison, with common initialization, fixed AdamW/data budgets and one globally selected bundle. Increased capacity, fewer examples per branch and different active parameter-update counts remain explicit limitations. New R68 TEST was excluded; historical TEST is exposed. R58 and eight legacy B-task semantics remain deferred; the full18 single-completed-solver70% goal remains active and unachieved.