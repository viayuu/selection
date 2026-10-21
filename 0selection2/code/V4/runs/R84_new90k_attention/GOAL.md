# R84: New-90k Data And Useful Method Recall

Updated: 2026-10-18. Project goal: active, not achieved.

Current progress:

- Data release complete: `selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`.
  All 54 splits ready; 1,620,000 rows; 8,010,000 finite retained costs;
  no old TRAIN row moved to a held-out split. Distribution PDF delivered.
- Direct-attention implementation complete; focused CPU checks passed.
- Additive focal objective implemented; zero-weight objective/gradient/RNG
  equivalence and numerical tests passed.
- Cache preparation completed at 01:36 CST; 19,005,947,954 bytes were staged
  once to each allocation's local cache. Only identical old inputs reuse geometry;
  all labels and TRAIN statistics belong to the new release.
- All four fresh runs are training on four distinct authorized GPUs. The first
  full-18 update cycle passed with finite losses/gradients and zero AMP retries.
  A/C use the two 3090s; B/D use the two 4090s. The recorded startup utilization
  snapshot was 97-100%, not a guarantee of continuous utilization.
- CPU monitor `r84_goal_watch` updates `goal_progress.json` every 60 seconds.
  After all locked TEST/LIB/PDF stages, it writes the paired comparison PDF/CSV
  under `comparison_artifacts/`. It does not launch unplanned configurations.
- No R84 validation/TEST result exists at this handoff; the goal is unresolved.

## Scope

Organize all 90,000 newly generated instances per task, excluding the original
10,000. Cover all 18 tasks and retain the established B-filtered, single-ReLD
candidate pools. `reld_mt` is RELD_MOEL; no solver reruns or cost mixing.

Use 81,000 TRAIN, 4,500 validation and 4,500 TEST instances per task. Preserve
the previous validation/TEST source identities, and extend each held-out set
only from previously unused instances. Split extensions use input metadata
(scale, distribution, capacity type), seed 2, without winner-based selection.
Previously reported TEST examples are development-exposed, not a blind set.

Run a matched original-selector control and a direct-attention decoder. The
latter predicts candidate probabilities from attention logits without the
final scoring MLP. Additional training objectives require a concrete literature
motivation and a recorded protocol before the first optimizer update.

## Optimization Goal

Improve useful instance-dependent selection, including methods with genuine
winner support but zero or near-zero selections. Fewer zero-count methods alone
is not success. Evaluate full-18 macro Top1, actual relative regret and mean
cost, plus per-method true-winner recall, selection precision, introduced errors
and paired cost changes. Do not enforce uniform method frequencies or quotas.

All fitting uses TRAIN. Validation selects one checkpoint and one global rule
per version. Formal TEST and LIB reports use that locked model. No per-task
checkpoint stitching, TEST-driven recalibration or automatic coefficient grids.
Candidate acceptance requires better macro Top1 without higher macro mean cost
or actual regret against the matched control. Report all per-task changes.

## Initial Execution

| Arm | Final selection head | Extra training penalty |
| --- | --- | --- |
| A_baseline | Original scoring MLP/maximin | None |
| B_attention_direct | Mean multi-head QK logits, then softmax | None |
| C_focal | Original scoring MLP/maximin | Additive focal |
| D_attention_focal | Mean multi-head QK logits, then softmax | Additive focal |

The focal arms retain the original objective and add
`0.35 * mean((1 - p_true)**2 * (-log(p_true)))`, averaged over the two
dropout forwards. The modulation remains differentiable. Its fixed coefficient
is active from the first update; there is no class weight, frequency target,
additional warmup, coefficient search or inference calibration. This is a
CE-plus-focal adaptation, not the original paper's standalone focal objective.
See `docs/research/r84_zero_selection_literature.md` for motivation and limits.

- Seed: 2; logical batch: 1024; natural task/example sampling.
- Shared fresh initialization where architecture parameters correspond.
- Original AdamW/dropout/R-Drop and update-axis schedule are the control.
- 57,600 successful updates; validation every 1,440; warmup 4,320.
- Idle authorized GPUs may run independent arms in parallel.
- Final artifacts: PDF, machine-readable CSV, checkpoints, protocol and logs.
- Each completed meaningful version receives one consolidated GitHub commit
  when publication is requested; no intermediate commit churn.

## Continued Research

After each completed round, inspect validation-based mechanism evidence and
the complete published results with the stated TEST-development boundary.
Select a substantively different, evidence-backed next hypothesis when useful;
do not automatically extend budgets, scan penalties or retry negative variants.
Keep this goal unresolved until its acceptance criteria are met or the user
pauses it. Training launch or literature discovery does not complete the goal.

Paper research and improvement ideas: `gpt-6-astra`, reasoning `max`.
Implementation, experiment planning, code review, results and monitoring:
`gpt-6.1-sol`, reasoning `max`.

The application-level historical goal remains paused and unchanged. Creating a
new application goal returned `unfinished goal`; this project file records the
newly authorized work without falsely completing or resuming the old R58 goal.
