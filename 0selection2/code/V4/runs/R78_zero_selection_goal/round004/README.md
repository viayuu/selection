# R78 Round004

Approved bounded CPU fit on frozen R77 full-B ep28/u34776 saved TRAIN/VAL scores.
Implementation: code/V4/r78_frozen_score_cost_head.py.
Launcher: bash code/V4/run_v4_r78_frozen_score_cost_head.sh start.

All18 tasks /89 candidate entries /513 zero-initialized D parameters. Masked FP32
softmax at T1; original FP64 TRAIN regret and task scale; FP64 head/loss. Fixed
unit prior ||D||^2/(2N), fresh AdamW weight_decay0, seed2 natural permutations,
1380 successful logical batch1024 updates/task, cosine .01 to boundary0.
No gradient/optimum certificate or budget extension.

Only two synthetic checks and the real first TRAIN batch finite check (which
is update1 of the declared budget). The single all18 matrix bundle is locked
before the only fitted VAL endpoint. B, actual fitted and whole-rule fallbackB
metrics are separate. CE remains frozen-B CE, not softmax(-g).

No GPU/backbone/solver/API/TEST/LIB calls; W&B is offline with system stats
disabled. Gate failure closes only this finite route and retains B. Gate pass
requires a separate owner approval before any formal TEST/LIB release. Overall
file goal stays active; no native or legacy app-goal status is mutated.

## Completed Execution

PID4183038 on admin01 /CPU threads2. Both the PID and tmux session were checked
absent after completion; no jobs were stopped. Launch-to-results27.480857s,
TRAIN fitting21.500315s. Exact process-exit wall time was not captured; offline
W&B cleanup followed result writing. See final_receipt.json.

Complete native TRAIN70114 rows/task (1262052 total) and VAL3895 rows/task
(70110 total), all18 tasks /89 candidate entries. Each task completed1380
successful batch1024 updates,24840 total; continuous permutations consumed
1413120 row draws/task,25436160 total. No extra updates or dropped boundary tail.

The single candidate failed all three fixed VAL gates. Actual fitted TRAIN/VAL
results remain distinct from B and whole-version fallbackB. This finite route
is closed negative; no TEST/LIB, extra budget, temperature/accuracy variant or
automatic next trial. File goal remains active, waiting for Astra evidence
direction, expected NSS TRAIN/VAL coverage only.

## Output Index

- RESULTS.md: final B /actual fitted /fallbackB TRAIN and VAL table and gate.
- results.json: complete all18 native metrics, cold-method summary and fit statistics.
- all18_TRAIN_VAL_metrics.csv: per-task native Top1 /cost /regret /frozen-B CE /VS.
- native_method_counts.csv: all89 task-candidate native supports and selection counts.
- decision_changes.csv: corrected /harmed /wrong-to-wrong counts and FP64 positive/negative cost changes per task.
- decision_change_costs_by_category.csv: separate FP64 cost changes for those three switch categories; no cross-task raw-cost pooling.
- locked_heads.npz and head_lock.json: unique frozen whole18 matrix bundle, identities, TRAIN scales and pre-VAL lock.
- protocol.json and launch_receipt.json: fixed plan /source /optimizer /sampler /launch evidence.
- final_receipt.json and progress.json: full24840 budget, complete row counts, PID exit and workflow disposition.
- loss.jsonl and train.log: bounded TRAIN fitting trace and final execution log.
- CHECKS.md and first_batch_check.json: two synthetic checks and the budget-included real first batch finite record.
- wandb/: local offline tracking only; no external API or dashboard release.
