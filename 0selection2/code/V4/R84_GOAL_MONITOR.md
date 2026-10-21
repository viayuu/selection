# R84 CPU Goal Monitor And Matched Comparison

This module owns only `r84_goal.py`, its focused tests, `goal_progress.json`,
the watcher receipt/lock, and `comparison_artifacts/`. The main thread owns
`runs/R84_new90k_attention/GOAL.md`. No goal API, GPU query, model import,
training launch, inference, solver execution, or automatic retry is performed.

User-facing progress: `runs/R84_new90k_attention/goal_progress.json`.
The authorized watcher is running in tmux `r84_goal_watch`; its process receipt
is `runs/R84_new90k_attention/goal_watch_receipt.json`, with quiet output in
`goal_watch.log`. After all four arms finish, the cross-arm primary report is
`runs/R84_new90k_attention/comparison_artifacts/COMPARISON.pdf`.

## Commands

Activate the experiment environment first:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
python -B -m code.V4.r84_goal status
python -B -m code.V4.r84_goal watch
python -B -m code.V4.r84_goal report
```

`status` reads small JSON receipts and updates `goal_progress.json`; it imports
neither numpy, matplotlib nor torch. `watch` polls every60 seconds, quietly.
Only A's final VAL lock can trigger the one-time target freeze. Only all four
locked full TEST, LIB, and per-arm PDF completions can trigger a comparison.
The watcher ends after comparison or an explicit error; it never retries a
failed training/report stage. `--max-polls 1` is a bounded one-shot watcher.

The main thread authorized a persistent CPU watcher after implementation. It
can be hosted in tmux `r84_goal_watch`, with output `goal_watch.log` in the R84
root. Do not create duplicate watchers; `goal_watch.lock` prevents a duplicate
active process. This persistent watcher does not authorize a fifth experiment.

## Fixed Analysis

The four arms are `A_baseline`, `B_attention_direct`, `C_focal`, and
`D_attention_focal`. All use fresh initialization, seed2, logical batch1024,
and the same81000/4500/4500 population per task. Saved predictions are compared
with the release's FP64 costs and native winner. Decision ties use the recorded
ascending global solver ID. No checkpoint is loaded.

The target list is frozen from A's locked VAL: native winner count greater than
zero, chosen share <=0.1%. Zero and near-zero groups are distinguished. This
baseline-dependent list is a **posthoc descriptive mechanism analysis** with a
predeclared threshold. It neither changes TRAIN sampling nor supplies a new
checkpoint or inference rule. The same list is used on TEST without additions.

For every method, report true winner recall, selection precision, new false
picks and their regret contribution. For each task and candidate arm, report
corrected/harmed predictions, changes among both-wrong cases and signed net
cost/regret changes. Raw costs are not summed across different problems for
acceptance. ALL is the equal18-task macro mean; mixed raw mean_cost remains
descriptive. Undefined precision/recall uses an empty CSV field, not zero.

Empirical acceptance uses **unrounded VAL only**:

1. Macro Top1 strictly exceeds A_baseline.
2. Macro mean_cost does not exceed A_baseline.
3. Macro actual regret does not exceed A_baseline.

TEST has a separate published column for each already locked version. It is
never used to choose a checkpoint or an acceptance rule. Historical TEST rows
have participated in research; this is not a fresh blind-test claim. LIB metrics
and distribution PDFs belong to the per-arm reporter, and are not recreated by
this comparison code.

## Outputs

- `goal_progress.json`: current phases, updates, missing completions, errors and
  next action, with `goal_complete=false` throughout.
- `comparison_artifacts/COMPARISON.pdf`: compact full18, gate, target and paired
  task tables; PDF is the primary cross-arm deliverable.
- `metrics_by_problem.csv`: separate VAL/TEST, ALL and all18 tasks, including
  TRAIN-fitted SBS and oracle metrics.
- `paired_changes.csv`: per-task corrected/harmed/both-wrong cost changes.
- `method_precision_recall.csv`: all legal task/method groups for every arm.
- `target_list.json,csv`, `target_summary.csv`: frozen target population,
  winner rescues/losses, precision/recall and new false-pick costs.
- `paired_gates.csv`, `protocol.json`, `RESULTS.md`: unrounded gates, prediction
  and checkpoint identities, interpretation, and next action.
- `error.json`: actionable comparison failure, with no automatic retry.

If no candidate passes VAL acceptance, `next_action=research_handoff_needed`.
If one passes, `next_action=review_validation_gain`. Neither state declares the
continuing research goal achieved or starts future training. The main thread
decides future work independently after reading the completed evidence.
