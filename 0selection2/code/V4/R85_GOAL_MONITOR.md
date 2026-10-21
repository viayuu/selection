# R85 CPU Goal Watcher

`code.V4.r85_goal` reads small receipts every 60 seconds, quietly. It does not
start or retry any experiment, touch frozen core sources, notify anyone, or mark
the active research goal complete. An error is written to `goal_progress.json`
and stops the watcher for manual repair; successful experiments remain intact.

Progress: `code/V4/runs/R85_zero_selection_research/goal_progress.json`.
Watcher receipt: `goal_watch_receipt.json`, including its actual PID.
Tmux session: `r85_goal_watch`. The log is `goal_watch.log` in the same root.

After all four arms finish training, unique VAL lock, full TEST, both LIBs and
their per-arm PDFs, it creates `comparison_artifacts/COMPARISON.pdf`, metrics,
same-instance paired changes versus R84A, and B versus the matched R85A OVR
control. It reuses the reporter's saved-prediction loaders/comparison functions.

Targets are copied from the published R84 comparison `target_list.json`: 36
zero/near-zero groups and 4,625 VAL winners. Strict-zero and combined target
results are separate. Recall, precision, true rescues, lost winners, new false
picks and regret effects accompany counts. This list is posthoc/descriptive;
it does not change TRAIN sampling or checkpoint selection.

An empirical screen passes only if unrounded full18 VAL Top1 improves over
R84A while mean cost and actual regret do not increase. TEST reports the same
already locked checkpoint and does not select a checkpoint or rule. A small
pass keeps the goal active; an all-negative result requests
`research_handoff_needed`. All18 macros are equally weighted. All labels use
saved native/FP64 winners on the exact NEW90k release (4500 held-out rows/task).

Run in `easynco`, from `/public/home/shiys/0selection2`:

```bash
python -B -m code.V4.r85_goal status
python -B -u -m code.V4.r85_goal watch --interval 60
python -B -m code.V4.r85_goal report
```

`status` never reads checkpoints or prediction arrays. `report` is CPU-only and
requires all completions. It does not modify any experiment's core files.
