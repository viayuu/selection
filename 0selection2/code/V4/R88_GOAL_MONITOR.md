# R88 CPU Watcher

Quiet 60-second polling reuses `r84_goal.arm_snapshot` for
`A_canonical_anchor` and `B_execution_anchor`. Only small saved stage receipts
are read while waiting; no weights, GPU probes, inference or training.

- Root: `code/V4/runs/R88_execution_role_inputs`.
- Launch root: `code/V4/runs/R88_launch`.
- Ready requires both arms' completed training, unique VAL lock, full TEST,
  LIB and per-arm PDF receipts/files, as defined by the existing shared helper.
- Queue waiting is not failure. Explicit stage errors stop the watcher and
  retain the original evidence for manual intervention.
- When both arms are ready, execute once:
  `python -u -m code.V4.r88_reporting compare --root <root> --device cpu`.
- Normal completion requires `comparison_artifacts/protocol.json` with
  `complete=true` and nonempty `comparison_artifacts/COMPARISON.pdf`.
- No automatic stage/command retries, including unfinished comparison attempts
  left by an interrupted watcher. Goal remains `active_unresolved` throughout.
- R85/R87 monitors, training cores, launchers and frozen locks are untouched.

From `/public/home/shiys/0selection2`, after activating `easynco`:

```bash
python -u -m code.V4.r88_goal watch --interval 60
python -m code.V4.r88_goal status
python -m unittest code.V4.test_r88_goal
```

The watcher runs in `tmux r88_goal_watch` with CUDA hidden and numerical CPU
threads capped at two. Its only directly owned run artifacts are:

- `<root>/goal_progress.json`
- `<root>/goal_watch_receipt.json` (actual CPU watcher PID)
- `<root>/goal_watch.log` (quiet while waiting)
- `<root>/goal_watch.lock` (CPU singleton only, not a frozen experiment lock)
- `<root>/goal_compare_receipt.json` and `goal_compare.log` after readiness
- `<root>/goal_watch_error.json` only for watcher/comparison exceptions

The existing CPU reporter owns the comparison PDF/CSV/protocol outputs.
