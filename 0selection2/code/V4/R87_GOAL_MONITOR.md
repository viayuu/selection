# R87 CPU Watcher

This quiet watcher reuses `r84_goal.arm_snapshot` every 60 seconds. It reads
small saved receipts only; it never reads model weights, probes GPUs, launches
training/inference, retries failed stages, or completes the research goal.

- Arms: `A_shared_residual`, `B_private_residual`.
- Root: `code/V4/runs/R87_private_method_residual`.
- Launcher receipts: `code/V4/runs/R87_launch`.
- B waiting for predecessor delivery is normal, not an error.
- Ready requires both arms' completed training, unique VAL lock, full TEST,
  LIB and per-arm PDF receipts/files, using the existing shared helper.
- On readiness it executes exactly once:
  `python -u -m code.V4.r87_reporting compare --root <root> --device cpu`.
- Normal exit requires `comparison_artifacts/protocol.json` with
  `complete=true` and a nonempty `comparison_artifacts/COMPARISON.pdf`.
- An explicit stage or comparison error stops the watcher and retains receipts
  for manual repair. It never retries automatically, including after restart
  with an unfinished/failed comparison-attempt receipt.
- Goal state remains `active_unresolved`, even after comparison succeeds.

Run from `/public/home/shiys/0selection2`, in `easynco`:

```bash
python -u -m code.V4.r87_goal watch --interval 60
python -m code.V4.r87_goal status
python -m unittest code.V4.test_r87_goal
```

The long-lived command runs in `tmux r87_goal_watch`, with CUDA hidden and
numerical CPU threads capped at two. Progress and watcher-owned receipts:

- `<root>/goal_progress.json`
- `<root>/goal_watch_receipt.json` (actual watcher PID)
- `<root>/goal_watch.log` (quiet while unchanged)
- `<root>/goal_watch.lock` (CPU watcher singleton only, not a training lock)
- `<root>/goal_compare_receipt.json` and `goal_compare.log` after readiness
- `<root>/goal_watch_error.json` only on watcher/comparison exceptions

No training protocol, launcher, frozen source, or R85 watcher is modified.
