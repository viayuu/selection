# R89 CPU Monitor (Prepare Only)

The monitor is prepared but **not started**. Main owns the actual start after
the experiment controllers are launched. No real run receipt is created by
preparation or by the temporary-root tests.

- Arms: `A_adamw`, `B_muon_hybrid`.
- Root: `/public/home/shiys/0selection2/code/V4/runs/R89_optimizer_geometry`.
- Launch root: `/public/home/shiys/0selection2/code/V4/runs/R89_launch`.
- Quiet interval: 60 seconds; CPU-only, CUDA hidden, numerical threads capped
  at two. Reuses `r84_goal.arm_snapshot`, including the common 57,600-update
  completion receipt and unique VAL / full TEST / LIB / per-arm PDF requirements.
- Saved phases are evidence of recorded progress, **not process or GPU liveness**.
- Missing data/receipts and new optimizer setup phases are normal waiting, not
  failures. Only explicitly failed stage receipts or monitor/comparison errors
  stop the monitor for manual intervention.
- No GPU launch, inference, optimizer/model loading, automatic retries or
  research-goal completion. Goal stays `active_unresolved`.
- No R88/R87/R85 files or processes are changed.

## Reporter Contract

The runner-owned reporter landed during preparation. Its actual CLI and
completion output paths were confirmed from source (without executing it):

```bash
python -u -m code.V4.r89_reporting compare --root <root> --device cpu
```

If both arms finish before the reporter appears, the monitor normally waits
in `waiting_for_reporter`. Once available it executes the command only once.
Success requires return code zero, `comparison_artifacts/protocol.json` with
`complete=true`, and nonempty `comparison_artifacts/COMPARISON.pdf`.
An unfinished/failed comparison-attempt receipt is retained and never retried
automatically, including after a monitor restart.

## Main-owned Start

From `/public/home/shiys/0selection2`, first activate `easynco`:

```bash
python -B -u -m code.V4.r89_goal watch --interval 60
python -B -m code.V4.r89_goal status
python -B -m unittest code.V4.test_r89_goal
```

Main may run the first command in `tmux r89_goal_watch` and redirect its quiet
output to `<root>/goal_watch.log`. Do not interpret the preparation as a start
authorization or as evidence that a monitor is already running.

Only after an explicit invocation, monitor-owned run artifacts are:

- `<root>/goal_progress.json`
- `<root>/goal_watch_receipt.json` (actual PID when started)
- `<root>/goal_watch.lock` (CPU singleton only, not a frozen experiment lock)
- `<root>/goal_compare_receipt.json` and `goal_compare.log` after readiness
- `<root>/goal_watch_error.json` only for monitor/comparison exceptions

The existing CPU reporter owns the comparison PDF/CSV/protocol outputs.
