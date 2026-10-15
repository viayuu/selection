# Round003 Bounded CPU Execution Checks

Owner: Solmax. Five synthetic unittest cases passed in1.658s on easynco.
Raw output: `synthetic_checks.log`. Shell syntax: `bash -n` passed once.
No previous18-test suite, gradient certification, GPU probe, profile or extra
code-review loop was run. This is same-family execution evidence, not an
independent external review or a claim of GPU readiness.

1. Original FP64 tiny gaps survive before conversion; a_t is taskwide and
   averages valid-K row means, not an extra per-instance normalization.
2. Centered masked softplus is score-shift invariant; invalid candidates have
   no gradient; all-tie TRAIN scale gives an exactly zero risk block.
3. Control equals the inherited objective; CE/compare/KL/focus parts are
   unchanged; both branches perform two forwards with all labels excluded.
4. Exact1024 synthetic rows accumulated at micro128 or1024 give one fresh
   AdamW step and one gradient clip; the two-forward convention is preserved.
5. Late LR/endpoints match the fixed34776..49680 axis; task counts are828
   each; incomplete18 VAL and any positive raw cost-gate violation are rejected.

CPU preparation completed2026-10-14T21:40:04.864107+00:00, using existing TRAIN
cost arrays and full-B TRAIN/VAL metadata, no model forward or TEST labels.
All18 tasks retain70114 TRAIN rows and89 total native task-candidate entries.
Each original cost array is FP64; r and a_t are computed there before immutable
FP32 risk supervision is saved. The complete14904-rate FP64 shared LR sequence
is saved once: first2.4543907940693327e-5, final1e-6. See `prepared.json`,
`task_scales.json`, `late_lr.npy` and `prepare.log`.

The genuine continuation update count remains0. Hardware approval is false;
NSS has priority. One future approved-GPU smoke is pending, not executed or
implicitly passed. No model weights from a smoke may become trial initialization.
The global file workflow is active/ready_pending_hardware. Legacy app paused
and historical child app blocked flags were not changed or re-queried.
