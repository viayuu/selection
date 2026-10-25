# R111 Narrow Prelaunch Review

Reviewed 2026-10-25 03:17 UTC. Scope: PLAN and the actual R111 model,
runner, launcher and reporter; only their necessary shared call sites.
Same-family provisional engineering review, not efficacy evidence.
No Astra was invoked; the required Sol/max provider route was not independently
attested here, so this file must not be described as verified model routing.

## Findings By Priority

- P0/P1 blocking defects: none found in this narrow review.
- P2 warning: no GPU/NCCL run was performed. CPU checks and source reasoning
  establish engineering readiness, not observed two-GPU execution or efficacy.
- Resolved during review: the reporting owner's synthetic LIB fixture originally
  used four rows against the helper's fixed 49/100 populations. The owner fixed
  the fixture; the same three reporting tests now pass. No production-code
  defect was inferred from that fixture failure.

Minimum required code fixes: **0**. No additional audit or test matrix requested.

## Critical Checks

- Proper conditional NLL, no hard route or labels in forward: the objective
  strips supervision before forward and applies NLL directly to joint logP
  ([runner](/public/home/shiys/0selection2/code/V4/r111_experiment.py:50)).
  Every non-singleton legal family computes two private full-node EncoderLayers
  before pooling; non-reentrant checkpointing preserves RNG. Root reads shared
  features only; absent families never enter an empty conditional softmax
  ([model](/public/home/shiys/0selection2/code/V4/r111_model.py:123)).
  Wrong-family private classification gradients are zero; this is not a claim
  of zero AdamW decay/momentum updates.
- DDP normalization is correct: each rank draws its half of the same natural
  1024-row stream; local512 uses the existing R76 AMP update with microbatch
  factor `2 * rows / 1024`, followed by DDP averaging. One coordinated clip/step
  and overflow replay are retained
  ([runner](/public/home/shiys/0selection2/code/V4/r111_experiment.py:181)).
  Snapshot RNG gathering is collective; checkpoint/evaluation/status writes are
  rank0-only, including the first snapshot. Rank1's wait matches rank0's next
  snapshot collective; no unmatched collective or duplicate writer found
  ([runner](/public/home/shiys/0selection2/code/V4/r111_experiment.py:157)).
- Seed2 fresh R84 tensors and TRAIN-only geometry are validated; shared stem
  parameters are copied, not a trained selector. Fixed 57600/4320 updates,
  global1024/local512 and caps match PLAN. Strict-first minimum full VAL
  macro_vs_sbs selects one checkpoint; locked TRAIN/VAL, then TEST and LIB,
  use that same state without reselection
  ([runner](/public/home/shiys/0selection2/code/V4/r111_experiment.py:221),
  [finalize](/public/home/shiys/0selection2/code/V4/r111_experiment.py:290),
  [report bindings](/public/home/shiys/0selection2/code/V4/r111_reporting.py:94)).
- Launcher parsers match all five actual stage commands. Both original R110
  worker/controller pipelines must finish at rc0, including PDF; existing UUID
  mutexes are opened without replacement and held through CPU reporting.
  No new allocation, R109/R110 interruption, implicit resume or cache rebuild
  is on this path
  ([commands](/public/home/shiys/0selection2/code/V4/r111_launch.py:114),
  [predecessor](/public/home/shiys/0selection2/code/V4/r111_launch.py:165),
  [mutexes](/public/home/shiys/0selection2/code/V4/r111_launch.py:87)).

## Evidence And Judgment

CPU-only existing tests: model **6/6**, launcher **3/3**, reporting **3/3**.
`bash code/V4/run_v4_r111.sh check-cli`: all five stage parsers passed.
Reviewed source SHA256 prefixes: model `98a87147aefa`, runner `a0e23195c92b`,
launcher `e5d79fd8e20e`, reporter `84d0a6d35c83`. The launcher/report final
small edits and reporting-owner fixture fix were included before closeout.

**READY, 9/10 for narrow engineering prelaunch readiness.** Actual dispatch
still requires main's explicit ready evidence, both complete original R110
pipelines and original mutex release. Reaching update57600 alone is not release.
This review neither writes launch authorization nor claims empirical benefit.
Only this review file was edited; no GPU work, code fix or Git mutation.
