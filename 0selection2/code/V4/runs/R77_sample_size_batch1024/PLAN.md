# R77 Launch State

TRAIN-only CPU preparation and focused CPU checks are permitted now.
GPU launch is formally user-approved; the resource agent confirmed both cards idle
and the shared/local caches readable. Do not repeat resource/cache probes.
No R74 resumption, new solver costs, cache rebuild, DDP or git commit.

Approved allocation: job1475 / gpu05, two independent RTX4090 workers.
A: GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be.
B: GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d.
Each srun step:4 CPUs /9G; threads2 / prefetch2 / local R76 cache;
global1024 and large micro352. No other GPU UUIDs are permitted.

Fixed budget49680; linear warmup3726 then cosine1e-4 to1e-6.
Common FULL-TRAIN SBS VAL criterion; own-TRAIN SBS reporting-only.
VAL1242; TRAIN every5 VAL points and selected best; one full locked TEST/arm.

Launch after approval:
`R77_APPROVED=1 bash /public/home/shiys/0selection2/code/V4/run_v4_r77.sh start`

## Running Receipt

Both arms are now training. Slurm20.11 default step GRES claimed both job GPUs;
the queued B request and a failed zero-GRES attempt performed no B updates.
A was interrupted after this R77 run saved its own successful-update2484
checkpoint, then resumed that same R77 state at update2485. This is not reuse of
R76/historical selector weights and not a Git commit. Optimizer, scaler, RNG and
shuffle cursors were restored; architecture and optimization formulas did not change.
Both replacement steps request one GPU each:1475.17 (A),1475.18 (B).
Original logs are preserved; no sampling, init, LR, loss or budget rule changed.

tmux session:`R77_sample_size_batch1024`.
Controller pane PID2426415; monitor pane PID2426412.
gpu05 Python PIDs:A638471, B638565.
At the first parallel throughput receipt:A2880, B360 successful updates;
~0.24s/update, ~4300samples/s, ~3.5-4.5h including evaluation overhead.
CPU smoke:10 tests passed. Paired parameter/task/LR/budget/common-SBS invariants
were checked from both actual training protocols, all equal.

Monitor writes `monitor_status.json` every30s and stops with actionable failure
or paired completion. Each training worker finishes the fixed budget, best lock,
one full TEST, and saved-prediction analysis without interactive supervision.

## Context Continuation

Read root `STATUS.json` / `STATUS.md` first. A separate `root_status` tmux window
refreshes these every5 minutes, including current counters, complete VAL,
latest common-update A/B metrics, TRAIN curve assessment and live read-only GPU
telemetry for the two approved UUIDs. It never pauses or restarts a training process.

The `delivery_guard` CPU window waits for the existing final summary writers to
exit, checks saved full-budget/one-shot-TEST/unique-best artifacts, and writes
the final ALL + four families +18 task rows and `FINAL_DELIVERY.json`. No model
inference or TEST is repeated. Existing TEST-start claims must never be removed.

Healthy running processes need observation only. No new preparation, capacity
scan, solver run or GPU launch is required after context compaction. Keep both
arms at49680 successful updates and never use TEST to alter model/checkpoint rules.

## Canonical Success Gate

Root `STATUS.json` / `SUCCESS_RECEIPT.json` / `finalcomparison.csv` are fixed paths.
The `success_gate` CPU observer waits for both pipelines and controller exit0,
checks saved40 VAL points, exact49680/per-task budgets, unique minimum-VAL locks
and one full TEST/arm. It writes success only after read-only NVML confirms both
approved4090 UUIDs idle with no compute processes. No GPU is reassigned here.

Before SUCCESS_RECEIPT exists and release is verified, do not assign these cards
to NSS. Once it exists, notify the user immediately; NSS/user may then reallocate.
The final CSV includes all18/families native Top1/regret, common/own TRAIN-fixed
SBS,10000 vs70114 unique TRAIN rows/problem and the common successful-update axis.
All final reporting reads saved metrics, not a second model TEST.
