# R81B one-shot queue (proposal alias: requestedR80B)

Only `B_cross_instance_rank_seed2` is authorized. Existing R80/B_batch512 and
all other R80 processes/receipts are read-only to this queue. No A, extra seed,
new allocation, direct SSH training, automatic restart, or Git submission.

## Current Gate

**ARMED, pending** as of 2026-10-15 15:30:12 UTC (23:30:12 Asia/Shanghai).
Main-thread final ready was received; `prepared/code_ready.json` confirms 15/15
objective/runner CPU tests and TRAIN support readiness, with the stable
`code/V4/R81_INTERFACE.md` contract. `queue_ready.json` was created at
2026-10-15 15:29:40 UTC. Do not repeat the one-shot arm command.

Detached tmux `r81b_once` runs on admin01, Python PID `2786026`, pane `%212`.
All four R80 full-pipeline receipts remain `running`, so no candidate has been
claimed and no R81 training or CUDA context has started. The queue checks only
resources every 60 seconds; remote GPU queries wait until a full R80 pipeline is
terminal. It does not collect model metrics or occupy GPU memory while pending.

The original preparing-only PID `2764255` was safely replaced before arm after
the NVML context fix. Its receipts are preserved as
`queue_tmux_before_nvml_fix.json` and `queue_status_before_nvml_fix.json`;
`queue_standby_refresh.json` records the no-ready/no-claim guard, exact old/new
PIDs and loaded source hash. No R80 process or allocation was changed.

The machine-readable current state is `queue_status.json`; the exact argv and
locked budget are in `queue_plan.json`. `queue_selftest.json` records the small
pure-CPU queue tests. These are not loss equivalence tests or training results.
Runner/objective preparation and tests remain owned by their respective workers.

Queue checks: **10/10 PASS**, pure CPU, no training subprocess or CUDA context.
The actual runner and reporter parsers accepted all four candidate-card argv with
runtime `/tmp` cache, batch1024, steps49680 and warmup3726. Receipts:
`queue_selftest.json`, `queue_cli_check.json`, `queue_cpu_baseline.json`.
An actual duplicate-loop invocation was rejected with exit 2 by the global
flock; the then-current standby remained alive. CUDA was not initialized.
`queue_nvml_node_probe.json` preserves the bounded host-query exit-6 evidence;
`queue_nvml_container_probe.json` records the corrected read-only resource probe
PASS through the immutable existing allocation container. No training probe ran.

## Commands

Every shell starts in the repository with the easynco environment activated:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
bash code/V4/run_v4_r81.sh dry-run
bash code/V4/run_v4_r81.sh selftest
bash code/V4/run_v4_r81.sh standby
bash code/V4/run_v4_r81.sh status
```

`standby` is idempotent while the same tmux exists. A missing tmux with an
existing receipt is NOT automatically restarted. Duplicate queue processes are
rejected by a global flock; `queue_claim.json` is created with exclusive create
before any launch and permanently prevents a second attempt, including failure.

After main has explicitly confirmed objective/gradient tests AND runner prepare
ready, the queueworker executes this command with the actual main-thread evidence
and runner preparation receipt (not before that confirmation):

```bash
bash code/V4/run_v4_r81.sh selftest
bash code/V4/run_v4_r81.sh arm --objective-tests-passed --runner-tests-passed --preparation-receipt /public/home/shiys/0selection2/code/V4/runs/R81_method_conditioned_auc/prepared/prepared.json --ready-evidence 'ACTUAL MAIN-THREAD READY EVIDENCE'
bash code/V4/run_v4_r81.sh status
```

Equivalent arm helper: `bash code/V4/runs/R81_method_conditioned_auc/launch_arm.sh
'ACTUAL MAIN-THREAD READY EVIDENCE'`. This must only run after explicit final ready.

The ready gate freezes source/preparation hashes. Later source changes block
launch; they do not silently authorize modified code. No GPU probe training is
used. `cpu-baseline` optionally captures old per-GPU CPU affinities once via two
small existing-allocation CPU-only steps; it never queries CUDA/NVML.

CPU-only baseline was successfully captured from both existing allocations on
2026-10-15. Both container IDs, 20 GiB limits and ten-CPU allowed sets match.
Some historical host-step affinities are outside the current allocation cpuset;
these are not reused. The queue always intersects with current allowed CPUs and
excludes all live partner process affinities, without changing any R80 affinity.

## Resource Gates

The candidate order is gpu03 GPU0, gpu03 GPU1, gpu05 GPU0, gpu05 GPU1. All are
already user-authorized; whichever first passes every gate is claimed once:

| Old R80 group | Job / Node | Authorized GPU UUID |
| --- | --- | --- |
| A_ce_only | 1465 / gpu03 | GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 |
| C_lr3e4 | 1465 / gpu03 | GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d |
| B_batch512 | 1475 / gpu05 | GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be |
| D_code_embedding | 1475 / gpu05 | GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d |

1. Old R80 launcher receipt is terminal `complete` or `failed`. Complete must
   include train, full TEST, and LIB/PDF with exit 0. A training marker, utilization
   0, or the absence of a process for an instant never suffices. Failed R80 is
   preserved and never restarted. Host process scan must show no live old worker
   or stage, including container namespace PIDs.
2. Original job is still RUNNING, same owner/node, CPU10/mem20G/GPU2. Existing
   allocation-owned container must be running with its exact original ID and
   <=20 GiB memory cap. No new allocation is requested.
3. NVML (`nvidia-smi`, not CUDA) reports no compute process on that exact UUID and
   at most 512 MiB used. Utilization is informational, never the idle criterion.
   Host/container ordinal indices may differ; immutable UUID is authoritative.
   Receipts store the observed `nvml_index` separately from the authorized logical
   index. Both node-side and in-container checks use this same UUID/type rule.
   A bounded read-only CPU-step probe on 2026-10-15 confirmed direct host NVML
   fails (exit 6). Node probes therefore run `nvidia-smi` through the verified
   allocation-owned container, by immutable container ID; not in the CPU cgroup.
4. Four CPUs must be in both the job cpuset and the existing container cpuset,
   disjoint from the live partner. Prefer saved old per-GPU affinity; use remaining
   permitted CPUs only if needed. Each pipeline requests CPU4/mem9G, so two fit
   the existing CPU10/mem20G allocation.
5. After exclusive claim, repeat the resource checks in the CPU srun child and
   check cgroup, UUID, affinity, ready hashes and GPU idleness inside the container
   immediately before the first runner child. A mismatch fails closed, no retry.

Actual transport is `srun --jobid=1465|1475 --overlap --exact --gres=none
--cpus-per-task=4 --mem=9G` into `sudo -n /usr/bin/ai_docker exec -u UID` of the
existing container. The CPU-only srun itself never runs CUDA. GPU work occurs
only inside the allocation-owned container, with exactly one UUID in
`CUDA_VISIBLE_DEVICES`. No SSH training or inherited GPU2-GRES blocking step.

Containers are identity-locked:

- gpu03: `1465_gpu03`, `b8ced5d8a8cdafdc5006f827a0d36c319c61056a65316af4ff9937453fa0635c`.
- gpu05: `1475_gpu05`, `76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6`.

Reuse `/tmp/shiys_R76_R45A_single_reld_mt_cache` in the container: 375 files,
17,026,558,189 bytes. Validate metadata identity and array sizes, never rebuild,
copy, run solvers, or inspect heldout cost values for scheduling.

## Pipeline And Inspection

Budget: seed2, fresh untrained R77 identical initialization, logical batch1024,
49680 successful updates, warmup3726, LR1e-4 cosine to1e-6. Objective, unique VAL
selection, no TEST/LIB tuning and inference semantics follow R81_INTERFACE and
the runner preparation. Training CLI runs through unique VAL lock and full TEST;
the reporting CLI then does both LIBs and PDF using that same lock/checkpoint.
The GPU reservation spans the entire sequence, never only the training stage.
No `--resume` or automatic retry is passed. Every stage requires exit 0.

W&B is enabled, project `selector`, entity
`yjkds-southern-university-of-science-technology`. Use standard environment,
settings and netrc credential lookup. Missing credentials mean explicit recorded
offline mode, with no login prompt or key request; credentials are never logged.

`queue_status.json` and `launch_*.json` contain tmux, controller/worker/child PID,
UUID, affinity, stage, command, log path, exit and reservation state. After launch,
see `B_cross_instance_rank_seed2/train.log` and `reporting.log`. Queue tmux is
`r81b_once`; `tmux capture-pane -p -t r81b_once -S -40` shows launcher output.
The launcher only waits for its stages and records exits, without collecting new
model metrics, starting monitoring daemons or iterating on the experiment.
