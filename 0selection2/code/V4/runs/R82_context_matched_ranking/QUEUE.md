# R82B One-Shot Resource Queue

Only `B_context_matched_rank_seed2` is permitted. No A, extra seed, resume,
automatic retry, new allocation, solver run, R80 modification or Git submission.

## Readiness And Priority

Current standby: created 2026-10-16 00:23:36 Asia/Shanghai, tmux `r82b_once`,
pane `%213`, PID `2884671`. It is unarmed and has not claimed a GPU or started
training. Queue CPU tests: 7/7 PASS. Actual train/report CLI: all8 argv PASS.
These are resource/CLI checks, not a substitute for main's objective and runner
tests or TRAIN pair-coverage confirmation. The arm helper reruns the small
queue/CLI checks against the final source hashes before creating readiness.

One bounded, manually requested read-only coexistence probe also passed:
gpu05's second UUID was idle (2 MiB, no compute process); live R81 CPUs were
`0,1,72,73` and the R82 proposed CPUs were `2,3,4,74`, with no overlap.
`queue_coexist_probe.json` stores the real allocation/container/NVML evidence.
It created no CUDA context or training process. The unarmed background queue
itself still performs no automatic remote probe and cannot launch before ready.

The detached standby session is `r82b_once`. Before main-thread final readiness
it is **unarmed**: no automatic resource probe, CUDA context, GPU reservation or training.
Main must first confirm focused objective/runner tests and TRAIN-only batch1024
context-pair coverage. `prepared/prepared.json` and `prepared/pair_coverage.json`
are hashed with the exact source files when arming. Code or receipt changes
thereafter block launch rather than silently authorizing different code.

Earlier R81B has priority. R82B waits for a valid R81 first claim, then excludes
that claimed UUID and can use another genuinely idle authorized card. Both
queues hold the **same per-UUID flock** under
`runs/R81_method_conditioned_auc/queue_gpu_GPU-*.json` throughout training,
unique-VAL checkpoint selection, full TEST, both LIBs and PDF reporting.
This includes intervals between stages when GPU utilization might temporarily
be zero. No existing worker is interrupted or restarted.

## Existing Resources Only

| Original R80 slot | Existing job/node | Exact GPU UUID |
| --- | --- | --- |
| A_ce_only | 1465/gpu03, RTX 3090 | GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 |
| C_lr3e4 | 1465/gpu03, RTX 3090 | GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d |
| B_batch512 | 1475/gpu05, RTX 4090 | GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be |
| D_code_embedding | 1475/gpu05, RTX 4090 | GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d |

A slot qualifies only after the full old R80 pipeline is terminal, no old
stage remains alive, the original allocation/container identities still match,
NVML reports no compute process and at most 512 MiB used, and four CPUs are
available within the allocation/container cpuset without overlapping a live
partner. UUID, not the possibly different host/container ordinal, is authoritative.

Transport reuses the proven allocation-bound route:
`srun --jobid=1465|1475 --overlap --exact --gres=none` with CPU4/memory9G,
then `sudo -n /usr/bin/ai_docker exec -u UID` into the original verified
allocation container. Direct SSH training is never used. CUDA receives only
the single claimed UUID; the worker validates its container cgroup and CPU
affinity before launching. CPU-only srun is never used for direct CUDA access.

The existing container-local frozen cache
`/tmp/shiys_R76_R45A_single_reld_mt_cache` is reused without copying or rebuilding.
W&B uses standard existing credentials, or explicit recorded offline mode if
none exist. No credentials are printed or saved by this queue.

## Protocol And Commands

Fresh seed2, logical batch1024, 49,680 successful updates, warmup3,726,
CMR maximum weight0.10, unchanged R77 base objective and maximin inference.
The runner owns training, VAL selection and full TEST. The reporter owns both
LIBs and PDF, using that same selected checkpoint. Only exit0 advances stages.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
bash code/V4/run_v4_r82.sh dry-run
bash code/V4/run_v4_r82.sh selftest
bash code/V4/run_v4_r82.sh cli-check
bash code/V4/run_v4_r82.sh standby
bash code/V4/run_v4_r82.sh status
```

Only after actual main-thread final readiness:

```bash
bash code/V4/runs/R82_context_matched_ranking/launch_arm.sh 'ACTUAL MAIN-THREAD FINAL-READY EVIDENCE'
```

`queue_status.json` distinguishes preparing, waiting for R81 priority, waiting
for another free GPU, launching, running and terminal states. `queue_claim.json`
permanently prevents a second launch, including after failure. A missing/dead
tmux with an existing receipt is not automatically restarted.

After launch, progress is in
`B_context_matched_rank_seed2/train.log`, `status.json` and `updates.jsonl`;
`reporting.log` follows training and TEST. Receipts include actual UUID,
container/host PID, stage, CPU affinity and exit status. The queue polls only
resources every60s until launch and then waits for its children; it does not
run ongoing model monitoring or launch further experiments.

## R81 Resource-Only Repair

The real read-only probe identified the platform's stdout prefix:
`Distribute Task Container.` and `task_id:1475`. R81 formerly parsed these as
CSV GPU rows and therefore could not claim an idle card. The resource-only
repair strips exactly this known prefix and still rejects unexpected output.
It also adds the shared UUID flock and queued-worker CPU-affinity accounting.
Model, objective, data, initialization, budget and prepared artifact hashes
were unchanged. Original readiness and pre-claim evidence were preserved in
R81 `queue_resource_fix_before.json`; refresh is documented in
`queue_resource_fix_refresh.json`. Only the unclaimed standby was replaced,
not any training process. R81 subsequently claimed gpu05's first UUID and
started its first training attempt. R82 remains subject to its own final gate.
