# R113/R114 Deferred Wave

This plan is not main approval. The controller may wait before the final source
freeze. It does not call srun, Docker, a solver, or CUDA until `ready.json` exists.
Do not create that receipt until the R113 finalize/reporting owner and the R114
queue owner have confirmed their final sources. R112 finisher is separate.

## Jobs And Resources

Priority is C_transition, B_snapshot, A_scratch, R114_labels. C/B use the first
available existing 4090 slots. A and R114 use the remaining slots. Jobs progress
independently; one failure never cancels a healthy peer. Every step requests
4 CPUs and 9 GiB in allocation 1465/gpu03 or 1475/gpu05, using
`srun --overlap --exact --gres=none`, then the original allocation container via
`ai_docker exec -u 2022`. No new allocation or SSH is permitted.

The container worker verifies its cgroup, full container ID, uid, allocation,
CPU affinity and physical UUID. It holds the existing original r81 mutex inode
through every child. CUDA visibility is a single physical UUID. Busy mutexes,
live compute processes, >512 MiB GPU memory, or occupied slot CPUs defer only
before initialization. No process is stopped. No existing mutex is rewritten.

Each R113 job runs `r113_witness` exactly once before formal work. This is the
bounded four-TRAIN-stress-batch check, plus one B/C process-pretrain batch, not
R112's witness transport. Failure preserves its receipt and ends that job.
The protocol owns selector/pretrain caps and evaluation size identically on
all GPUs; no launcher cap override/shrink is supported.

A: capacity -> train -> finalize. B/C: capacity -> pretrain -> train -> finalize.
Finalization is the runner's unique locked-checkpoint TEST/timing/LIB/PDF path,
including `process_diagnostics_complete.json` and its JSON/CSV/PDF artifacts.
The shared `process_diagnostic_inputs.json` is frozen with all runtime sources.
The existing `/tmp/shiys_R84_new90k_cache` is read-only, never copied or rebuilt.

R114 is one GPU worker, not an allocation or another GPU queue. Main must supply
its final grouped `plan.json` and per-task timeout (3600 seconds); no historical
plan is auto-selected. The worker directly invokes `code.V4.r114_run_queue
--plan P --manifest M --deployment-lock L --host HOST --gpu-uuid UUID
--cpu-ids LIST --timeout-seconds 3600 --execute --finalize`. The queue owns
necessary runtime gates, per-problem fail-fast, partial mask export/assembly
and read-only response analysis. Missing/blocked work is not success.
There are no extra solver preflight jobs or probe-cost label substitutions.
`R114_CPU_IDS` is the first two assigned slot CPUs; the scheduler is pinned with
taskset and OMP/MKL/OPENBLAS/NUMEXPR are all set to one numerical thread.

## CPU Checks

All commands below run on admin01. The wrapper activates easynco itself.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
python -B -m unittest code.V4.test_r113_wave -v
bash code/V4/run_v4_r113_wave.sh plan
bash code/V4/run_v4_r113_wave.sh status
```

## Start Waiting, Without Approval

This standalone tmux command is safe before main approval and before R112 ends.
Do not run a second controller over existing controller/attempt receipts.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
tmux new-session -d -s r113_r114_wave 'bash /public/home/shiys/0selection2/code/V4/run_v4_r113_wave.sh wait --poll-seconds 60'
tmux set-option -t r113_r114_wave remain-on-exit on
bash code/V4/run_v4_r113_wave.sh status
```

## Main Ready, After Final Source Freeze

The R114 owner confirmed the following exact plan and timeout. `check-cli`
performs finite CPU-only parser/protocol checks and reports
`protocol_sha256`, `source_set_sha256`, and `r114.plan_sha256`. It does not run
the witness, train, bind a GPU, or execute runtime solver gates.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
R114_PLAN=/public/home/shiys/0selection2/code/V4/runs/R114_paired_responses/preparation/grouped_execution/queue_v1/plan.json
R114_TIMEOUT=3600
bash code/V4/run_v4_r113_wave.sh check-cli --r114-plan "$R114_PLAN" --r114-timeout-seconds "$R114_TIMEOUT"
```

Main then supplies the three exact reviewed hashes and dated test/freeze
evidence. Creating this receipt authorizes the already-waiting daemon to
dispatch as slots become idle. Do not run this just to test the interface.

```bash
PROTOCOL_SHA256=678685bd573d3c034fb17d6410fc089c63302867b88c4eb3c67dfd91f86b7a3e
R114_PLAN_SHA256=b314d2d643bbac142b6890ae6aa3b8792a7a6469be1c217af63bb7c219120ea2
bash code/V4/run_v4_r113_wave.sh ready \
  --r114-plan "$R114_PLAN" --r114-timeout-seconds "$R114_TIMEOUT" \
  --confirm-main-protocol "$PROTOCOL_SHA256" \
  --confirm-source-set "$SOURCE_SET_SHA256" \
  --confirm-r114-plan "$R114_PLAN_SHA256" \
  --runner-tests-passed --evidence "$MAIN_FREEZE_AND_TEST_EVIDENCE"
```

## States And Recovery

`waiting_main`: no approval; no resource probes or GPU dispatch.
`waiting_resources`/`resource_busy`: original mutex/slot busy, or a worker proved
resource contention before starting any witness/formal child. Only this case
may defer and choose another allowed slot. Every transport attempt is retained.
`running`: initialized capacity/formal work; never automatically retried.
`complete`: the job worker completed every required stage at exit 0.
`failed`: terminal, including a missing worker completion receipt.
R114 `blocked` (exit 2) preserves finalized partial masks/analysis, is terminal,
and is distinct from a transport/resource failure. It is never automatically retried.

Receipts/logs are under `preparation/wave_dispatch`; capacity receipts are
`preparation/capacity_<arm>.json`. Tokens are not printed in status/log argv.
Read `status` or tmux output for progress. Source/protocol/inode drift is fatal,
not permission to refreeze. Interruptions stop new dispatch but wait for owned
children; they never kill peers or release a live worker's mutex early. Existing
controller/job/witness receipts forbid automatic restart. Inspect and request a
new explicit main decision rather than deleting receipts to retry.
