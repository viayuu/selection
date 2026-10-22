# R95 Execution

## Resources

| Arm | Fixed slot | Existing allocation | Predecessor |
| --- | --- | --- | --- |
| A uniform replacement | 3090_0 / GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 | 1465 gpu03 | R92A full TEST/LIB/PDF |
| B corrected adaptive | 4090_0 / GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be | 1475 gpu05 | R92B full TEST/LIB/PDF |

R94A/B keep their healthy slot1 workers. No new allocation, no ordinary-SSH
training, no GPU remapping, no automatic retry. Each arm uses the existing
allocation-owned container, four disjoint allowed CPUs, 9GiB reservation,
one exact CUDA UUID and the shared UUID lock. Reuse local frozen R84 caches;
no data generation, solver decoding, cost relabeling or R58 work.

## Commands

All invocations first activate `easynco`:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
```

Main writes `main_ready.json` only after focused CPU tests and source/CLI
integration, then executes the following lifecycle. These commands are not
evidence that training has already started; actual receipts are authoritative.

```bash
python -B -m code.V4.r95_launch specs --main-ready-evidence code/V4/runs/R95_adaptive_example_sampling/main_ready.json --runner-tests-passed
python -B -m code.V4.r95_launch prepare --spec code/V4/runs/R95_launch/specs/A.json --main-ready-evidence code/V4/runs/R95_adaptive_example_sampling/main_ready.json --runner-tests-passed
python -B -m code.V4.r95_launch prepare --spec code/V4/runs/R95_launch/specs/B.json --main-ready-evidence code/V4/runs/R95_adaptive_example_sampling/main_ready.json --runner-tests-passed
python -B -m code.V4.r95_launch start --spec code/V4/runs/R95_launch/specs/B.json --main-ready-evidence code/V4/runs/R95_adaptive_example_sampling/main_ready.json --runner-tests-passed
python -B -m code.V4.r95_launch queue --spec code/V4/runs/R95_launch/specs/A.json --main-ready-evidence code/V4/runs/R95_adaptive_example_sampling/main_ready.json --runner-tests-passed --interval 60
python -B -m code.V4.r95_goal watch --publish-github --interval 120
```

The queue/watch commands run in detached tmux sessions, not a foreground SSH
shell. Training includes the unique VAL lock and full TEST; the explicit TEST
stage is idempotent, followed by same-GPU LIB and CPU PDF. Exit-code zero is
required before the next stage. W&B is enabled offline because the established
environment has no credentials. No notification services.

## Progress And Delivery

- `code/V4/runs/R95_launch/{A,B}/controller.json` and `worker.json`: actual execution/stage/PIDs.
- `code/V4/runs/R95_launch/A/queue_status.json`: waiting versus first start.
- `code/V4/runs/R95_adaptive_example_sampling/{A,B}/status.json`: successful updates and current phase.
- `history.json`, `updates.jsonl`: full validation and optimization traces.
- `proposals/updateXXXXXXX.json`: own-TRAIN source identity, gradient proxy and sampling diagnostics.
- `goal_progress.json`: full TEST/LIB/PDF completion, not early efficacy.
- `github_publication.json`: one final curated version commit/push receipt.

Research used `gpt-6-astra/max` only for papers/idea limits. Implementation,
focused review, execution, metrics and delivery use requested Sol workers.
Tool model requests are recorded; actual backend identity is not independently
attested. No success claim before same-lock final full18 and recovery evidence.

## Accepted Start 2026-10-22

95 focused CPU tests passed in7.3seconds; all9 runner/report CLI commands parse.
Both arms retain all1842953 original fresh parameters, common hash
`dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93`.
Sol's focused review found no P1/P2 issues. Actual preparation is compatible
with both runner and reporter. No costly duplicate GPU preflight was added.

B started2026-10-21T22:18:11UTC in allocation1475-owned container, namespace
PID1508159 / verified host PID1093732, CPUs0,1,2,3. The first full18 task cycle
has finite loss and gradients with zero AMP retries. At22:19:46UTC B had234
successful updates at about0.368seconds/update. These are warm natural rows;
the first adaptive proposal is not used until the scheduled TRAIN7200 pass.

R92A, R94A, R94B and R95B are the four real advancing GPU workers. A's queue
waits for R92A's full delivery, not merely a training-complete flag. Receipts:
`launch_startup_check.json`, `main_ready.json`, launcher worker/controller
records. Sessions `r95_B`, `r95_queue_A`, `r95_goal_watch` are detached.
One instantaneous GPU snapshot included idle gaps; this is not a claim of
constant100percent utilization or a model-quality result.

## Automatic A Handoff And First Adaptive B Phase

R92A finished full TEST/LIB/PDF at23:09:43UTC and its completed version was
pushed as`172f075d7`. A's queue recognized the predecessor at23:10:07UTC;
the accepted A controller started23:10:23UTC without stopping other workers.
A's first full18 update cycle has finite metrics and zero AMP retries;
verified namespace PID2899721 / host PID3086900, CPUs12,13,14,15. Actual
fresh A/B initialization hashes match exactly. See `A_startup_check.json`.
The old queue status is a handoff receipt, not a live training-status source.

B's own complete TRAIN7200 pass generated all18 proposal tables from81000
rows/task. The first adaptive full18 cycle (update7218) completed with finite
loss0.42077748 and gradient norm0.54786428. Mean inverse weight0.99694450,
mean batch-unique fraction0.99213325; duplicates were retained. These are
runtime connectivity facts, not variance-reduction or efficacy claims.
Four actual GPU groups are now R94A/B and R95A/B.
