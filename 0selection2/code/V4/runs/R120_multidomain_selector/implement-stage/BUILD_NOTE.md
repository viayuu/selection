# Build Progress

## Current State

Snapshot: 2026-10-29 20:28 CST. The latest explicit user continuation authorizes
implementation, necessary data completion and the four existing allocated GPUs.
No previous experiment was stopped. Formal 24-task model training is running:
all 12,970 native MKP labels completed, complete-pool export/preparation and
host-local frozen input staging succeeded. Four-rank training reached 96
successful updates with finite loss/gradients and zero AMP retries. See
`../startup_recovery.md` for the earlier zero-update network startup failure
and preserved evidence.

| Rung | Feature | Acceptance | Status |
| --- | --- | --- | --- |
| F0 | Actual native records to typed model and CPU loss/update | Three-domain CPU training spine with finite loss/gradient | Passed, separate smoke root |
| F1 | Six raw schemas, FP64 objectives, directions, ties and observed masks | Data/supplement unit tests and real-record forwards | Passed; immutable ready-three export preserved |
| F2 | Shared typed relation/global encoder and full legal candidate scoring | Model/objective CPU tests | Passed, including dense routing arc values and explicit execution roles |
| F3 | Exact global-batch DDP, validation lock and reporting interfaces | Runner/report tests, including unequal two-rank GLOO accumulation | Passed |
| F4 | Complete the declared supplement and train one 24-task selector | All 16 annotation receipts, immutable complete export, successful formal train | All labels/export complete; four-GPU formal training reached update 96; no final result yet |
| F5 | Locked TEST/LIB, PDFs and one curated GitHub publication | Pipeline stage receipts and report completion | Automatically scheduled after F4; not yet a result |

## Run Record

- The main model suite completed with exit 0: 39 tests passed in 17.23 seconds.
  Command: `PYTHONPATH=/tmp/shiys_r120_test_deps:$PWD OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m pytest -q code/V4/test_r120_model.py code/V4/test_r120_objective.py code/V4/test_r120_experiment.py code/V4/test_r120_reporting.py`.
  Later report-only additions retain their own targeted test receipt.
- Data/supplement tests passed independently; no solver environment or shared
  Python environment was upgraded for the isolated pytest dependency.
- CPU supplement production completed: BPP has FFD/BFD/MBS_prime;
  SMTWTP has EDD/WSPT/ATC; MKP has both deterministic baselines and an incomplete
  native 20-ant, 100-iteration AntSystem column. Public benchmarks stay external.
- Four authorized GPUs are completing the latter column, with four independent
  one-CPU processes per GPU. All original native iterations remain intact.
  The assigned two RTX3090 and two RTX4090 each reported 99% utilization in the
  latest queried snapshot; concurrent annotation time is not exclusive latency.
- `r120_full24_pipeline` waits for specific per-worker completion receipts,
  exports a new immutable complete pool, prepares and trains the shared model,
  evaluates locked TEST/LIB, builds PDFs, and publishes one curated commit.
  `r120_mkp_full_pool` owns the current annotation workers.
- Annotation is now complete. The current controller is
  `r120_full24_pipeline_host`, running the same protocol on allocation GPU
  steps after the documented transport-only recovery. At update 96 the
  logical batch is 1024, world size is four, and sample exposures are 98,304.
  W&B run `361td5ju` is explicitly offline because no credential was configured.
- Astra's scientific handoff is complete, including its PDF. Sol owns the
  implementation, protocol, execution and result/report checks.

## Interpretation Boundaries

- This is a new 24-task baseline, not a matched ablation against an old routing
  model and not implementation evidence for the later D1/D2 research proposals.
- Routing keeps native-winner Top1 and the original relative-cost metrics.
  New tasks use direction-aware any-best Top1, expose canonical-label accuracy
  separately, and preserve exact ties and zero objectives. ALL24 never averages
  raw distance, makespan, bin count, profit and tardiness together.
- L2D/DAN weights are separate deployment actions, not independent algorithm
  families and not a free per-instance best-checkpoint operation.
- Repeated small-task streams are explicitly task-balanced. No query label,
  objective, route or solution certificate enters the forward inputs.
- The R84 NSS reference has matching evaluation populations, but historical
  R79 weights trained on 70,114 examples per task; it is not a new90k retraining.
- B-problem scenario repair and replacement of the original routing cost table
  remain deferred. This version does not silently claim R58 was completed.

## Deferred And Blockers

The remaining critical path is formal model training/evaluation. The first
zero-update network failure is preserved and explicitly recovered. A failed stage is preserved
and stops the controller instead of being silently retried or changing the pool.
The shared workspace contains unrelated historical changes; publication is
limited to this completed version's curated small files, not raw data/weights.

See `../LAUNCH.md` and `../pipeline_status.json` for commands and live progress.
