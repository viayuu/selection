# R94 Queue and Progress

Prepared on 2026-10-22 Asia/Shanghai. **B is now training; A waits for R93A's
full delivery.** B's first startup was explicitly interrupted to fix measured
input read amplification. Its logs/failures/initial checkpoint are preserved
under `startup_slow_io_attempt1`; the accepted run starts fresh, not resumed.
R92A/B and R93A were not stopped. No new allocation or implicit retry.

| New arm | Frozen donor | Slot | Required predecessor |
| --- | --- | --- | --- |
| A | Random native ReLD encoder | gpu03 / 3090_1 / job1465 | R93A complete train + locked TEST + LIB + PDF |
| B | R48-pinned pretrained native ReLD encoder | gpu05 / 4090_1 / job1475 | R93B complete train + locked TEST + LIB + PDF |

The live CPU queues are `r94_queue_A` and `r94_queue_B`. They poll saved
predecessor delivery receipts every60 seconds, then call the one-shot launcher.
The launcher separately requires the actual UUID to be idle, four disjoint
allocated CPUs and9GiB memory headroom, inside the existing allocation-owned
container. Ordinary SSH sessions are not used for training. Each worker holds
the existing cross-version UUID flock. A failed predecessor or new stage stops
the affected arm; no automatic restart or duplicated attempt.

## Evidence

- `cpu_checks.json`:99 focused CPU tests passed, no skips, including tiny actual
  random/pretrained native encoder witnesses and cache/real-VAL-lock regressions.
- `main_ready.json`: final main authorization. Two actual fresh-state
  constructions and runner/report protocols agreed; both trainable SHA256 are
  `9db3df55a91ca0f331880cfcc4688ffab82b83bb66e80f9cebe9d6978e903cb4`.
- `../R94_launch/specs/A.json` and `B.json`: thirteen exact CLI commands parsed.
- `../R94_launch/A/plan.json` and `B/plan.json`: source hashes frozen; no resource
  probe or GPU execution was performed during that preparation.
- `../R94_launch/A/queue_status.json` and `B/queue_status.json`: queue/start state.

`IO_PIPELINE_RECOVERY.pdf` records the resource-only correction. A default
~26MB feature gather read1.2-2.7GB physically. RANDOM advice alone reduced bytes
but was slower and was rejected; precise requested-range asynchronous prefetch
reduced two diagnostic gathers to0.10-0.11seconds. The restarted training's
first270 updates averaged0.425seconds versus7.408seconds for the old first54
updates. These are startup observations under co-resident jobs, not model
efficacy or a controlled hardware benchmark. Feature values, row order, model,
loss, initialization and the scientific protocol are unchanged. The preserved
old startup logged108 successful updates (at most125 before termination) but
its only durable checkpoint was update0; those discarded updates are not
presented as resumed state or hidden from compute accounting.

Only the additional early projection is newly trainable. Original fresh R84
parameters and its TRAIN-only geometry statistics are reused, not a trained
selector checkpoint. Exact-N donor output is cached after FP32 extraction and
FP16 quantization. Per-arm TRAIN/VAL sidecars require about41.075GiB. The
read-only container disk observation showed217GiB/209GiB available; the queue
does not reserve or fill that space before its predecessor completes.

Native batching has floating-point, not bitwise, equivalence. The bounded
pretrained CPU comparison uses rtol2e-5/atol3e-5; this is not a model-efficacy
result. The exact checkpoint pin and source provenance are recorded by the
donor. No solver decoder, new route, new cost label or R58 relabeling runs.

## Pipeline

The one-shot GPU pipeline is:

```text
exact-N TRAIN/VAL input features -> train -> unique VAL lock + complete TEST
-> idempotent TEST -> full uncached/cached timing -> native LIB
-> ADMIN CPU PDFs
```

All original natural logical1024/seed2 training controls remain fixed. The
feature-preparation batch128 is not the training batch. TEST sidecars can be
created only after the real R94 VAL lock. Complete-stage logs go under
`../R94_launch/A/` or `B/`; training history and status go under this root's
`A/` or `B/`. Feature-task progress is also visible in the node-local sidecar
`metadata.json` files. Cached readout latency is never called full inference.

## Commands

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
python -B -m code.V4.r94_goal status
tmux attach -t r94_queue_A
tmux attach -t r94_queue_B
```

After a queue launches, the controller sessions are `r94_A` / `r94_B`.
`r94_goal_watch` quietly checks completion every120 seconds. It only compares
after both same-checkpoint full TRAIN/VAL/TEST/timing/LIB/PDF deliveries pass,
then publishes one curated version commit/push under the common Git mutex.
It excludes raw arrays, weights, caches, credentials and W&B stores. Research
goal remains active: successful launch or fewer zero-selection groups is not
evidence of improved method selection.
