# R87 resource-only handoff

Preparation only: no R87 training, tmux, new allocation, cache copy, or
preprocessing has started. Main owns actual starts after runner CPU-ready.
Runner: Einstein `01a154d3-23a9-7463-97b3-5be3b7ca0719`; model: Wegener.
Model output root is `code/V4/runs/R87_private_method_residual`.

| Arm | Existing allocation | Exact GPU UUID | Large micro / eval / logical batch |
| --- | --- | --- | --- |
| A_shared_residual | 1475 / gpu05 slot0 | GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be | 352 / 256 / 1024 |
| B_private_residual | 1475 / gpu05 slot1 | GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d | 352 / 256 / 1024 |

Both arms use seed2, 57,600 successful updates, warmup4,320, evaluation1,440,
the existing NEW90k data, TRAIN-only normalization/SBS, and fresh common random
initialization. No R84/R85 trained weights initialize R87. GPU work remains
inside container `76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6`.
The two arms each reserve four disjoint allocated CPUs and nine GiB.

## Predecessor and queue rules

A waits for R85C; B waits for R85D. Each predecessor must have complete rc0
worker stages `train_VALlock_TEST` and `LIB`, complete rc0 controller stages
`allocation_GPU_pipeline` and `PDF`, and `postprocess_status.phase=complete`.
The existing R85C receipts already satisfy this; R85D was still training at
handoff. Main reported C's PID644375 absent / slot0 idle and D PID645535 live
on slot1. These are main's resource observations, not a new GPU audit here.

A/B never fall back to another GPU. A queued B does not claim GPU/CPU resources
until D's complete report rc0. PID exit or an idle UUID alone does not release
this gate. A failed predecessor stops the queue with preserved evidence; there
is no implicit repair or retry. After this gate, the established per-UUID flock,
allocation/container/cgroup identity, live GPU and CPU checks apply at claim.
No R85 process or either 3090 is stopped or modified.

## Cache and W&B

Dataset:
`/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`

Preparation: `code/V4/runs/R84_new90k_attention/prepared`.
Both arms reuse `/tmp/shiys_R84_new90k_cache` in the existing gpu05 container.
The already completed R85 TRAIN/VAL cache identity evidence is reused; no
321-file scan, array hashes, cache rebuild, or data transfer occurs. Entry only
checks the identity stamp and train/val directory existence. TEST is consumed
only by the runner after its unique VAL lock.

At worker entry, standard W&B lookup reads existing environment/settings/netrc.
No key is logged or copied into receipts. If no credential exists, actual train
argv explicitly uses offline mode; `wandb_lookup.json` records the reason.
The template's `auto` placeholder resolves in the actual container, not from
an assumed admin01 login. Project `selector`, entity
`yjkds-southern-university-of-science-technology` remain enabled.

## Commands

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
bash code/V4/run_v4_r87_launch.sh self-test
bash code/V4/run_v4_r87_launch.sh specs
bash code/V4/run_v4_r87_launch.sh gates
```

`specs` writes two independent specs under `runs/R87_launch/specs/` and parses
actual runner/reporter argv when those modules become available. Missing CLI
is explicitly pending, never a fabricated PASS. `spec_parse.json` and
`launcher_CPU_tests.json` provide the CPU handoff. There is no training on
import, self-test, spec generation, or preparation.

When runner/model/reporting sources are ready, main freezes each plan:

```bash
bash code/V4/run_v4_r87_launch.sh prepare --spec /ABSOLUTE/FINAL_SPEC.json
```

Only main's explicit authorization starts a single arm controller:

```bash
bash code/V4/run_v4_r87_launch.sh start --spec /ABSOLUTE/FINAL_SPEC.json \
  --runner-tests-passed --main-ready-evidence 'MAIN_PROVIDES_FINAL_READY'
```

Controllers are `tmux r87_A_shared_residual` and `r87_B_private_residual`.
After readiness, a B controller may wait safely for D's report completion.
The pipeline is train/uniqueVALlock/completeTEST, same-checkpoint LIB, then CPU
PDF on admin01. First nonzero stage stops later stages and preserves receipts.
No notifications, Git operations, automatic retries, or new allocation occur.

R87 sources and the frozen R84/R85 helpers it actually executes are hashed in
the ready plan. Historical modules/receipts are read-only. Runtime records and
logs belong only to `code/V4/runs/R87_launch/<arm>/` and the R87 model outputs.
