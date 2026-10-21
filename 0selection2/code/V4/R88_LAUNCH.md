# R88 preparation-only resource handoff

No R88 training or tmux controller is started by this preparation. Main owns
actual starts after model/runner/launcher CPU-ready. Runner is Einstein; model
is Wegener. Only R88 sources, tests, and launch receipts are written here.

| Arm | Allocation | UUID | Required predecessor |
| --- | --- | --- | --- |
| A_canonical_anchor | job1465 / gpu03 / 3090_0 | GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 | R85 A_ovr_control |
| B_execution_anchor | job1465 / gpu03 / 3090_1 | GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d | R85 B_gradient_balance |

Both arms use 3090 microcap352, eval256, logical batch1024, seed2, 57,600
successful updates, warmup4,320, evaluation1,440. Each takes four disjoint
allocated CPUs and nine GiB, within the existing ten-core / twenty-GiB job.
The container is
`b8ced5d8a8cdafdc5006f827a0d36c319c61056a65316af4ff9937453fa0635c`.
No gpu05/4090 slot is eligible; R87 remains untouched.

The R88 model owner implements the TSP marker in existing input slot3: A uses
the minimum-x then minimum-y canonical anchor, including exact ties; B uses
the actual input node0. No extra parameters/losses are added. All18 tasks
still train/evaluate; other17 input paths remain unchanged. This launcher only
routes the locked runner and records execution controls.

## Full-delivery gate

Each arm waits for its own R85 predecessor's full rc0 worker and controller
stages: train/uniqueVALlock/TEST, LIB, and PDF. The report's postprocess phase
must also be complete. A missing PID or idle GPU alone is insufficient.
Failed predecessors preserve evidence and stop the queue without repair or
retry. Only after the gate does the stable per-UUID flock and live resource
claim run. No earlier training process is stopped or migrated.

## Existing cache and credentials

Reuse the NEW90k release:
`/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`
and preparation `code/V4/runs/R84_new90k_attention/prepared`.
Use the existing gpu03 container cache `/tmp/shiys_R84_new90k_cache`.
The completed R85 identity evidence is reused; entry checks only the cache
stamp and train/val directories. No copy, preprocessing, array hashing, or
repeated file scan occurs. Initialization is the common saved fresh seed2
state, not a trained R84/R85/R87 checkpoint.

W&B remains enabled. Worker standard environment/settings/netrc lookup resolves
the template mode into explicit online/offline argv. No key is recorded; no
credential gives explicit offline with a reason in `wandb_lookup.json`.

## Main commands

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
python -B -m unittest code.V4.test_r88_launch
bash code/V4/run_v4_r88_launch.sh specs
bash code/V4/run_v4_r88_launch.sh gates
```

`specs` parses all eight exact train/test/lib/report argv against the actual
R88 parsers and checks main's `R88_execution_role_inputs/protocol_lock.json`.
It does not create ready gates, claims, tmux, or source freezes. When owners
declare source readiness, main prepares each independent spec:

```bash
bash code/V4/run_v4_r88_launch.sh prepare --spec /ABSOLUTE/FINAL_SPEC.json
```

The plan hashes R88 sources and the unchanged R84/R85/R87 helpers it executes.
Only main later starts:

```bash
bash code/V4/run_v4_r88_launch.sh start --spec /ABSOLUTE/FINAL_SPEC.json \
  --runner-tests-passed --main-ready-evidence 'MAIN_PROVIDES_FINAL_READY'
```

Sessions are `r88_A_canonical_anchor` and `r88_B_execution_anchor`. Receipts and
logs belong to `code/V4/runs/R88_launch/<arm>/`; models belong to
`code/V4/runs/R88_execution_role_inputs/<arm>/`. Stages are training with unique
VAL lock and TEST, same-checkpoint LIB, then CPU PDF on admin01. There are no
automatic retries, new allocations, Git operations, or notifications.
