# R89 preparation-only resource handoff

Main owns actual preparation/source freeze and both starts. This launcher
preparation never starts training, creates a tmux controller, or claims a GPU.
Only new R89 launcher files, tests, and R89 launch receipts are owned here.

| Arm | Existing allocation | Fixed GPU UUID | Required predecessor |
| --- | --- | --- | --- |
| A_adamw | job1475 / gpu05 / 4090_0 | GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be | R87 A_shared_residual |
| B_muon_hybrid | job1475 / gpu05 / 4090_1 | GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d | R87 B_private_residual |

Both arms use the existing allocation-owned container
`76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6`.
The CPU entry is `srun --jobid=1475 --overlap --exact --gres=none`; actual
GPU work runs inside that container with one explicit UUID. Each arm claims
four disjoint CPUs and nine GiB. No new allocation, GPU outside an allocation,
direct SSH GPU launch, or automatic retry is allowed. R88 on gpu03 is untouched.

## Matched controls

Both arms train the same full18 original R77 selector with R88's
`B_execution_anchor` TSP slot3 rule and no new model parameters. Both retain
the original CE/comparison/risk/R-Drop objective. Only the optimizer differs.
Use seed2, logical batch1024, large TSP/CVRP microcap352, evaluation batch256,
57,600 successful updates, warmup4,320, and evaluation every1,440 updates.
NS5 BF16 internal calculations and FP16 model forward must be compatible
with the existing environment; this launcher does not upgrade Torch.

Readiness and configuration do not prove Muon executed. After main starts,
the bounded first-full18 witness must tie actual host/container PID, UUID and
cgroup to finite PART_KEYS. For B it must additionally identify the real
optimizer types/groups, `muon_step_applied > 0` (a per-step indicator, not a
cumulative counter), positive updated-matrix count, and finite measured
direction/momentum RMS. No validation wait is required. A real failure preserves the original
evidence and is reported to main, never used to restart a healthy arm.

## Full-delivery gate

Each arm waits for its R87 predecessor's complete rc0 worker stages
`train_VALlock_TEST` and `LIB`, controller stages `allocation_GPU_pipeline`
and `PDF`, and completed postprocess receipt. The R87 round comparison must
also have a complete rc0 `goal_compare_receipt.json`. Only then can the stable
per-UUID flock and resource claim run. An idle GPU or exited train PID alone
does not release the slot. Failed predecessors/comparison are not retried.

## Existing cache, initialization, and W&B

Reuse `/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`,
R84's `prepared` directory, and existing gpu05 container cache
`/tmp/shiys_R84_new90k_cache`. Only the saved identity stamp and train/val
directory existence are checked. No copying, array scan/hash, preprocessing,
or TEST reading during training preparation. Initialization is the saved fresh
seed2 state, never a trained R84/R87/R88 selector.

W&B is enabled for `selector` / `yjkds-southern-university-of-science-technology`.
The worker uses the standard environment/settings/netrc credential lookup.
When no existing credential is found, argv explicitly selects offline and
`wandb_lookup.json` records the reason without recording secrets.

## CPU preparation and main-only start

Cache provenance is the existing `code/V4/runs/R85_launch/resources.json`;
R87 supplies predecessor-delivery receipts, not a new resource snapshot.
R89 calls the generic prepare directly so these paths remain independent.
After the prelaunch path correction, existing specs stay valid: main only
reruns both `prepare` commands to freeze the corrected R89 source. The failed
A partial plan is preserved as
`code/V4/runs/R89_launch/prepare_A_failed_partial_plan_20261020T033107Z.json`.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
python -B -m unittest code.V4.test_r89_launch
bash code/V4/run_v4_r89_launch.sh specs
bash code/V4/run_v4_r89_launch.sh gates
```

`specs` is run only after the final runner/reporter are present. It parses eight
actual train/test/lib/report argv against their parsers and reads main's
`R89_optimizer_geometry/protocol_lock.json`. This does not freeze sources,
create ready gates or controllers, or start training. Specs record all final
R89 runtime sources plus unchanged model/training/reporting helpers; main's
later `prepare` hashes those small source files, not cache arrays.

Main alone prepares and starts each final spec after owner CPU readiness:

```bash
bash code/V4/run_v4_r89_launch.sh prepare \
  --spec /public/home/shiys/0selection2/code/V4/runs/R89_launch/specs/A_adamw.json
bash code/V4/run_v4_r89_launch.sh prepare \
  --spec /public/home/shiys/0selection2/code/V4/runs/R89_launch/specs/B_muon_hybrid.json
bash code/V4/run_v4_r89_launch.sh start \
  --spec /public/home/shiys/0selection2/code/V4/runs/R89_launch/specs/A_adamw.json \
  --runner-tests-passed --main-ready-evidence 'MAIN_PROVIDES_FINAL_READY'
bash code/V4/run_v4_r89_launch.sh start \
  --spec /public/home/shiys/0selection2/code/V4/runs/R89_launch/specs/B_muon_hybrid.json \
  --runner-tests-passed --main-ready-evidence 'MAIN_PROVIDES_FINAL_READY'
```

Sessions: `r89_A_adamw`, `r89_B_muon_hybrid`. Receipts/logs belong to
`code/V4/runs/R89_launch/<arm>/`; model outputs belong to
`code/V4/runs/R89_optimizer_geometry/<arm>/`. Pipeline: train with unique
VAL checkpoint lock and TEST, same-checkpoint LIB, then CPU PDF. Every next
stage requires rc0. No Git operations, external notifications, or persistent
monitor are added by the launcher owner.
