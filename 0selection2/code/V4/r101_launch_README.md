# R101 Dispatch And Delivery

Activate `easynco` before every command. Main owns scientific settings, exact
stage argv, ready evidence and actual start. This launcher never extracts a
teacher and never stops a healthy predecessor.

## Final Mapping

| Arm | Teacher | Fixed Slot | Required Predecessor |
| --- | --- | --- | --- |
| B | Existing pretrained R94B | `4090_0` | Entire R98_A controller/GPU/CPU/PDF rc0 |
| A | Existing random R94A | `4090_1` | Entire R100_A controller/GPU/CPU/PDF rc0 |

Both use the original allocations/containers, four CPUs and 9 GiB per worker.
Shared CPU claim files stay under `runs/R98_parallel_dispatch`; GPU locks stay
at the original R81 mutex inodes. Private adapters do not rebind frozen modules.

## Main-Supplied Spec

Use `schema_version=1`, `run_id=R101_A|R101_B`, `run_dir=MODEL_ROOT/ARM`,
`common_root=MODEL_ROOT`, one matching fixed slot, `wandb_mode=offline`,
`cache_kind=r84_inputs`, the existing R84 `cache_source`, and
`cache_root=/tmp/shiys_R84_new90k_cache`. `MODEL_ROOT` is the absolute directory
`/public/home/shiys/0selection2/code/V4/runs/R101_hint_pretraining`.

`science_protocol` must name main's existing locked R101 seed2 full18 protocol,
with frozen TRAIN81000/VAL4500/TEST4500, logical batch1024, Stage-I12800 and
Stage-II57600 separately. `hint_pretraining.updates` is supported directly.
The launcher does not generate or alter the scientific protocol.

Provide exactly these direct Python-module argv stages, in this order:

| Stage | Device | Entry |
| --- | --- | --- |
| pretrain | gpu | Main's R101 pretrain CLI |
| train | gpu | R101 original full18 downstream CLI |
| TEST | gpu | R101 locked TEST CLI |
| timing | gpu | R101 student-only timing CLI |
| LIB | gpu | `code.V4.r101_reporting lib --lib-batch-size 2` |
| PDF | cpu | `code.V4.r101_reporting report` |

Use `{python}`, `{device}`, `{cache_root}`, `{common_root}`, `{teacher_root}`
placeholders where appropriate. Ready preparation parses every exact command
and freezes its source closure; files added later do not change the snapshot.
No teacher flags belong in TEST/timing/LIB/PDF. Final model loading and forward
must remain student-only, even if teacher or pretrain files are unavailable.

## Teacher Staging

B's `teacher_root=/tmp/shiys_R94_B_node_features`. B prepare/start has no
dependency on A or a shared export receipt. Its worker calls the existing
`r101_features.stage_teacher` with `target_root=None, variant=B` and no source
receipt: native16 TRAIN metadata/inputs are validated, with no teacher payload
copy, hash pass, extraction or encoder execution. An optional metadata-only
`teacher_receipt` may pin the bridge discovery; it is NOT a copy source receipt.

A's `teacher_source=MODEL_ROOT/teacher_cache/A` and
`teacher_root=/tmp/shiys_R101_A_teacher`. Its export must already be complete.
The default `teacher_receipt` is
`MODEL_ROOT/preparation/A_teacher_export_receipt.json`. The worker CPU-stages
only existing TRAIN metadata and teacher arrays through the feature bridge,
with its bounded streaming/hash verification, before pretraining starts.
Teacher roots are parents of `train`, never the `train` subdirectory itself.
No teacher cache, checkpoint, prediction arrays or W&B files are published.

## CLI

From the project root, main can arm/start B independently:

```bash
conda activate easynco
python -B -m code.V4.r101_launch prepare --spec code/V4/runs/R98_parallel_dispatch/specs/R101_B.json --evidence "MAIN_CPU_TEST_EVIDENCE" --runner-tests-passed
python -B -m code.V4.r101_launch start --spec code/V4/runs/R98_parallel_dispatch/specs/R101_B.json --evidence "MAIN_CPU_TEST_EVIDENCE" --runner-tests-passed
```

After A's export and exact argv are ready, prepare A identically and use `queue`
instead of `start` to wait for R100's whole pipeline. Main should put this long
CPU wait in tmux. `predecessor --arm A|B` only inspects saved receipts.
Queue and launch attempts are one-shot; failures are retained, never repaired
or automatically retried. Source changes after ready require main review.

The CPU delivery watcher is explicit opt-in:

```bash
conda activate easynco
python -B -m code.V4.r101_goal --watch --publish-github
```

It requires both complete Stage-I endpoints, fresh non-encoder lineage and
reset optimizer/scaler/RNG/streams, original1842953-parameter student-only
deployment, both full57600 downstream budgets, unique native VAL locks,
full18 TEST, both native LIBs, timing and six per-arm PDFs. It then invokes
the saved-data CPU comparison once and publishes one curated version commit
and push through the existing `codex-publication.lock`. No weights, raw arrays,
teacher/runtime caches, predictions or W&B assets enter the allowlist.
Unchanged progress stays quiet. Research goal remains `active_unresolved`.
