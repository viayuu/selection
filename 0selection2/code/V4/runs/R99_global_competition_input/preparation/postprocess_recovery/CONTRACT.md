# Frozen R99 Recovery Contract

Module: `code.V4.r99_postprocess_recovery`.
Schema: `r99_postprocess_recovery_v1`.
All commands run from `/public/home/shiys/0selection2` in `easynco`.
The default `--root` is the original `R99_global_competition_input` run.

## CPU Sequence

After the main-owned original reporter LIB and private timing stages finish:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
python -B -u -m code.V4.r99_postprocess_recovery report --arm A --device cpu --timing-revision disclosed-online-v1 --max-seconds 1800
python -B -u -m code.V4.r99_postprocess_recovery report --arm B --device cpu --timing-revision disclosed-online-v1 --max-seconds 1800
python -B -u -m code.V4.r99_postprocess_recovery compare --device cpu --max-seconds 1800
python -B -u -m code.V4.r99_postprocess_recovery finalize --arm A --device cpu --bind-val-sha
python -B -u -m code.V4.r99_postprocess_recovery finalize --arm B --device cpu --bind-val-sha
python -B -u -m code.V4.r99_postprocess_recovery finalize --device cpu
python -B -u -m code.V4.r99_postprocess_recovery validate --device cpu
```

A/B reports are independent and may run concurrently. Each stage refuses to
overwrite a prior attempted stage receipt. `bind-val` is also a standalone
CPU command; `finalize --arm A/B --bind-val-sha` runs that binding explicitly.
Completed finalization and successful binding are idempotent.

## R104 Gate

The terminal gate file is:

`/public/home/shiys/0selection2/code/V4/runs/R99_global_competition_input/preparation/postprocess_recovery/recovery_complete.json`

Do not gate on file existence or change the failed original dispatch status.
Use the read-only CPU validator, which raises on missing, changed, foreign,
incomplete, or falsely equivalent evidence:

```python
from pathlib import Path
from code.V4.r99_postprocess_recovery import validate_recovery_complete

receipt = validate_recovery_complete(Path(
    '/public/home/shiys/0selection2/code/V4/runs/R99_global_competition_input'))
```

Required global fields and exact values:

| Field | Contract |
| --- | --- |
| `schema` | `r99_postprocess_recovery_v1` |
| `experiment` | `R99` |
| `complete`, `same_checkpoint_verified`, `original_failure_receipts_preserved` | `true` |
| `exit_status`, `state` | `0`, `complete` |
| `cli_module` | `code.V4.r99_postprocess_recovery` |
| `checkpoint_locks` | A/B full original `final_test_lock.json` dictionaries |
| `original_failure_sha256` | A/B identity digests described below |
| `arms` | A/B `{path, sha256, checkpoint_lock, original_failure_sha256}` |
| `comparison` | `{path, sha256}` of `comparison_stage.json` |
| `stages_completed` | `['A_timing_LIB_PDF_VAL_binding', 'B_timing_LIB_PDF_VAL_binding', 'CPU_AB_comparison']` |
| `original_inference_equivalence` | Honest measured result; `false` for this approved recovery |
| `training_rerun`, `full_test_rerun`, `feature_copy`, `feature_rebuild`, `decision_rule_changed`, `efficacy_reused_as_exact_online` | `false` |

Each child is `<arm>/recovery_complete.json`, with the same schema/rc0/state
checks, full `checkpoint_lock`, `checkpoint_sha256`, `state_hash`,
`original_failures`, `timing`, `lib`, `pdfs`, hashed `report_receipt`, and hashed
`val_binding`. Its exact `stages_completed` list is:

```json
["original_locked_full_TEST", "recovery_timing", "original_reporter_LIB", "private_CPU_full_report", "metadata_only_VAL_checkpoint_binding"]
```

`original_failures` maps `worker.json`, `controller.json`, `timing.log`, and
`allocation_GPU_pipeline.log` to their original `{path, sha256}` records.
`original_failure_sha256` is SHA256 of that dictionary serialized by
`r99_features.digest` (sorted keys, compact separators). The original worker
and controller remain failed; their bytes/hashes must not change.

## Evidence And Outputs

The CPU report reads linked original checkpoint/evaluation/LIB inputs and
writes only `report_view/<arm>/full_model_statistics/*`. Reports use a private
validator and an explicit numerical-boundary PDF page. The timing path must
remain `mixed`, policy `default`; no approximate-equality flag substitutes
for the recorded bitwise result. Cached full-TEST efficacy is explicitly not
claimed as exact online efficacy. LIB uses the unchanged original reporter
and the same checkpoint; it is not rerun by this module.

The only authorized original-file write is the explicit post-report binding
of `final_metrics.json` at `val.identity.checkpoint_sha256`: present-null to
the locked checkpoint SHA. It requires identical update/state/modelState
across the original cached VAL and locked TEST. The original bytes are saved
as `<arm>/final_metrics.before_val_sha_binding.json`; only the `null` byte span
is replaced. `<arm>/bind_val_stage.json` proves before/after hashes. Original
archived VAL identity, predictions, weights, and numerical metrics stay
unchanged. The private report view retains the original unbound metadata so
its archive identity checks remain valid.

`validate_recovery_complete` recursively verifies all arm receipts, original
failure hashes, checkpoint hashes, protected saved predictions, approved
timing revision and timing file, original rc0 LIB receipts, full PDFs/report
protocol, metadata-only binding, and same-lock A/B comparison. It never loads
a model or starts CUDA. R104 must accept this explicit recovery proof, not
pretend the original failed worker/controller finished successfully.
