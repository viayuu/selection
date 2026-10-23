# R102 Delivery Recovery Contract

This is postprocessing continuation, not retraining or a successful restart of
the failed original pipeline. Main owns allocation, execution and proofs. This
helper allocates nothing, launches no subprocess, creates no watcher and does
not publish or invoke Git. Original A/B receipts and the stopped watcher remain
verbatim failed. No GPU execution was performed while implementing this helper.

## CLI

Run from `/public/home/shiys/0selection2`, always after:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
export PYTHONDONTWRITEBYTECODE=1
ROOT=/public/home/shiys/0selection2/code/V4/runs/R102_full_type_pretrained_fusion
BASE=$ROOT/preparation/postprocess_recovery
python -B -m code.V4.r102_postprocess_recovery describe --root "$ROOT" --device cpu
```

The public commands are `gpu`, `finish-gpu`, `report`, `compare`, `describe`. Common options:
`--root` is always the ORIGINAL canonical R102 root, never the view;
`--arm A|B`, `--device`, `--cache-root`, `--feature-root`,
`--routing-feature-root`, optional `--proof`, `--protocol-lock` and
`--expected-uuid`. Runtime path options, when supplied, MUST equal the original
arm protocol, not a replacement cache or feature recipe. `--protocol-lock`
defaults to `$ROOT/protocol_lock.json` even though runtime root is the view.
`--proof` defaults to `$BASE/array_equality_proof.json`.

Main's original-allocation entry invokes the following ONLY after both original
wrappers/controllers are terminal. It must expose exactly the arm's original
GPU UUID as `cuda:0`, with original CPU affinity and original container. The
helper verifies CUDA's actual device UUID, not only the environment string.

```bash
python -B -u -m code.V4.r102_postprocess_recovery gpu \
  --root "$ROOT" --arm A --device cuda:0 \
  --cache-root /tmp/shiys_R84_new90k_cache \
  --feature-root /tmp/shiys_R102_A_extra_nodes \
  --routing-feature-root /tmp/shiys_R94_B_node_features \
  --protocol-lock "$ROOT/protocol_lock.json" \
  --expected-uuid GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be
```

B uses `--arm B`, `/tmp/shiys_R102_B_extra_nodes`, and
`GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d`. This documentation does not assign,
launch, kill or acquire GPU resources. Main/admin sidecar owns those actions.

### Explicit Post-TEST Completion

Main explicitly authorized `finish-gpu` after BOTH first full TESTs completed
and the helper's EXTRA bitwise timing assertion stopped at TSP. The original
`r102_experiment.measure_inputs` has no bitwise/pick-equivalence gate. This
command restores that unchanged original timing procedure, not a new numerical
recipe. The historical UUID normalization remains intact.

```bash
python -B -u -m code.V4.r102_postprocess_recovery finish-gpu \
  --root "$ROOT" --arm A --device cuda:0 \
  --cache-root /tmp/shiys_R84_new90k_cache \
  --feature-root /tmp/shiys_R102_A_extra_nodes \
  --routing-feature-root /tmp/shiys_R94_B_node_features
```

B changes arm and original feature root exactly as above. Same original UUID,
CPU affinity and container checks apply. `finish-gpu` NEVER calls TRAIN, TEST,
TEST feature preparation or shadow creation. It verifies the already completed
same-locked first TEST, native4500/all18 population, VAL/model-state identity,
original source pins, weights and cache signatures. Actual first-TEST result
and archive SHAs and actual prior timing failure/TSP-observation SHAs are pinned
per arm in `KNOWN_TIMING_FAILURES`; unknown/modified causes are rejected.

Old `first_error.json`, `timing_checks/TSP.json`, `timing_started.json`,
`gpu_entry.json`, `gpu_uuid_format_correction_entry.json`, first TEST JSONs and
all TEST archives remain byte-for-byte. New stages are `timing_observation`
then `lib_completion`, with exclusive new start markers. New full18 observations
go to `$BASE/{arm}/timing_observations`; no `require_alignment` hard gate is
applied there. Original aligned-input, finite-score, real-donor-hook-count
checks still apply. No TF32, grouping, inference or normalization change occurs.

Success writes the SAME `$BASE/{arm}/gpu_complete.json`, adding
`completion_kind=post_test_original_timing_observation`,
`stage_exit_statuses={timing_observation:0,lib_completion:0}`, immutable
`prior_timing_failure` artifact SHAs, `test_rerun=false`, all18 observation
artifacts and `online_efficacy_bitwise_certified=false`. Provenance is NEW
`finish_gpu_provenance.json`; another failure writes NEW `post_test_error.json`
and prevents further work without overwriting the historical first error.
`checked_completion`, report and compare accept this exact SHA-bound declared
exception only, never blanket-ignore failed receipts. Admin/Jason owns the new
GPU-entry receipt and publisher; this helper does not touch those components.

Numerical boundary: cached first-full-TEST efficacy is NOT certified as
bitwise-equivalent online efficacy. Original timing measures online
donor+selector and cached selector separately. Observed float/FP16/pick
differences concern first128 rows/task ONLY, not whole-TEST agreement. CPU
reporting adds `full_model_statistics/numerical_boundary.md` and a standalone
`numerical_boundary.pdf` with this disclosure and all18 sampled comparisons,
and records these artifacts in report provenance/completion. The original
mandatory report/PDF generator and inference factory remain unchanged.

After the corresponding GPU receipt exists, CPU stages are separate:

```bash
python -B -m code.V4.r102_postprocess_recovery report --root "$ROOT" --arm A --device cpu
python -B -m code.V4.r102_postprocess_recovery report --root "$ROOT" --arm B --device cpu
python -B -m code.V4.r102_postprocess_recovery compare --root "$ROOT" --device cpu
```

Reports accept original reporter options `--nss-root`, `--historical-root`,
`--r94-root`; native LIB batch is fixed to the original value 2. No new NSS
training/inference is started. CPU reporting can run on a shared-filesystem
login node: it verifies original shared receipts/archives, not node-local
`/tmp` caches. The GPU process checks those caches' unchanged file signatures.

## Evidence And Outputs

Consume the existing proof without inventing a new hash infrastructure:
`schema=r102_cross_node_test_array_identity_v1`, `complete=true`,
`all_arrays_equal=true`, `problem_count=18`, `array_count=107`, and the existing
`nodes.gpu03/gpu05.{path,sha256}` raw manifests. Validate their actual file SHAs,
all problem/array names, all raw byte counts/SHA pairs, per-problem counts and
totals, reference-matching gpu03 headers, and the exact four gpu05 differences.
The existing `per_problem` records contain `arrays_equal`, `array_count`,
`bytes`, `gpu05_different_header_fields`. The existing aggregate fields
`gpu03_header_matches_frozen_reference`, `other_gpu05_header_fields_equal`,
`allowed_build_receipt_fields`, `total_bytes_per_node` are also checked.
Proof validation and `describe` do not open array payloads or import Torch.

Only `created_utc`, `completed_utc`, `seconds`, `test_lock_root` can differ, and
only with the exact actual/reference values recorded by the proof. All 18
TEST array files, including raw inputs, geometry, n, coord_dist, FP64 costs and
native winner, must be byte-identical across nodes. Main's proof establishes
payload equality; GPU continuation also checks live metadata, NPY headers and
file sizes, and monitors file signatures without rehashing array payloads.

The process-local adapters retain original schema/legal-pool validation and
do not mutate `RUNTIME_FIELDS` or any source file. R94 routing digests may bind
only the exact proved reference OR actual header digest; donor, original TEST
lock, direct source directory/input paths/specifications/sizes and FP16 schema
are still checked by original functions. No metadata/digest is rewritten.

`VIEW=$BASE/report_view` contains tiny byte-copied A/B metadata, the identical
original `best.pt` symlink and read-only-use TRAIN/VAL archive links. It does
NOT inherit the original `test_started.json`, failed status, error receipt or
any evaluation-directory link that could redirect TEST writes to the original.
An existing shadow is rejected. No large cache/weight copy occurs. New native
TEST features go to `$BASE/{arm}/test_features/test`; routing remains a direct
read-only reference to the original common R94B features. The CLI feature root
remains the original fixed path for authorization; an explicit private path
adapter redirects only the newly created TEST sidecars. Original feature roots
and all TRAIN/VAL features remain untouched.

`gpu` calls the original R102 TEST, timing and LIB functions in that order.
R102's original complete model factory restores all checkpoint tensors/buffers
and original protocol. Recovery additionally verifies the selected model hash
and update against the original VAL archive. No optimizer is constructed or
restored; no TRAIN updates, checkpoint choice, calibration, labels, solvers or
per-problem assembly are added. A is locked to update 33120 / SHA
`10694b4d0fed80f1baec0d5e7de15cceec210eca5dfa846623783c4b9d6e7a2c`;
B to update 20160 / SHA
`7df434c762a0e0b7726ab67211f63f51b55d1f86d108ec307a0aede0060f921f`.
Original source SHA pins are verified, not changed.

VAL capture identities currently contain `checkpoint_sha256=null`. This is
NOT asserted to be a completed capture-time SHA binding. Original archive and
summary remain unchanged, including null. The recovery provenance records the
null and the verified selected update/protocol/full-state binding; strict load
must reproduce that full model state before TEST/timing/LIB. Original reports
accept this existing VAL identity and do not require a fabricated archive SHA.

Donor feature extraction remains original FP32/TF32-off to FP16, stable exact
real-N batches of 128, original normalization, arguments, legal candidate pools
and native FP64 winner semantics. TEST retains the original entry's Torch
configuration; timing restores the process's initial defaults, as the original
separate timing entry would; LIB uses its original configure function.
The historical `gpu` attempt observes the original online forward without regrouping. Any observed
FP16 feature/mask/valid-score bit drift or changed pick is written to
`$BASE/{arm}/timing_checks/{problem}.json` and fails immediately. Approximate
equality or identical picks cannot silently excuse drift; LIB/report cannot
continue in that historical mode. The explicit `finish-gpu` exception above
supersedes this EXTRA gate only for the exact observed prior timing failures;
it does not claim numerical equality. No alternative inference implementation is substituted.

Main/admin consumes only real successful helper receipts:

| Receipt | Meaning |
| --- | --- |
| `$BASE/{arm}/origin.json` | Hashes of original failures, locks, archives, source proof and immutable assignments; not completion. |
| `$BASE/{arm}/gpu_complete.json` | `schema=r102_postprocess_recovery_v1`, `complete=true`, `exit_status=0`; first completed full TEST, timing, LIB with same checkpoint/state, hashed outputs and view path. |
| `$BASE/{arm}/report_complete.json` | Same schema/lock; original CPU report validated, required PDFs hashed. |
| `$BASE/compare_complete.json` | Both real reports validated, original CPU comparison completed. |
| `$BASE/{arm}/first_error.json` | Immutable FIRST continuation exception/traceback; only the explicit SHA-bound post-TEST exception may continue. |

Each stage has an exclusive `*_started.json`. No automatic retry/resume occurs,
even if an interrupted stage left partial new artifacts. The original marker
means an attempted, failed pre-feature TEST, not a completed TEST. The new marker
and archive explicitly document its first completion in the separate view.
Original failed worker/controller/status/watcher files are never rewritten or
reinterpreted as complete. Main must not mark delivery complete without the
real GPU/report/compare receipts. A new recovery failure needs an explicit
human/main decision; this helper never self-repairs or starts a replacement.

## Bounded CPU Verification

```bash
python -B -m unittest code.V4.test_r102_postprocess_recovery -v
python -B -m code.V4.r102_postprocess_recovery describe --root "$ROOT" --device cpu
```

Tests use small synthetic metadata, mocked stage dispatch, and a CUDA-init
tripwire. They do not load real weights, open large caches or execute TEST,
timing, LIB, reporting, allocation, Slurm, GPU processes, or Git. Actual GPU
execution, timing equivalence and final PDFs are main-owned pending evidence,
not claimed by these tests.
