# R104 Runner Preparation

Scope: only r104_experiment.py, r104_reporting.py, their two focused test files,
and this note. Model files, PLAN.md, protocol_lock.json, main_ready.json,
launch/goal code, and historical/live sources are not edited.

Main's existing lock is authoritative: fresh seed2, NEW90k full18, logical1024,
57600 successful updates, warmup4320, VAL1440, TRAIN7200; original AdamW
1e-4 -> 1e-6, WD1e-4, clip1, dropout.1, two-view stateless R-Drop. Both arms
use TSP/CVRP128, all other tasks512, eval128. No OOM auto-shrink, cache
construction/copy, teacher/features/routes/new targets, or historical reruns.

Implementation choices: use a private R84 engine with its own local micro_caps;
bind final selected-VAL summaries only after exact checkpoint/model-state,
protocol-state and update verification, keeping capture-time archive identity
untouched. Reporting converts the bound summary back to its preserved capture
identity only while validating/loading that original archive. Timing includes
the complete standalone larger model on original GPU-resident inputs, no donor.
R89A AdamW, R89B Muon and R84A are historical references, not matched controls.

## Integration Ready

Status: CPU integration draft complete, 2026-10-23. Owned source is ready for
main/launcher integration. No GPU witness, formal training, TEST, timing, LIB,
PDF rendering, or Git operation was run here. Main retains scientific/readiness
lock ownership; launch/goal ownership is separate. The model files are unchanged.

Runner exports `ARMS = NEURAL_ARMS = ('A_width256', 'B_depth8')`, `BUDGETS`,
`PARAMETER_COUNTS`, `required_lock_fields()`, `protocol_lock(args)`,
`private_engine()`, `make_model`, `initialize_shared`, `make_objective`,
`update_batch`, `evaluate`, `load_checkpoint` (also `load_model` and
`from_checkpoint`), `authorize_test`, `ensure_test_cache`, `train`, `test`,
`timing`, `parser`, and `main`.

`private_engine()` is an independent import of R84. Only its local
`micro_caps` and final-summary `write_json` are overridden. Its train/evaluate
functions therefore both use the same complete fixed cap map. No shared R84,
R99, R101, R102, or historical module global is patched. Original successful
update accounting, natural task/row streams, AMP finite-gradient replay, real
micro-row/logical-row normalization, clipping and stateless R-Drop are retained.
R84's five-VAL-point TRAIN cadence is exactly7200 updates; absent selected-TRAIN
predictions are evaluated at the same uniquely selected model, not reselected.

`required_lock_fields()` matches main's existing protocol_lock.json. Paths
must match its `data_root`, `prepared_root`, and `primary_cache`. Other
main-owned provenance/resource fields are preserved in the embedded lock.
`protocol()` records actual per-arm dimensions/counts, copied versus independent
capacity counts, seed104002, the full microbatch map, eval128, original loss/LR,
TRAIN7200, physical TSPnode0, raw-input-only semantics, maxima499/500/<=101,
the exact VAL binding rule and the non-matched-comparison boundary. Counts:
original1,842,953; wide7,288,969 with1,417 copied; deep3,426,057 with1,842,953 copied.
The complete initialization receipt is saved in each arm's initialization.json.

## Selected VAL Binding

Before writing final_metrics.json, `bind_selected_val` checks the immutable
checkpoint file SHA, exact checkpoint model-state SHA (parameters AND buffers),
protocol state hash, selected successful update, and original archived VAL
identity. Only the final VAL summary receives the checkpoint SHA. It preserves
`archive_capture_identity` plus an explicit `checkpoint_binding` receipt with
rule `exact_selected_checkpoint_model_state_protocol_state_and_successful_update`.
The original update archive, copied selected_best archive, identity.json and
metrics.json retain their capture-time null checkpoint SHA. No historical
administrative repair is inherited or applied.

The reporter verifies this binding, then uses the preserved capture identity
for archive-equality checks. TEST, TRAIN diagnostics and native LIB predictions
are still tied to the same VAL lock/model state. Existing R84 TEST arrays must
already be complete for every task; missing arrays fail rather than trigger
preparation or copying.

## CLI

Run from `/public/home/shiys/0selection2`, first activate easynco. Replace the
arm with `B_depth8` for the independent depth arm. CUDA assignment belongs to
the launcher; these commands are documentation, not executed launches.

```bash
conda activate easynco
python -m code.V4.r104_experiment describe --arm A_width256
python -m code.V4.r104_experiment train --arm A_width256 --device cuda:0 --gpu-type 3090 --large-micro-cap 128 --eval-batch-size 128 --wandb-mode offline
python -m code.V4.r104_experiment test --arm A_width256 --device cuda:0
python -m code.V4.r104_experiment timing --arm A_width256 --device cuda:0
python -m code.V4.r104_reporting lib --arm A_width256 --device cuda:0 --lib-batch-size 2
python -m code.V4.r104_reporting report --arm A_width256 --device cpu
python -m code.V4.r104_reporting compare --device cpu
```

Defaults: root `code/V4/runs/R104_capacity_scaling`, prepared root
`code/V4/runs/R84_new90k_attention/prepared`, fixed NEW90k data root from the
main lock, direct cache `/tmp/shiys_R84_new90k_cache`. Override `--root`,
`--protocol-lock`, `--prepared-root`, `--data-root`, or `--cache-root` only with
the same main-locked identities/paths. Training supports `--resume` and
`--state-hash`/`--expected-state-hash`; no budget/LR/cap-search flags exist.
The original train engine automatically proceeds to locked TEST on completion;
the explicit test command is idempotent using saved locked predictions.

Reports reuse original full18 metric/distribution PDFs: native/FP64 labels,
fixed aligned method order, same-axis choices versus true winners, per-dataset
model Top1, TEST then LIB for TSP/CVRP, and ALL/per-task SBS/Oracle/regret.
NSS4 reuses existing exact4500 TEST and native LIB reference predictions without
retraining/rerunning NSS. R89A (`--r89a-root`), R89B (`--r89-root`) and R84A
(`--historical-root`) are saved historical comparisons. Their aligned cost
changes and fixed36 rescues/losses/false picks/net cost are written as CSV and
`target_recovery_and_train.pdf`; count/initialization/timing boundaries appear
there and in report provenance. `compare` emits saved-evidence CSV/provenance
only, explicitly not a parameter-matched comparison.

Timing uses first128 locked TEST inputs/task, one warmup plus three synchronized
forwards of the whole actual model. It includes TSP role handling, geometry,
distance processing, all instance/joint layers and the original score path.
GPU-resident input timing excludes I/O, transfer, training and solver execution;
it is not end-to-end latency and uses no donor/cached-feature latency proxy.

## Minimal CPU Checks

```bash
conda activate easynco
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m unittest code.V4.test_r104_experiment code.V4.test_r104_reporting -v
```

7/7 passed in1.152s. Covered actual main-lock compatibility/CLI/schedule,
private train/eval cap resolution without shared mutation, per-arm protocol
dimensions/counts, row-weighted successful-update replay, explicit null-capture
VAL binding with wrong-state rejection, true rescue/false-pick/net-cost
arithmetic, and complete standalone all18 GPU-timing receipt validation.
No publication-polish or exhaustive campaign audit is claimed. No model
integration bug requiring a change to the frozen model files was found.
