# R121 Learner Finish API v1

Owner scope: only `r121_finish.py`, `test_r121_finish.py`, and this document.
No collector, learner, transport, publisher, shared reporter or trained source
is modified. No new experiment or inference rule. Main alone queues execution.

## Commands

The default root is the experiment root, NOT its `principal` subdirectory:
`code/V4/runs/R121_structural_families/preparation/learner_v1`.

`capture-train` accepts the existing `r121_experiment` common arguments, including
all release/binding, natural/family cache, capacity and logging arguments. It
requires completed14400 training and the unique original natural-VAL lock BEFORE
loading weights or input caches. It uses `best.pt` once in eval/inference mode,
calls the unchanged `r121_experiment.evaluate(..., 'train', 'selected_best', ...)`,
and separately records family predictions from the SAME model state. No optimizer,
collector, solver, feature rebuild, GPU allocation or checkpoint selection.

Main's assigned GPU/CPU environment is retained. This is a ready command, not an
authorization for this implementation worker to execute it:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
python -B -u -m code.V4.r121_finish capture-train \
  --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/learner_v1 \
  --device cuda:0 --gpu-type 4090 \
  --cache-root /tmp/shiys_R84_new90k_cache \
  --test-cache-root /tmp/shiys_R84_new90k_cache \
  --family-labels /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_explicit_cpu_v2/family_labels.jsonl \
  --family-cache /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/learner_v1/family_cache \
  --release /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1/main_r120_release.json \
  --binding /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_persistent_v1/main_binding_lane0.json \
  --micro-cap 352 --eval-batch-size 256 --lib-batch-size 1 --wandb-mode offline
```

After the existing TEST and both native LIBs plus the capture are complete:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
CUDA_VISIBLE_DEVICES='' python -B -m code.V4.r121_finish finish \
  --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/learner_v1 \
  --device cpu
```

`finish` calls the unchanged saved-only `r121_experiment.finish` first. Its original
`principal/delivery_manifest.json` and main PDFs retain their original schema and
filenames. Additional outputs live only under `principal/learner_finish/`.
`finish` never loads a model, forwards, opens feature caches or collects labels.

## Publisher Contract (Curie)

Final sidecar: `principal/learner_finish/reports/receipt.json`.
Capture: `principal/learner_finish/capture_train_receipt.json`.
Family forward: `principal/learner_finish/family_capture_receipt.json`.

Every receipt has `schema = r121_learner_finish_v1`, `experiment = R121`,
`arm = principal`, `checkpoint_sha256`, `selected_update`, `state_hash`,
`model_state_sha256`, `complete`, and `input_files_sha256` (absolute path -> SHA256).
Identity must match the existing unique `principal/final_test_lock.json` and the
TEST model-state identity. Source hashes are recorded at runtime, after main and
the training-curve owner finish their separate prelaunch source freeze.

The final receipt additionally has:

- `phase = complete_post_TEST_learner_finish`; `all18 = true`.
- `train_complete = true`; `train_rows_per_problem = 81000`.
- `train_eligible_rows_per_problem`: exact existing isolation counts by task.
- `TRAIN`, `TRAIN_ELIGIBLE`, `VAL`, `TEST`: native metric macros, task-equal.
- `family_diagnostics_complete = true`, `family_baselines = [zero_potential,
  family_train_mean_potential]`; no invented actual-MAE column.
- `fixed36_count = 36`, `fixed36_target_file` with path/hash, and
  `historical_references` with pinned R84A / R94B provenance and unmatched-control
  disclosures. Historical paired correction/cost tables are VAL/TEST only;
  historical full TRAIN is not assessed by this wrapper and TRAIN is not
  silently compared with a historical sampled TRAIN subset.
- `base_delivery_manifest`: path/hash of the unchanged base delivery receipt.
- `capture_train_receipt`, `family_capture_receipt`: path/hash/bytes records.
- `artifacts`: relative filename -> `{path, sha256, bytes}` for ALL supplemental
  PDF/CSV/protocol files, not just PDFs. Preserve the base artifact set separately.
- `source_sha256`, `model_inference = false`, `GPU_execution = false`,
  `solver_execution = false`, `training = false`, `checkpoint_reselection = false`,
  `research_goal_complete = false`. `complete` means delivery, NOT scientific benefit.

Required supplemental files: `R121_MECHANISM.pdf`, `natural_metrics.csv`,
`natural_fixed36_error_budgets.csv`, `family_diagnostics.csv`,
`target_recovery.csv`, `target_recovery_summary.csv`, `paired_changes.csv`,
`historical_references.csv`, `protocol.json`. The final receipt signs all of them.
Publication should verify every captured input hash and artifact hash; no transport
or generic publisher implementation is changed by this worker.

## Definitions And Boundaries

TRAIN has two explicit populations: full original81000/task (normalization and
TRAIN-fitted SBS unchanged), and the saved eligible-gradient subset after the
existing known family-val source-record isolation. Never call them the same set.
Native int64 winner / FP64 costs and current legal pools are retained. ALL averages
tasks equally; raw cross-task cost means are descriptive only.

Fixed36 is the frozen historical R84A VAL target list, never reselected. Reuse the
R122 pure-array recovery definitions: true recall, corrected and harmed native
decisions, new false selections, cost and regret deltas. Winner-group cost
contributions and predicted-method false-pick groups use their respective explicit
denominators. Zero-selected-group reduction is descriptive, NOT success.
The original target-list SHA256 is pinned in the new wrapper:
`406ded5a21d8b73c2a41b87dc2bc3cc14330663ae1aa4f59ca6fbfde83770fe4`.

Family targets are the actual native `center(log(cost)) / .01`. Compare actual
saved potential predictions with (a) zero potential and (b) per-method mean
potential fitted ONLY on unique family_train endpoints. The latter is identical
at both endpoints, so its response is exactly zero. Report both unique-endpoint
Huber and original-edge-slot-weighted endpoint Huber, plus paired response Huber.
Response uses changed minus original from the same centered potential. Family_val
does not fit the mean, select a checkpoint or update weights. Query costs/labels,
family metadata and partition tags never enter model forward.

Known source-record/selected-geometry gradient exclusion is NOT exhaustive content
dedup, independent family benchmarking, or an exposure-matched historical control.
TEST was exposed during research. No claim of accuracy/regret improvement follows
from reduced auxiliary Huber or broader selection alone.

## CPU Verification

Only bounded synthetic fixtures are executed by this worker; no actual TRAIN,
family model inference, solver, GPU or Git command is run. The fixture suite covers
common argument compatibility, pre-model lock failure, label-isolated variable
microbatches, TRAIN-only baseline fitting, unique versus repeated edge weighting,
source-mapped eligibility, genuine fixed36 rescue/false picks, and the saved-only
PDF/receipt path with model loading/forwarding forbidden.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -B -m unittest code.V4.test_r121_finish -v
```
