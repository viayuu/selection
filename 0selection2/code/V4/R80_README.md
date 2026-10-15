# R80: Training Progress And Locked Reports

Workspace: `/public/home/shiys/0selection2`.
Run root: `code/V4/runs/R80_training_and_code_ablation`.

Every shell must initialize conda and activate `easynco` first:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate easynco
```

## Integration Contract (Laplace / Main Runner)

The launcher owns GPU assignment and training. The reporter never starts or
resumes training, calls a solver/embedding API, or selects a checkpoint.
After the arm's training and full TEST succeed, call:

```bash
python -m code.V4.r80_reporting --root code/V4/runs/R80_training_and_code_ablation --arm A_ce_only
```

Arms are `A_ce_only`, `B_batch512`, `C_lr3e4`, `D_code_embedding`.
The arm directory is `ROOT/ARM`. Reporter output is `ROOT/ARM/full_model_statistics`.
LIB inference defaults to CPU; the launcher may pass `--device cuda:0` within its
existing `CUDA_VISIBLE_DEVICES` allocation. No GPU discovery or allocation occurs.

Expected runner interface (R77-compatible, but R80 factory is mandatory):

- `code.V4.r80_experiment.load_checkpoint(path, device)` returns `(model, state)`.
  It must use the same R80 model factory and restore every checkpoint buffer,
  including solver source embeddings, solver IDs, geometry mean/std.
- `ROOT/ARM/protocol.json`: `experiment=R80`, `arm=ARM`, `model_params`,
  `data_root`, `baselines` (FULL TRAIN-fitted SBS per problem),
  `max_successful_updates`, `checkpoint_rule`.
- `training_complete.json`: `successful_updates` equals the arm budget.
- `final_test_lock.json`: `experiment=R80`, `arm=ARM`, `checkpoint`,
  `checkpoint_sha256`, `epoch` (one-based), `selected_update`,
  `selected_without_test=true`, `per_problem_assembly=false`, `rule`,
  `state_hash` (canonical JSON SHA256 of protocol), `inference_rule`.
  R80 checkpoint `state['epoch']` is also one-based (unlike R77); no +1 conversion.
- `test_results.json`: `complete=true`, `checkpoint_sha256`, `epoch`,
  `selected_update`, `result.per_problem` / `result.macro`.
- `final_metrics.json`: same checkpoint identity; `val.per_problem` / `val.macro`.
- TEST archive: `evaluation/test/selected_best/rank000.npz`.
- VAL archive: `evaluation/val/selected_best/rank000.npz`, or the selected
  `evaluation/val/updateNNNNNNN/rank000.npz` (exact tag from `best_eval.json`).
- Each archive uses `PROBLEM__{logits,indices,costs,winner,pool,pool_ids,nodes}`;
  costs are native FP64, winner is native `ind`, pool IDs are stable global IDs.
  Optional `pred` must agree with locked score argmax/global-ID tie-breaking.
- Each evaluation directory has `identity.json`, copied into metrics `identity`:
  R80 arm/split, `successful_updates`, `state_hash`, `model_state_sha256`.
  TEST also records the locked `checkpoint_sha256`. Selected VAL and TEST must
  have exactly the same model-state hash, including buffers.

The reporter verifies the on-disk checkpoint SHA256 and requires the R80 loader;
it never falls back to an R77/R76 checkpoint or their LIB prediction files.
The Laplace runner and Mill wrapper have been checked against this interface.
Mill's `--root ROOT --arm ARM --device cuda:0` call is accepted unchanged.
`send_input` is not exposed to this worker; this README is the shared handoff.
Interface adjustments are confined to the reporter.

## Four Arms

All arms: seed 2, from scratch, new single-RELD full-data 18-problem release.

| Arm | Change | Logical Batch | Successful Updates |
|---|---|---:|---:|
| A_ce_only | CE only | 1024 | 49680 |
| B_batch512 | batch size | 512 | 99360 |
| C_lr3e4 | peak LR 3e-4 | 1024 | 49680 |
| D_code_embedding | R45B source embedding + learned ID | 1024 | 49680 |

## Progress (One-Shot, No Monitor Daemon)

```bash
python -m code.V4.r80_reporting --root code/V4/runs/R80_training_and_code_ablation --status
```

This reads existing status/receipt JSON files and writes `ROOT/STATUS.md` only.
`ROOT/PROGRESS.md` is the stable progress entry. Per-arm `status.json` and logs
remain the runner's authoritative training records; reporting has a separate
`postprocess_status.json`.

```bash
tail -n 40 code/V4/runs/R80_training_and_code_ablation/A_ce_only/train.log
tail -n 40 code/V4/runs/R80_training_and_code_ablation/A_ce_only/reporting.log
tmux list-sessions
```

W&B project: `selector`; entity:
`yjkds-southern-university-of-science-technology`.
Use the existing runner's W&B URL or local `wandb/latest-run/logs/debug.log`.
Offline runs remain local; `wandb sync ROOT/ARM/wandb/offline-run-*` is an
explicit user action, not something the status command starts.

## Delivery And Retry

Each arm delivers `MODEL_STATISTICS.pdf`, `selection_distributions.pdf`,
per-problem/ALL metrics CSV, NSS four-dataset comparison CSV, distribution counts
CSV, and provenance. `training_and_method_diagnostics.pdf` adds saved TRAIN loss,
VAL traces and VAL/TEST/LIB Top1/cost comparisons against each single method.
TSP/CVRP pages have four panels in order: LIB model, LIB
native winner, TEST model, TEST native winner. Other problems have TEST two-panel
pages. Model-panel Top1 is per-instance native-winner agreement, never the largest
selection frequency. Winner panels carry no model accuracy.

SBS is fixed on FULL TRAIN. Oracle is the per-instance minimum over the legal
candidate pool. ALL is the equal-weight average of the 18 problem metrics;
cross-problem raw cost averages are descriptive only. `vs_SBS` / `vs_Oracle`
are ratios of means; actual regret is the mean of per-instance relative regrets.

NSS reuses R79 predictions: native TSP TEST Top1 is 44.338896%, whereas historical
FP32-argmin Top1 is 43.953786%. These definitions are never merged. New-data TEST
and original native LIB versions/pools are identified separately. Existing TEST
has partly informed multi-round R&D; it is not described as untouched blind data.

On reporting failure, keep training/checkpoints/TEST untouched. Inspect
`ROOT/ARM/postprocess_status.json` and rerun only the reporter command above.
Checked same-lock LIB predictions are reused. If LIB inference itself was
unfinished, keep the recorded `--device` and LIB batch size on retry (the launcher
uses `--device cuda:0` inside its assigned allocation). Fully saved LIB predictions
can be reported on CPU. No background monitor is needed.

Reporter checks passed: CLI/compile, synthetic 18-problem PDFs (22/18 pages),
mock same-lock LIB reuse, one-based R80 epoch, strict checkpoint buffers,
wrong-lock refusal, failure preservation, and malformed status handling.
These checks did not execute real TEST/LIB inference or `test_r80.py`.

## Implementation Decisions

| ID | Decision | Source |
|---|---|---|
| I-01 | Load only the R80 worker's unified checkpoint factory, strict buffer check | user |
| I-02 | Separate postprocess status, preserve completed training on any failure | user |
| I-03 | Default LIB device CPU; GPU assignment belongs to launcher | default |
| S-01 | Native/FP64 `ind` is primary; FP32 argmin is explicitly auxiliary | user |
| S-02 | Main SBS identities fixed on FULL TRAIN; ALL equal problem weights | user |
| S-03 | No TEST/LIB-based checkpoint, threshold, or inference-rule selection | user |
