# R109 Reporting Preparation

Status: implementation final; 25 reporting CPU tests passed, including synthetic
five-page root PDF rendering. Ready for main's source freeze. Not a production
evaluation, launch approval, timing receipt, completed model report, or commit.
After handoff this reporting worker will not edit these R109 files again.

## Scope And Interface

- Owned files: `code/V4/r109_reporting.py`, `code/V4/test_r109_reporting.py`,
  and this file. Do not change other workers' model, runner, launcher, or locks.
- Direct R107 port, using private R80/R84/R85/R87 module copies. Historical
  module globals are never rebound. No historical factory fallback.
- Factory: `code.V4.r109_experiment.load_checkpoint(path, device='cpu')`.
  Required runner APIs: `ARMS`, `BUDGETS`, `PARAMETER_COUNTS`, `MODEL_FACTORY`,
  `VAL_BINDING_RULE`, `INFERENCE_RULE`, `selected_capture_proof`,
  `ORIGINAL_FRESH_SHA256`, `BRANCH_INIT_SEED`, and `CAPS.eval_batch_size`.
- The factory is lazy: CLI/help/status do not construct models or read labels.
  `report` and `compare` consume saved evidence on CPU without inference or
  checkpoint reselection. Only launcher-owned `lib` may perform LIB inference.

## Scientific Boundary

- Arms: `A_tied_late` (2,305,033 trainables) and `B_private_late`
  (10,220,553). Not parameter/FLOP-matched; complete state hashes/dimensions
  are not identical. Common original fresh base: 1,842,953 trainables.
- Each candidate refines shared original H4 with two graph layers before
  pooling. A ties weights; B uses 21 banks keyed by global method ID.
  Identical initial branch prototypes and common base give the same initial
  eval function. Preserve original two joint layers, score MLP/maximin, and
  original R77 CE/comparison/risk/two-forward RDropObjective.
- No new route/path fields, labels, donor, teacher, or context. Main owns
  the witness and final caps. Runner now exposes 128/1024/eval128; reporting
  reads its locked eval count rather than hardcoding an independent cap.
- Retain immutable selected VAL capture plus exact checkpoint/state/update
  binding and same-lock complete TRAIN/VAL/TEST/LIB metrics.
- Native/FP64 ind, unchanged legal pools, TRAIN-fitted SBS, task-equal ALL.
  TEST is development-exposed, not a new blind holdout.
- Combined18 distribution PDF: each dataset has two colored bars on one
  axis (model selection versus true winner), fixed candidate order, counts,
  percentages, and actual instance-wise model Top1. TSP/CVRP TEST then LIB.
- NSS4: TSP TEST, CVRP TEST, TSPLIB, CVRPLIB from the same current lock.
  Reuse historical R79 weights and saved new4500 TEST/native LIB predictions;
  never retrain NSS or substitute old3895 TEST metrics.
- R109 CSV writer emits UTF-8 without BOM; readers use `utf-8-sig`. NSS4 rows
  explicitly contain `data_version`, actual aligned `candidate_pool`, and
  native winner/checkpoint metadata. TEST names the frozen new90k release;
  each LIB names its original native dataset and historical R79 comparison.
  Re-read and validate the actual CSV before report completion. Missing
  provenance, wrong lock, nonfinite metrics or incomplete NSS4 is a failure;
  no delivery gate is weakened. Frozen R108 source is untouched.
- Keep frozen R84 target36 true rescues/losses, false picks and aligned
  population net cost/regret separate from selection counts. Saved
  R89A/R89B/R84A references are required; R94B is optional and isolated.

## CLI Handoff

Invocation examples for the owning launcher, not executed by this worker.
Run from `/public/home/shiys/0selection2`; use only its assigned CUDA device.

```bash
conda activate easynco
python -B -m code.V4.r109_reporting lib --arm A_tied_late --device cuda:0 --lib-batch-size 2
python -B -m code.V4.r109_reporting report --arm A_tied_late --device cpu
python -B -m code.V4.r109_reporting lib --arm B_private_late --device cuda:0 --lib-batch-size 2
python -B -m code.V4.r109_reporting report --arm B_private_late --device cpu
python -B -m code.V4.r109_reporting compare --device cpu
python -B -m code.V4.r109_reporting status --arm A_tied_late
```

Default root: `code/V4/runs/R109_method_conditioned_nodes`; override `--root`.
Reference overrides retain R107's `--historical-root`, `--r89a-root`,
`--r89-root`, `--nss-root`, and optional `--r94b-root`.

Order: main's full selected TRAIN/VAL and locked TEST, main's timing, `lib`,
CPU `report`, then paired `compare` after both locks are available.
Timing requires `method_conditioned_node_refinement_included=True`, complete
standalone model, real assigned CUDA execution, no cached-feature proxy,
no input labels and no donor. Sample count follows main's `CAPS.eval_batch_size`;
all18 rows have three positive finite repetitions and the correct denominator.
Excludes dataset I/O, CPU transfer, training and solvers, so not end-to-end.

Per-arm outputs: `full_model_statistics/MODEL_STATISTICS.pdf`,
`selection_distributions.pdf`, `learning_curves.pdf`,
`training_and_method_diagnostics.pdf`, `target_recovery_and_train.pdf`,
metrics/method/distribution/NSS/timing/recovery CSVs, and `protocol.json`.
Completion is published only after all five required PDFs exist and are nonempty.
Paired outputs: root `comparison_artifacts/*_B_vs_A.csv` and `protocol.json`,
plus root `MODEL_COMPARISON.pdf` and `RESULTS.md`. The combined PDF has five
pages: ALL summary, TRAIN/VAL/TEST each with ALL+18 paired rows, and frozen36
recall/rescues/losses/false-pick costs plus population net cost/regret.
It is produced only after both per-arm complete same-lock reports, all five
per-arm PDFs and all59 TRAIN/VAL/TEST/LIB metric rows exist. Root completion
metadata is written only after both combined PDF and Markdown are nonempty.

## CPU Verification

```bash
conda activate easynco
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m unittest code.V4.test_r109_reporting -v
```

Reporting result: 25 tests passed (2.668 seconds in the recorded run; test time,
not model latency). Coverage includes private historical-module isolation,
exact selected VAL/TRAIN binding without archive rewrites, tiny CPU factory
count/hash checks, unequal complete state/count requirements, main-driven
all18 timing, real runner input-only timing/cached-input reader with mocked
arrays, actual runner selected-capture proof on synthetic checkpoint tensors,
NSS4 dataset provenance and BOM handling, fixed36 true rescue/false-pick
cost/regret, per-arm completion gating, and synthetic five-page root PDF plus
Markdown rendering. Synthetic files are confined to temporary directories.

CLI help smoke passed. Installed CJK font confirmed as Noto Sans CJK SC;
the in-memory combined TSP/CVRP plots and synthetic PDF rendered without
missing-glyph warnings. Only existing Matplotlib/Pyparsing deprecations were
seen. Main separately reports its model/runner/launch/goal/report tests pass;
that is main's evidence, not a GPU or formal run performed by this worker.
No GPU, real cache opening/generation, training, launch, real evaluation,
production report generation, or Git command was performed here.

Final regression: R109 reporting25 + historical R107 reporting18 + historical
R108 reporting21 = 64 CPU tests passed in 3.384 seconds. Real runner import
smoke confirmed the exact factory, arm counts 2,305,033/10,220,553, main's
128/1024/eval128 caps, R109's private CSV writer, and the unchanged historical
CSV writer binding. No further R109 reporting edits after this handoff.
