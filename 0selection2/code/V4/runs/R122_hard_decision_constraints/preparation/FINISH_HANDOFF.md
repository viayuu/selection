# R122 Locked Finish Contract

Only NEW `r122_finish.py` and `test_r122_finish.py` plus this handoff are owned.
The running runner/objective, approved plan, preparation/source hashes and all
historical experiment files stay unchanged. No GPU, solver, copy, or Git launch
is authorized in this implementation session. Parent executes after full TEST.

## Decisions Declared Before Implementation

| ID | Decision | Source |
| --- | --- | --- |
| F1 | Exactly one current BASE/model checkpoint across all18 tasks, TEST and both LIBs. | user |
| F2 | Standard r80_lib input/geometry/inference implementation; finish factory adapter derives legacy epoch solely from selected_update/720, verifies the VAL lock and does not rewrite best.pt. | local helper compatibility |
| F3 | CPU-only report uses saved predictions; LIB is an explicit parent-assigned CUDA command. No full TEST retry, checkpoint selection, solver or NSS training/inference. | user |
| F4 | TRAIN controller budgets and independent VAL selection thresholds are separate; report selected feasibility/fallback without changing or hiding failures. | D2 / frozen protocol |
| F5 | Natural FULL18 and native-winner FIXED36/OUTSIDE36 error/regret partitions use full-problem denominators, equal-task ALL. Fixed36 is the historical immutable list, not a dynamic zero set. | user |
| F6 | Fixed36 true TP/rescues/lost TP and false/new false picks are all reported; zero-count reductions are not success. False picks are grouped by predicted method, not native-winner partitions. | user / existing recovery helper |
| F7 | R84A is architecture-matched only, not matched budget/objective. R94B is model_matched=false; both are historical same90k references, never newly matched controls. | user |
| F8 | Four NSS rows reuse R84 historical70114-trained NSS on aligned new4500 TEST/native LIB; missing results are explicitly missing, inconsistent existing results fail. | AGENTS / user |
| F9 | TRAIN sample curves are sampled interval optimization feedback; full TRAIN evaluation is drawn only at actual saved evaluations. No fabricated dense TRAIN trace. | existing R122 history |
| F10 | PDF is primary; CSV/JSON auxiliary. Current18 distribution pages contain20 dataset panels, TEST before LIB for TSP/CVRP, fixed candidate order, paired choice/native-winner bars and true per-instance model Top1. | AGENTS |

## Queued Entry Points

From `/public/home/shiys/0selection2`, only after the runner's complete14400 TEST:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && python -B -m code.V4.r122_finish finish --device cuda:0 --lib-batch-size 2
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && CUDA_VISIBLE_DEVICES='' python -B -m code.V4.r122_finish report --device cpu
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && CUDA_VISIBLE_DEVICES='' python -B -m unittest -v code.V4.test_r122_finish
```

The finish command runs locked LIB then saved-only CPU reporting, never train or
TEST. A missing training/TEST completion returns not-ready; no automatic retry.
Outputs are under `model/lib_evaluation` and `model/full_model_statistics`.
No model result or report is produced until the real gates pass. GPU commands
above are handoff commands, not actions executed by the implementation agent.

## Delivered Implementation

- `lib` uses the unchanged current factory through an epoch-only in-memory
  adapter. Original native LIB inputs and `attach_geometry` are used by the
  standard helper. No checkpoint or model/buffer rewrite is performed.
- `report` is CPU/saved-only. The main PDF includes full18 TRAIN/VAL/TEST
  metrics, separate TRAIN constraints and VAL thresholds, explicit feasibility
  or fallback, native LIB and four-dataset NSS tables, historical comparisons,
  all18 distribution pages (20 dataset panels), and an actual-history overview.
- Separate PDFs: `selection_distributions.pdf`, `learning_curves.pdf` (overview
  plus18 actual per-problem histories), `training_and_method_diagnostics.pdf`,
  and `fixed36_recovery.pdf`. Existing NSS predictions also get correctly paired
  distribution plots. No dense full-TRAIN history is fabricated.
- CSVs retain metrics, method Top1/cost comparisons, native-winner error/regret
  partitions, separate constraint roles, exact NSS alignment, R84A/R94B paired
  changes, fixed36 per-method true TP/rescues/losses/false picks and contributions.
- `report_complete.json` means delivery completed, NOT science success. Failed
  TRAIN constraints and/or infeasible VAL selection remain explicit in PDFs,
  `constraint_status.csv`, `science_claims.json` and completion receipts.
- Missing full train/TEST receipts stop before any LIB operation with exit75.
  Missing NSS/historical references are labeled missing, not replaced by fake
  metrics or a new run; inconsistent existing references fail closed.

CPU checks use only small synthetic fixtures and layout previews. Matplotlib's
installed pyparsing deprecation warnings are unrelated to metrics; no missing
CJK glyph warnings were observed. If the GPU container lacks the system CJK
font, the existing shared Noto cache is a fallback, or pass `--font` explicitly.

11 CPU tests pass, including a tiny saved-only artifact spine with mocked gates
and renderer. The real paired-bar/table layout was separately rendered and
visually inspected using clearly labeled synthetic examples. This is not an
actual R122 TEST/LIB/report execution. See `FINISH_CPU_CHECKS.json` for preserved
training hashes and the exact verification scope.
