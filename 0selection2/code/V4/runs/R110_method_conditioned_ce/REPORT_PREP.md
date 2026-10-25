# R110 Reporting Preparation

Scope: `r110_reporting.py`, `test_r110_reporting.py`, and this note only.
No other source, runner, model, protocol lock, execution plan, launch, GPU,
production report, or Git operation belongs to this worker.

## Source And Interface

- Standalone port of finalized R109 reporting, re-read after Poincare's
  `REPORT_PREP.md` states that no further R109 edits will follow.
- Source SHA256: `r109_reporting.py`
  `b5313ff03bfb02e6498480716546164751b29b443fa116dc9b52bc56f4b83f34`;
  `test_r109_reporting.py`
  `327efa2341f2c4e55eadd08bf45c4aa59b1cda69f88da16d2a61763650bb85b6`.
- Root: `code/V4/runs/R110_method_conditioned_ce`.
- Arms: `A_tied_ce`, `B_private_ce`; factory:
  `code.V4.r110_experiment.load_checkpoint`.
- Restore original model params unchanged: `r109_arm=A_tied_late` or
  `B_private_late`, respectively, and the original `r109_spec`. Never invent
  `r110_arm` inside the restored model parameter dictionary.
- Counts come from the runner/model: A 2305033, B 10220553, original 1842953.
  Common fresh original tensors and identical late prototypes, not identical
  complete state hashes or total parameter counts.
- Lazy CLI: `status`, `lib`, `report`, `compare`. `report` and `compare` use
  saved evidence on CPU; this worker never executes production stages.
- Historical R89A/R89B/R84A saved references retain R109's defaults. R94B,
  R93A, R109A and R109B are explicit optional saved references, never implicit
  dependencies. Missing/invalid optional evidence is disclosed and isolated.

## Scientific And Delivery Boundary

TRAIN optimizes one-pass, unweighted, legal-pool raw-utility CE with weight 1.
Comparison, relative-risk and RDrop/KL weights are zero; no priors, class
weights, task context, calibration, auxiliary targets or quota. Saved evaluation
maximin CE is a diagnostic, not the training raw-utility CE objective.
Per-arm PDFs, curve legends, provenance and root comparison state this boundary.

Preserve full18 selected TRAIN/VAL/TEST, native/FP64 winner labels, TRAIN-fitted
SBS, task-equal ALL, fixed36 true rescues/losses and false picks versus full
population cost/regret. Preserve combined same-axis model-choice/true-winner
bars, actual instance-wise Top1, fixed method order, TSP/CVRP TEST then LIB,
NSS TSP TEST/CVRP TEST/TSPLIB/CVRPLIB, all five per-arm PDFs, and root
`MODEL_COMPARISON.pdf` plus `RESULTS.md` after both complete reports.

NSS readers use `utf-8-sig`; writers emit BOM-free UTF-8. All four serialized
rows must carry `data_version`, `candidate_pool`, native winner rule and the
same current checkpoint/state/update/factory identity. No old TEST substitution
or NSS retraining.

Main's witnessed training/eval caps are 256/1024/eval256. Standalone timing is
an independently recorded sampling contract: the inspected R110 runner uses
128 input-only instances per problem and exports `validate_timing_record`.
Reporter delegates to that API and also checks actual sample policy and
per-instance denominators, rather than equating timing samples with eval cap.
It requires complete method-conditioned node refinement, all18 tasks, no donor,
input labels, cached-feature proxy or end-to-end-latency claim.

## Decisions Before Implementation

| Decision | Choice | Source |
| --- | --- | --- |
| Port source timing | Final R109 snapshot after its explicit handoff | User |
| Shared model identity | Keep mapped `r109_arm` and `r109_spec` | User |
| Loss plot meaning | One-pass utility CE; inactive components explicitly zero; eval maximin CE diagnostic | User |
| Optional history | Explicit saved-reference paths, not mandatory R109 results | User |
| Timing sample count | Runner validator plus actual receipt, separate from eval batch | Current runner API |
| NSS serialized schema | BOM-tolerant read and strict four-row provenance validation | User |

## Verification

Runtime reporter handed off as stable, with no further runtime edits by this
worker. `r110_reporting.py` SHA256:
`a3d42b3e26f7a67372848848a87401219aab4e25f019f68f81069673be15740d`.

CLI `--help` smoke passed, exposing `lib`, `report`, `compare`, `status`, both
CE arms and optional `--r94b-root`, `--r93a-root`, `--r109a-root`,
`--r109b-root`. These optional paths name saved arm directories. The real
factory remains lazy; no checkpoint or label access is needed by status.

`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
MKL_NUM_THREADS=2 python -B -m unittest code.V4.test_r110_reporting -q`:
25/25 CPU contracts passed in 2.540 seconds. This is the finalized R109's
bounded synthetic suite ported to R110, including BOM/schema and same-current
identity checks, fixed36/net-cost metrics, capture preservation, tiny factory
guards, combined bars and a temporary five-page root PDF/Markdown.

Installed CJK family checked: Noto Sans CJK SC. Only existing
Matplotlib/Pyparsing deprecation warnings appeared. Synthetic artifacts stay
in temporary directories and are not model evaluation or production timing
evidence. Main's separately reported launch/goal/report and runner/objective
passes remain main's evidence, not an additional audit performed here.

No GPU, production LIB/report/compare, real inference, training, launch,
protocol update, result audit or Git operation was performed by this worker.
Only the three owned files were created. Main may freeze and start immediately;
future notes, if needed, will not modify this runtime source.
