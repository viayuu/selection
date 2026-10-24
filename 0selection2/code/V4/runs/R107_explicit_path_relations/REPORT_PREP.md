# R107 Reporting Preparation

Status: implementation and 18 CPU tests passed. Not an evaluation or delivery receipt.
No R107 training, GPU inference, timing run, LIB run, real PDF, or commit is
performed by the reporting worker. Only synthetic CPU tests are authorized here.

## Scope And Decisions

- Owned files: `code/V4/r107_reporting.py`, `code/V4/test_r107_reporting.py`, and
  this file. Historical reporting/experiment files remain unchanged.
- Reuse private R80/R84/R85/R87 reporting modules, following R105's full18
  reporting pipeline. Never rebind globals in historical modules themselves.
- Arms: `A_edge_factorized` and `B_connected_paths`. Describe the contrast as
  factorized versus joint connected-path computation, not MoE FFN/routing.
- Same complete fresh initialization and total parameter counts must be proved
  from the actual per-arm factory/initialization evidence. Counts are not guessed
  here. Equal active functional freedom or equal FLOPs are not implied.
- Preserve fresh R84/R88B backbone, original RDropObjective, 57,600 updates,
  logical batch 1024, full18 frozen data/pools, and one whole18 VAL-selected
  checkpoint per arm across full TRAIN/VAL/TEST/LIB. No donor or new target.
- Retain frozen R84 target36, rescues/losses/false picks, population-aligned net
  costs and regret, combined method-selection/true-winner plots, and NSS4 table.
- Historical R89A/R89B/R84A references remain mandatory saved references.
  R94B is optional and must never block the base report when absent or invalid;
  record its availability/error explicitly, never invent its metrics.
- No routing receipt, routing summary CSV, entropy plot, or routing launch stage
  is required. Timing must include graph/path construction inside the complete
  model and exclude solver execution. This is not dataset-I/O-inclusive timing.

## Interface Check

The main worker's `r107_experiment.py` has appeared. Reporting uses its
`load_checkpoint`, `selected_capture_proof`, `VAL_BINDING_RULE`,
`INFERENCE_RULE`, `MODEL_FACTORY`, `PARAMETER_COUNTS`, and timing schema.
The runner and model source interfaces have both appeared. The factory is
imported lazily: CLI/status and CPU mock tests do not construct a real model.
No dependency is silently substituted with an
older experiment/model, and parameter counts are read at evidence-validation
time, not duplicated from MODEL_PREP estimates.

Timing matches main's `explicit_path_computation_included=True` schema. It does
not request `all_four_experts_computed` or `routing_diagnostic`. Complete forward
timing includes graph/path construction; no cached-feature proxy is accepted.

## CLI Handoff To Main And Gauss

Admin recipient: Gauss `01a169ad-31af-78a1-8986-a1856bda6c85`.
These are invocation examples for the owning launcher, NOT commands executed by
the reporting worker. Run from `/public/home/shiys/0selection2`. The launcher
supplies its explicitly assigned CUDA device; do not auto-select another GPU.

```bash
conda activate easynco
python -B -m code.V4.r107_reporting lib --root code/V4/runs/R107_explicit_path_relations --arm A_edge_factorized --device cuda:0 --lib-batch-size 2
python -B -m code.V4.r107_reporting report --root code/V4/runs/R107_explicit_path_relations --arm A_edge_factorized --device cpu
python -B -m code.V4.r107_reporting lib --root code/V4/runs/R107_explicit_path_relations --arm B_connected_paths --device cuda:0 --lib-batch-size 2
python -B -m code.V4.r107_reporting report --root code/V4/runs/R107_explicit_path_relations --arm B_connected_paths --device cpu
python -B -m code.V4.r107_reporting compare --root code/V4/runs/R107_explicit_path_relations --device cpu
python -B -m code.V4.r107_reporting status --root code/V4/runs/R107_explicit_path_relations --arm A_edge_factorized
```

Default saved references: `--historical-root` = R84 `A_baseline`,
`--r89a-root` = R89 `A_adamw`, `--r89-root` = R89 `B_muon_hybrid`,
`--nss-root` = R84 `nss_reference`. Optional:
`--r94b-root code/V4/runs/R94_pretrained_node_fusion/B`. Omit it for the base
report; if supplied but unavailable/incompatible, record the reason and continue
without inventing R94B metrics. No reference inference or training is invoked.

Dependency order: main's locked full TEST and full selected TRAIN/VAL archives,
then main's standalone timing, then reporter `lib`, then CPU `report`.
`compare` consumes both arms' saved TRAIN/VAL/TEST and complete initialization
proofs. A missing/mismatched required artifact is a failure, not a completed PDF.
`lib` produces its own same-checkpoint/model-state/update completion metadata.
No routing stage or summary is required. Existing launcher LIB/PDF options match
this CLI; no launch spec/source-freeze edits are performed by this worker.

## CPU Verification

```bash
conda activate easynco
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m unittest code.V4.test_r107_reporting -v
```

Result: 18 tests passed (unittest body: 0.762 seconds). This is test-suite time,
not model inference latency. Coverage includes private historical-module
isolation, exact selected VAL/TRAIN binding without archive rewrites, actual
factory count/hash checks with a tiny CPU model, same complete A/B initialization
and count requirements, all18 complete-forward timing validation, fixed36 target
identity, paired rescues/losses/false-pick costs and regret, same-checkpoint NSS4,
optional R94B failure isolation, and no premature PDF-completion metadata.

The initially reported plot error was a test-only NumPy/MagicMock shape issue:
NumPy interpreted a list of MagicMock axes as nested array-like objects. The
fixture now explicitly allocates a one-dimensional object array and assigns
each mock axis individually. No historical plotting source was changed.

Also drew real two-candidate TSP and CVRP plots on the in-memory Agg canvas.
Verified TEST then TSPLIB/CVRPLIB, aligned candidate order, two colored bar
series per axis, legend and native/FP64 labeling, and actual prediction/winner
Top1=25.00% despite a 100% most-selected-method frequency. Figures were closed
without saving any image or PDF. Installed CJK font confirmed with `fc-match`:
`/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc`. Only pre-existing
Matplotlib/Pyparsing deprecation warnings appeared, not missing-glyph warnings.

CLI `--help` and real runner factory import/arms/count-map/binding API smoke
checks passed; the import smoke did not construct a model or run inference.
Whitespace checks passed for both owned Python files. Historical R105/R84/R85/R87
report sources have no worktree diff. No actual checkpoint/data/NSS archives
were opened, no full pipeline or latency evidence was produced, and no formal
protocol, launch, GPU probe, ready receipt, source-freeze update or commit was
performed. Main owns its pending capacity witness and final microcap lock; this
reporter does not freeze or duplicate those caps.

The exact CLI is shared above for main/Gauss. The reporting worker has no
directed inter-agent messaging tool, so no private-message delivery is claimed.
