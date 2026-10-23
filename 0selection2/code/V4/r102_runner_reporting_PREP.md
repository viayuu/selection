# R102 Runner and Reporting Preparation

Implementation only. No launch, GPU discovery, real training/inference, large
cache copy, package installation, Git operation, publication, credentials or
external notifications. User-authorized internal owner coordination only.
Existing R99/R101 code and running processes remain untouched. Main owns the
model, objective, PLAN and final protocol; another owner owns donor, features,
transfer and launcher. This note is not a protocol lock or launch authorization.

## Interfaces Declared Before Implementation

Runner: `python -m code.V4.r102_experiment {prepare,train,test,timing,describe}`.
Options: `--root`, `--arm A|B`, `--prepared-root`, `--data-root`, `--cache-root`,
`--feature-root`, `--routing-feature-root`, `--protocol-lock`, `--gpu-type 4090`,
`--large-micro-cap 352`, `--eval-batch-size 256`, `--device`,
`--wandb-mode offline`, `--resume`, `--state-hash`, `--log-every`.
Run directories: `ROOT/A` and `ROOT/B`.
Default root: `code/V4/runs/R102_full_type_pretrained_fusion`.
Default extra features: `/tmp/shiys_R102_A_extra_nodes` (or `B`).
Default common routing features: `/tmp/shiys_R94_B_node_features`, reused
DIRECTLY and read-only in the original gpu05 container, with no transfer or
R94 metadata rewrite. Main's resource-only correction places A on4090_0 after
R101B and B on4090_1 after R101A complete their full pipelines; no launch here.

Reporting: `python -m code.V4.r102_reporting {lib,report,compare,status}` with
`--root`, `--arm`, `--device`, `--lib-batch-size`, `--nss-root`,
`--historical-root` (R84A target36 anchor), `--r94-root` (historical R94B).
`report` and `compare` are strictly CPU saved-evidence operations. `lib` is an
explicit future inference stage, never started by report/compare.

Model API: `make_model(params, arm, state_dict=None, *, donor=None)`, `SPEC`,
`ARMS=('A','B')`, `ARM_VARIANTS={'A':'extra_random','B':'full_pretrained'}`.
The runner delegates model construction and never changes projection seeds.
The objective is late-bound as `r102_objective.RDropObjective`.
Feature API is exactly the user-specified `prepare_features` and
`FrozenFeatureCache` contract; no historical feature globals are changed.
Actual provenance is a full donor dictionary with `components.TSP/ATSP/VRP`.
All18 receipts live under the arm extra root. Routing receipts point to
`routing_directory` under the original R94B root, with `source_kind` equal to
`r94b_reference`; no routing payload is copied or re-extracted.

The final main-owned lock must contain `experiment='R102'`, `locked=true`,
`arms=['A','B']`, `seed=2`, `task_scope='full18'`,
`rows_per_problem={'train':81000,'val':4500,'test':4500}`, `batch_size=1024`,
`logical_global_batch=1024`, `max_updates=57600`, `warmup_updates=4320`,
`eval_every=1440`, `train_eval_every=7200`, `large_micro_cap=352`,
`eval_batch_size=256`, `donor_variants` as above, `data_root`, `prepared_root`.
Additional main-owned provenance fields are preserved, not inferred.

## Assumption Ledger

| ID | Interface choice | Decision | Source |
| --- | --- | --- | --- |
| I1 | Objective export name not specified | Use the existing R94-style `RDropObjective` name; fail if unavailable | default |
| I2 | Feature receipts container | Require an all18 task map, accepting either direct map or `splits[split]`; validate every receipt | default |
| I3 | Report historical references | R84A anchors fixed target36; R94B is a historical reference, never matched | user |
| I4 | Inference timing scope | First128 aligned TEST rows/task, one warmup and three measured GPU-resident batches; not end-to-end or singleton | user |
| I5 | TRAIN preparation | Both TRAIN/VAL receipt sets mandatory; source batch128; no on-the-fly missing-cache fallback | user |
| I6 | Original R84 lock lacks explicit selection split | Private R102 `lock_best` retains original unique first-tie minimum selection and adds `selection_split='val'`, required by feature TEST authorization | integration |

## Acceptance Plan

- Small CPU fixtures only: CLI, protocol/config gates, exact added parameter
  names/shapes/counts, feature receipt coverage, TEST lock boundaries, private
  engine isolation, external-file-free checkpoint restoration call contract.
- Timing wiring uses a tiny synthetic donor/selector, not a real model: both
  cache fields removed on every online call; hook execution count checked;
  cached selector reported separately and rejected as online total.
- Saved-array report fixtures: genuine rescue/loss/new false-pick cost, strict
  matched A/B routing provenance, all18 timing scope, native LIB49/100,
  same-axis winner/selection plots and all18 metric schemas.
- No numerical model-quality claim. Main reviews final model/feature integration
  and locks the protocol before any future authorized launch.

## Verified Preparation Status

`python -B -m unittest code.V4.test_r102_experiment code.V4.test_r102_reporting -q`
passed21/21 in4.459s in `easynco`. Fixtures are small CPU state containers and
saved synthetic arrays only; CUDA initialization is guarded, native donor
constructors/weight reads are replaced where needed. No native encoder,
selector training, real inference, TEST/LIB evaluation or production PDF was run.
Only temporary synthetic learning-curve PDFs and in-memory distribution figures
were rendered. No CJK missing-glyph warnings; pre-existing Matplotlib/pyparsing
deprecation warnings remain. Installed Noto CJK file and a Chinese glyph are
verified directly, not inferred from a stale Matplotlib font cache.

Both arms explicitly record `external_solver_pretraining=True` because their
common VRP donor is pretrained; `extra_type_pretraining` is true only for B.
Initialization checks original names/hash, exact three projection shapes/counts
and main-owned isolated seeds; routing projection remains fresh seed940003.
Checkpoint restore explicitly passes `initialize_from_checkpoint=False` and
saved provenance, then checks every registered state tensor including donor.

Completion receipts are fail-closed: full18 timing needs actual outer-donor
hook counts4/0 and both cache fields removed; LIB needs native49/100; reports
need the same unique lock, NSS4, fixed36 and all five nonempty PDFs. Comparison
requires BOTH completed whole-arm reports with equal routing provenance and
all non-runtime/non-extra-donor controls identical. R94B remains historical.

Main-owned final protocol JSON, real native integration and resource readiness
remain for main/other owners to review. No ready flag, source freeze, queue,
watcher, launch, automatic terminal retry or publication was created here.

## Bounded P2 Review Fix

`_donor_controls` now first validates the exact arm-specific boolean
`external_pretraining_by_component` map, then masks ONLY TSP/ATSP flags;
VRP=true and the full routing provenance remain strict and unmodified.
Report fixtures now use the actual full provenance constructor with native
encoder/weight I/O mocked; positive A/B and wrong/missing/extra/nonboolean flag cases pass.
Only `python -B -m unittest code.V4.test_r102_reporting -q` ran:12/12 PASS in5.615s.
Changed reporting/tests/this note only; no GPU/Git or full-suite rerun. Frozen again for main/Mendel narrow recheck.
