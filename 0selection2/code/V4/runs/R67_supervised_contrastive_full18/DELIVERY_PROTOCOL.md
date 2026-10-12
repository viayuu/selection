# R67 Delivery Preparation

Prepared code only. The existing r67_train/Slurm1465.96 job, frozen six launch
files, source snapshot, protocol, selection design and40+40epoch budget are
unchanged. These delivery files are separate from the training hash list.

## Future Sequence

Parent first records ACTUAL successful Slurm/tmux terminal evidence in
`training_terminal_receipt.json`, after the complete A1->A2->B1->B2 pipeline and
both result files plus validation_candidates.json/paired_training_complete.json
exist. Required receipt fields: state=COMPLETED, exit_code=0:0,
tmux_terminal=true, tmux_session=r67_train, slurm_step=1465.96. Retain actual
elapsed_seconds and original scheduler evidence when recording the receipt.

The following commands are FUTURE commands, not authorization to run them now:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
CUDA_VISIBLE_DEVICES= python -B -m code.V4.r67_frozen_evaluation --stage freeze --authorize-test
tmux new-session -d -s r67_final_test 'R67_PARENT_GPU_BATON=locked-test bash code/V4/run_v4_r67_test.sh >> code/V4/runs/R67_supervised_contrastive_full18/test_pipeline.log 2>&1'
```

Freeze is CPU-only and refuses unfinished/nonterminal training, changed launch
sources or existing locks/TEST markers. It chooses ONE complete stage2 saved
averaged-x selector per arm by max INTEGER validation native correct/18000,
earliest exacttie; official arm tieA. It preserves any existing identical unique
selected-epoch checkpoint and published best, recording full absolute paths,
file/state hashes, stage/global epoch, mode, original18 scope and full legal IDs.
Costs, TEST, parameter mode and task-wise assembly never choose checkpoints.

The future TEST launcher requires that parent lock AND an explicit locked-test
baton. It uses only existing Slurm1465/gpu03/index1 UUID
GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d in tmux r67_final_test. An ordinary
assigned-UUID-only idle check occurs just before that future launch, waiting30s
if occupied. No allocation, device switching or stopping another client.

An exclusive final_test_started.json permits one attempt. Each locked model gets
ONE complete original18 TEST, FP32batch128 with104-row tails, live raw inputs,
one NativeUtilitySelector forward per query and no optimizer-mode transition.
There is no projector, stage1 checkpoint, composite/reference model, cached TEST
score, reselection, separate latency benchmark or solver run. Original FP64
costs/native winners/full legal pools/global-ID exactties are preserved. Labels
and costs remain outside forward. Full per-instance NPZ includes original row
indices, predictions/logits, native winners, raw FP64 costs, pool names/IDs and sizes.

## After Actual TEST Terminal0

test_process_receipt.json records the actual evaluator return code; it is NOT
Slurm terminal evidence. Parent separately writes test_terminal_receipt.json
using actual terminal state: state=COMPLETED, exit_code=0:0, tmux_terminal=true,
tmux_session=r67_final_test, the ACTUAL future slurm_step matching the launch/
start/process receipts, and original_full18_fulllegalpool_tests_per_frozen_model=1.
Only then, in the same conda/PYTHONPATH/threads environment, run ONCE:

```bash
CUDA_VISIBLE_DEVICES= python -B -m code.V4.r67_analysis
```

Analysis has an exclusive analysis_started.json. It reads saved JSON/updates;
it performs no inference, NPZ prediction replay, retrospective TEST or solver
operation. It reports R67A/B and fixed R45A38,R60B39,R64A/B,R65A/B,R66A/B official
saved reports. Historical policies/exposed TEST and missing saved historical
TRAIN are disclosed; no missing reference metrics are fabricated.

Outputs include comparison/selected/method metrics, availability and deltas;
task/family/ALL stage2 TRAIN/VAL curves and selected TRAIN/VAL/TEST; best/final and
last5 VAL summaries; separate correctly labeled A-classification/B-SupCon stage1
objectives; stage1/2 LR/gradient/AMP/counters/runtime CSV/figures; compute_summary,
analysis/report and delivery receipts. Stage1 has no validation curve. TRAIN is
complete only on its predeclared every5/final/selected stage2 epochs.

Eachstage planned9600updates/7.2Mrows; eacharm19200/14.4M; pair38400/28.8M. Actual
measured logs/counters are authoritative on interruption/failure. Transition
receipts retain terminal x->identical whole fresh head/fresh optimizer/scaler.
There are no inherited pretrained selectors. A projector/B score head are
intentionally unused in stage1. Timing is ordinary complete evaluation wall
time, not singleton latency. No severe-overfit/noise/ceiling claim follows.

Numeric70% is reported only for the parent's official locked complete TEST.
A supported70% claim additionally requires parent objective_audit.json with
passed=true, scope="whole objective", final_test_lock_sha256 and
test_results_sha256 matching those actual files. Missing audit means pending,
not an automatic pass; parent owns whole-goal closure and review state/memory.

## Local Assets

Code/reports/configs/receipts/CSV curves/figures/review records may later be
curated into one completed-version commit by parent. Weights, NPZ/raw costs,
raw update logs, W&B and regenerable schedules remain LOCAL under concurrent
code/.gitignore cleanup; nothing is force-added. Git alone does not contain all
inference assets. These preparation commands perform no Git operation.
R58 remains fully deferred; the eight legacy B-task feasible-domain mismatches
remain unresolved and disclosed, not silently fixed or used to veto native70%.
