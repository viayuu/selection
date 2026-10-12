# R68 Delivery Preparation

Code preparation only. The existing r68_train/Slurm1465.99 training, six frozen
launch files, source snapshot, protocol, schedule and budget are unchanged.
Delivery sources have a separate identity; never add them to the training hash
list. Nothing here authorizes TEST or analysis during preparation.

## Future Parent Sequence

First require BOTH actual pipelines to finish40 epochs, full40 original18 VAL
histories, both result.json files, validation_candidates.json and
paired_training_complete.json. The original tmux must be gone and the actual
owned Slurm1465.99 step must be COMPLETED/0:0. A live RUNNING/0:0 is not success;
poll timeouts do not authorize restart. Genuine failure stops delivery.

Parent records training_terminal_receipt.json with state="COMPLETED",
exit_code="0:0", tmux_terminal=true, tmux_session="r68_train",
slurm_step="1465.99", actual elapsed_seconds and original scheduler_evidence.
Keep observation UTC and actual command output. Process receipts alone are not
scheduler evidence. Then, only with future explicit parent authorization:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
export PYTHONPATH=/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/python_compat:/public/home/shiys/0selection2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export WANDB_MODE=offline WANDB_PROJECT=selector
export WANDB_ENTITY=yjkds-southern-university-of-science-technology
CUDA_VISIBLE_DEVICES= python -B -m code.V4.r68_frozen_evaluation --stage freeze --authorize-test
tmux new-session -d -s r68_final_test 'R68_PARENT_GPU_BATON=locked-test bash code/V4/run_v4_r68_test.sh >> code/V4/runs/R68_sharpness_aware_finetuning/test_pipeline.log 2>&1'
```

CPU freeze uses maximum INTEGER original18 validation native correct/18000,
earliest epoch exacttie; official arm max, tieA. It locks one complete unique
best_checkpoints/epochNNN.pt per arm and the identical published best.pt, with
absolute paths, file/model hashes, fine-tuning/global epoch and ordinary-weight
metadata. Costs and TEST never select; no task-wise assembly or alternate mode.

R68 checkpoints contain the full NativeUtilitySelector at ordinary UNPERTURBED
AdamW weights. They are loaded directly: no ScheduleFree mode transition,
averaging, projector, stage1 checkpoint, inherited trained utility head, R66
additive model or head transplant. Inheritance was only R67B terminal stage1
averaged-x TRUNK plus the entire fresh head670003 before matched fine-tuning;
the final R68 deployment itself has ordinary weights.

The future TEST launcher uses only existing allocation1465/gpu03/index1 UUID
GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d, tmux r68_final_test and the explicit
locked-test baton. Its assigned-UUID-only idle check occurs immediately before
actual launch, waiting30s if busy. No new allocation, other GPU/client/queue,
device switching, client stop or network preflight is allowed.

An exclusive final_test_started.json allows ONE attempt. Each frozen model
gets one complete original18 TEST with FP32batch128 and104-row tails: one live
raw-input selector forward/query, unchanged full legal pools/native winners,
utility argmax and ascending-global-ID exacttie. Labels/costs stay outside
forward. Preserve complete per-instance NPZ with original row indices, logits,
predictions/native winners, raw FP64 costs, pool names/globalIDs and sizes.
Deployment uses one completed solver/query; this evaluation executes zero
solvers. No reference inference/retest, cached TEST scores, prediction replay,
separate singleton latency benchmark or TEST reselection. Failed/partial TEST
preserves its marker and does not authorize another attempt.

## After Actual TEST Terminal0

Only after the ACTUAL new owned Slurm step is COMPLETED/0:0 and r68_final_test
tmux is gone, parent writes test_terminal_receipt.json. Required fields are
state="COMPLETED", exit_code="0:0", tmux_terminal=true,
tmux_session="r68_final_test", actual slurm_step matching launch/start/process
receipts, actual elapsed_seconds, original scheduler_evidence, observation UTC,
and original_full18_fulllegalpool_tests_per_frozen_model=1.
test_process_receipt.json is evaluator exit evidence, not scheduler evidence.
In the same activated conda/PYTHONPATH/thread1 environment run ONCE:

```bash
CUDA_VISIBLE_DEVICES= python -B -m code.V4.r68_analysis
```

The exclusive analysis_started.json prevents automatic replay. Analysis reads
saved reports/update logs; it performs no model inference, NPZ replay, solver,
retrospective TEST or historical checkpoint selection. It reports full18
task/family/ALL selected TRAIN/VAL/TEST nativeTop1/2/3, CE, mean_cost, vsSBS and
actual regret. ALL retains equal-task macro; original cost definitions remain.
Best/final/last5 VAL and actual matched common TRAIN/VAL epochs are reported.
Last5 means epochs36-40, not uncertainty over seeds. Every VAL is complete;
TRAIN is evaluated every5/final/selected-best, with no invented dense curve.

Fixed historical R45A38,R60B39,R64A/B,R65A/B,R66A/B,R67A/B official saved reports
are reused with original checkpoint/mode policies and provenance. Historical
TEST exposure makes cross-version comparisons descriptive. Missing historical
TRAIN is disclosed, never filled by inference. R68 has no new stage1 curve;
inherited R67 SupCon is not mislabeled CE or given a stage1 validation curve.

Outputs include comparison/selected/method metrics, availability/deltas;
learning/task/family/optimization/task-optimization/update CSV curves;
stage2_summary/last5_validation/matched_common_epochs; figures, REPORT.md,
reference_provenance/compute_summary/analysis and terminal/delivery receipts.
Training diagnostics retain ordinary first and SAM perturbed loss/CE/KL,
LR, first/second gradient norms, clip, effective perturbation norm/exact
restoration/RNG replay, AMP scale/retries, successful/attempted view rows,
actual walltime/peakmemory and one base optimizer step per successful update.
Full evaluation walltime is not singleton latency; no extra sharpness audit.

Each new arm40epochs/9600updates/7.2Mselected rows; new pair19200/14.4M. Count
inherited R67B stage1 9600updates/7.2Mrows ONCE separately. A14.4M/B28.8M
successful view rows, plus actual retry attempts. Rows/updates are matched,
FLOPs are not. Real256/task x3=768, tails16x3=48; exact R67 stage2 global41-80
task/row/dropout slice. Actual saved counters/receipts are authoritative. Both
new optimizers/scalers were fresh; all selector parameters trainable. This is
one fixed-seed optimization contrast from a shared contrastive pretrained trunk,
not a from-scratch encoder experiment. No severe-overfit/noise/ceiling claim.

Numeric70% can be reported only for the parent's official locked complete
original18 TEST. A supported70% claim additionally requires parent
objective_audit.json with passed=true, scope="whole objective", matching
final_test_lock_sha256 and test_results_sha256. Missing audit remains pending;
parent owns the whole-goal audit/closure. SAM>=A but below R60B is limited
progress, not completion or a70% promise.

## Local Assets and Ownership

Code/reports/configs/receipts/CSV curves/figures/reviews may later be published
once by parent on whole-version completion. Weights, NPZ/raw costs/IDs, raw
update logs, W&B and regenerable schedules/caches remain LOCAL under userignore;
nothing is force-added. Git alone does not contain all inference assets.
Preparation performs no Git operations or changes to parent state/memory.
R58 remains fully deferred. Eight legacy B-task feasible-domain mismatches are
unresolved and disclosed, not repaired or used as a native70% metric veto.
