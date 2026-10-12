You are the requested GPT-6.1-Sol/max implementation/execution agent for R67.
Read the ONE adopted actual Astra/max PLAN ONLY in full:
.aris/traces/R66_closure_R67_plan.response.md.
Implement promptly, do ONE focused necessary CPU check, and launch the formal
paired experiment. You own only isolated R67 source/launcher/README/check and
code/V4/runs/R67_supervised_contrastive_full18/. You are not alone: parent is
closing and committing R66 on CPU. Do not modify/revert other work, do not
commit, launch agents/reviewers, run TEST, regenerate reference predictions,
run solvers/R58, or perform a GPU capacity/learning-rate/pilot sweep.

Active goal: original full18/full legal pools/native winner/single completed
solver macro TEST Top1 70%, nightmare loop100. Few checks, prompt training;
R58 fully deferred and eight legacy B-task semantics remain disclosed. No task
narrowing, new data/labels/augmentation, expert, source embedding or retrieval.
Completed R64/R65/R66 code/checkpoints/results are immutable.

Exact adopted recipe, with operational clarification:
- Two freshly initialized R64 NativeUtilitySelectors, experiment seed2, copied
  identical shared parameters; no R66 base/member or trained checkpoint reuse.
  Preserve d128,4instance+2joint layers,4heads,FFN512,score256,dropout.1,
  handcrafted34+ID, original TRAIN-normalized geometry, every original input
  field, directed ATSP information, legal pool and native winner.
- During stage1 call encode_pairs ONCE per dropout view, reuse those features
  for the original utility head and representation. Masked candidate mean of
  the 384-dimensional pair features, L2 normalize, projector384->256->128 with
  ReLU, L2 normalize output. Projector seed670002, explicitly copied A/B.
- A_classification_pretrain_seed2 stage1: two-forward native CE + legal FP32
  symmetric RDrop KL, coefficient .35*min(stage_successful_update/720,1).
- B_contrastive_pretrain_seed2 stage1: ONLY supervised contrastive Eq2. Both
  dropout views are anchors, all other same-native-winner views are positives
  including the other view of the same input. Exclude only self diagonal;
  denominator is all nonself views from THIS actual task batch. No cross-task
  comparisons. Stable FP32 logsumexp, mean positive log-probabilities OUTSIDE
  log, temperature .1, fixed loss multiplier .1. No detach of either branch.
  Targets only construct loss masks, never enter forward.
- Attribution clarification: the author implementation multiplies loss by
  temperature/base_temperature (default base_temperature .07), NOT invariably
  by temperature. Our fixed multiplier .1 is a PROJECT loss-scale choice
  (equivalent base_temperature1), not a claimed author-default reproduction.
  Preserve the locked numeric .1, no coefficient search or silent substitution.
  Primary sources: https://papers.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf
  and https://github.com/HobbitLong/SupContrast/blob/master/losses.py.
- Stage1 is40 FIXED epochs, terminal averaged-x trunk transferred to stage2.
  NO stage1 validation checkpoint selection. For speed, do not evaluate a
  nontrained B utility head on full validation in stage1: record its actual
  contrastive objective/gradient/updates. A projector and B utility score head
  are unused by their stage1 objective and naturally have no gradient; that is
  expected, not grounds for a new pilot or extra head/loss.
- Stage2 DISCARD projector in BOTH arms; replace the ENTIRE scalar score_head
  MLP with the same ordinary fresh init once, seed670003, explicitly copy its
  state across arms (not just last layer, not zero init). No trained head reused.
  Reset optimizer AND scaler and stage update/warmup/RDrop counters. Fine-tune
  ALL selector parameters40epochs with the SAME native CE/RDrop as A stage1.
  Paper's frozen linear stage2 is NOT reproduced: fully trainable fine-tuning
  and dropout-only views are explicit project adaptations.
- Each stage/arm exact official R65 ScheduleFree AdamW implementation/config:
  LR3e-4/internal720-successful-update warmup PER STAGE, then constant;
  WD1e-4,betas(.9,.999),eps1e-8,r0,weight_lr_power2,inner_momentum0.
  Train y/eval-save x, no external double warmup/controller/earlystop/PCGrad.
  Ensure stage transfer/head reset starts from saved/current eval-x, not y/z;
  fresh stage2 optimizer treats transferred x as initialization.
- Real256 independent rows/task,3distinct tasks/update,effective768; final
  16/task/effective48. One shared80epoch R64-style seed2 task/row/dropout plan,
  full180k original TRAIN rows each epoch, no taildrop/accumulation/microbatch.
  Same plan A/B, stage1epochs1-40 and stage2global41-80. Each stage9600updates/
  7.2Mpresentations; each arm19200/14.4M, pair38400/28.8M. Record stage/global
  counters explicitly. Contrastive actual512views/taskbatch (tail32).
- Mean three same-state task gradients FP32, clip once1, one optimizer update.
  Whole-triple AMP retry restores RNG/same rows, updates no counters until
  success. Do not silently skip/trim a batch or change calibrated256.
- Stage2 EVERYepoch complete original18validation, FP32batch128 and tails.
  Complete eval-mode TRAIN every5/final/selected stage2epochs. Select max
  INTEGER native correct count/18000, earliest exacttie; officialarm max/tieA.
  Save complete deployment selector in averaged-x, projector excluded, with
  unambiguous mode and original utility/globalID exacttie/fullpool rule.
  No TEST until parent freezes one best checkpoint perarm after BOTH complete.
  Inference ONEselectorforward/ONEfinalsolver. Include source/protocol/args,
  update/history/gradient/objective/counter logs, runtime, stage transition,
  selected prediction arrays as existing runners. W&B offline selector/entity.
- Do not commit the redundant >50MB row-schedule cache; keep its seed/config
  and local regenerate recipe. No large intermediate/checkpoint publication
  needed now; actual run data stay local until curated completed-version commit.

Reuse existing R64/R65 encoder, loss, data scheduling, optimizer/retry machinery
where sensible; don't edit them. Suggested independent r67_model.py,
r67_objective.py,r67_experiment.py,r67_cpu_check.py,run_v4_r67.sh,R67_README.md.
One CPU check should cover scalar-reference Eq2 positive/self masking and
two-view gradients, target-isolated forward, identical shared init, terminal-x
stage transition, identical whole-head reset and fresh optimizer/scaler state.
Synthetic fixtures ONLY for unit checks; production uses original rawdata.
No exhaustive audit, training-overfit pilot or repeated timing.

Only GPU: existing Slurm1465, gpu03/index1 UUID
GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d. Use easynco/established bridge
PYTHONPATH, threads1, tmux r67_train. Check ONLY that UUID normally for idleness
immediately before actual launch, wait if occupied. Never SSH/allocate/move/
stop any other job/client/queue/GPU. R66 TRAIN1465.94 and TEST1465.95 already
terminal0, do not replay them. No W&B login/network preflight.

Launch A stage1+stage2 then B stage1+stage2 ONCE; persistent pipeline.log.
After first finite FORMAL successful update, write handoff.json with command,
tmux/Slurm/PID, realbatch, memory, currentstage, CPUcheck, changedfiles and next
normal-monitor instruction. Return with all needed local exec sessions settled
but tmux experiment alive. Actual blocker: exact evidence/minimum remedy only;
do not silently change recipe or start alternatives. Parent handles later
delivery/TEST/review/Git. Do not alter global review state/memory concurrently.
