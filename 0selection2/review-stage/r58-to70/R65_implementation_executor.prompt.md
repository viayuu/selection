You are the requested GPT-6.1-Sol/max implementation/execution/monitor role.
Parent ADOPTS exactly the actual Astra/max R65 plan in:
code/V4/runs/R64_pcgrad_full18/reviews/R64_closure_R65_plan.response.md
Read it directly and implement promptly. R64 is fully completed/committed before
this invocation; never resume it or alter its source/snapshots/artifacts. Full18
single-solver native70% goal is ACTIVE, nightmare/100rounds, same-family/provisional.
R58 completely DEFERRED: no constraints/labels/scenario fixes or solver calls.

ONE major mechanism: official schedule-free AdamW versus matched constant-LR
AdamW. BOTH fresh identical seed2 NativeUtilitySelector (exact r64_model.py),
original18/fullpools/native winners, handcrafted34+ID, nativeCE+sameRDrop. No
PCGrad in either arm; three same-state task gradients arithmetic mean FP32,
global clip1 and one step. Reuse original TRAIN-only geometry/data cache and
R64 task+row schedule generation, not trained weights/optimizer/cursor state.
Real256/task, effective768;40cycles/240updates/180000presentations per epoch,
tail16/task/effective48, fixed40E=9600updates/7.2Mpresentations per arm. No
accumulation/microbatch, scheduler/controller/earlystop, grid or new sizing.
Both LR3e-4 with720successful-update linear warmup, then constant; WD1e-4,
betas(.9,.999),eps1e-8,foreachFalse; RDrop .35*min(u/720,1). Both same fresh
init/data/dropout stream. A ordinaryAdamW, B officialAdamWScheduleFree with
warmup_steps720,r0,weight_lr_power2,inner_momentum0. B internal warmup must NOT
also get external warmup (no double ramp). Same effective LR is logged honestly.

Follow the actual official optimizer, not a handwritten approximation or wrapper.
Use an existing matching installed implementation if available; otherwise keep
only the necessary official implementation+license in isolated new R65 files,
record normal upstream provenance and use apply_patch for manual file creation.
Do not upgrade/change the shared conda environment or build a dependency matrix.
Official source:
https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/adamw_schedulefree.py
https://github.com/facebookresearch/schedule_free

Implement correctly the two parameter modes. B's gradients use y after optimizer
train(); validation/TRAIN evaluation and saved deployment best use averaged x
after optimizer.eval(). Serialize resumable last model+optimizer consistently in
eval mode, restoring optimizer train mode once before continuation. No choice
among x/y/z after looking at scores. LayerNorm backbone needs no BatchNorm pass.
Preserve whole-triple nonfinite AMP retry/RNG restoration, no sample/update/ramp
advancement until successful. Do not project or normalize per-task gradients.

New files under code/V4, root runs/R65_schedule_free_full18/. Reuse local helpers
and patterns without modifying completed R64 files or unrelated user work. Save
PLAN/protocol/source snapshot, paired fresh initialization and exact schedule,
both configurations, periodic resumable checkpoints, per-update finite/gradient/
LR/throughput/retry observations, normal W&B offline selector logging, fullval
eachE, evalTRAIN each5/end/selected, raw curves and val predictions. Integer native
count max outof18000 selects eachbest, exacttiesearlierepoch. Officialarm by that
same count, tieA, notcost. Inferencemaxdirectutility/globalIDexactties/fullpool.

Exactly ONE focused CPU check for schedule-free eval/train roundtrip and consistent
checkpoint/resume is enough; also verify this code reuses actual selected batch.
No repeated test suites, GPU preflights/timing/sizing, integrity audits or review
gates. Timing comes from normal first formal successful update. Immediately launch
the fresh paired pipeline after this necessary check, A then B. Ordinary assigned
UUID idle check ONLY at actual launch; if unavailable wait, never switch devices.

Keep tool output concise: use focused source reads and structured JSON summaries,
not giant result/history/state dumps. Necessary implementation work is authorized;
do not turn source provenance or normal input loading into another audit gate.

Sole assigned GPU: gpu03/index1 UUID GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d,
existing Slurm1465. New tmux r65_train. Do not touch anotherGPU/client/process/
allocation/queue or old jobs. Use existing R64 launcher pattern safely but new
R65 paths/handles/baton. Every shell activates easynco via source conda.sh.
Read AGENTS.md if needed; do not modify it. No Git add/commit/push in this worker.

TEST is NOT yet authorized: implement the same later validation-only mechanical
freeze/unique lockedtest entry, no automatictest. After both complete parent will
lockuniqueweights/rule/officialarm then authorize ONE fulloriginaltest perarm.
Historical references are reused, no reference inference reruns. No analysis
before actual completion and parentauthorization. No R66, subagents/newreview.

Return at first formal launch/finiteupdate+live tmux receipt, with exact command,
step/PID, changedfiles, CPUcheckresult, actualsize/LR/loss/gradient/time/memory,
and clear ownership handoff. Leave ordinary longtraining in tmux, settle any
own exec sessions. Update persistent state/memory with actual launch evidence
and preserve existing fields. Do not call PLAN8ready performanceacceptance or
promise70%. Legacy B-task feasible-domain mismatch remains unresolved/disclosed.
