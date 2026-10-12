# Adopted R65 execution protocol

Actual Astra/max plan: `code/V4/runs/R64_pcgrad_full18/reviews/R64_closure_R65_plan.response.md`.
Plan8/ready is plan-only, performance unverified; same-family/provisional.
The full18 single-solver native70% goal is ACTIVE, nightmare/100rounds.

One mechanism: official non-wrapper AdamWScheduleFree versus ordinary AdamW.
Both fresh identical seed2 exact R64 NativeUtilitySelector, original18/fullpools/
native winners, handcrafted34+ID, existing original TRAIN-only geometry/data cache.
Reuse R64 schedule generation, no trained weights/optimizer/cursor state.

Both use three unchanged-state task gradients, arithmetic mean FP32, global clip1
and one step. No PCGrad or task-gradient normalization. Real256/task/effective768,
40 cycles/240 updates/180000 presentations each epoch, tail16/task/effective48.
Fixed40E=9600 updates/7.2M presentations per arm; two dropout passes=14.4M forward rows.
No accumulation, microbatch, scheduler, controller, early stop, grid, or new sizing.

Both use LR3e-4 with720-successful-update linear warmup then constant, WD1e-4,
betas(.9,.999),eps1e-8,foreachFalse. RDrop `.35*min(u/720,1)` with successful update u.
A uses external warmup; B exclusively uses official warmup_steps720, r0,
weight_lr_power2, inner_momentum0, constant optimizer baseLR3e-4. Log actual
effective LR. Same fresh state, selected task/row schedule and dropout stream.
Whole-triple nonfinite AMP retry restores RNG, reuses actual selected batches and
does not advance samples/updates/ramps until success.

B trains at y after optimizer.train(); validation/evalTRAIN and deployment best
use averaged x after optimizer.eval(). Save resumable last model+optimizer in eval
mode and restore train mode once before continuation. No x/y/z score-based choice.
LayerNorm needs no BatchNorm pass.

Full FP32 batch128 validation each epoch; full evalTRAIN each5/end/selected. Save
initialization/schedule/configuration/source/provenance, periodic last checkpoints,
offline W&B selector, per-update finite/gradient/LR/throughput/retry observations,
raw curves and all validation predictions. Maximum integer native_correct/18000
selects best with exact ties earlier. Official arm same count, tieA, never cost.
Inference max direct utility/fullpool, exact globalID ties, one completed solver.

Exactly one focused CPU mode/resume+selected-batch check, then formal A followed
by B in tmux r65_train. Assigned gpu03/index1 UUIDcb626bdb, existing Slurm1465.
Only actual-launch assigned-UUID idle check; wait if occupied, never switch.
No GPU preflight/timing/sizing, integrity audit or new review gate. First normal
formal successful update supplies timing. No environment changes or Git action.

TEST remains unauthorized. After both arms complete, parent uniquely freezes
validation-selected weights/rule/official arm, then separately authorizes ONE
full original TEST per arm through the exclusive locked TEST entry. No automatic
TEST/analysis and no reference inference reruns. R64 remains immutable. R58 is
completely deferred. Legacy B-task feasible-domain mismatch remains unresolved.
