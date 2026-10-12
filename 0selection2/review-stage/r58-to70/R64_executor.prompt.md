You are GPT-6.1-Sol, reasoning effort max, implementation/production executor.
Implement and launch the ONE adopted R64 experiment below. The planner/reviewer
was GPT-6-Astra/max. You are not alone in this shared workspace: preserve all
other changes, staged user W&B files, old runs and queues. Do not spawn agents.

Authoritative completed TRAIN/validation-only plan (read it in full):
code/V4/runs/R63_recurrent_workspaces/reviews/R63_closure_R64_plan.response.md
Its eight numbered R64 specifications are adopted, not a menu for a search.
R63 was completed and pushed as15363d0ec. Its TEST is terminal rc0,1465.88;
no other parent-owned GPU job is running. Do NOT read R63 TEST outputs/errors or
RESULTS.md to tune R64. The plan did not use new TEST evidence.

Write new R64 files only (suggested names):
code/V4/r64_model.py, r64_gradients.py, r64_experiment.py, r64_analysis.py,
test_r64.py, run_v4_r64.sh, R64_README.md, and
code/V4/runs/R64_pcgrad_full18/ (PLAN.md, protocol, initialization/batches,
run artifacts and curves). Do not refactor old modules, tools, AGENTS or R63.
Use apply_patch for manual edits. Existing helpers/frameworks should be reused.

Two fresh, identical model initializations, seed2:
A_mean_gradient_seed2 versus B_pcgrad_seed2. R45A dual-stream architecture,
handcrafted34+ID, d128/four instance layers/two joint layers/four heads/FFN512,
local geometry fitted on original TRAIN. Native utility logits directly from
encode_pairs+score_head; no maximin, risk, pairwise objective, pretraining,
routes, solver execution, memory, source embedding, per-task expert or new labels.
Use nativeCE plus existing symmetricRDrop, .35 ramp3, for BOTH groups.

Batch construction must actually implement the adopted semantics: per epoch
239 cycles, each with all18 distinct tasks shuffled and partitioned into six
triples. Each task contributes its next shuffled42 examples, final4 retained.
All three task gradients are at the SAME parameter state, then ONE AdamW step.
Exactly1434 successful updates/180000 presentations per complete epoch; maximum
57360updates/7.2M presentations per arm. Typical effectivebatch126, final12.
If the existing paired_schedule helper is not cycle-based, implement this small
schedule explicitly rather than claiming it already has these semantics.

PCGrad only on the named shared dense modules specified by Astra. Verify the
actual corresponding parameter names, do not silently project an empty subset.
For each task, project a mutable copy against the other ORIGINAL gradients in
independently shuffled order; skip zero denominator. Separate RNG6402. Average
projected gradients for B; arithmetic mean for A. Other parameters use ordinary
mean in both. No per-task norm normalization; weight decay is not projected.
Record original conflicts/norms and projection magnitude during NORMAL training.

Follow the plan's FP32 unscaled-gradient/AMP and whole-triple retry semantics.
No optimizer/sample/update advancement on overflow; restore all relevant RNG,
including projection order. One global clip1 after reduction, one fresh AdamW
step, LR1e-4/WD1e-4/betas(.9,.999)/epsilon1e-8. Step warmup3*1434updates.
A controls the subsequent common LR/epoch horizon; B replays it exactly, max40,
min15, patience3 LRhalf/patience8stop/min_delta.001percentagepoints. Save the
shared trace, not merely similar rules. Same paired batches/dropout draws.

Fix the documented R63 floating-tie issue PROSPECTIVELY: validation tasks all
have1000rows, so rank checkpoints by INTEGER total native correct counts; exact
ties retain earlier epoch, never cost. Do not alter R63's locked A34 or test.
Full original full18/fullpool val each epoch; complete eval-mode TRAIN every5
and at termination/selected checkpoint, no online batch substitution. Report
nativeTop1/2/3,CE,mean_cost,actualregret,task/family results and actual compute.

Only brief CPU checks for projection, complete paired batch coverage, and exact
ties are needed. Do not run a GPU preflight/timing pilot or repeat audits. After
these pass, proceed DIRECTLY to formal training; first real update logs finite
gradient/retry/memory/throughput data, not a discarded pilot. No reviewer gate.

Sole authorized GPU: GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d, RTX3090,
gpu03/index1, existing Slurm1465 allocation. Parent authorizes the R64 train
baton R64_PARENT_GPU_BATON=pcgrad-full18. Use the existing R63 launcher pattern,
easynco, PYTHONPATH compatibility export, offline W&B projectselector/entity
yjkds-southern-university-of-science-technology, threads1, and tmux r64_train.
Every command must source conda.sh and activate easynco. Check ONLY this UUID's
idleness at actual launch; if occupied, wait, never switch GPUs or touch clients.
Do not change allocations, queues, other GPUs/processes, MPS or environments.
No GPU-hour cap. Preserve unresolved legacy B semantics; R58 is fully deferred.

Implement resume-safe normal training using existing patterns, not a new generic
framework. Record actual launches, successful updates, retries, wall, best/final
and saved predictions. Normal selected evaluation may include singleton/batch
timing once; never add a standalone timing pass. Analysis should support later
locked test reporting, but do NOT run analysis or TEST automatically.

NO TEST authorization yet: after both arms finish the parent freezes each unique
validation-selected best, inference rule and official arm by validation(tieA),
then authorizes ONE complete original TEST per frozen model. R60B/R45A references
will reuse existing official tests; do not rerun them. No test-driven reselection.
Do not git add/commit/push; one whole-version commit happens only after completion.

Return promptly AFTER implementation, the brief CPU tests, tmux formal launch
and first successful real update. Include changed files, CPU test outcome,
actual tmux/Slurm/GPU receipt and first-update metrics. Do not wait for all
epochs or run ongoing review loops; the parent will continue monitoring. Make
sure any exec_command session you started has settled or is properly handed
off via the tmux receipt before your final response. If a real blocker occurs,
report the exact failing path and minimal remedy, without scope expansion.
