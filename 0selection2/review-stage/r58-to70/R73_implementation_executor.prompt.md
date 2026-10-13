You are SAME actual GPT-6.1-Sol/max execution agent, thread
01a13212-f99c-72a3-9279-db14f0073262. Your R72 delivery actually ended wrapper0
at03:17:52UTC. Actual Astra/max primary VAL-only plan8ready and fresh delivery
9ready/noactionabledefects completed; ONE R72publication9512d912c is pushed.
Parent adopts EXACT R73 approved_plan.md, planned with NEW R72 TEST excluded.

Read FULL code/V4/runs/R73_expansion_teacher_training/approved_plan.md. This is
ONE new full18 warm-started student experiment, with fixed R72 teacher/control.
No new architecture, solver labels, extra seed, confidence/search/calibration,
reference inference, R58, otherGPU/allocation, Git/reviewer/subagent/nextplan.
Do not edit historical sources, checkpoints, outputs, AGENTS/skills/data/registry
or user changes. Manual edits apply_patch. Add isolated R73 files only; suggested
r73_data.py, r73_experiment.py, r73_gradients.py if truly needed, r73_cpu_check.py,
run_v4_r73.sh, R73_README.md. Reuse existing factories/official optimizer/helpers,
not near-duplicate whole frameworks. Root runs/R73_expansion_teacher_training/.

Teacher/control A, exact frozen R72 bundle consensus_complete.pt:
- SAME full3 R60B39/R67B36global76/R70B23global63, complete states/buffers and
  original input paths, eval/FP32/no_grad, integer doubled-midrank Borda/globalID
  ties/full original legal pools. Use pred_global_id as HARD pseudo-target.
- No softmax of Borda scores, confidence filter, teacher refresh, new members or
  class balancing. No actual solver executions. Teacher errors are not labels
  measured from solving; hard targets discard ranking uncertainty.
- A saved VAL8923/18000. No new control standalone VAL/TEST/TRAIN, component
  evaluation, prediction recount or historical checkpoint selection. Teacher
  forward permitted ONLY for required expansion-input pseudo-target generation.
  No teacher pass over original native TRAIN/VAL/TEST. Later A TEST object reuse
  belongs to parent delivery authorization, not this implementation phase.

Student B, one original shared selector, NOT a bank:
- Strictly restore complete R67B36/global76 best.pt, saved averaged-x state,
  including trained utility head and all original normalization buffers.
  Original NativeUtilitySelector/d128/instance4/joint2/heads4/FFN512/readout256/
  dropout.1/handcrafted34+ID; no projector/task-specificbank/newhead. All selector
  parameters trainable, normalization statistics unchanged. Fresh optimizer and
  scaler; never resume old optimizer or invent a random fresh-model comparison.
- This is a warm-started teacher-training PACKAGE versus historical 3policy A,
  not exposure/capacity/compute-matched or isolated data-coverage causality.

Data/teacher cache, before student training:
1. Original18 native TRAIN180000rows/10000perproblem and native winners untouched;
   original full legal candidate pools/VAL18000. No R49 binary splits/subsets.
2. Use existing R62 raw-input cache/source IDs10000..99999, exactly90000/task,
   1.62Mtotal. Isolated input-only reader: node/matrix/known metadata only. Do NOT
   read expansion auxiliary costs or observation masks into this experiment,
   forward or objective. No new input generation/solver/winner/cost annotation.
   Preserve coordinates/orientation/customerorder/physicalroles/constraints,
   directed ATSP, true n and existing original TRAIN normalization. Existing
   label-free geometry helper/cache can be used; never refit normalization.
   Stream expansion batches from CPU raw arrays/cache to avoid retaining all
   expansion graphs/geometry on GPU; only required current batch is resident.
3. ONE formal teacher pass, fixed original task order and ascending sourceIDs,
   batch128,18tails16:12672querybatches/38016memberforwards/4.86Mmemberrows.
   Save static global pseudo-winner IDs plus source/task/pool metadata locally.
   Validate legal alignment as part of this formal pass, no extra teacher pilot
   or native/reference forward. Save all rows before marking cache complete.
   Release teacher models/graphs before student construction/training. Record
   actual teacher counters/wall/memory separately; do not call this new truth.

Exact training:
4. Reuse R67 seed2 stage2 native task/row schedule, global epochs41..80. Per task,
   concatenate five permutations of90000 expansion indices from ONE private
   NumPy generator730002+problemID and consume first400000. Each expansionrow
   occurs4times and40000/task a fifthtime. SourceIDs remain original10000+index.
   Three tasks/update, each real256native+256aux rows, corresponding16rowtails,
   no microbatching or gradient accumulation across optimizer updates.
5. Native batch has two ordinary dropout forwards. C=mean native CE of both;
   R=existing legal symmetric KL, both paths differentiable. Equal-sized aux
   batch has ONE dropout forward and P=mean CE against detached cachedteacher
   globalID mapped into its correct legal localcolumn. Exact L_task =
   C + .35*min(u/720,1)*R + P. No auxKL/risk/pair/MSE/mixup/other objective.
   CE/KL FP32/legal candidates; supervision never passed through forward.
6. Official installed ScheduleFree config: LR3e-4, internalwarmup720,
   betas(.9,.999),eps1e-8,WD1e-4,foreachFalse,r0,weight_lr_power2,
   inner_momentum0. Use existing library/version/factory, no installation/search.
   Train at y, evaluate/save averaged x; freshoptimizer/scaler with warm-started
   model. No additional external LR warmup/controller/earlystop beyond this plan.
7. Native ordinary RNG setup follows original stage2 seed2. Aux dropout private
   seed730003+18*(u-1)+problemID, restore ordinary CPU/CUDA RNG after aux so it
   doesn't consume the native stream. Do not claim bitwise historical-mask
   equivalence without actually preserving it. Retry whole3taskupdate on any
   nonfinite with restored RNG/reducedAMPscale and unchanged optimizer/cursors;
   retain20attempts, record attempts/retries separately. No skipped samples.
8. Backpropagate/release native graphs before aux forward. Accumulate combined
   per-task gradients FP32 at SAME parameters, divide by3, clip ONCE at1, ONE
   optimizer.step. No update between tasks/streams. Exactly40epochs/9600success
   updates:7.2Mnative+7.2Maux presentations,14.4Mtotal and21.6Mstudentencoder
   views. Distinct180000native+1.62Mexpansioninputs; only native measuredwinners.

Selection/evaluation, still NO TEST authorization now:
9. Complete original native VAL everyepoch, frozen x/eval/FP32/batch128/tails104,
   all18x1000/fulllegalpools. Earliest MAXINTEGER correct selects ONE shared
   complete checkpoint, officialBstrictly>8923elsefixedA. Keep original native
   TRAIN evalevery5/final/selectedbest, not training-mode accuracy or pseudo
   agreement substituted for real generalization. Log nativeCE/KL/pseudoCE/total,
   Top1/2/3/cost/actualregret, LR/gradient/retries/successvsattemptedviews/time/
   memory; sparse TRAIN remains sparse. No test/model/member/taskwise selection.
10.B deploys one shared selector/fullpoolutilityargmax/globalIDties, then ONE
   completed solver. Original native TEST is later ONLY after parent lock of
   unique whole checkpoint/officialarm, exactly once for newB regardlessVALgain;
   A reuses its already authorized R72 TEST object unchanged. No TEST/freeze/
   deliveryanalysis authorized in this implementation phase.

Minimum work and actual launch:
11.ONE focused synthetic CPU invocation only: global/local target/task alignment,
   target isolation, native+aux gradient reduction and whole-update RNG recovery.
   No real checkpoint/data/model preflight, GPU capacity/pilot/size sweep,
   reference replay or more checks. Then DIRECT formal teacher cache+training.
12.Ownedtmux r73_train, existing Slurmallocation1465/nodegpu03, soleindex1 UUID
   GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d. Check ONLY assignedUUID at actual
   launch; if busy wait same device. No otherGPU/client/queue query or allocation.
   conda easynco first, existing python_compat bridge/threads1/W&Boffline selector
   entity yjkds-southern-university-of-science-technology. Long formal process can
   include cache then training sequentially in ONE job; teacherstate released
   before student. Exactactualjob ID/PID fromlaunch, never predeclare nextID.
13.Record protocol/source snapshots/lockedrecipe/teacher+student identities before
   formal pass. Preserve actual failure without automatic restart or recipe
   changes. No broad environment/package/network survey, timecap or GPU-hourcap.
14.After implementation, one CPU invocation and actual first formal teacher batch
   onassignedGPU, return PROMPTLY with real handoff/job/tmux/PID/counts/time/
   memory/parametercounts. Preserve original running formal job; do not wait in
   LLM turn for40epochs or duplicate/restart it. State clearly whether only
   teacher cache is running versus actual first student update observed. Parent
   monitors smallstatus and later delegates isolated conditional delivery while
   the original process lives. Set created foreground execsessions settled
   before returning; intentionally owned formal tmux remains live.

Original70goal ACTIVE/unachieved, minimum12600/18000 native full18 originalTEST.
R58/eightlegacyBsemantics deferred/unfixed/notnumericveto. Only truthfully report
formal work actually observed. No Git by you; parent publishes once/completed
version after actual primary+fresh reviews and objective audit.
