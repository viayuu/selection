# Adopted R64: PCGrad on the shared dense full18 classifier core

Authority: `../R63_recurrent_workspaces/reviews/R63_closure_R64_plan.response.md`,
read in full before implementation. Its eight numbered specifications are fixed.
The parent authorizes TRAIN/validation only with baton `pcgrad-full18`.

1. Identical fresh seed2 R45A dual-stream models: handcrafted34+ID, d128,
   four instance layers, two joint layers, four heads, FFN512, head256, dropout.1.
   Direct utility logits from encode_pairs+score_head, original-TRAIN geometry.
2. Original10000 TRAIN/task,18 tasks,22 identities,128 legal deployments. NativeCE
   plus existing symmetric R-Drop, ramp0 to.35 over the first4302 updates. Preserve
   task/constraint inputs and full directed ATSP matrices; labels outside forward.
3. Exactly239 independently shuffled all18-task cycles/epoch; six distinct-task
   triples/cycle. Each task consumes its next42 shuffled rows, final4 retained.
   Three gradients at the same parameter state, one update:1434/180000 per epoch.
4. Only B projects the seven fixed shared dense prefixes. Mutable task copies
   project against ORIGINAL other gradients in independently shuffled orders,
   skipping zero denominators. RNG6402 is separate. Other gradients use arithmetic
   mean, without per-task normalization or projected weight decay.
5. Finite unscaled FP32 buffers before projection; AMP overflow retries the entire
   triple and restores every RNG without advancing optimizer/sample/update state.
   One global clip1, one AdamW step:LR1e-4,WD1e-4,betas(.9,.999),epsilon1e-8.
6. Step warmup4302. A controls common actual LR trace/horizon; B replays it exactly.
   Patience3 LRhalf, patience8 stop, min15/max40, min_delta.001percentage points.
   Maximum57360 updates/7.2M presentations per arm. Select strict highest INTEGER
   validation native correct/18000; exact ties retain earlier epoch, never cost.
7. FP32 eval, full legal pool, highest direct utility, stable global-ID ties. After
   both arms finish the parent freezes both selected bests, inference rule, and
   official arm by validation (tieA), then authorizes ONE original TEST/model.
8. Normal training records losses, task/shared norms, conflicts, projection change,
   clipping/retries, wall/memory/throughput. Full val every epoch; complete eval-mode
   TRAIN every5 and termination/selected checkpoint. Save predictions and curve
   data; later explicit analysis supports nativeTop1/2/3,CE,cost,regret,SBS,task/family
   results and required plots. Singleton/batch timing only in normal selected eval.

Execution: easynco, threads1, compatibility PYTHONPATH, offline W&B selector /
yjkds-southern-university-of-science-technology; tmux r64_train; existing Slurm1465;
gpu03/index1 UUID GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d only. Check that UUID at
actual launch and wait if occupied. No pilot, reviewer gate, TEST authorization,
automatic analysis, git operation, allocation/queue/client/environment change.
Only brief CPU projection/coverage/tie checks precede formal training. Preserve
all other changes and artifacts, R63's frozen A34, and unresolved B semantics.
R58 is deferred. R63 TEST outputs/errors/RESULTS.md are not read to tune R64.
