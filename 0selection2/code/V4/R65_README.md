# R65 schedule-free optimizer pair

Adopted plan: `runs/R64_pcgrad_full18/reviews/R64_closure_R65_plan.response.md`.
Plan8/ready is plan-only. Performance is unverified, same-family/provisional.
The full18 single-solver native70% goal remains active and unachieved.

A uses ordinary AdamW and B the unmodified official AdamWScheduleFree from
`r65_official_schedulefree/`, with the upstream commit and license recorded there.
Both freshly initialize the exact R64 NativeUtilitySelector at seed2, original18
full pools and native winners, handcrafted34+ID, same TRAIN-only geometry cache,
same task/row schedule and update-index dropout seeds. No trained R64 state is used.

Each update averages three same-state FP32 task gradients, clips once at1, and
steps once. Real256/task, effective768, tail16/task/effective48; no accumulation
or microbatching. Both complete40 epochs,240 updates/epoch,180000 presentations
per epoch:9600 updates and7.2M presentations per arm. Native CE from two dropout
forwards plus symmetric KL has weight `.35 * min(successful_update/720,1)`.
LR is `3e-4 * min(successful_update/720,1)`, then constant. A warms up externally;
B uses only its official internal720-step warmup. Both use WD1e-4,
betas(.9,.999),eps1e-8,foreach=False. B uses r0,weight_lr_power2,inner_momentum0.
Whole-triple AMP retries restore RNG and reuse the selected tensors without
advancing rows, updates, or ramps.

B calls optimizer.train() for gradients at y and optimizer.eval() for averaged x
deployment. Every last checkpoint serializes eval weights and eval optimizer
state consistently. Continuation returns to train mode once before updates.
No x/y/z selection is allowed. The LayerNorm backbone needs no BatchNorm pass.

Full FP32 validation at batch128 runs every epoch. Full eval TRAIN runs every5,
at the end and at the selected checkpoint. Strict maximum integer native correct
out of18000 selects each best, with exact ties retaining the earliest epoch.
Official-arm candidates use the same count, tieA. Cost never selects the model.
Inference is maximum direct utility over the full legal pool, exact ties by global
ID, with one completed solver downstream. This training process invokes no solver.

One focused CPU check precedes formal training:

```bash
bash code/V4/run_v4_r65.sh cpu-check
R65_PARENT_GPU_BATON=schedule-free-full18 bash code/V4/run_v4_r65.sh train
```

The formal command runs A then B. Run it inside tmux `r65_train`. The launcher
uses existing Slurm1465 on gpu03 and only index1 UUID
`GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`; an occupied assigned device waits.
Ordinary first successful formal update supplies timing; no GPU sizing/preflight.
Artifacts live under `runs/R65_schedule_free_full18/`, including periodic resumable
checkpoints, raw curves/predictions and offline selector W&B logs. `--resume`
continues an incomplete arm and skips completed arms.

After both finish, `validation_candidates.json` exposes the validation-only
mechanical candidate freeze. Parent must uniquely lock models, inference rule and
official arm in `final_test_lock.json` and explicitly authorize TEST. Only then
may `R65_PARENT_GPU_BATON=locked-test bash code/V4/run_v4_r65.sh test` run. Its
exclusive `final_test_started.json` permits only one original TEST per frozen arm.
There is no automatic TEST, analysis, reference inference, review, or Git action.
Historical references are reused. Legacy B-task feasible-domain mismatch remains
unresolved/disclosed. R58 is completely deferred; no R66 is proposed here.
