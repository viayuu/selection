# R78 Round003 Execution Preparation

Research specification: `../review-stage/FULLB_MONOTONE_COST_PLAN.md`.
Execution/code/checks: Solmax. No further Astra research or review is requested.

## Fixed Matched Trial

- All18 tasks, full70114 TRAIN rows/task, natural class frequencies, seed2.
- Both arms load locked R77 full-B ep28/u34776, SHA256
  `31c0c75d66b1bde6ad533909a9c04b0036e80af4c4a5775b01b957ce310e6d1b`.
- Both start fresh AdamW: betas(.9,.999), eps1e-8, weight_decay1e-4;
  no source optimizer or RNG restoration. Fresh seed2 task/order/shuffle streams.
- Both use the complete shared `late_lr.npy` sequence at logical34777..49680.
  The inherited R-Drop warmup position remains34776, so its weight stays.35.
- Exactly14904 successful updates/arm,828/task, one optimizer update per1024
  real rows; inherited clip1, AMP overflow replay, two-dropout objective averaging.
- Reuse R77 CPU gather/pinned double-buffer async prefetch and immutable caches.
  Both arms use TSP/CVRP micro352, other tasks1024; inference eval batches256.
- Control calls the original single objective unchanged. Treatment retains.35
  CE,.35 compare and.35 R-Drop and replaces only.02 risk with the specified
  centered-logit cost-weighted softplus. No new head, feature or inference rule.
- Every r and task scale a_t is constructed from original FP64 TRAIN costs.
  Prepared r/a_t is then FP32 for inherited loss reductions. An all-tie task
  has exactly zero weights/block. Invalid methods are excluded from centering
  and risk averaging. No per-instance extra normalization or coefficient scan.
- Supervision, including the new risk weights, is removed before both forwards.

## Evaluation And Gate

Only complete18 VAL at continuation+4968/+9936/+14904 (epochs4/8/12), logical
u39744/u44712/u49680. Reuse locked B's saved TRAIN/VAL as zero-update fallback.
No preparation forward, VAL-target scaling, TEST/LIB stage or TEST rule design.

Among eligible endpoints use native macro Top1, then lower actual regret,
then earlier continuation update. Cost<=11.39555834354421 and actual
regret<=0.40132674544775454%; Top1 must strictly exceed the unrounded locked B.
No tolerances, taskwise assembly, frequency gate or rounded comparisons.
Treatment-specific positive also requires Top1>selected matched control.
Report signed cost/regret differences against control without claiming Pareto
dominance. Control-only gains are continuation/reset findings, not softplus gains.

Selected new checkpoints each receive one full TRAIN report. Saved predictions
support the paired TRAIN/VAL changes, exact cost ties, native fixes/breaks,
wrong-to-wrong switches and descriptive zero/near-zero method counts.
No formal TEST release is authorized by this preparation or trainer.

## Minimal Checks And Pending Hardware

Run only the five CPU synthetic checks in `code/V4/test_r78_fullb_cost.py`:
FP64/task-scale construction; centered masked/zero-block risk; unchanged control
and two label-free forwards; exact1024 single-step accumulation; late LR/endpoints
and complete18 raw gate. Do not rerun earlier18-test suites or gradient audits.

After explicit user hardware allocation, permit one disposable GPU/TSP treatment
smoke at batch1024 and micro352. No VAL or model checkpoint; the scientific
continuations restart from B with fresh seed2/optimizer, not smoke weights.
This smoke is pending and is not evidence of GPU readiness now.

NSS four-card DP migration has priority. `hardware_allocation.pending.json`
is deliberately unapproved with no assigned GPU. Do not launch, query capacity,
reuse stale R77 approvals or kill processes. A future allocation must explicitly
approve each arm/node/job/UUID; launcher uses tmux plus an existing-job exact
srun step, four CPUs, two threads and9GiB per run. One run/card; only the four
previously user-approved UUIDs are allowed, no SSH GPU programs.

CPU preparation: `bash code/V4/run_v4_r78_cost.sh prepare`.
CPU tests: `CUDA_VISIBLE_DEVICES='' python -B -m unittest code.V4.test_r78_fullb_cost`.
Workflow target: `active / ready_pending_hardware`. Native child app-goal flags
are not resumed or completed; legacy/root goal remains paused and unmodified.
