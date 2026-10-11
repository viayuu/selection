# R59 precommitted performance experiment

Date:2026-10-11. Performance round1/100. Astra/max plan014 reviewed8/10, ready
for implementation only, same-family/provisional. No performance result exists
at the time of this plan.

User scope: skip R58 entirely, retain the B semantic issue as pending, continue
the70% single-solver full18-task performance goal on the designated available3090.

## Mechanism

Fully trainable routing-pretrained ReLD-MTL encoder transfer to the full-pool
selector; a random encoder with the same architecture is the minimum control.
Native encoder is shared by CVRP and15 MVRP tasks, fused with raw/local geometry
and task conditions. TSP/ATSP keep the original encoding path. No frozen-readout
subtask, local binary expert, solver decoding or new-label generation.

## Locked development rules

- Original train/validation only; unchanged18 tasks,22 global solvers and128
  legal problem/solver deployments. Expansion inputs and labels are excluded.
- Seed2, matched non-native initialization and task/instance schedules.
- Batch128 including tails, AdamW, WD1e-4, gradient clip1.
- Native encoder LR3e-5 in both arms; other parameters1e-4. Preserve ratio.
- Original comparison/risk/R-Drop, three-epoch warmup and existing15--40-epoch
  stopping/controller rules. No loss or learning-rate search.
- Checkpoint strict minimum validation macro_vs_sbs_pct.
- Advance if B minus R45A Top1>=3pp, B minus A>=2pp, actual regret no worse than
  both. R45A validation reference49.1167%, regret0.926048%.
- NO test reads or test evaluation in this round, irrespective of result.
- Focused checks: native encoder parity, padding, complete candidate outputs,
  label isolation, paired initialization and transferred-layer gradients.
- Record first-epoch throughput and actual runtime;4--8 GPUh is an estimate for
  both arms/evaluation, not a measured result or compute limit.

## Reporting boundary

All18-task validation and family curves, FP64 prediction replay, solver choices,
corrected/harmed/both-wrong cost changes. Neither a positive plan review nor the
screen proves70% or an unseen-test gain. External pretraining provenance is
incomplete. Old B semantic conflicts are not repaired by this experiment.
R58 remains deferred; no adapter repair, cost table generation or R58 baseline.
