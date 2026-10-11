# R62: Source-Separated Portfolio-Performance Pretraining

R61 route pretraining did not improve the incumbent R60B. R62 changes the
training evidence: saved comparisons across the auxiliary solver portfolio,
then transfer to the untouched original full18/full-pool selection task.
There is no new architecture, solver execution, annotation or test access.

## Completed Result

Both arms completed 25,344 pretraining updates and 40 full-pool fine-tuning
epochs (56,880 updates). A selected epoch39: validation Top1 49.0278%,
actual regret 0.925225%. B selected epoch24: Top1 48.8500%, regret 0.931233%.
Neither exceeds the R60B incumbent49.3611%/0.922789%. The fixed advancement
screen failed. This rejects this source-separated transfer recipe, not larger
datasets in general or the possibility of stronger selection.
Full metrics and paired decisions are in `runs/R62_portfolio_performance_pretraining/RESULTS.md`.
No test data was accessed.

## Same-Source Data Comparison

A uses a fixed size-stratified nested10k NEW inputs/task, replayed to match
B's actual-size schedule. B uses the available new90k inputs/task. Both read
the same frozen snapshot of saved auxiliary costs, not old versus new recipes.
Incomplete observations mask only the pair objective. Every original legal
candidate still participates in the forward; no partial-table winner is made.

The auxiliary deployments can differ from legacy labels in inference budgets,
sampling and constraints. Their observations are separate-source pretraining
knowledge, not replacements or certification of the original cost table.
Source limitations, existing VRPLTW repair and legacy B inconsistency remain
explicit. R58 is deferred and no cost-table solving is performed.

## Training

Start both original R45A ScoreDifferenceSelector models from identical fresh
seed2 parameters. Stage1 trains all original representation modules with
per-query mean BCE over observed finite unequal FP64 cost pairs. No full-pool
winner, risk, R-Drop or route objective is inferred from incomplete rows.

Use two complete new-input passes, batch128 with tails, matched successful
updates and actual sizes. AdamW1e-4/WD1e-4, dropout0.1, clip1; first-pass
linear warmup then cosine decay to1e-5. Transfer the final fixed-budget state.

Reset only the utility score head to identical fresh values and reset the
optimizer/scaler. Fine-tune all parameters on original full-pool labels with
the unchanged R43 objective/R-Drop. A controls the common15--40-epoch LR trace
and horizon; both independently select strict-minimum validation macro_vs_sbs_pct.
Only original single-solver inference is used after training.

Advance B only with native full18 Top1 at least52.3611111%, at least A+2pp,
and actual regret no worse than A and R60B's0.9227890652%. This is a continuation
screen, not70% achievement. No test data is read even if it passes.

## Execution

Outputs: `runs/R62_portfolio_performance_pretraining/`, arms `A_subset_seed2`
and `B_full_seed2`. Each has `pretrain/` plus full-pool fine-tuning artifacts.
Use easynco and the sole designated available3090; preserve all other GPUs,
user queues and historical data. There is no GPU-hour cap.

```bash
bash code/V4/run_v4_r62.sh plan
bash code/V4/run_v4_r62.sh train --resume
bash code/V4/run_v4_r62.sh analyze
```

Formal training runs in tmux. Keep checks to the necessary loader/target
boundaries; record first-step gradients and throughput during actual training,
not a separate GPU preflight. Analyze once after both arms finish, then commit
and push the whole completed version once. No rescue coefficient/seed search.
