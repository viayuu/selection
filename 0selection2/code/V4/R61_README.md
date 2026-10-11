# R61: Dense Saved-Route Behavior Pretraining

R60 gave a small positive validation change, not a substantial improvement.
Its validation-cost-best B epoch39 is the locked incumbent: Top1 49.3611111%,
actual regret 0.9227890652%. R61 tests a different supervision mechanism.

## Completed Result

Both arms completed 21,120 pretraining and 45,504 full-pool fine-tuning updates.
A selected epoch24: validation Top1 48.8111%, actual regret 0.924975%.
B selected epoch32: validation Top1 49.0778%, actual regret 0.929035%.
B does not improve the incumbent and fails the precommitted continuation screen.
This rejects this saved-route pretraining recipe, not an accuracy ceiling.
See `runs/R61_behavior_pretraining/RESULTS.md` for curves and paired outcomes.
No test data was accessed.

## Two Stages, One Matched Comparison

Pretrain the original instance encoder on the two archived ReLD solution
structures, then transfer only that encoder into the original R45A/R43A
full18/full-pool selector. Train the complete selector, not a gated pair expert.

- A uses old-TRAIN route replay; B uses additional existing TRAIN routes.
- Stage1 reuses R54's customer successor/END targets and normalized loss.
  Only the encoder and two training-only route heads are optimized.
- Both arms have 21,120 successful pretraining updates, batch128 with tails,
  matching task and actual node counts. B traverses new90k/task twice.
- Stage2 discards route heads and uses unchanged original full-pool labels,
  score differences, maximin logits, comparison/risk objective and R-Drop.
- The remaining selector parameters start fresh and identically; all parameters
  are trainable. A controls the common fine-tuning LR trace and stopping horizon.

Raw attributes, task conditions, original solver ID/manual-feature branch and
legal candidate pools are unchanged. No route, source-instance ID or current
answer enters the encoder. At deployment there is no route head or solver probe.
Geometry statistics come only from original TRAIN and are shared.

## Existing Assets Only

The data worker resolves actual saved tour bodies and input identities while
converting them into compact targets. Expected expanded coverage is 90k new
rows/task across15 MVRP, excluding the original10k prefix. These are
problem-instance observations, not a claim of independence across task variants.
Original R54 TRAIN routes provide the replay control. Validation/test routes
are not pretraining examples.

There is no new solver run, instance generation, annotation job, API call or
cost-table replacement. Missing required routes stop the experiment rather than
silently shrinking a candidate/task set or generating substitutes. Existing
small calibration and VRPLTW repair limitations stay explicit. The legacy B
semantics discrepancy remains unresolved; R58 remains deferred.

## Training And Decision

Stage1: seed2, AdamW1e-4/WD1e-4, dropout0.1, clip1; one-pass linear warmup,
then cosine decay to1e-5 in the second pass. Transfer the final encoder without
validation selection. Stage2: original batch128/full coverage, warmup3,
AdamW1e-4/WD1e-4 and 15--40-epoch controls. Select strict minimum original
validation macro_vs_sbs_pct independently in each arm.

Continue only if B reaches at least52.3611111% full18 native Top1, beats A by2pp,
and has actual regret no worse than A and the incumbent. This is a continuation
screen, not the70% goal. No test access occurs in this round.

## Execution And Outputs

Use easynco and tmux on the sole designated available3090. Preserve other GPUs,
user queues and historical files. Execution order is A pretrain, A fine-tune,
B pretrain, B fine-tune. There is no GPU-hour cutoff.

Outputs are under `code/V4/runs/R61_behavior_pretraining/`. Each arm has a
`pretrain/` directory for stage1 and ordinary best/history/prediction files at
its root for stage2. The complete comparison, curves and paired decision changes
are generated once after both arms finish.

```bash
bash code/V4/run_v4_r61.sh plan
tmux new-session -d -s r61_train 'env R61_PARENT_GPU_BATON=behavior-pretraining bash code/V4/run_v4_r61.sh train --resume'
bash code/V4/run_v4_r61.sh analyze
```

Keep correctness work to a few focused conversion/mask tests and one real
pretraining update, then start formal training. Do not add audit/review gates,
timing pilots or parameter searches. Commit and push the whole completed version
once, including code, configuration, selected checkpoints and results.
