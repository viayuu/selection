# R125: remove BQ and LEHD, retrain a classic source selector

## Question

Does one fresh full18 source-embedding selector work better on the reduced
candidate scenario? This is a candidate-set experiment, not proof that decoder
family imbalance caused earlier under-selection. Removing methods changes both
the Oracle and the classification task; an increased Top1 alone is insufficient.

## Frozen Scope

- Source: `selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`.
- Same 81,000 TRAIN / 4,500 VAL / 4,500 TEST instances per problem.
- Remove BQ and LEHD from every legal pool: TSP 8 -> 6, CVRP 10 -> 8.
- Keep the other 16 tasks, the B8 exclusions, and the MOEL-only `reld_mt` alias.
- Keep all instances, including those whose former winner was removed.
- Recompute FP64 native winner, stable ranking, Oracle and TRAIN-fitted SBS.
- Original scenario and historical results are read-only; no solver reruns.

## One Model

- Classic R80D architecture: four frozen 1024-dimensional source views; per-role
  projection to 32 dimensions; shared fusion; trainable method ID; 128-dimensional
  method tokens; four-layer instance encoder; two synchronous dual-stream layers;
  original QKV decoder, shared score MLP, score differences and maximin.
- Frozen source vectors are reused, not regenerated. No pretrained selector
  weights, no new experts, auxiliary heads, retrieval, quotas or calibration.
- Fresh seed 2, original R77 common random initialization, R84 TRAIN-only input
  geometry statistics, logical batch 1024, microbatch 256 for TSP/CVRP and 1024
  for other tasks. Real sample-count weighting; one optimizer step per batch.
- AdamW: peak LR 1e-4, end LR 1e-6, weight decay 1e-4, gradient clip 1, dropout .1.
- Original CE .35 + comparison .35 + relative risk .02 + R-Drop ramp to .35.
- 57,600 successful updates; 4,320 update warmup; full VAL every 1,440 updates.
- Each task receives 3,200 successful updates. Shuffle tails are carried over,
  not dropped. Full TRAIN diagnostics every five VAL evaluations.
- Select the strict minimum full18 VAL macro_vs_sbs_pct against reduced-pool
  TRAIN-fitted SBS; first exact tie wins. No TEST/LIB-based changes.

## Delivery

Lock one checkpoint, then complete TEST, TSPLIB and CVRPLIB. Report VAL and TEST
per-task/equal-task ALL Top1, mean cost, actual regret, TRAIN SBS and Oracle.
Provide paired selection/winner distribution PDFs with model Top1 in every title,
CSV exports, learning curves, and complete-model forward timing. Reuse saved NSS
scores on the reduced pool when aligned; identify this as fixed-model reranking,
not NSS trained after candidate deletion. Missing matched references stay missing.

R80D is the architecture reference but used an older data volume and holdouts;
its published full-pool result is not a matched causal control. The strongest
within-run comparison is against same-scenario TRAIN-fitted baselines. Historical
TEST/LIB have informed research and are not a pristine blind test.

Use one idle allocation-owned GPU, persistent tmux and W&B. Preserve failures;
never automatically repeat a failed fresh training run. Publish one curated
version to GitHub after training, locked evaluation and reports are complete.
