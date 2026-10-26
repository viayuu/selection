# R116: full18 subset-dependent elimination

## Why This Version

Astra's completed post-major-paradigm report proposes a finite zero-new-label
fallback (D3). R115's predeclared cross-base response gate failed, so its
annotation/neural expansion stops. R116 is a separate finite whole18 experiment,
not an additional response-model tuning round.

Only the two authorized RTX3090 UUIDs in gpu03/job1465 may execute new GPU work.
No 4090, new allocation, solver execution, instances or labels.

## Matched Comparison

- A_static: compute full-set elimination logits; each remaining subset only
  masks and renormalizes those fixed preferences.
- B_subset: recompute nonlinear candidate interaction with the remaining set.
- Same fresh seed2 complete parameters, R84 TRAIN-only geometry, frozen90k
  data/pools, natural tasks/instances, logical1024 batch and update budget.
- Both evaluate every subset with the same head tensor shapes and use exact
  log-space dynamic programming over ALL elimination paths. Neither deploys
  greedy elimination, a scalar-score/maximin bypass, thresholds or quotas.
- Four-layer128 node encoder and two-layer original bidirectional interaction;
  only the decision object becomes an elimination distribution. The new
  64/32-width set head is identical in both arms. No donor/code/context/probe.

## Supervision And Budget

One dropout forward; fixed loss = 0.5 native champion NLL + 0.5 mean elimination
NLL over K-1 removals. Full ranking comes from original FP64 TRAIN costs, with
ascending global ID ties, except native winner retained first inside its exact
minimum tie group. Labels enter only losses, never subset scores/query inputs.
No R-Drop, extra risk, cost head or reweighting. This is a new matched objective,
not a loss-isolated comparison with historical R77/R94.

AdamW lr1e-4, weight_decay1e-4, clip1, dropout0.1;4266 successful warmup updates,
then cosine to1e-6. Each arm28440 successful updates=1580/task=19.9743 full TRAIN
exposures. Full VAL every1422updates; full TRAIN every7110 and selected-best.
Caps/eval batch are fixed in main protocol after one necessary capacity witness;
no automatic OOM shrinking. Shared singleGPU row-weighted microbatch semantics.

Capacity completed on the authorized first3090: four shape corners, one actual
forward/backward/AdamW update each, 7.3seconds total including startup. Locked
microbatch choice: TSP/CVRP/ATSP256, other15 tasks512; logicalbatch1024 and
evaluation256. Worst tested CVRP n500/K10 used7,179MiB allocated. A/B full
fresh-parameter hashes matched. These throwaway steps never initialize formal
training. This is not evidence of selection quality or final throughput.

Select maximum whole18 VAL macroTop1, ties minimum actual regret then earliest.
Lock one checkpoint and inference rule, then full TEST/LIB once; saved outputs
may be reused for reports but never reselection. Preserve old exposed holdouts.

Total GPU hard cap28hours,14/arm INCLUDING fixed evaluation/report GPU stages.
If incomplete at the cap, retain partial receipt/checkpoints, not a negative
full-budget result. Do not expand seeds/layers/thresholds to rescue this version.

## Decision And Delivery

Relative matched A: B needs VAL Top1 +0.20pp, regret harm<=0.005pp, vsSBS not
worse, every task regret harm<=0.05pp. Report historical R94B only as context.
Deployment p95 including subset DP must be <=2x an ordinary point selector to
recommend deployment. Better tail-ranking NLL or fewer zero-selected methods
alone are not success. Report full18 metrics, frozen zero-list true rescues and
false picks/net cost, champion vs fullranking NLL, timing/memory, PDFs with
paired selected/native-winner charts and same-checkpoint NSS fourdataset table.

One curated GitHub commit/push after completed-version reporting. Training and
reporting must remain monitored under the active goal; no automatic new arms.

The final CPU publisher inherits a process-local `core.autocrlf=false` through
`GIT_CONFIG_COUNT=1`, `GIT_CONFIG_KEY_0=core.autocrlf`,
`GIT_CONFIG_VALUE_0=false`. The repository/global Git setting is not changed.
This preserves generated CSV bytes and their recorded SHA256 in the commit,
avoiding the historical CRLF-to-LF publication-manifest mismatch. This is a
report/publication setting only, with no effect on any GPU training process.

Primary ranking-model reference: Ragain and Ugander,
[Choosing to Rank](https://arxiv.org/abs/1809.05139). Exact elimination DP and
whole18 neural solver-selection use here are project-specific choices, not
claimed performance results of that paper.
