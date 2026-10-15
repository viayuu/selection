# R78 | 真实会赢但模型零选择：六轮实验回顾

[PDF](ZERO_SELECTION_RETROSPECTIVE.pdf) | [CSV](round_summary.csv)

各轮基线不同，非同条件排行榜。候选均指实际结果，不是 fallback。

| Round | Baseline | VAL Top1 baseline -> actual (%) | VAL regret baseline -> actual (%) | VAL cost baseline -> actual | Status |
|---|---|---:|---:|---:|---|
| 001 | R76 ep18 | 69.236913 -> 69.379546 | 0.464503 -> 0.467179 | 11.402474922 -> 11.402575925 | inconclusive; 1/18 converged; cost gate failed |
| 002 | R76 ep18 | VAL not evaluated | VAL not evaluated | VAL not evaluated | inconclusive; 14/18 converged; VAL not evaluated |
| 003 | R77B ep28/u34776 | B 70.894309 C 70.814434 T 70.828698 | B 0.401327 C 0.410926 T 0.413111 | B 11.395558344 C 11.395855343 T 11.395991602 | closed negative; 0/6 endpoint gates; finite budget only |
| 004 | R77B ep28/u34776 | 70.894309 -> 67.947511 | 0.401327 -> 0.492726 | 11.395558344 -> 11.407802464 | closed negative; all 3 gates failed |
| 005 | R77B ep28/u34776 | 70.894309 -> 70.753102 | 0.401327 -> 0.402753 | 11.395558344 -> 11.395610637 | closed negative; all 3 gates failed |
| 006 | R77B ep28/u34776 | 70.894309 -> 70.801597 | 0.401327 -> 0.412334 | 11.395558344 -> 11.395979870 | closed negative; all 3 gates failed |

## Round 001
Frozen R76 logits plus per-task sum-zero intercepts; natural TRAIN CE + ||b||^2/(2N); seed2, batch1024.
540,000 CPU updates (30,000/task), batch1,024; only 1/18 passed two successive convergence checks.
VAL (3,895/task, 18 tasks) zero/winner pairs: 54 -> 54; corrected/harmed/wrong-to-wrong: 740/640/975.
TEST: False; final: retain R76 identity; no certified converged optimum claim.
[Results](../../round001/RESULTS.md) | [Protocol](../../round001/RESEARCH_HANDOFF.md)

## Round 002
Frozen R76, K-1 intercepts; FP64 cost-weighted quadratic mean[(r/c)*(1+h+b)^2]+||b||^2/(2N), c=TRAIN mean regret; projected SVRG.
253,000 CPU updates + 271 full-gradient snapshots (not updates); 14/18 converged, 4 hit 30,000; no candidate VAL.
TRAIN (70,114/task, 18 tasks) zero/winner pairs: 55 -> 21; corrected/harmed/wrong-to-wrong: 143169/503105/not recorded.
TEST: False; final: retain R76 identity; analytical TRAIN reference not deployed.
[Results](../../round002/RESULTS.md) | [Protocol](../../round002/protocol.json)

TRAIN only; candidate VAL was not evaluated because 18/18 convergence was required.
TRAIN R76 -> fitted Top1: 69.369646% -> 40.849743%; regret: 0.460650092% -> 2.826220296%; mean_cost: 11.373004840 -> 11.653491213.
Analytical b* reference: TRAIN Top1=40.849505409%, regret=2.826224389%; 27 prediction disagreements with fitted; not deployed.

## Round 003
Matched full-model continuation from B: control original loss; treatment replaces only 0.02 risk with 0.02*mean_valid_K[(r/a_t)*softplus(z-mean_valid_K z)]; CE/compare/RDrop each remain 0.35.
29,808 GPU updates total (14,904/arm), batch1,024; fixed +4,968/+9,936/+14,904; 0/6 endpoint gates passed.
VAL +4968 only (3,895/task, 18 tasks) zero/winner pairs: 30 -> 27; corrected/harmed/wrong-to-wrong: not recorded/not recorded/not recorded.
TEST: False; final: retain whole R77B; no sufficient-convergence claim.
[Results](../../round003/FINAL_RESULTS.md) | [Protocol](../../round003/EXECUTION_PLAN.md)

Actual VAL endpoints; every cost/regret worse than locked B, 0/6 gates passed. Final fallback zeros are not actual trajectory changes.
| Update | Arm | VAL Top1 (%) | VAL regret (%) | VAL mean_cost |
|---:|---|---:|---:|---:|
| 4968 | control | 70.921409214 | 0.405973048 | 11.395716839270 |
| 4968 | treatment | 70.897161603 | 0.407708383 | 11.395848830432 |
| 9936 | control | 70.850092711 | 0.408499816 | 11.395744982192 |
| 9936 | treatment | 70.832976751 | 0.410676066 | 11.395886356713 |
| 14904 | control | 70.814434460 | 0.410925983 | 11.395855343252 |
| 14904 | treatment | 70.828697761 | 0.413110693 | 11.395991601929 |

## Round 004
Frozen B cost head: g=1-p+Dp; FP64 relative-cost regression + fixed unit prior; no backbone update.
24,840 CPU updates; 1,380/task; batch 1,024; fit 21.50s
VAL (3,895/task, 18 tasks) zero/winner pairs: 30 -> 23; corrected/harmed/wrong-to-wrong: 3276/5342/1406.
TEST: False; final: retain whole R77B.
[Results](../../round004/results.json) | [Protocol](../../round004/protocol.json)

## Round 005
Fixed 0.5/0.5 average of A/B T=1 probabilities; original preprocessing; no training or weight search.
0 updates; one cached VAL screen, 70,110 rows; CPU 0.290s
VAL (3,895/task, 18 tasks) zero/winner pairs: 30 -> 33; corrected/harmed/wrong-to-wrong: 751/850/843.
TEST: False; final: retain whole R77B.
[Results](../../round005/results.json) | [Protocol](../../round005/protocol.json)

## Round 006
Original two-view loss + 0.10 cost-profile neighborhood representation loss; unique final14904 endpoint.
14,904 GPU updates; 828/task; batch 1,024; single 4090; 4,858.43s
VAL (3,895/task, 18 tasks) zero/winner pairs: 30 -> 26; corrected/harmed/wrong-to-wrong: 641/706/586.
TEST: False; final: retain whole R77B.
[Results](../../round006/results.json) | [Protocol](../../round006/prepared.json)

## Existing R76 / R77B TEST reference
52 -> 26 zero-selected winner-method groups; only 703 / 14,077 (4.9940%) fixed original missed-winner rows truly corrected, with 13,374 still missed. TSP/LEHD remains 0/637.
R76/R77 changed batch and training conditions jointly; this is not a batch-only causal result or an accepted R78 intervention.
[Existing source](../R76_R77_TEST_COLD_COUNTS.md)

Rounds001/002 remain inconclusive, with convergence not certified. No accepted R78 candidate; retain B; the long-term goal is unresolved.
Read historical JSON/CSV/Markdown only. No model/NPZ loading, training, inference, experiment reruns, literature search, goal changes, report edits or commits.
