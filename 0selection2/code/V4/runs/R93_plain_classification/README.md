# R93: Plain Classification Learning Control

This is an engineering control, not a claimed novel paper mechanism. R91 changed
probability links but retained comparison/risk/R-Drop. R93 instead asks whether
the original full18 selector can learn stronger instance discrimination when its
only target is the proper, native winner classification loss.

| Arm | Training objective | Dropout |
| --- | --- | --- |
| A | One forward, unweighted legal-masked utility CE | 0.1 |
| B | Identical CE and original complete model | 0.0 |

No pairwise comparison, risk, R-Drop, code embedding, route target, retrieval,
prefix probing, cost regression, class quota, output bias, margin, or new head.
B still has AdamW weight decay1e-4 and is not called unregularized. Both share
every fresh R84 seed2 parameter, natural sample sequence, TRAIN-only geometry
statistics, logical batch1024, micro352 and eval256. There is no trained warm start.

The original score MLP and maximin inference remain; utility and maximin have
the same within-instance argmax order. TRAIN CE is on utility; any reused
maximin CE in evaluation is explicitly a separate diagnostic, not the training
loss. TSP's known physical-node0 execution role is identical in A/B.

Each arm uses57600 successful updates, warmup4320, original cosine learning rate
1e-4 to1e-6, clip1, validation every1440, and full TRAIN every7200 plus selected.
Choose one global checkpoint strictly by full18 validation macro_vs_sbs_pct,
then perform TEST/LIB and generate PDF. No test-dependent checkpoint/threshold
selection or per-task model stitching. TEST is development-exposed, not blind.

Historical R89B and R84A are references, not newly matched original-loss
controls. A/B isolate dropout under pure classification, not which individual
old objective caused a limitation. A greater fitting rate is not itself a
success: require natural Top1, costs/regret, fixed36 missed-winner recall and new
false picks. Do not automatically extend an unsuccessful result into a loss or
dropout coefficient grid.

A runs on a released3090 and B on a released4090. Code, data, fresh weights and
update budgets align, but hardware is not bitwise matched; tiny A/B differences
must not be treated as a causal dropout finding.

Use the two GPUs released by the completed R92 tree arms. The healthy R92 graph
hierarchy runs remain untouched. PDF runs on ADMIN01 CPU; W&B is enabled offline.
Publish one curated version commit after complete evaluation and reports.
