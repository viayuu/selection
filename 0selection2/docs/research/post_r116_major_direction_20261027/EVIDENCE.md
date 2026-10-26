# Completed evidence for scientific follow-up

Collected by Sol from completed version files; not a request for Astra result
verification. Main objective remains whole18 improvement, not a local binary.

## Fixed Scenario

`selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`: every problem
81,000 TRAIN/4,500 VAL/4,500 TEST. Eight B tasks exclude RF/MoSES; MTL is removed,
MOEL retained as reld_mt. Previous holdouts were exposed to R&D. R58 all-solver
constraint unification remains deferred. Two existing RTX3090s, no new4090work.

The prior complete Astra report is published in86900ed3be5152a7aa2601862ea96f07fbe6c670.
R109/R112/R113 completed without joint system breakthroughs. R113's typed
synthetic-process knowledge is not native candidate execution. R114 completed
16-task response acquisition, not a downstream neural response model.

## R115: Small Cross-Base Response Screen

Published cc3f039d74d17f2b683013e7ea92e4201e6b14eb. Only512 TRAIN bases,
2,048 endpoints/1,536 pairs. Two base-disjoint folds16 bases/problem each;
same-base families/arms stay together. One fixed Ridge(alpha1) per problem/family,
714 features=357 existing input statistics plus endpoint changes; fit-fold-only
standardization and fixed non-Oracle cost scale. Two controls: zero, fit-fold mean.

| Condition | Ridge MAE | Zero MAE | Fit-fold mean MAE |
|---|---:|---:|---:|
| CVRP-demand |1.256850|0.816604|0.839936|
| VRPL-demand |2.732097|1.464198|1.774381|
| OVRPL-demand |2.800651|1.992098|2.018942|
| OVRPTW-tw_center |5.062264|2.521884|2.559576|

Units=%fixed TRAIN-fold scale, NOT selector regret. Predeclared continuation gate
failed. No new solves, validation/TEST reads or GPU. This only bounds a small,
high-dimensional linear screen; it is not a proof against neural response learning.
R114 directed/random perturbation amplitudes were not strictly matched.

## R116: Complete Matched Whole18 Experiment

Published dfa66a773d40707c91a0355646e45d495aaf7871, remote main confirmed.
Both fresh seed2, same complete parameters/data order, logical1024,28440 updates
(19.9743 TRAIN exposures), four128 node layers/two joint layers. A_static uses
fixed full-set elimination preferences; B_subset recomputes a nonlinear remaining-
set context. Same all-subset head tensor shapes, exact log-space ALL-path champion
DP in BOTH arms, not greedy vs DP. No solver execution/newlabels.

Loss in both=.5 native champion CE+.5 per-elimination mean full-ranking NLL;
no RDrop/pairwise/risk. Historical R77 comparisons are not isolated loss ablations.
Unique whole18 VAL maximumTop1 checkpoint, cost tie-break; formal TEST/LIB locked.

| Model | Selected update | TRAIN Top1 | VAL Top1 | VAL regret | TEST Top1 | TEST regret | TEST mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_static |28440|71.085597%|70.897531%|0.410795%|71.134568%|0.404762%|11.374935341|
| B_subset |25596|71.107270%|70.933333%|0.412386%|71.255556%|0.402782%|11.374748341|
| R94B historical |locked earlier|see report|see report|see report|71.577778%|0.379115%|11.372233605|

All costs/metrics same frozen scenario/nativeFP64 labels. ALL equal-task,
cross-problem raw mean cost descriptive only. TEST TRAIN-fittedSBS cost11.385299246;
Oracle cost11.329932369. B-A TEST corrections796, harms698, net98/81,000;
regret delta-0.001980316pp. B's CVRP TEST Top1+.533333pp but regret+.048287148pp.

R116's declared VAL acceptance failed: Top1+.035802469pp vs required+.20pp,
regret+.001591311pp, vsSBS+.000411689pp (worse). No TEST-based promotion.
Full-DP inference p95 max task ratio vs point selector A1.09533/B1.08556, within2x.
Both ranking likelihood and champion NLL improved modestly without a breakthrough:
TEST mean ranking NLL A0.583865989/B0.564418179; champion NLL A0.698717020/B0.693839211.

### Fixed36 Zero/Near-Zero Groups

List was frozen on R84A VAL, NOT selected anew on R116. On TEST it covers4,598
native winners. True selected winners: R84A13(0.2827%recall), R94B117(2.5446%),
R116A10(0.2175%), R116B18(0.3915%). BvsA rescues12/losses4; BvsR94 rescues6/losses105.
B selects a fixed36 method34times with18true/16false, precision52.9412%.
The low genuine recall is not solved by a smaller zero-selected-group count.
TSP B TEST: LEHD winner762/4500, selected0; DIFUSCO winner334, selected0.
All method/partition costs and both-wrong contributions are in target_recovery*.csv.

### Same-Lock B Four-Dataset Context

| Dataset | NSS Top1 | B Top1 | NSS mean cost | B mean cost |
|---|---:|---:|---:|---:|
| TSP new4500 TEST |45.888889%|47.088889%|7.970400719|7.972263670|
| CVRP new4500 TEST |45.977778%|47.133333%|18.004802368|18.020067315|
| TSPLIB49 |20.408163%|24.489796%|8.217907074|8.211730043|
| CVRPLIB100 |70.000000%|44.000000%|67.101111717|67.633942080|

NSS is saved historical R79 weights evaluated on these same datasets, not newly
retrained on90k. No cross-version stitching. B is not recommended for deployment.

## Coverage Matters

R94B's donor is one frozen pretrained ReLD encoder, CVRP+15MVRP only; TSP/ATSP
have no donor features. R50 only compared two ReLD encoders on OVRPTW. Neither
tests all heterogeneous candidates' own pretrained computations. This is a code-
coverage fact, NOT evidence that adding more donors will solve selection.

The next research should use the above distinctions and all previous negative
evidence, not describe another static-ID/code-embedding or local expert variant
as a fundamentally new test. Actual next protocol/budget/execution remains Sol's.
