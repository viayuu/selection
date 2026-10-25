# R110: Method-Conditioned Nodes With Pure CE

This is the objective counterpart to R109, not a second new architecture.
The four-GPU study crosses graph-weight sharing with the training objective.

| Objective | Tied candidate graph weights | Private candidate graph weights |
| --- | --- | --- |
| Original composite, two dropout forwards | R109A | R109B |
| Pure legal-pool utility CE, one forward | R110A | R110B |

R109 retains the original mixed CE/comparison/risk/RDrop; R110 removes those
additional objectives and uses only unweighted raw-utility CE. R93 already
tested pure CE on the original shared model. The research question here is its
interaction with candidate-specific graph computation, not whether CE is new.
The number of dropout forwards, probability link and training compute differ;
do not call the cross-version comparison a one-loss-term-only ablation.

## Model And Inputs

Use exactly code.V4.r109_model, mapping A_tied_ce to A_tied_late and
B_private_ce to B_private_late. Four original shared instance layers plus two
graph attention/FFN updates per legal candidate, before masked mean/max pooling.
Candidate-specific summaries enter the original score MLP and the original two
joint layers. No branch-gradient-independence claim: joint/context paths retain
cross-candidate coupling. Method-private parameters are keyed by global IDs,
not candidate position, supervision, routes or predicted winner.

Same R84 original fresh seed2 and branch prototype initialization as R109;
same initial evaluation function for tied/private arms. Parameters are A2,305,033
and B10,220,553, not matched counts/FLOPs or complete state hashes. No trained
selector initialization, donor, route targets, new features, query outcomes,
TRAIN label contexts, method quota, calibration or posthoc output correction.

## Fixed Execution And Evaluation

Frozen NEW90k:18 tasks, TRAIN81,000/VAL4,500/TEST4,500 each, same members and legal
candidate exclusions. Keep single original MOEL value named reld_mt. No solver
reruns, new labels, re-splitting or R58 work. Reuse warm original R84 input cache
and TRAIN-only normalization/statistics without rebuilding or copying.

Seed2, logicalbatch1024, natural task/sample schedule. AdamW peak1e-4/end1e-6,
weight decay1e-4, norm clip1, dropout.1.57,600 successful updates, warmup4,320,
full VAL every1,440 and TRAIN every7,200. Training objective is one-forward
`cross_entropy(raw_utility masked to the legal pool, native TRAIN winner)`.
No class weights, risk, comparison, RDrop, cost regression or extra head.
Inference is unchanged utility difference/maximin argmax. Recorded evaluation
CE is the existing maximin diagnostic, distinct from raw-utility training CE.

Main's actual one-forward private-model B256/1024 TRAIN-input witness passes:
TSP n49917,334MiB, CVRP n50019,726MiB, ATSP n1005,845MiB and VRPBLTW n1015,244MiB
peak allocated. Fixed microcaps TSP/CVRP256, all other tasks1024, eval256.
Logicalbatch remains1024; no automatic shrink or terminal failure relaunch.

Strict minimum whole18 VAL macro_vs_sbs_pct, first exact tie, common full-TRAIN
SBS. Lock one checkpoint per arm; full TEST and both LIBs use that same lock.
Report full18 and ALL Top1, mean_cost, SBS/Oracle cost, vs_SBS/vs_Oracle, actual
regret, fixed36 recalls/false picks/net costs, NSS4, runtime and combined
model/winner distribution PDFs with modelTop1 in titles. No per-task stitching,
TEST/LIB thresholds or checkpoint reselection. TEST is development-exposed.

Whole-version completion creates one curated GitHub publication. GPU slots are
the original R108 two4090s on existing allocation1475/gpu05, after original
pipelines exit rc0; CPU-only reporting recovery does not justify re-running GPU
stages. Retain original UUID mutexes, CPU quartets and allocation container.
All engineering is Sol/max; Astra/max supplies literature/mechanism ideas only.
