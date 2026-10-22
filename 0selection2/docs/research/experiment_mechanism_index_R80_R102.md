# R80-R102 Mechanism Ledger

Snapshot: 2026-10-22 12:34 UTC. This continues the existing R60-R79 index.
Machine-readable companion: `experiment_mechanism_index_R80_R102.csv`.
It is an internal overlap index, not a new benchmark, leaderboard or result
audit. Read only the referenced small README/PLAN/RESULTS/history summaries;
no weights, prediction arrays, source data, solver or new model was used.

## Scope And Evidence

- NEW70114 means the original new-only single-ReLD/B-filtered release, each
  task TRAIN70114 / VAL3895 / TEST3895. NEW90k is the frozen current release,
  each task TRAIN81000 / VAL4500 / TEST4500. These are not causal same-data
  comparisons merely because all versions have18 tasks and seed2.
- Complete metrics mean the referenced saved locked results exist; they do
  not erase an original report failure. R83's original PDF-stage failure and
  R100's explicit CPU-only recovery remain historical evidence.
- R99/R101 running status was checked against each actual controller PID
  and advancing update logs in this goal turn. R102 queue receipts are
  waiting_predecessor, not training. These observations expire; use their
  live progress files for the present state. A PLAN alone proves no result.
- No new checkpoint selection, threshold, model mixture, TEST evaluation,
  inference rerun, training or publication is performed by this ledger.
  Existing TEST has informed research; it is not a new blind holdout.
- Each CSV row is one version, not one method or independently controlled
  experiment. Different arms, historical controls and diagnostics stay distinct.
  Native winner frequency, actual model picks and true winner recall differ.

## Training And Representation Families

| Versions | What has actually been tried | Boundary |
| --- | --- | --- |
| R80 | CE-only, smaller batch, higher peak LR, static source-text plus ID | Four different changes; R80A is NOT an original-loss control for AUC |
| R81B | Same-method winner/nonwinner logistic AUC over full logical batch | Historical R77B reference, no newly trained A |
| R82B | Context-matched continuous-relative-regret cross-instance ranking | Different from global AUC; no new matched A |
| R83 | Restrict source-text vectors to encoder/decision model code | Not numeric pretrained features on the current instance |
| R84 | Original versus direct-attention scoring, with/without focal CE | Matched NEW90k release; old-release results are separate |
| R85 | OVR, EQL-style gradient balance, cost-profile KL, candidate subsets | Not just four coefficients of the same CE |
| R87 | Shared versus method-private residual parameters | Does not replace R69's earlier complete task-private bank history |
| R88 | Geometric anchor versus actual TSP physical node0 role | Full18 training but only the TSP input slot is changed |
| R89 | AdamW versus Muon hidden-matrix optimization | Small matched single-seed gain; not a recovered-winner solution |
| R91 | Change CE/risk/R-Drop probability links | Retains complete comparison loss and original argmax path |
| R92 | Hard/learned graph hierarchy and raw-statistic tree families | Tree bank is full18 but not equal-neural-compute control |
| R93 | Pure utility CE, one forward, dropout0.1 versus0 | Removes comparison/risk/R-Drop rather than merely moving CE |
| R94 | Random/pretrained native routing nodes before trainable graph layers | Full18 selector; donor covers16 routing interfaces, not TSP/ATSP |
| R95 | Gradient-proxy adaptive exposures with inverse-probability correction | Preserves expected pre-clip empirical gradient, not all optimizer behavior |
| R99 | Local row-distance versus global Sinkhorn input messages | Running; no final efficacy result |
| R101 | Random/pretrained node hints, then encoder-only transfer and reset | StageI complete; full18 StageII running; no deployment teacher |
| R102 | Extend numeric native donor coverage to TSP and ATSP | Queued; both arms already contain identical routing pretraining |

## Other Regimes Are Not More SGD Models

| Version | Mechanism | What it must not be called |
| --- | --- | --- |
| R86 | Small frozen-readout radius/angle counterfactual | New normalized-head training or full TEST result |
| R90 | Existing-probability-link reasoning and full TRAIN/VAL fitting curves | New training or proof of a unique regularization/capacity cause |
| R96 | Answer-aware overlap of21 saved VAL models | Deployable ensemble, winner oracle score or accuracy ceiling |
| R97 | Frozen public TabICL with full historical TRAIN winner context | Scratch neural training or labels absent from all forward context |
| R98 | Frozen TabICL per-method expected relative gap | Full-pool winner probability or a classification objective |
| R100 | Frozen independent pair contexts and fixed mean-win/Borda | Joint champion probability or a newly trained comparison encoder |

R81B here refers to `R81_method_conditioned_auc`, not the distinct
`R81_current_label_deployment` provenance document. That document is not an
extra trained AUC model and does not certify R58 scenario repair.

## Decisions This Index Changes

Calling a proposal balanced, hard-example, cost-sensitive, retrieval,
probability-link, independent-expert, hierarchical, contrastive, recurrent,
pretrained or relational does not by itself establish a new project mechanism.
Identify what object is learned, what information changes, and what the final
decision must use. In particular, R63 already has candidate-state recurrence,
R69 already has a complete18-task private bank, and R71 already models a
joint performance distribution under an older scenario.

The completed NEW90k full-TRAIN trajectory evidence shows small aggregate
TRAIN/VAL gaps, while several well-supported winner methods are suppressed
on TRAIN too. This limits the blanket severe-unseen-overfitting narrative;
it neither proves label randomness, an accuracy ceiling nor that more data
or a different learning mechanism cannot help.

Completed weighting/ranking/private-parameter variants sometimes lower the
strict-zero count, but fixed-target true rescues remain small and false picks
must be counted. Method-choice frequency is not the objective. R94 adds
more distinct MVRP rescues than the recent context banks, but has online
encoding cost and little target recovery outside that covered domain.
This supports asking the already-queued R102 coverage question, not assuming
its answer or launching further variants before current evidence exists.

Only whole-model VAL-selected TEST/LIB results can support acceptance of a
finished version. The queued/running rows contain no such result. No healthy
job was interrupted; no new allocation, solver label or R58 work is implied.
