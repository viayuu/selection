**No actionable TRAIN/validation implementation defect found. Integrity: 9/10, ready within this scope. Original 70% performance: 4/10, not ready.** No corrective experiment or replay is warranted. This does not assess R73’s TEST delivery.

`review_independence=same-family; acceptance_status=provisional`

The [terminal receipt](/public/home/shiys/0selection2/code/V4/runs/R73_expansion_teacher_training/training_terminal_receipt.json) records original `1465.109 COMPLETED/0:0`, 7,631 seconds, with the training process and owned tmux gone. The completion record and history support all **40 epochs, 9,600 successful updates and both 7.2M-row streams**.

The inspected implementation preserves the adopted design:

- The [input reader](/public/home/shiys/0selection2/code/V4/r73_data.py:18) excludes auxiliary costs and observation masks. Teacher targets retain task, source-row and global-candidate alignment.
- The [training path](/public/home/shiys/0selection2/code/V4/r73_experiment.py:265) releases the frozen teacher before strictly restoring complete R67B36, including its trained head and buffers. Optimizer/scaler state starts fresh; all 1,842,953 selector parameters remain trainable.
- The [gradient path](/public/home/shiys/0selection2/code/V4/r73_gradients.py:34) implements native mean CE plus ramped legal-pool RDrop plus auxiliary CE, releases native graphs before auxiliary computation, restores private RNG and retries the entire update before advancing counters. Saved evaluations use averaged `x`; selection uses integer counts and earliest ties.

**R73 gains 36 validation decisions, +0.200000pp, over its fixed teacher/control.** Epoch23/global63 uniquely maximizes validation accuracy, so the predeclared rule selects B. It also exceeds historical R60B by74 decisions/+0.411111pp and its R67B warm start by87/+0.483333pp. These are small, single-run package gains, without an exposure-matched control or independent replication. [Saved result](/public/home/shiys/0selection2/code/V4/runs/R73_expansion_teacher_training/B_teacher_training_seed2/result.json)

| Record | VAL Correct / Top1 | CE | Mean Cost | Actual Regret |
|---|---:|---:|---:|---:|
| Fixed R72 teacher/control | 8923 /49.572222% | 1.220912* | 11.168430 | 0.925862% |
| Selected B23 | **8959 /49.772222%** | 1.252106 | **11.167655** | **0.917354%** |
| Final B40 | 8932 /49.622222% | 1.286271 | 11.168001 | 0.919657% |
| Epochs36–40 mean | 8927 /49.594444% | 1.282046 | 11.168055 | 0.920444% |

*Teacher CE is the fixed Borda-score diagnostic; it is not directly comparable calibrated probability quality. Last-five counts are **8932, 8922, 8924, 8925, 8932**. Their mean describes epochs, not a checkpoint, ensemble or seed uncertainty. Selected validation was retained from its original epoch evaluation; no validation recount selected another model.

| Family | Teacher Top1 | B23 Top1 | Change |
|---|---:|---:|---:|
| TSP | 38.700% | 38.300% | −0.400pp |
| CVRP | 49.600% | 50.600% | +1.000pp |
| ATSP | 78.400% | 78.900% | +0.500pp |
| MVRP | 48.373333% | 48.540000% | +0.166667pp |

Nine tasks improve: **CVRP+1.0, ATSP+0.5, VRPB+1.1, VRPTW+0.4, OVRPTW+1.1, OVRPB+0.3, VRPBL+1.2, VRPBTW+1.7 and OVRPBTW+1.0pp**. Seven regress: **TSP−0.4, OVRP−0.7, VRPL−0.6, OVRPL−0.1, VRPLTW−0.7, OVRPBL−0.8 and OVRPBLTW−1.4pp**. OVRPLTW and VRPBLTW tie.

The [complete history](/public/home/shiys/0selection2/code/V4/runs/R73_expansion_teacher_training/B_teacher_training_seed2/history.json) shows additional fitting with limited validation movement. Complete native TRAIN accuracy rises **50.865556% at epoch5 →52.215556% at selected23 →53.567222% at40**. From23→40, validation loses27 decisions while CE rises1.252106→1.286271; the final TRAIN/VAL gap is3.945pp. Optimization pseudo-target CE falls0.564979→0.423589 across training. Better imitation and fitting therefore do not establish sustained native improvement. TRAIN measurements exist at5/10/15/20/25/30/35/40 plus the selected checkpoint; missing epochs must remain missing.

Actual expenditure separates as follows:

- **Teacher:** one1.62M-input pass,12,672 query batches,38,016 member forwards,4.86M member-row forwards and eighteen16-row tails; **699.395791s**. No native teacher pass, component evaluation, measured labels or solver executions.
- **Student:** preparation **49.247368s**; training-run wall **6,876.811431s**, including its evaluations and checkpoint work. New exposure is7.2M native plus7.2M auxiliary presentations,21.6M successful encoder-view rows; attempted views total21,604,080 with six retries. Distinct inputs are180,000 native plus1.62M expansion rows; only the native rows supply measured full-pool winners.
- **Evaluation:** forty complete validation passes total720,000 rows/73.926404s. Eight scheduled TRAIN passes plus selected-TRAIN evaluation total1.62M rows/163.762191s. These times are already inside student wall time. The original process records7,627.687051s overall; scheduler wall is7,631s. The single synthetic CPU check cost6.490444s separately.

Historical R60/R67/R70 training remains separate expenditure. R67B36’s inherited state contains40 pretraining plus36 fine-tuning epochs; those updates are not new R73 exposure. Shared R67/R70 pretraining must be counted once. Deployment requires **one shared selector forward and one completed solver**, without the teacher. Selected validation’s1.841943s is a whole cached-input evaluation measurement, not singleton latency or a measured deployment speedup.

**Next-plan verdict: no defensible launch-ready R74 plan from the evidence inspected.** I investigated adaptive acquisition of bounded incomplete solver states as the most substantive remaining direction; this is a research recommendation, not experiment adoption.

Its hypothesis would be that choosing which incomplete execution state to observe, conditioned on the current instance and acquired states, adds useful decision information. R73 expanded input coverage but supplied no independently measured winners on those inputs. That motivates examining additional information, without establishing that missing information caused the plateau.

[Janisch et al., problem definition and Equations1–2](https://ojs.aaai.org/index.php/AAAI/article/download/4287/4165) support sequential feature acquisition and stopping. Their tabular error-plus-feature-cost objective would require adaptation to a fixed acquisition budget and our unchanged native target. [Renau and Hart, Sections3–4 and6](https://arxiv.org/html/2401.12745v1) support short-trajectory algorithm selection, but study three continuous optimizers and complete-point fitness evaluations. Their results do not establish full18 routing feasibility or justify exposing completed routing solutions.

The local obstacle is concrete: [R51’s initializer](/public/home/shiys/0selection2/code/V4/r51_probe.py:90) explicitly constructs **OVRPTW**, and its features summarize two ReLD solvers at fixed steps1/5/10. Its completed binary comparison lost2.6pp validation and incurred1.1961× pipeline latency. Adaptive acquisition differs from that fixed probe, but changing prefix length or pooling would repeat the rejected mechanism. A full18 acquisition menu, usable existing TRAIN states, continuation semantics and total cost remain unestablished.

Other completed evidence prevents treating a familiar module as a new solution: R39/R43/R55/R71 constrain more cost supervision; R45/R47/R50/R53–56 constrain representation, retrieval and pair-specific proposals; R60/R62 constrain broader auxiliary supervision. R64–70 give mixed optimization and generalization results. R66 specifically compared standalone-`g` versus composite-`F+g` **training**, with both deploying `F+g`; its weak ensemble benefit remains counterevidence. R72/R73 now add modest gains, not a universal diagnosis.

Partial-outcome learning also remains unready. A strict R60 pair comparison leaves six of seven candidates possible. Even assuming all39 small-sample-agreeing columns were compatible and co-observed, additional exclusions remain sparse: at least seven candidates survive for TSP, eight for CVRP, all five for ATSP, and five or six for MVRP. Four/eight-instance agreement does not certify full-column compatibility. [PRODEN’s Section3 condition](https://proceedings.mlr.press/v119/lv20a/lv20a.pdf) requires every false label to have some chance of exclusion; the extra fixed-pair observations alone do not meet it. Original complete labels can anchor learning, but merely substituting a partial-label loss for R60’s comparison loss is insufficient justification for a major new experiment.

The minimum proposed next question is **read-only feasibility**, not a pilot: *Do existing TRAIN artifacts and their generating sources provide a bounded, non-completing acquisition action for every original task, with row/deployment identity, retained state and recorded continuation/cost evidence?* A compact present/absent/unknown table with exact sources would resolve whether a full18 action menu can be specified. Do not generate missing states, evaluate models, reconstruct final costs or launch a solver. I have not executed this query. Until that question is answered, exact initialization, training budget and inference workload cannot honestly receive a plan-ready verdict.

**Memory update:** R73 completed40 epochs/9,600 updates without an actionable inspected TRAIN/VAL defect. Unique B23/global63 reached8959/18000,49.772222%, +36/+0.20pp over fixed R72 and +74 over R60B. Nine tasks improve, seven regress and two tie; selected cost/regret improve slightly. Final TRAIN53.567222% versus VAL49.622222%, worsening validation CE and falling pseudo-target loss retain a generalization concern without proving teacher-error causality, noise or a ceiling. Teacher caching, complete warm start, fresh optimizer, target isolation, recovery and integer selection are supported. Adaptive acquisition remains a distinct but unready research direction; partial supervision retains ambiguity and compatibility limits. No successor is adopted or launched.

**NEW R73 TEST, locks, predictions and prohibited delivery outputs were not read.** Historical TEST exposure and the previously disclosed early VAL prefix remain acknowledged; conclusions use the complete permitted TRAIN/VAL records. No execution, replay, edits, Git or delegation occurred. R58 remains deferred. Goal acceptance still requires an official locked original18 TEST with **at least12600/18000 correct**.