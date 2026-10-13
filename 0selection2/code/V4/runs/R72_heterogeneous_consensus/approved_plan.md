Approved by actual GPT-6-Astra/max from completed R71 TRAIN/validation only.
Primary actual CLI0:2026-10-13T02:28:13UTC. Plan8/10 ready, PLAN ONLY before parent adoption.
NEW R71TEST excluded; historical TEST exposure disclosed. Full raw source follows.

**No actionable TRAIN/validation implementation defect found. Integrity: 9/10, ready within this scope. Original 70% performance: 4/10, not ready.** This does not review or approve R71’s TEST delivery.

`review_independence=same-family; acceptance_status=provisional`

The saved [terminal receipt](/public/home/shiys/0selection2/code/V4/runs/R71_joint_performance_mixture/training_terminal_receipt.json) records original Slurm1465.105 `COMPLETED/0:0`, 4,076 seconds, and `r71_train` absent. The complete history contains all40 validation epochs and uniquely selects epoch31. **Validation retains fixed R70B23 as official A.**

The inspected [model](/public/home/shiys/0selection2/code/V4/r71_model.py:24), [gradient path](/public/home/shiys/0selection2/code/V4/r71_gradients.py:12) and [training/evaluation path](/public/home/shiys/0selection2/code/V4/r71_experiment.py:286) implement the approved TRAIN-only normalization, canonical candidate order, installed mixture likelihood, two-view NLL/D, fixed Sobol integration, restored cost units, target isolation and integer checkpoint selection. Whole-update retries and inactive-parameter handling are supported. Zero initial trunk gradients follow correctly from the zero final-layer weights. No repair or replay is warranted.

| Record | TRAIN Top1 | VAL Correct / Top1 | VAL Native CE | VAL Density NLL/D |
|---|---:|---:|---:|---:|
| Reused A23 | 50.891111% | 8709 /48.383333% | 1.121139 | Not applicable |
| New B31, selected | 46.063333% | **8371 /46.505556%** | 1.175962 | −0.128058 |
| A40, final | 58.138889% | 8158 /45.322222% | 1.232391 | Not applicable |
| B40, final | 45.993333% | 8247 /45.816667% | 1.163053 | −0.165196 |

Sources: original [A result](/public/home/shiys/0selection2/code/V4/runs/R70_latent_mixup/B_latent_mixup_seed2/result.json) and [B result](/public/home/shiys/0selection2/code/V4/runs/R71_joint_performance_mixture/B_joint_mixture_seed2/result.json). **B loses338 decisions, −1.877778pp against A, and514 decisions, −2.855556pp against R60B.** Selected validation regret also worsens:1.024641% versus0.988830%; regret remains disclosure, not the target.

| Selected Family | A23 Top1 | B31 Top1 | B−A |
|---|---:|---:|---:|
| TSP | 36.300% | 36.000% | −0.300pp |
| CVRP | 49.000% | 44.100% | −4.900pp |
| ATSP | 78.000% | 78.400% | +0.400pp |
| MVRP | 47.173% | 45.240% | −1.933pp |

Only **ATSP+0.4pp, OVRPTW+1.6pp and VRPBLTW+0.2pp** improve; the other15 tasks regress. The largest losses are OVRP−5.3pp, VRPBL−5.2pp and CVRP−4.9pp.

R71 did learn winner information: validation accuracy rises from29.188889% at epoch1 to46.505556% at its selected epoch. Nevertheless, improved density fitting did not produce stronger native discrimination than the existing policies. Between selected31 and final40, TRAIN density NLL/D improves−0.160860→−0.216729 and validation improves−0.128058→−0.165196, while native accuracy declines on both splits. Negative continuous-density NLL is legitimate; it is neither classification CE nor proof of accurate point-cost predictions or calibration.

The final TRAIN/validation accuracy gap is only **0.176667pp**. R69’s large classification generalization gap is absent here. B’s last-five validation mean is46.011111%, below A’s46.292222%, although B has lower native CE. Its final89-decision advantage over A does not overturn the selected-checkpoint loss. These observations reject this bounded density-learning package’s selection benefit; they do not identify a universal bottleneck, label noise or an accuracy ceiling.

New B completed **9,600 updates/7.2M selected rows**,14.4M successful versus14,402,560 attempted encoder-view rows, four retries and **4,019.743262s** arm wall time. Peak allocation was15.748332GiB. All18 branches received1,600 active updates/400,000 rows. The1.389960s CPU check and5.965898s discarded capacity operation are separately recorded, with formal state recreated afterward. Historical A and inherited pretraining remain separate expenditure. B’s38,231,040 versus A’s33,173,154 parameters make this an objective/readout/inference package comparison, not an isolated covariance experiment.

**Recommend R72: one fixed heterogeneous full18 consensus experiment. Plan: 8/10, ready, PLAN ONLY.** Parent adoption and execution remain subsequent actions.

The major observed failure is that successive single-policy learning changes have not surpassed R60B. R39/R43/R55 and nowR71 constrain another cost-supervision proposal. R45/R47/R50/R51/R53/R54/R56 constrain another embedding, retrieval, trajectory-summary or pair-specialist intervention. R64–65 produced limited optimization gains; R67 andR70 modest matched gains; R68–69 negative comparisons. This supports a bounded test of combining already learned decisions before funding another training mechanism.

The directly relevant primary source is [Tornede et al., §§5.1/Eq.14,6.3 and6.6](https://link.springer.com/article/10.1007/s10994-022-06161-4): Borda aggregates complete candidate rankings, and heterogeneous voting improved aggregate ASlib penalized-runtime performance. Its target differs from native Top1; stacking results were weaker. [Lakshminarayanan et al., §2.4 and§3.4](https://papers.nips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) support neural prediction aggregation, but use probability averaging. The proposed ordinal combination follows Tornede, not an exact deep-ensemble reproduction.

**R66 remains material counterevidence.** It compared native CE/RDrop on standalone`g` against composite`F+g`, with both deploying`F+g`; it was not native CE versus regret routing. Its independent sum gained only eight validation decisions over its base. R72 instead combines three complete policies from different training mechanisms with fixed ordinal aggregation. Their task summaries do not establish complementary instance errors, and R67/R70 share pretraining. Consensus can dilute the strongest member.

1. **Frozen components and control.** B contains exactly **R60B39, R67B36/global76 and R70B23/global63**, on every query. Use their existing complete saved checkpoints: R60 andR67 `best.pt`, and R70 `best_checkpoints/epoch023.pt`; all three files exist. Reconstruct their original architectures from saved arguments and strictly load complete states. R67 uses saved averaged-`x` weights without an optimizer transition or projector; R70 retains its complete18-branch common-epoch bank. Every parameter and normalization buffer is frozen. A is unchanged R60B39, **8885/18000 validation correct**. Preserve its historical cost-selected checkpoint; no reselection.

2. **Inputs and supervision boundary.** Use original full18 inputs, legal pools and existing preprocessing. Preserve each member’s TRAIN-fitted normalization, directed ATSP information, geometry and constraint fields. Known problem identity selects R70’s branch. Costs, winners, routes and auxiliary targets never enter member forwards or aggregation. No transformations, additional runtime fields, new labels or solver calls.

3. **One exact inference rule.** Preserve each member’s original returned `logits`: R60’s score-difference/maximin output and R67/R70’s direct utilities. Align legal columns by ascending global solver ID and exclude padding. For member`m` and candidate`j`, define doubled midrank `q_mj = 2*G_mj + E_mj + 1`, where`G` counts strictly higher legal scores and`E` counts equal legal scores, including itself. Sum `Q_j = q_1j+q_2j+q_3j` in integer arithmetic. Select minimum`Q`; final ties use smallest global ID. This retains every legal candidate and fixes exact-score ties without fitted weights or temperatures.

4. **Objective and numerics.** There is no optimized loss, calibration, optimizer or trainable aggregation. Run sequential FP32 evaluation with dropout disabled and no gradients. For reporting CE only, use the fixed score `z_j=−Q_j/6` and stable legal-pool log-softmax; label this **Borda-score CE**, not calibrated probability quality. Density NLL is inapplicable. Nonfinite legal scores or inconsistent candidate sets stop the evaluation and preserve the failure; no member dropping or alternative aggregation.

5. **Exact new budget.** **Zero training epochs, zero optimizer updates and zero TRAIN forward rows.** Run one complete18,000-row validation evaluation of B, batch128 with104-row task tails:144 query batches and432 member forwards, totaling54,000 member-row forwards. Each query executes three selectors and ultimately **one completed solver**. R60 runs only as B’s required member; A receives no separate reference evaluation. Reuse saved component metrics. Record ordinary evaluation wall time including all three forwards and aggregation; report historical training separately, counting shared inherited pretraining once.

6. **Selection and TEST.** Record the fixed membership, checkpoints, preprocessing and rule before validation. There is one candidate and no epoch, subset, weighting or task-wise selection. Official B requires **more than8885 correct**; otherwise official A remains. After the parent freezes the complete bundle and official arm, execute **one original full18 TEST for B regardless of validation improvement**, using all three live members. Reuse A’s historical TEST unchanged. This adds another54,000 member-row forwards, with no standalone component reruns or cached-TEST shortcut.

7. **Minimum check and stopping rule.** One synthetic CPU check covers column permutations, legal masks, exact-score midranks, final global-ID ties and exclusion of targets from the forward interface. Then proceed directly to the formal validation pass on the designated3090, without a GPU pilot or reference replay. If B does not beat8885, reject this fixed bundle’s selection benefit; do not search subsets or aggregation coefficients afterward. Complete its predeclared frozen TEST/reporting once. Any gain supports this particular bundle, not proven error independence or a70% forecast.

**Memory update:** R71 completed40 epochs with no actionable inspected TRAIN/validation defect. B31 achieved8371/18000,46.505556%, losing338 decisions against fixed R70B23 and514 against R60B; validation retains A. Three tasks improve and15 regress. Density likelihood improves into late training while native accuracy remains weak; final TRAIN45.993333% and validation45.816667% do not reproduce R69’s large classification gap. New expenditure is9,600 updates/7.2M rows/4,019.743262s, four retries, with unequal parameter capacity and separately discarded capacity work. Recommend one fixed R60B39/R67B36/R70B23 Borda-consensus experiment with zero training and reused R60B39 control. R66’s weak ensemble result and unestablished instance complementarity remain explicit counterevidence. No successor was launched.

**NEW R71 TEST and all prohibited delivery artifacts were excluded throughout.** Historical TEST exposure remains disclosed. No model execution, prediction replay, GPU/job query, edits, Git or subordinate agents occurred. R58 and legacy B discrepancies remain deferred. Goal success still requires an actual locked original18 TEST with **at least12600/18000 correct**; integrity and plan readiness do not complete that goal.
