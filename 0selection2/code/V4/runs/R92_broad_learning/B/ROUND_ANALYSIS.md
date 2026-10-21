# R92B | Completed Round Analysis

2026-10-22 | B only | seed 2 | existing completed results, no rerun

**Verdict:** R92B does not improve on either historical reference. Deterioration is widespread but dominated by six tasks; fixed-target recovery remains negligible. The measured TRAIN/VAL gaps do not establish overfitting.

**Locked scope.** Learned 32/8 hierarchy, link 0.1 + entropy 0.01, common full18 R77 main objective. All 57,600 updates completed; the unique checkpoint is update 51,840, selected strictly by minimum full-VAL macro vs_SBS, first tie wins. Its SHA256 was checked once against the lock and matches: `1573447afc314a77c511a73dc14bc612ff5648b409851877e2255d2cb40369b9`. TEST/LIB and original PDFs are complete. No inference rule or checkpoint was reselected.

## 1. System Results

| Model | Split | Top1 (%) | mean_cost | Regret (%) | vs_SBS (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| R84A | VAL | 71.085185 | 11.416611477 | 0.403494 | -0.194190 |
| R89B | VAL | 71.248148 | 11.416294284 | 0.400676 | -0.197193 |
| R92B | VAL | 69.558025 | 11.419872283 | 0.437580 | -0.159682 |
| R84A | TEST | 71.235802 | 11.373843044 | 0.394847 | -0.197053 |
| R89B | TEST | 71.419753 | 11.373470452 | 0.389646 | -0.201446 |
| R92B | TEST | 69.787654 | 11.376775960 | 0.429720 | -0.162234 |

R92B fails the existing VAL empirical screen against both references. TEST Top1 changes are -1.448148 pp vs R84A and -1.632099 pp vs R89B; regret increases by +0.034873 and +0.040074 pp. Against R89B, 2,965 instances are corrected but 4,287 harmed: net -1,322 correct.

| R92B split | TRAIN-fitted SBS cost | Oracle cost | vs_Oracle (%) |
| --- | ---: | ---: | ---: |
| VAL | 11.427978473 | 11.371691902 | 0.428351 |
| TEST | 11.385299246 | 11.329932369 | 0.421391 |

All three models use the same NEW90k (20261018) data, 81,000/4,500/4,500 per task, and frozen legal pools. Labels are native int64 raw `ind`, verified against FP64 argmin, never FP32 argmin. ALL averages 18 task metrics equally; cross-task raw mean_cost is descriptive only. Per task, vs_SBS and vs_Oracle are ratios of mean costs, while actual regret is `100 * mean((selected_cost - oracle_cost) / oracle_cost)`. Oracle is the per-instance legal-pool minimum; SBS is fitted on full TRAIN, not post-hoc held-out data.

## 2. Where Deterioration Sits

| TEST task | R92B Top1 (%) | Delta vs R84A (pp) | Delta vs R89B (pp) | Regret delta vs R89B (pp) |
| --- | ---: | ---: | ---: | ---: |
| ATSP | 68.911111 | -7.800000 | -8.000000 | +0.191508 |
| OVRPL | 35.600000 | -6.244444 | -6.022222 | +0.155348 |
| OVRP | 36.400000 | -4.311111 | -4.088889 | +0.095827 |
| VRPL | 34.244444 | -2.911111 | -4.066667 | +0.057764 |
| CVRP | 45.066667 | -2.044444 | -3.577778 | +0.095212 |
| TSP | 44.911111 | -1.488889 | -2.200000 | +0.002596 |

TEST Top1 is lower on 16/18 tasks against each reference. Cost and regret worsen on 17/18 tasks vs R84A and 16/18 vs R89B. The six tasks above account for 95.14%/95.16% of net Top1 loss and 82.82%/82.94% of net regret deterioration vs R84A/R89B. Thus severity is concentrated, not the existence of regressions. Against R89B, the only Top1 gains are tiny: VRPBL +0.044444 pp and OVRPBL +0.022222 pp; VRPBL still has worse cost/regret. VAL Top1 is lower on 15/18 tasks vs R84A and 17/18 vs R89B (one tie).

<!-- PAGE 2 -->

## 3. TRAIN Versus VAL, Selected and Last

| Update / state | Split | Top1 (%) | Eval CE | Regret (%) | vs_SBS (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| 51,840 / selected | TRAIN | 69.757750 | 0.772185 | 0.428064 | -0.164684 |
| 51,840 / selected | VAL | 69.558025 | 0.778813 | 0.437580 | -0.159682 |
| 57,600 / last | TRAIN | 69.730316 | 0.772405 | 0.428972 | -0.163829 |
| 57,600 / last | VAL | 69.533333 | 0.778930 | 0.438046 | -0.159142 |

These are full-dataset eval-mode metrics, not minibatch losses. TRAIN minus VAL Top1 gaps are +0.199726/+0.196982 pp; VAL minus TRAIN CE gaps are 0.006627/0.006525 (selected/last). Both splits finish nearly flat and slightly worse. Selected TRAIN also trails R84A/R89B by 1.486145/1.933402 pp Top1, with worse cost/regret on all 18 tasks. This does not support large TRAIN/VAL divergence or identify the cause of weaker fit.

## 4. Genuine Fixed-36 Recovery

The target list is frozen from **R84A's locked VAL**: selected frequency <=0.1% and native winner count >0; it is not redefined from R92B. Across 4,598 TEST target winners, R92B gets only **32 correct (0.695955% recall)**. Its 71 target picks contain 39 false picks (45.070423% precision). Only 5/36 groups have any correct target picks; 31 have none, and 28 remain entirely unselected.

| Reference | Old correct | Rescued | Lost | Net | New false | False pp | Winner delta pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R84A | 13 | 29 | 10 | +19 | 38 | +0.001901 | +0.000646 |
| R89B | 45 | 27 | 40 | -13 | 35 | +0.001827 | +0.003684 |

"False pp" is R92B's absolute regret contribution on newly introduced false target picks, not the change from the old pick. "Winner delta pp" is paired regret change on all true fixed-target winners. Both use whole-task denominators and equal 18-task weighting, not 4,598-only normalization; the subsets may overlap. Fixed-target winner regret worsens against both references.

ATSP/MATPOENET supplies 26 of the 27 rescues vs R89B: 26/370 winners recovered, but 56 picks include 30 false picks. ATSP overall still loses 8.0 pp Top1 and adds 0.191508 pp regret. TRAIN/VAL fixed-target recall is also only 0.868166%/0.864865%. A few zero-group rescues are therefore not broad recovery or system success.

## 5. LIB and Interpretation Boundary

| Dataset | NSS Top1 (%) | R92B Top1 (%) | NSS mean_cost | R92B mean_cost |
| --- | ---: | ---: | ---: | ---: |
| TSP TEST (4,500) | 45.888889 | 44.911111 | 7.970400719 | 7.971097341 |
| CVRP TEST (4,500) | 45.977778 | 45.066667 | 18.004802368 | 18.025317620 |
| TSPLIB (49) | 20.408163 | 22.448980 | 8.217907074 | 8.206105738 |
| CVRPLIB (100) | 70.000000 | 49.000000 | 67.101111717 | 67.492862062 |

NSS reuses historical R79 weights, not matched NEW90k training. TEST uses NEW4,500 per task; LIB uses original native datasets, with the same TSP/CVRP legal pools and the same single R92B lock. R92B LIB regret is 0.779640%/0.767800% and vs_SBS is -0.416599%/+0.368052% for TSPLIB/CVRPLIB. LIB evidence is mixed, not a generalization-success claim.

**Conclusion:** retain R92B as a completed nonpassing result, not a successful recovery model. R84A/R89B are same-data historical references, not matched-architecture or equal-compute causal controls; these observations cannot isolate hierarchy, link, or entropy effects. A is still training, so no R92 A/B conclusion is available. TEST has informed earlier R&D and is not an untouched blind test. No new experiments, TEST-based selection, or prediction reruns were performed.

**Evidence and CSV.** Existing [metrics](/public/home/shiys/0selection2/code/V4/runs/R92_broad_learning/B/full_model_statistics/metrics_by_problem.csv), [paired R84A](/public/home/shiys/0selection2/code/V4/runs/R92_broad_learning/B/full_model_statistics/paired_changes_vs_R84A.csv), [paired R89B](/public/home/shiys/0selection2/code/V4/runs/R92_broad_learning/B/full_model_statistics/paired_changes_vs_R89B.csv), target-recovery/VAL-screen/NSS CSVs, history/final-metrics/lock JSONs, and historical final metrics. Companion CSV: 59 locked TRAIN/VAL/TEST/LIB rows and two last-history ALL rows. Blanks mean unavailable/not applicable. Last-history rows are diagnostics only, never alternative TEST models.
