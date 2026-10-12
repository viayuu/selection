# R63: Native-Winner Classification With Recurrent Node Workspaces

Seed2; original18 tasks, all legal candidates and original native winners.
Training, checkpoint selection and the official arm are controlled by validation only.
Completed locked TEST is reported separately; it never reselects a checkpoint or arm.
No auxiliary data, solver execution, new annotation or R58 operations.

## Validation Top1-selected outcomes

| Model | Problem | N | Top1 % | Top2 % | Top3 % | Mean cost | vs SBS % | Regret % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R45A historical | ALL | 18000 | 49.1167 | 82.4444 | 92.7611 | 11.168517 | -1.0700 | 0.926048 |
| R60B incumbent | ALL | 18000 | 49.3611 | 82.2389 | 92.7222 | 11.168642 | -1.0735 | 0.922789 |
| R63A | ALL | 18000 | 48.4944 | 81.7556 | 92.2667 | 11.175508 | -1.0229 | 0.976334 |
| R63B | ALL | 18000 | 47.3444 | 81.0889 | 91.8222 | 11.177965 | -1.0032 | 0.995003 |

## Checkpoint identities

| Model | Checkpoint | Epoch | SHA256 | Lock status |
|---|---|---:|---|---|
| R45A historical | `/public/home/shiys/0selection2/code/V4/runs/R45_solver_code_embeddings/A_handcrafted_seed2/best.pt` | 38 | d1d435ef8fe15331c151b287d5bb2deedd498d669747dcb3f0aa1532b097848c | final TEST locked |
| R60B incumbent | `/public/home/shiys/0selection2/code/V4/runs/R60_expanded_partial_supervision/B_expanded_seed2/best.pt` | 39 | 127f15ca5aa585a6006f72beabf0a1ad1eb948f444e62e4c032f173a18debe25 | final TEST locked |
| R63A | `/public/home/shiys/0selection2/code/V4/runs/R63_recurrent_workspaces/A_static_memory_seed2/best.pt` | 34 | 95827a6fd15edb4d54282961af4fb0885d4c4168ecbdd2b8f3d35eb8bfd0e95b | final TEST locked |
| R63B | `/public/home/shiys/0selection2/code/V4/runs/R63_recurrent_workspaces/B_communicating_memory_seed2/best.pt` | 12 | 91d47ac4b84a62af71ba8a2c933c17809cd6bb8f34b81ff0de608257260520d0 | final TEST locked |

Official R63 arm: **R63A**, chosen by validation native Top1; ties A.
Each R63 arm retains its own unique validation-best checkpoint within its actual completed budget.
R60B remains its existing validation-cost-best checkpoint (epoch39); it is not reselected.
R45A uses its historical official TEST and original checkpoint lock; no rerun.

Numerical tie caveat: A epochs32 and34 both selected8729/18000 native winners
correctly. Floating-point macro averaging produced0.4849444444444444 versus
0.4849444444444445, so the original strict-float comparison selected34 rather
than honoring the intended earliest-on-exact-count-tie rule. A34 was locked
before TEST and remains the reported checkpoint; no epoch32 TEST or reselection
is performed. R64 will fix this prospectively using integer correct counts.
This affects the tie rule, not the reported R63 Top1 or its negative outcome.

## Locked final TEST

Final TEST receipt status: **complete**.
Both training arms must finish before the parent freezes final_test_lock.json and authorizes TEST.
Exactly one complete original18 TEST per R63A/R63B/R60B model; no TEST-based re-selection.
The analysis reads only saved FP64 predictions and cannot initiate or repeat inference.

| Model | Problem | N | Top1 % | Top2 % | Top3 % | Mean cost | vs SBS % | Regret % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R63A | ALL | 18000 | 47.3389 | 81.4111 | 92.1889 | 11.175934 | -0.9805 | 0.997570 |
| R63B | ALL | 18000 | 46.5167 | 80.6111 | 91.6000 | 11.179953 | -0.9460 | 1.033201 |
| R60B incumbent | ALL | 18000 | 48.2444 | 81.7222 | 92.5111 | 11.168404 | -1.0357 | 0.940486 |
| R45A historical | ALL | 18000 | 47.9111 | 81.7444 | 92.4333 | 11.168366 | -1.0320 | 0.943221 |

Official R63A TEST native Top1: 47.3389%.
Official locked TEST70% target hit: **False**.
Official minus R60B TEST Top1: -0.9056pp.
R63B minus R63A TEST Top1: -0.8222pp.
R45A reference: reused original official full18 TEST; no rerun.

The fixed R60B reference TEST is the explicit addition; other R59--R62 no-test histories remain unchanged.
Costs/regret are disclosed outcomes, never a veto on native Top1 or a basis for TEST re-selection.

## Validation families

| Model | Family | N | Top1 % | Top2 % | Top3 % | CE | Mean cost | vs SBS % | Regret % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R45A historical | ALL | 18000 | 49.1167 | 82.4444 | 92.7611 | 1.089768 | 11.168517 | -1.0700 | 0.926048 |
| R45A historical | TSP | 1000 | 38.5000 | 61.1000 | 75.3000 | 1.630474 | 7.914080 | -0.5494 | 0.561555 |
| R45A historical | CVRP | 1000 | 47.9000 | 69.2000 | 83.3000 | 1.338356 | 18.202897 | -0.4998 | 0.676407 |
| R45A historical | ATSP | 1000 | 78.5000 | 96.8000 | 99.3000 | 0.589599 | 1.634751 | -2.3169 | 0.313291 |
| R45A historical | MVRP | 15000 | 47.9467 | 83.7933 | 94.1200 | 1.070494 | 11.552105 | -1.0596 | 1.007841 |
| R60B incumbent | ALL | 18000 | 49.3611 | 82.2389 | 92.7222 | 1.092423 | 11.168642 | -1.0735 | 0.922789 |
| R60B incumbent | TSP | 1000 | 37.6000 | 60.2000 | 75.8000 | 1.633501 | 7.918297 | -0.4964 | 0.624154 |
| R60B incumbent | CVRP | 1000 | 49.3000 | 69.3000 | 83.6000 | 1.334916 | 18.199659 | -0.5175 | 0.654440 |
| R60B incumbent | ATSP | 1000 | 78.4000 | 96.8000 | 99.3000 | 0.590395 | 1.634628 | -2.3242 | 0.306340 |
| R60B incumbent | MVRP | 15000 | 48.2133 | 83.6000 | 94.0200 | 1.073653 | 11.552199 | -1.0657 | 1.001685 |
| R63A | ALL | 18000 | 48.4944 | 81.7556 | 92.2667 | 1.101019 | 11.175508 | -1.0229 | 0.976334 |
| R63A | TSP | 1000 | 38.5000 | 60.8000 | 74.8000 | 1.645675 | 7.914117 | -0.5490 | 0.565057 |
| R63A | CVRP | 1000 | 48.1000 | 69.0000 | 80.9000 | 1.364814 | 18.211336 | -0.4537 | 0.731023 |
| R63A | ATSP | 1000 | 78.2000 | 97.1000 | 99.3000 | 0.587307 | 1.635014 | -2.3012 | 0.328658 |
| R63A | MVRP | 15000 | 47.2067 | 82.9800 | 93.7200 | 1.081370 | 11.559912 | -1.0073 | 1.063285 |
| R63B | ALL | 18000 | 47.3444 | 81.0889 | 91.8222 | 1.131424 | 11.177965 | -1.0032 | 0.995003 |
| R63B | TSP | 1000 | 37.2000 | 60.8000 | 74.7000 | 1.750623 | 7.918382 | -0.4954 | 0.631103 |
| R63B | CVRP | 1000 | 44.0000 | 63.0000 | 78.4000 | 1.490614 | 18.227257 | -0.3667 | 0.797159 |
| R63B | ATSP | 1000 | 77.5000 | 96.9000 | 99.4000 | 0.602043 | 1.635503 | -2.2719 | 0.360061 |
| R63B | MVRP | 15000 | 46.2333 | 82.5933 | 93.3533 | 1.101491 | 11.561482 | -0.9949 | 1.074782 |

## TEST families

| Model | Family | N | Top1 % | Top2 % | Top3 % | CE | Mean cost | vs SBS % | Regret % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R63A | ALL | 18000 | 47.3389 | 81.4111 | 92.1889 | 1.106735 | 11.175934 | -0.9805 | 0.997570 |
| R63A | TSP | 1000 | 36.0000 | 61.7000 | 75.4000 | 1.603456 | 8.045447 | -0.4630 | 0.657324 |
| R63A | CVRP | 1000 | 45.5000 | 68.1000 | 80.9000 | 1.408035 | 18.262847 | -0.3790 | 0.729315 |
| R63A | ATSP | 1000 | 76.3000 | 96.4000 | 99.5000 | 0.602998 | 1.630554 | -2.3433 | 0.366208 |
| R63A | MVRP | 15000 | 46.2867 | 82.6133 | 93.5733 | 1.087116 | 11.548531 | -0.9642 | 1.080228 |
| R63B | ALL | 18000 | 46.5167 | 80.6111 | 91.6000 | 1.137600 | 11.179953 | -0.9460 | 1.033201 |
| R63B | TSP | 1000 | 35.3000 | 59.7000 | 73.5000 | 1.678036 | 8.046383 | -0.4514 | 0.680085 |
| R63B | CVRP | 1000 | 41.7000 | 64.1000 | 78.1000 | 1.514141 | 18.283094 | -0.2686 | 0.791234 |
| R63B | ATSP | 1000 | 75.5000 | 96.2000 | 99.5000 | 0.661888 | 1.632362 | -2.2350 | 0.488364 |
| R63B | MVRP | 15000 | 45.6533 | 82.0667 | 93.1800 | 1.108182 | 11.551821 | -0.9382 | 1.109196 |
| R60B incumbent | ALL | 18000 | 48.2444 | 81.7222 | 92.5111 | 1.102036 | 11.168404 | -1.0357 | 0.940486 |
| R60B incumbent | TSP | 1000 | 36.7000 | 61.7000 | 74.6000 | 1.622206 | 8.045406 | -0.4635 | 0.658364 |
| R60B incumbent | CVRP | 1000 | 48.3000 | 69.6000 | 82.4000 | 1.382379 | 18.249144 | -0.4538 | 0.622201 |
| R60B incumbent | ATSP | 1000 | 77.6000 | 96.3000 | 99.5000 | 0.611708 | 1.629807 | -2.3881 | 0.322834 |
| R60B incumbent | MVRP | 15000 | 47.0533 | 82.8933 | 93.9133 | 1.081357 | 11.540461 | -1.0224 | 1.021690 |
| R45A historical | ALL | 18000 | 47.9111 | 81.7444 | 92.4333 | 1.101119 | 11.168366 | -1.0320 | 0.943221 |
| R45A historical | TSP | 1000 | 35.9000 | 61.7000 | 75.2000 | 1.605020 | 8.044507 | -0.4746 | 0.650396 |
| R45A historical | CVRP | 1000 | 48.1000 | 69.2000 | 82.2000 | 1.388907 | 18.247945 | -0.4603 | 0.613832 |
| R45A historical | ATSP | 1000 | 77.5000 | 96.2000 | 99.5000 | 0.609865 | 1.629876 | -2.3839 | 0.327377 |
| R45A historical | MVRP | 15000 | 46.7267 | 82.9533 | 93.7933 | 1.081090 | 11.540551 | -1.0171 | 1.025758 |

Each candidate has its own persistent node workspace and eight tied recurrent rounds.
A reads initial neighbor states; B reads previous-round neighbor states. Both update their own states.
Graphs use input distances/direction, valid-node masks, depot and special-token connections only.
Both train native CE plus symmetric R-Drop and select strict maximum full18 native Top1.
No maximin, comparison/risk loss, route output or independent binary expert is used.

## Actual training budgets

| Arm | Best epoch | Stop epoch | Successful updates | Run hours |
|---|---:|---:|---:|---:|
| R63A | 34 | 40 | 56880 | 9.7785 |
| R63B | 12 | 15 | 21330 | 3.6523 |

User amendment: A completed40 epochs; B stopped after15 complete epochs and full validation.
Only the first15 epochs share batches and the A-controlled LR trace; A40/B15 is not a matched-compute contrast.
Effective batch128 uses fixed microbatch16 with sample-weighted accumulation.
The two dropout forwards and all microbatches produce one optimizer update per original batch.
Architecture parameters, initialization, common-prefix batches, graph construction and recurrence count are matched.

## Common-prefix validation (descriptive only)

| Arm | Prefix epochs | Last-prefix Top1 % | Best-prefix epoch | Best-prefix Top1 % |
|---|---:|---:|---:|---:|
| R63A | 15 | 46.4667 | 14 | 47.1889 |
| R63B | 15 | 46.9056 | 12 | 47.3444 |

These existing history values do not reselect, replace or evaluate any checkpoint.

## Selected-checkpoint train and validation

| Arm | Split | Top1 % | CE | Mean cost | Actual regret % |
|---|---|---:|---:|---:|---:|
| R63A | train | 50.4822 | 1.059193 | 11.175543 | 0.924336 |
| R63A | val | 48.4944 | 1.101019 | 11.175508 | 0.976334 |
| R63B | train | 47.4606 | 1.122022 | 11.185829 | 1.003570 |
| R63B | val | 47.3444 | 1.131424 | 11.177965 | 0.995003 |

Training evaluation uses eval mode and complete original TRAIN, not online batch accuracy.

## Decision and limits

Best selected arm by native Top1: R63A.
Validation70% target hit: **False**; validation alone does not establish the target.
Best minus incumbent Top1: -0.8667pp.
B minus A Top1: -1.1500pp.
Contrast scope: descriptive unequal-budget A40/B15 contrast; not a matched-horizon mechanism claim.
Best minus incumbent regret: +0.053545pp.

Costs/regret are disclosed, not an added veto on the user native-Top1 target.
The+3pp progress and B-A+2pp mechanism milestones do not replace70%.
Either arm may satisfy the validation target; unequal A40/B15 results do not establish a matched-budget communication advantage.
The official arm is frozen by validation before TEST; a better contrast-arm TEST does not change it.
The70% goal is not declared achieved by validation alone; official locked TEST target status is separate.
Comparison with historical cost-trained models changes objective/readout as well as computation.
This single-seed experiment with disclosed actual budgets cannot establish a universal accuracy ceiling.
Legacy B semantic mismatches remain unresolved; this is not a repaired common-task benchmark.

## Selection latency

Per-arm inference_timing.json records normal final-validation timing, including graph construction.
Inputs and geometry are already cached. Disk decoding, cache preparation and checkpoint loading are excluded.
Each original query is evaluated once: first16 per task at batch1, then the remainder batched.
Single-query times include gather, graph construction, selector forward and output transfer.
ALL single-query quantiles pool the same number of observations per task, not per scale.
No solver is run for selection; selector time is not full solver or end-to-end solve time.

| Arm | First query median ms | Later batch1 median / p95 ms | Batched queries/s | Graph seconds | Forward seconds |
|---|---:|---:|---:|---:|---:|
| R63A | 9.618 | 9.387 / 9.958 | 2146.11 | 0.675 | 10.211 |
| R63B | 9.660 | 9.351 / 9.994 | 2166.86 | 0.671 | 10.129 |

Full per-task timing is saved in inference_timing.csv; no standalone timing pass was added.

## Decision changes

| Split | Before | After | Category | N | Macro cost delta | Macro regret delta pp |
|---|---|---|---|---:|---:|---:|
| val | R45A | R63A | corrected | 1552 | -0.01965977 | -0.170811 |
| val | R45A | R63A | harmed | 1664 | +0.02376970 | +0.202941 |
| val | R45A | R63A | both_wrong_cheaper | 419 | -0.00432030 | -0.030835 |
| val | R45A | R63A | both_wrong_dearer | 500 | +0.00720159 | +0.048991 |
| val | R45A | R63A | both_wrong_equal | 2 | +0.00000000 | +0.000000 |
| val | R45A | R63A | unchanged | 13863 | +0.00000000 | +0.000000 |
| val | R45A | R63A | all | 18000 | +0.00699122 | +0.050285 |
| val | R45A | R63B | corrected | 1741 | -0.02286293 | -0.187970 |
| val | R45A | R63B | harmed | 2060 | +0.02877318 | +0.236388 |
| val | R45A | R63B | both_wrong_cheaper | 540 | -0.00468350 | -0.035546 |
| val | R45A | R63B | both_wrong_dearer | 650 | +0.00822208 | +0.056083 |
| val | R45A | R63B | both_wrong_equal | 0 | +0.00000000 | +0.000000 |
| val | R45A | R63B | unchanged | 13009 | +0.00000000 | +0.000000 |
| val | R45A | R63B | all | 18000 | +0.00944883 | +0.068954 |
| val | R60B | R63A | corrected | 1560 | -0.01998180 | -0.171185 |
| val | R60B | R63A | harmed | 1716 | +0.02444080 | +0.210718 |
| val | R60B | R63A | both_wrong_cheaper | 440 | -0.00449829 | -0.033386 |
| val | R60B | R63A | both_wrong_dearer | 497 | +0.00690477 | +0.047398 |
| val | R60B | R63A | both_wrong_equal | 2 | +0.00000000 | +0.000000 |
| val | R60B | R63A | unchanged | 13785 | +0.00000000 | +0.000000 |
| val | R60B | R63A | all | 18000 | +0.00686549 | +0.053545 |
| val | R60B | R63B | corrected | 1900 | -0.02422579 | -0.201952 |
| val | R60B | R63B | harmed | 2263 | +0.03096576 | +0.260407 |
| val | R60B | R63B | both_wrong_cheaper | 595 | -0.00548752 | -0.042967 |
| val | R60B | R63B | both_wrong_dearer | 675 | +0.00807066 | +0.056726 |
| val | R60B | R63B | both_wrong_equal | 0 | +0.00000000 | +0.000000 |
| val | R60B | R63B | unchanged | 12567 | +0.00000000 | +0.000000 |
| val | R60B | R63B | all | 18000 | +0.00932310 | +0.072214 |
| val | R63A | R63B | corrected | 1918 | -0.02543464 | -0.226426 |
| val | R63A | R63B | harmed | 2125 | +0.02809117 | +0.246964 |
| val | R63A | R63B | both_wrong_cheaper | 595 | -0.00729699 | -0.052282 |
| val | R63A | R63B | both_wrong_dearer | 616 | +0.00709807 | +0.050412 |
| val | R63A | R63B | both_wrong_equal | 2 | +0.00000000 | +0.000000 |
| val | R63A | R63B | unchanged | 12744 | +0.00000000 | +0.000000 |
| val | R63A | R63B | all | 18000 | +0.00245761 | +0.018669 |
| test | R45A | R63A | corrected | 1539 | -0.02031525 | -0.172719 |
| test | R45A | R63A | harmed | 1642 | +0.02451804 | +0.204203 |
| test | R45A | R63A | both_wrong_cheaper | 410 | -0.00375721 | -0.027061 |
| test | R45A | R63A | both_wrong_dearer | 507 | +0.00712263 | +0.049927 |
| test | R45A | R63A | both_wrong_equal | 0 | +0.00000000 | +0.000000 |
| test | R45A | R63A | unchanged | 13902 | +0.00000000 | +0.000000 |
| test | R45A | R63A | all | 18000 | +0.00756822 | +0.054350 |
| test | R45A | R63B | corrected | 1753 | -0.02240548 | -0.187937 |
| test | R45A | R63B | harmed | 2004 | +0.02943152 | +0.245447 |
| test | R45A | R63B | both_wrong_cheaper | 517 | -0.00507656 | -0.035049 |
| test | R45A | R63B | both_wrong_dearer | 709 | +0.00963755 | +0.067519 |
| test | R45A | R63B | both_wrong_equal | 1 | +0.00000000 | +0.000000 |
| test | R45A | R63B | unchanged | 13016 | +0.00000000 | +0.000000 |
| test | R45A | R63B | all | 18000 | +0.01158703 | +0.089980 |
| test | R60B | R63A | corrected | 1537 | -0.02009669 | -0.171035 |
| test | R60B | R63A | harmed | 1700 | +0.02515982 | +0.211237 |
| test | R60B | R63A | both_wrong_cheaper | 440 | -0.00442768 | -0.032261 |
| test | R60B | R63A | both_wrong_dearer | 509 | +0.00689440 | +0.049143 |
| test | R60B | R63A | both_wrong_equal | 0 | +0.00000000 | +0.000000 |
| test | R60B | R63A | unchanged | 13814 | +0.00000000 | +0.000000 |
| test | R60B | R63A | all | 18000 | +0.00752984 | +0.057084 |
| test | R60B | R63B | corrected | 1863 | -0.02316675 | -0.194284 |
| test | R60B | R63B | harmed | 2174 | +0.03032232 | +0.257298 |
| test | R60B | R63B | both_wrong_cheaper | 555 | -0.00504574 | -0.036591 |
| test | R60B | R63B | both_wrong_dearer | 725 | +0.00943882 | +0.066291 |
| test | R60B | R63B | both_wrong_equal | 1 | +0.00000000 | +0.000000 |
| test | R60B | R63B | unchanged | 12682 | +0.00000000 | +0.000000 |
| test | R60B | R63B | all | 18000 | +0.01154865 | +0.092715 |
| test | R63A | R63B | corrected | 1893 | -0.02542687 | -0.220813 |
| test | R63A | R63B | harmed | 2041 | +0.02861352 | +0.250519 |
| test | R63A | R63B | both_wrong_cheaper | 558 | -0.00682917 | -0.048854 |
| test | R63A | R63B | both_wrong_dearer | 643 | +0.00766134 | +0.054779 |
| test | R63A | R63B | both_wrong_equal | 1 | +0.00000000 | +0.000000 |
| test | R63A | R63B | unchanged | 12864 | +0.00000000 | +0.000000 |
| test | R63A | R63B | all | 18000 | +0.00401881 | +0.035631 |

Cost and regret deltas retain equal-problem macro weights, not raw-cost or sample-count weighting.
Positive cost/regret deltas mean the after-model is worse. Both-wrong cheaper/dearer/equal changes are included.
Per-task changes are in decision_changes.csv; TSP/CVRP/ATSP/MVRP changes are in family_decision_changes.csv.

## Validation per problem

| Model | Problem | N | Top1 % | Top2 % | Top3 % | Mean cost | vs SBS % | Regret % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R45A historical | TSP | 1000 | 38.5000 | 61.1000 | 75.3000 | 7.914080 | -0.5494 | 0.561555 |
| R45A historical | CVRP | 1000 | 47.9000 | 69.2000 | 83.3000 | 18.202897 | -0.4998 | 0.676407 |
| R45A historical | ATSP | 1000 | 78.5000 | 96.8000 | 99.3000 | 1.634751 | -2.3169 | 0.313291 |
| R45A historical | OVRP | 1000 | 38.0000 | 65.8000 | 83.9000 | 7.451840 | -0.1316 | 0.662677 |
| R45A historical | VRPB | 1000 | 55.4000 | 91.7000 | 99.6000 | 9.332689 | -0.0811 | 0.695334 |
| R45A historical | VRPL | 1000 | 33.5000 | 59.2000 | 73.6000 | 12.399742 | -0.0602 | 0.451090 |
| R45A historical | VRPTW | 1000 | 47.0000 | 76.6000 | 92.9000 | 17.885999 | -4.1329 | 1.373462 |
| R45A historical | OVRPTW | 1000 | 51.0000 | 93.3000 | 100.0000 | 10.043002 | -0.0541 | 0.858606 |
| R45A historical | OVRPB | 1000 | 58.3000 | 96.8000 | 98.4000 | 6.316885 | -0.0806 | 0.906703 |
| R45A historical | OVRPL | 1000 | 40.2000 | 69.1000 | 86.2000 | 7.432119 | -0.0732 | 0.720632 |
| R45A historical | VRPBL | 1000 | 52.0000 | 91.6000 | 98.8000 | 9.359363 | +0.0010 | 0.815433 |
| R45A historical | VRPBTW | 1000 | 44.4000 | 81.0000 | 93.0000 | 19.040335 | -3.1999 | 1.635488 |
| R45A historical | VRPLTW | 1000 | 45.7000 | 81.7000 | 94.1000 | 17.617218 | -4.8189 | 1.236411 |
| R45A historical | OVRPBL | 1000 | 58.8000 | 95.5000 | 98.5000 | 6.300912 | -0.1481 | 0.903190 |
| R45A historical | OVRPBTW | 1000 | 50.0000 | 91.3000 | 99.7000 | 10.497878 | -0.0847 | 1.159206 |
| R45A historical | OVRPLTW | 1000 | 50.8000 | 92.7000 | 99.8000 | 10.127592 | +0.0038 | 0.885000 |
| R45A historical | VRPBLTW | 1000 | 44.0000 | 78.6000 | 93.4000 | 18.943402 | -2.9528 | 1.654986 |
| R45A historical | OVRPBLTW | 1000 | 50.1000 | 92.0000 | 99.9000 | 10.532592 | -0.0814 | 1.159401 |
| R60B incumbent | TSP | 1000 | 37.6000 | 60.2000 | 75.8000 | 7.918297 | -0.4964 | 0.624154 |
| R60B incumbent | CVRP | 1000 | 49.3000 | 69.3000 | 83.6000 | 18.199659 | -0.5175 | 0.654440 |
| R60B incumbent | ATSP | 1000 | 78.4000 | 96.8000 | 99.3000 | 1.634628 | -2.3242 | 0.306340 |
| R60B incumbent | OVRP | 1000 | 36.6000 | 65.4000 | 83.6000 | 7.454570 | -0.0950 | 0.704896 |
| R60B incumbent | VRPB | 1000 | 55.2000 | 91.8000 | 99.6000 | 9.333949 | -0.0677 | 0.719457 |
| R60B incumbent | VRPL | 1000 | 34.3000 | 58.6000 | 73.2000 | 12.398925 | -0.0668 | 0.443409 |
| R60B incumbent | VRPTW | 1000 | 46.7000 | 76.9000 | 93.0000 | 17.890947 | -4.1064 | 1.394335 |
| R60B incumbent | OVRPTW | 1000 | 51.5000 | 93.3000 | 100.0000 | 10.039644 | -0.0875 | 0.825358 |
| R60B incumbent | OVRPB | 1000 | 59.9000 | 96.8000 | 98.6000 | 6.312018 | -0.1576 | 0.834979 |
| R60B incumbent | OVRPL | 1000 | 40.7000 | 68.8000 | 85.9000 | 7.429528 | -0.1080 | 0.689648 |
| R60B incumbent | VRPBL | 1000 | 52.2000 | 91.3000 | 99.0000 | 9.359368 | +0.0011 | 0.822030 |
| R60B incumbent | VRPBTW | 1000 | 42.7000 | 80.2000 | 92.5000 | 19.043829 | -3.1821 | 1.659605 |
| R60B incumbent | VRPLTW | 1000 | 45.6000 | 81.0000 | 93.9000 | 17.622698 | -4.7893 | 1.255128 |
| R60B incumbent | OVRPBL | 1000 | 59.2000 | 95.5000 | 98.4000 | 6.301526 | -0.1384 | 0.900730 |
| R60B incumbent | OVRPBTW | 1000 | 50.8000 | 91.3000 | 99.8000 | 10.497273 | -0.0905 | 1.155130 |
| R60B incumbent | OVRPLTW | 1000 | 52.6000 | 92.7000 | 100.0000 | 10.122716 | -0.0443 | 0.837164 |
| R60B incumbent | VRPBLTW | 1000 | 42.8000 | 78.5000 | 92.9000 | 18.947743 | -2.9305 | 1.669868 |
| R60B incumbent | OVRPBLTW | 1000 | 52.4000 | 91.9000 | 99.9000 | 10.528245 | -0.1227 | 1.113532 |
| R63A | TSP | 1000 | 38.5000 | 60.8000 | 74.8000 | 7.914117 | -0.5490 | 0.565057 |
| R63A | CVRP | 1000 | 48.1000 | 69.0000 | 80.9000 | 18.211336 | -0.4537 | 0.731023 |
| R63A | ATSP | 1000 | 78.2000 | 97.1000 | 99.3000 | 1.635014 | -2.3012 | 0.328658 |
| R63A | OVRP | 1000 | 35.1000 | 65.6000 | 84.0000 | 7.455268 | -0.0857 | 0.705101 |
| R63A | VRPB | 1000 | 54.0000 | 91.8000 | 99.5000 | 9.337215 | -0.0327 | 0.747606 |
| R63A | VRPL | 1000 | 31.7000 | 55.6000 | 71.5000 | 12.404257 | -0.0238 | 0.487041 |
| R63A | VRPTW | 1000 | 42.6000 | 75.0000 | 91.0000 | 17.932578 | -3.8833 | 1.619328 |
| R63A | OVRPTW | 1000 | 50.4000 | 93.3000 | 100.0000 | 10.043037 | -0.0537 | 0.861316 |
| R63A | OVRPB | 1000 | 58.6000 | 96.8000 | 98.6000 | 6.316961 | -0.0794 | 0.922678 |
| R63A | OVRPL | 1000 | 40.2000 | 68.3000 | 85.7000 | 7.432675 | -0.0657 | 0.727844 |
| R63A | VRPBL | 1000 | 51.9000 | 91.4000 | 98.6000 | 9.361362 | +0.0224 | 0.852569 |
| R63A | VRPBTW | 1000 | 44.2000 | 78.7000 | 93.0000 | 19.052605 | -3.1375 | 1.712882 |
| R63A | VRPLTW | 1000 | 46.5000 | 80.2000 | 92.8000 | 17.648511 | -4.6498 | 1.398134 |
| R63A | OVRPBL | 1000 | 56.7000 | 95.5000 | 98.6000 | 6.308916 | -0.0213 | 1.021203 |
| R63A | OVRPBTW | 1000 | 49.9000 | 91.3000 | 99.9000 | 10.501237 | -0.0527 | 1.200857 |
| R63A | OVRPLTW | 1000 | 51.7000 | 92.7000 | 100.0000 | 10.126411 | -0.0078 | 0.891938 |
| R63A | VRPBLTW | 1000 | 42.8000 | 76.5000 | 92.8000 | 18.947804 | -2.9302 | 1.671487 |
| R63A | OVRPBLTW | 1000 | 51.8000 | 92.0000 | 99.8000 | 10.529837 | -0.1076 | 1.129285 |
| R63B | TSP | 1000 | 37.2000 | 60.8000 | 74.7000 | 7.918382 | -0.4954 | 0.631103 |
| R63B | CVRP | 1000 | 44.0000 | 63.0000 | 78.4000 | 18.227257 | -0.3667 | 0.797159 |
| R63B | ATSP | 1000 | 77.5000 | 96.9000 | 99.4000 | 1.635503 | -2.2719 | 0.360061 |
| R63B | OVRP | 1000 | 31.2000 | 62.7000 | 84.1000 | 7.460637 | -0.0137 | 0.783256 |
| R63B | VRPB | 1000 | 55.3000 | 91.7000 | 99.1000 | 9.332973 | -0.0781 | 0.697738 |
| R63B | VRPL | 1000 | 30.5000 | 55.4000 | 69.6000 | 12.405875 | -0.0108 | 0.506229 |
| R63B | VRPTW | 1000 | 42.5000 | 74.1000 | 90.8000 | 17.934627 | -3.8723 | 1.648004 |
| R63B | OVRPTW | 1000 | 48.1000 | 93.3000 | 99.9000 | 10.045223 | -0.0320 | 0.891601 |
| R63B | OVRPB | 1000 | 56.0000 | 96.8000 | 98.6000 | 6.320985 | -0.0157 | 0.975914 |
| R63B | OVRPL | 1000 | 38.2000 | 68.3000 | 86.1000 | 7.435202 | -0.0317 | 0.769470 |
| R63B | VRPBL | 1000 | 55.8000 | 91.4000 | 98.2000 | 9.353789 | -0.0586 | 0.758904 |
| R63B | VRPBTW | 1000 | 45.3000 | 79.4000 | 91.7000 | 19.052212 | -3.1395 | 1.674491 |
| R63B | VRPLTW | 1000 | 45.3000 | 77.8000 | 91.9000 | 17.663253 | -4.5702 | 1.454016 |
| R63B | OVRPBL | 1000 | 53.8000 | 95.5000 | 97.9000 | 6.306193 | -0.0644 | 0.995271 |
| R63B | OVRPBTW | 1000 | 49.0000 | 91.3000 | 99.9000 | 10.500326 | -0.0614 | 1.189731 |
| R63B | OVRPLTW | 1000 | 50.5000 | 92.7000 | 100.0000 | 10.127085 | -0.0012 | 0.895663 |
| R63B | VRPBLTW | 1000 | 42.9000 | 76.5000 | 92.6000 | 18.946698 | -2.9359 | 1.658636 |
| R63B | OVRPBLTW | 1000 | 49.1000 | 92.0000 | 99.9000 | 10.537157 | -0.0381 | 1.222804 |

## TEST per problem

| Model | Problem | N | Top1 % | Top2 % | Top3 % | Mean cost | vs SBS % | Regret % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R63A | TSP | 1000 | 36.0000 | 61.7000 | 75.4000 | 8.045447 | -0.4630 | 0.657324 |
| R63A | CVRP | 1000 | 45.5000 | 68.1000 | 80.9000 | 18.262847 | -0.3790 | 0.729315 |
| R63A | ATSP | 1000 | 76.3000 | 96.4000 | 99.5000 | 1.630554 | -2.3433 | 0.366208 |
| R63A | OVRP | 1000 | 35.4000 | 63.7000 | 82.9000 | 7.466332 | -0.0425 | 0.782677 |
| R63A | VRPB | 1000 | 51.3000 | 91.8000 | 98.6000 | 9.285952 | -0.0980 | 0.753824 |
| R63A | VRPL | 1000 | 30.7000 | 52.5000 | 69.2000 | 12.357620 | -0.0249 | 0.513794 |
| R63A | VRPTW | 1000 | 44.4000 | 76.7000 | 92.0000 | 17.674816 | -4.1271 | 1.494174 |
| R63A | OVRPTW | 1000 | 50.2000 | 90.8000 | 99.9000 | 10.114580 | -0.0861 | 0.807466 |
| R63A | OVRPB | 1000 | 54.0000 | 96.4000 | 98.9000 | 6.297906 | +0.0253 | 1.041663 |
| R63A | OVRPL | 1000 | 35.5000 | 67.4000 | 85.8000 | 7.447295 | -0.0853 | 0.728198 |
| R63A | VRPBL | 1000 | 54.0000 | 93.8000 | 99.0000 | 9.251404 | +0.0241 | 0.813779 |
| R63A | VRPBTW | 1000 | 42.4000 | 77.8000 | 93.3000 | 19.009969 | -2.6443 | 1.727355 |
| R63A | VRPLTW | 1000 | 48.9000 | 77.7000 | 92.0000 | 17.621254 | -4.3719 | 1.474635 |
| R63A | OVRPBL | 1000 | 56.3000 | 97.1000 | 99.1000 | 6.304324 | -0.0396 | 1.018284 |
| R63A | OVRPBTW | 1000 | 50.0000 | 90.3000 | 99.9000 | 10.602371 | -0.0624 | 1.239953 |
| R63A | OVRPLTW | 1000 | 49.4000 | 92.1000 | 99.8000 | 10.102459 | -0.0113 | 0.871579 |
| R63A | VRPBLTW | 1000 | 42.9000 | 78.6000 | 93.2000 | 19.094719 | -2.9590 | 1.691490 |
| R63A | OVRPBLTW | 1000 | 48.9000 | 92.5000 | 100.0000 | 10.596965 | +0.0396 | 1.244545 |
| R63B | TSP | 1000 | 35.3000 | 59.7000 | 73.5000 | 8.046383 | -0.4514 | 0.680085 |
| R63B | CVRP | 1000 | 41.7000 | 64.1000 | 78.1000 | 18.283094 | -0.2686 | 0.791234 |
| R63B | ATSP | 1000 | 75.5000 | 96.2000 | 99.5000 | 1.632362 | -2.2350 | 0.488364 |
| R63B | OVRP | 1000 | 32.4000 | 64.2000 | 83.0000 | 7.473093 | +0.0480 | 0.881480 |
| R63B | VRPB | 1000 | 49.4000 | 91.8000 | 98.6000 | 9.290671 | -0.0473 | 0.796785 |
| R63B | VRPL | 1000 | 29.9000 | 51.8000 | 69.2000 | 12.358356 | -0.0189 | 0.530033 |
| R63B | VRPTW | 1000 | 42.7000 | 75.3000 | 90.1000 | 17.707288 | -3.9510 | 1.651242 |
| R63B | OVRPTW | 1000 | 49.1000 | 90.8000 | 99.9000 | 10.118477 | -0.0476 | 0.866448 |
| R63B | OVRPB | 1000 | 53.9000 | 96.4000 | 98.6000 | 6.299778 | +0.0550 | 1.077510 |
| R63B | OVRPL | 1000 | 37.2000 | 65.4000 | 85.8000 | 7.449597 | -0.0544 | 0.775750 |
| R63B | VRPBL | 1000 | 54.7000 | 93.9000 | 99.0000 | 9.249145 | -0.0004 | 0.780899 |
| R63B | VRPBTW | 1000 | 43.9000 | 76.5000 | 92.2000 | 18.998511 | -2.7030 | 1.624447 |
| R63B | VRPLTW | 1000 | 44.2000 | 75.1000 | 90.8000 | 17.626299 | -4.3445 | 1.543082 |
| R63B | OVRPBL | 1000 | 56.8000 | 97.1000 | 98.8000 | 6.304837 | -0.0315 | 1.016436 |
| R63B | OVRPBTW | 1000 | 49.3000 | 90.4000 | 99.9000 | 10.607040 | -0.0183 | 1.283730 |
| R63B | OVRPLTW | 1000 | 49.4000 | 92.1000 | 99.9000 | 10.099671 | -0.0389 | 0.860165 |
| R63B | VRPBLTW | 1000 | 43.1000 | 77.7000 | 91.9000 | 19.101015 | -2.9270 | 1.735371 |
| R63B | OVRPBLTW | 1000 | 48.8000 | 92.5000 | 100.0000 | 10.593536 | +0.0072 | 1.214556 |
| R60B incumbent | TSP | 1000 | 36.7000 | 61.7000 | 74.6000 | 8.045406 | -0.4635 | 0.658364 |
| R60B incumbent | CVRP | 1000 | 48.3000 | 69.6000 | 82.4000 | 18.249144 | -0.4538 | 0.622201 |
| R60B incumbent | ATSP | 1000 | 77.6000 | 96.3000 | 99.5000 | 1.629807 | -2.3881 | 0.322834 |
| R60B incumbent | OVRP | 1000 | 36.3000 | 64.4000 | 83.5000 | 7.463258 | -0.0837 | 0.746540 |
| R60B incumbent | VRPB | 1000 | 51.9000 | 91.6000 | 98.8000 | 9.286472 | -0.0924 | 0.760413 |
| R60B incumbent | VRPL | 1000 | 30.7000 | 53.5000 | 70.8000 | 12.354032 | -0.0539 | 0.487837 |
| R60B incumbent | VRPTW | 1000 | 43.9000 | 76.9000 | 92.5000 | 17.646568 | -4.2803 | 1.351038 |
| R60B incumbent | OVRPTW | 1000 | 49.1000 | 90.7000 | 99.9000 | 10.116840 | -0.0638 | 0.843981 |
| R60B incumbent | OVRPB | 1000 | 57.9000 | 96.4000 | 99.2000 | 6.290626 | -0.0903 | 0.959597 |
| R60B incumbent | OVRPL | 1000 | 37.7000 | 68.1000 | 85.8000 | 7.443992 | -0.1296 | 0.680099 |
| R60B incumbent | VRPBL | 1000 | 55.3000 | 93.8000 | 99.2000 | 9.247352 | -0.0198 | 0.764807 |
| R60B incumbent | VRPBTW | 1000 | 44.7000 | 78.4000 | 93.1000 | 18.981866 | -2.7883 | 1.558258 |
| R60B incumbent | VRPLTW | 1000 | 45.4000 | 77.8000 | 92.9000 | 17.607895 | -4.4444 | 1.418817 |
| R60B incumbent | OVRPBL | 1000 | 57.7000 | 97.1000 | 99.3000 | 6.299206 | -0.1208 | 0.930909 |
| R60B incumbent | OVRPBTW | 1000 | 51.4000 | 90.4000 | 100.0000 | 10.606522 | -0.0232 | 1.259368 |
| R60B incumbent | OVRPLTW | 1000 | 50.6000 | 92.1000 | 100.0000 | 10.095838 | -0.0768 | 0.806133 |
| R60B incumbent | VRPBLTW | 1000 | 44.7000 | 79.8000 | 93.8000 | 19.074475 | -3.0619 | 1.577660 |
| R60B incumbent | OVRPBLTW | 1000 | 48.5000 | 92.4000 | 99.9000 | 10.591981 | -0.0075 | 1.179891 |
| R45A historical | TSP | 1000 | 35.9000 | 61.7000 | 75.2000 | 8.044507 | -0.4746 | 0.650396 |
| R45A historical | CVRP | 1000 | 48.1000 | 69.2000 | 82.2000 | 18.247945 | -0.4603 | 0.613832 |
| R45A historical | ATSP | 1000 | 77.5000 | 96.2000 | 99.5000 | 1.629876 | -2.3839 | 0.327377 |
| R45A historical | OVRP | 1000 | 37.4000 | 64.9000 | 83.1000 | 7.461105 | -0.1125 | 0.713015 |
| R45A historical | VRPB | 1000 | 51.7000 | 91.8000 | 98.8000 | 9.287440 | -0.0820 | 0.768149 |
| R45A historical | VRPL | 1000 | 30.2000 | 52.8000 | 69.6000 | 12.354608 | -0.0492 | 0.492605 |
| R45A historical | VRPTW | 1000 | 43.5000 | 77.1000 | 92.4000 | 17.657704 | -4.2199 | 1.411912 |
| R45A historical | OVRPTW | 1000 | 49.1000 | 90.8000 | 99.9000 | 10.114022 | -0.0916 | 0.818999 |
| R45A historical | OVRPB | 1000 | 55.1000 | 96.4000 | 99.2000 | 6.295778 | -0.0085 | 1.027877 |
| R45A historical | OVRPL | 1000 | 37.0000 | 68.6000 | 85.7000 | 7.444869 | -0.1178 | 0.694297 |
| R45A historical | VRPBL | 1000 | 54.7000 | 94.0000 | 99.1000 | 9.245516 | -0.0396 | 0.746196 |
| R45A historical | VRPBTW | 1000 | 44.3000 | 78.5000 | 93.5000 | 18.972974 | -2.8338 | 1.511844 |
| R45A historical | VRPLTW | 1000 | 44.5000 | 78.5000 | 92.8000 | 17.603499 | -4.4683 | 1.386995 |
| R45A historical | OVRPBL | 1000 | 57.4000 | 97.1000 | 99.1000 | 6.301951 | -0.0773 | 0.984069 |
| R45A historical | OVRPBTW | 1000 | 51.4000 | 90.3000 | 99.7000 | 10.608161 | -0.0078 | 1.265002 |
| R45A historical | OVRPLTW | 1000 | 50.2000 | 92.1000 | 100.0000 | 10.097594 | -0.0595 | 0.820988 |
| R45A historical | VRPBLTW | 1000 | 44.9000 | 78.9000 | 94.1000 | 19.071665 | -3.0761 | 1.574574 |
| R45A historical | OVRPBLTW | 1000 | 49.5000 | 92.5000 | 99.9000 | 10.591372 | -0.0132 | 1.169842 |
