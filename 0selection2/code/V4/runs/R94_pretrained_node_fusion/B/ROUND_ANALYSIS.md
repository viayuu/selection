# R94B Completed-Round Analysis
Results-analysis role; generated 2026-10-22T04:52:00+00:00. No external model/reviewer was invoked; this session has no verified gpt-6.1-sol routing, and no Astra was used.

## Verdict And Lock
Small positive historical-reference result, not a demonstrated pretrained-donor effect or automatic promotion. B's actual controller and worker both returned 0; full18 TRAIN/VAL/TEST, LIB, PDFs and timing stages completed. Timing completion is not timing validity (see below). R94A was still running at the request snapshot (~46k/57.6k); A was not read or evaluated here. No matched A/B claim is made.

Unique selected update **54,720/57,600**, epoch 38; strict minimum full18 VAL macro_vs_sbs_pct, first tie wins. Checkpoint `8e797150d22ea138db246536f36a1b7dfa6ef3b516ee065a3303a6b6e579c3ee`. No TEST/LIB reselection, per-problem stitching, calibration, new training, GPU use, Git or publishing in this analysis.

Same frozen `selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018` as R89B (main) and R84A (secondary): 81,000/4,500/4,500 per task, seed2, logical batch1024, same preparation hash, native/FP64-ind winners and legal pools. Eight B tasks retain only MTPOMO/MVMOE/reld_mt; RELD_MTL is removed, RELD_MOEL is aliased to reld_mt without minima merging. TEST partly informed prior R&D; not a new blind test. External solver pretraining adds knowledge/compute, and its original instance manifest/overlap is unavailable.

## Locked Full18 Metrics
| Split | Top1 % | Mean cost | SBS cost | Oracle cost | vs SBS % | vs Oracle % | Actual regret % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TRAIN | 71.947874 | 11.386290286 | 11.400702391 | 11.345073027 | -0.223076 | 0.361663 | 0.367808 |
| VAL | 71.428395 | 11.416278963 | 11.427978473 | 11.371691902 | -0.198659 | 0.388932 | 0.399079 |
| TEST | 71.577778 | 11.372233605 | 11.385299246 | 11.329932369 | -0.211169 | 0.371939 | 0.379115 |
ALL is the equal-weight mean of 18 task metrics; cross-task raw cost is descriptive, not a pooled performance objective. SBS was fitted on frozen full TRAIN; Oracle is the per-instance minimum over legal candidates. vs_SBS/vs_Oracle are task-level ratios of means, not mean per-instance actual regret.

## Historical Comparisons
| Reference | Split | dTop1 pp | dMean cost | dActual regret pp |
| --- | --- | --- | --- | --- |
| R89B | VAL | +0.180247 | -0.000015320 | -0.001597 |
| R89B | TEST | +0.158025 | -0.001236847 | -0.010531 |
| R84A | VAL | +0.343210 | -0.000332514 | -0.004415 |
| R84A | TEST | +0.341975 | -0.001609439 | -0.015731 |
R89B TEST: 1,756 corrected / 1,628 harmed = **+128** native correct, with 4,869 changed predictions. Top1 improves on 14/18 tasks, cost on 14/18, regret on 13/18. VRPBL has the largest Top1 gain (+0.9556 pp); VRPL regresses (-0.6444 pp, cost +0.000678), while OVRPTW/OVRPL/OVRPLTW improve Top1 but worsen cost and regret. Against R84A TEST: 1,913 corrected / 1,636 harmed = +277. These are single-seed historical comparisons, not statistical significance or a matched mechanism test.

## Fixed36 True Recovery
Targets remain the R84A locked-VAL zero/near-zero list (chosen <=0.1%, native winners >0); no TEST-based target reselection. Full18 TEST: 117/4,598 true target winners recovered (recall 2.5446%), 230 target selections, 113 false picks (precision 50.8696%); 12/36 groups have any correct pick, and 34/36 have TEST winner support. VAL: 126/4,625 correct, 238 selections, 112 false picks, recall 2.7243%, precision 52.9412%. TRAIN: 2,730/83,279 correct (3.2781%), 1,526 false picks.

| Reference | Split | Rescued | Lost | Net | New false | Target-winner dRegret pp | New-false regret burden pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R89B | VAL | 92 | 18 | 74 | 97 | -0.001885 | 0.009442 |
| R89B | TEST | 86 | 14 | 72 | 94 | -0.006998 | 0.007458 |
| R84A | VAL | 119 | 4 | 115 | 110 | -0.004942 | 0.010947 |
| R84A | TEST | 106 | 2 | 104 | 109 | -0.010036 | 0.008097 |
Rescued/lost are paired native-correct changes within the fixed target winner groups, not mere method-selection counts. 81/86 TEST rescues vs R89B occur in OVRPB, VRPBL, OVRPBL and OVRPLTW. Target-winner dRegret is the change on target-winner rows, normalized by all rows per task then macro-averaged. New-false burden is current regret on newly false-selected rows, not their baseline-relative change. These two subset quantities must not be added/subtracted into a total net regret; full18 dActual regret above is the net cost-quality evidence.

## Coverage And Fitting
Donor features cover CVRP + 15 MVRP tasks only; TSP/ATSP are raw-only. Versus R89B, covered16 TEST dTop1=+0.166667 pp, dCost=-0.001376926, dRegret=-0.011536 pp; raw2 dTop1=+0.088889 pp, dRegret=-0.002487 pp. Covered16 accounts for 86 rescued / 12 lost vs raw2 0 / 2. Covered16 VAL cost actually worsens by +0.000082329 despite full18 VAL cost improving slightly. Raw-only changes can arise from shared training/checkpoint differences, not direct donor execution.

Selected full-TRAIN Top1 71.947874% vs VAL 71.428395%: TRAIN-VAL gap **0.519479 pp**, with VAL-TRAIN actual regret +0.031272 pp. This is the full new90k cohort, not an old tiny training subset. It does not support a large tiny-cohort memorization story, establish a regularization cause, or show broad rare-winner recovery.

## LIB And Reused NSS
| Dataset | N | R94B Top1 % | NSS Top1 % | R94B cost | NSS cost |
| --- | --- | --- | --- | --- | --- |
| TSP TEST | 4500 | 47.0222 | 45.8889 | 7.970665 | 7.970401 |
| CVRP TEST | 4500 | 48.4667 | 45.9778 | 18.009808 | 18.004802 |
| TSPLIB | 49 | 24.4898 | 20.4082 | 8.212695 | 8.217907 |
| CVRPLIB | 100 | 64.0000 | 70.0000 | 67.140632 | 67.101112 |
Same R94B checkpoint throughout. NSS is reused R79 historical weights/predictions, NEW4500 TEST inference plus original LIB, not NEW90k retraining. Dataset/pool identities are retained in the existing NSS CSV. TSPLIB vs R89B: -2.0408 pp Top1, +0.010187 cost, +0.150375 pp regret; CVRPLIB: -1.0000 pp Top1, -0.097105 cost, -0.016568 pp regret. Against R84A, TSPLIB Top1 rises +4.0816 pp but cost worsens +0.003230; CVRPLIB Top1 rises +1 pp and cost falls -0.109740. LIB is mixed and small, not uniform transfer superiority. NSS cost is lower on both synthetic TEST tasks despite R94B's higher Top1; CVRPLIB also favors NSS on Top1 and cost.

## Timing Finding And Decision Boundary
**Code-supported diagnostic call-path/labeling error:** `r94_experiment.py:327-334` obtains `FrozenFeatureCache.batch` and calls `engine.inputs_only(batch)` for the nominal online branch. `r94_features.py:414-435` injects `nco_nodes`/`nco_node_mask`; `r84_experiment.py:26` imports `pairwise_objective.inputs_only`, whose lines 10-14 remove supervision but retain those fields. `r94_model.py:191-212` preserves them and calls the donor only under `if not cached`. Thus covered-task nominal donor+selector calls skip the online donor; TSP/ATSP skip it by design. The five inspected source hashes match the recorded protocol. A CPU-only extracted-helper sentinel check confirms raw/cached keys and objects are identical, without importing torch or a model.

Reported column differences span 0.000046 to 0.000737 ms/instance; they are not measured donor overhead. Cached GPU-resident selector timing remains a narrow diagnostic (128 fixed rows, batch timing/128, one warmup, three repetitions; excludes I/O/cache preparation), not independent-instance or full-system latency. `includes_native_encoder=true` / `donor_plus_selector_measured=true` are unsupported by this call path. No online donor latency is established. This finding does not invalidate the saved predictive metrics; no runtime or timing receipt was changed or rerun.

Retain the completed B as a small historical gain with limited true recovery, mixed LIB and unresolved online latency. Defer the matched pretrained-vs-random-donor conclusion until R94A completes under its existing lock. No automatic promotion, new variant, new split or additional run is authorized by this report.

Sources: B `protocol.json`, `final_test_lock.json`, `full_model_statistics/metrics_by_problem.csv`, `train_metrics_by_problem.csv`, paired/recovery CSVs vs R89B/R84A, `nss_comparison.csv`, `inference_timing.csv`; B actual controller/worker receipts; historical scalar CSVs and protocols. No raw predictions, neural forward passes or live A reads. Existing selection/learning-curve PDFs remain unchanged. Companion CSV: 65 rows (18 tasks + ALL/COVERED16/RAW2 for TRAIN/VAL/TEST; 2 LIB rows), with full metrics, both references, recovery and provenance. Missing LIB paired/target counts are blank, not zero.
