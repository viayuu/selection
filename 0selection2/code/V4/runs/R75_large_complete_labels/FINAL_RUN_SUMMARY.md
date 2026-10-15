# R75 Final Local Delivery

Completed at 2026-10-14 08:35:36 UTC / 16:35:36 Asia/Shanghai. One seed2, one group, approved RTX 3090 only.

**Overall outcome is negative against saved R65B:** VAL is **8345/18000 = 46.361111%**, down 425 correct / 2.361111 percentage points; the single final TEST is **8076/18000 = 44.866667%**, down 499 correct / 2.772222 points. Actual regret is also higher on both splits. This is an outcome for the frozen-snapshot training recipe, not a causal result about data size.

## Locked Checkpoint And TEST

Selected epoch **40**, using the strict maximum original full18 VAL native integer correct count, earliest epoch on an exact tie. The saved 40-epoch history confirms this maximum. The lock was written before TEST; no TEST-driven reselection, cost-based selection, per-task checkpoint assembly, or additional seeds.

Checkpoint: [single_seed2/best.pt](/public/home/shiys/0selection2/code/V4/runs/R75_large_complete_labels/single_seed2/best.pt). Saved SHA256: `31e1ef7696c0876ba9dfcf557ffd592e919634081289b764a6eeae30d9504cd6`; this delivery did not open checkpoint bytes. R65B architecture, fresh parameters, official AdamWScheduleFree averaged-x deployment. Inference remains FP32, highest direct utility over the full legal pool, exact ties by ascending global solver ID.

The completed pipeline records **official_test_passes = 1**: all 18 tasks, 1000 original TEST instances each, 18000 total. TEST reports scores against saved costs, not new solver executions. Historical R65B metrics are reused without inference or prediction replay.

## Selected And Final TRAIN / VAL

| Checkpoint Role | Epoch | Split | Native Correct/Total | Row Top1 % | Task Macro Top1 % | CE | Actual Regret % |
|---|---:|---|---:|---:|---:|---:|---:|
| selected | 40 | train | 915469/1786887 | 51.232619 | 51.181345 | 1.025647412 | 0.879254054 |
| selected | 40 | val | 8345/18000 | 46.361111 | 46.361111 | 1.249806288 | 1.727816927 |
| final | 40 | train | 915469/1786887 | 51.232619 | 51.181345 | 1.025647412 | 0.879254054 |
| final | 40 | val | 8345/18000 | 46.361111 | 46.361111 | 1.249806288 | 1.727816927 |

Selected and final are both epoch40. The table preserves their distinct saved sources, `best_eval.json` and the last `history.json` record; metric values agree, while saved TRAIN evaluation seconds differ. No epoch35/39 TRAIN values are substituted. TRAIN has unequal task sizes, so row accuracy and task macro differ. TRAIN and held-out metrics describe different datasets and cannot be read as a matched improvement comparison.

## Saved R65B Comparison

Both checkpoints are epoch40. Top1 below is the equal-task macro; VAL/TEST also equal row accuracy. Deltas are R75 minus R65B, in percentage points. ALL covers 18 tasks; MVRP covers the other 15 tasks.

| Scope | R75 VAL % | R65B VAL % | Delta VAL pp | R75 TEST % | R65B TEST % | Delta TEST pp |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 46.361111 | 48.722222 | -2.361111 | 44.866667 | 47.638889 | -2.772222 |
| TSP | 28.000000 | 38.200000 | -10.200000 | 28.400000 | 37.000000 | -8.600000 |
| CVRP | 49.400000 | 48.000000 | +1.400000 | 45.500000 | 46.400000 | -0.900000 |
| ATSP | 76.400000 | 77.400000 | -1.000000 | 75.300000 | 75.700000 | -0.400000 |
| MVRP | 45.380000 | 47.560000 | -2.180000 | 43.893333 | 46.560000 | -2.666667 |

ALL actual regret: VAL **1.7278169274948956% vs 0.9455277310434759%**; TEST **1.7008112093165524% vs 0.9782802196993414%**. ALL mean cost: VAL **11.340649034622196 vs 11.169639888890874**; TEST **11.328395167764304 vs 11.171845153964233**. Mixed-task mean cost has heterogeneous units and is descriptive. CVRP improves VAL Top1 but not TEST; every reported family has lower TEST Top1 than saved R65B.

[final_comparison_compact.csv](/public/home/shiys/0selection2/code/V4/runs/R75_large_complete_labels/final_comparison_compact.csv) contains 40 rows: R65B saved TRAIN/VAL/TEST, R75 selected TRAIN/VAL/TEST and final TRAIN/VAL, each at ALL/TSP/CVRP/ATSP/MVRP. It retains counts, row and macro Top1, macro Top2/Top3/CE, mean cost, actual regret, vs-SBS, source paths/pointers, and VAL/TEST deltas including relative Top1 change. Accuracy fields are fractions; regret/vs-SBS and relative change are percentages; `*_pp_*` fields are percentage points. TRAIN deltas are intentionally blank because training inputs differ. The five R75 TEST scope rows summarize one pass, not five passes.

## Completed Budget And Runtime

| Run | Epochs | TRAIN Rows/Epoch | Batch/Task | Normal Effective Batch | Updates/Epoch | Successful Updates | Presentations | Warmup Updates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R65B saved | 40 | 180000 | 256 | 768 | 240 | 9600 | 7200000 | 720 |
| R75 completed | 40 | 1786887 | 384 | 1152 | 1566 | 62640 | 71475480 | 4698 |

R65B settings are taken from its actual args/protocol, not inferred from R75. Every saved R75 epoch covers all retained rows: TSP 87904, CVRP 98983, and 100000 for each of the other 16 tasks, with tails retained. The sums match both completed budgets. Saved training reports 27 AMP retries, two runtime continuations and zero replayed logged successful updates.

At TRAIN completion, `training_wall_seconds = 15704.904015968554` covers **only the last resumed process segment**. `total_formal_pipeline_elapsed_seconds = 59754.377463` includes original cache/input preparation and both IO optimization pauses through training completion; it is not compute-only training time or the end-to-end TEST/analysis duration.

Runtime receipts compare unpaired 120-update windows, excluding the first six cold updates: the first IO continuation measured 2.371477 -> 0.408090 seconds/update (**5.811160x**); the second measured 0.415825 -> 0.411824 (**1.009717x**) but wall-throughput ratio **0.985696**, not a measured wall-throughput gain. These are local runtime observations, not whole-run, inference-latency or accuracy speedups. No additional benchmark was run for this delivery.

W&B was configured **offline**, project `selector`, entity `yjkds-southern-university-of-science-technology`. Online synchronization or a live dashboard is not claimed. User-requested model roles are Sol/max executor and Astra/max reviewer; this local numeric delivery does not claim a new final reviewer pass.

## Interpretation Limits And Sources

R75 input is the frozen **Oct13 20261013_1923 large snapshot**, retaining the original first10000 rows per task and adding new-deployment labels; original VAL/TEST remain unchanged. Known depot TW deadline differences, incomplete TSP/CVRP large-size tails, changed batch/update/warmup budgets and TRAIN-fitted geometry statistics prevent an otherwise-identical, pure data-volume interpretation. This is a single-seed historical comparison; the legacy eight B-task feasible-domain mismatch remains deferred/unfixed. No accuracy ceiling or random-label claim follows from these results.

The later **new-only B-filtered dataset is a separate future release, not R75 input**. Its six data tests are unrelated to R75 run evidence and are not used here.

Evidence: [pipeline_complete.json](/public/home/shiys/0selection2/code/V4/runs/R75_large_complete_labels/pipeline_complete.json), [final_test_lock.json](/public/home/shiys/0selection2/code/V4/runs/R75_large_complete_labels/final_test_lock.json), [training_complete.json](/public/home/shiys/0selection2/code/V4/runs/R75_large_complete_labels/training_complete.json), and per-row JSON sources in the compact CSV. Budget/protocol sources: [R75 protocol](/public/home/shiys/0selection2/code/V4/runs/R75_large_complete_labels/protocol.json), [R65B args](/public/home/shiys/0selection2/code/V4/runs/R65_schedule_free_full18/B_schedule_free_seed2/args.json), [R65 protocol](/public/home/shiys/0selection2/code/V4/runs/R65_schedule_free_full18/protocol.json). Runtime sources: [IO continuation receipt](/public/home/shiys/0selection2/code/V4/runs/R75_large_complete_labels/runtime_io_continuation1/optimization_measurements.json), [pipeline continuation receipt](/public/home/shiys/0selection2/code/V4/runs/R75_large_complete_labels/runtime_pipeline_continuation2/optimization_measurements.json).

This delivery only reads saved JSON/CSV and writes this note plus the compact CSV. Automatically generated RESULTS/comparison/history/protocol, trainer/data/GPU/job state and Git are untouched. No model or prediction replay, raw data/cache load, solver call, extra seed or additional experiment was performed.
