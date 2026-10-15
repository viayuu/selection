# User Tasks Delivery Index

2026-10-15. Existing results only: no new statistics, inference, checks or tuning for this index.

## 1. R77 Sample Size / Logical1024
Both A/B completed49680 successful updates, batch1024, seed2; A uses10000 and B70114 TRAIN rows/task.
TEST Top1 A/B:70.5691%/71.2252%; actual regret:0.412741%/0.394014%; full-data gain:+0.6561pp.
Historical R76 TEST Top1:69.4395%, but logical batch9856 and a different budget: NOT a strict matched control.
Source: [R77 results](/public/home/shiys/0selection2/code/V4/runs/R77_sample_size_batch1024/RESULTS.md).

## 2. R76 Missed-Winner Probabilities
52 zero-selected task-method combinations;14077 native winner rows.
TSP/LEHD:637 winner rows, conditional mean probability18.806%, yet zero selections.
Source in the original R76 run: [probability results](/public/home/shiys/0selection2/code/V4/runs/R76_R45A_single_reld_mt/missed_winner_probabilities/RESULTS.md).

## 3. R76 TRAIN-vs-LIB Distributions
Prediction-distribution TV: TSP12.84%, CVRP18.71%; R76 LIB Top1:28.57%/33% respectively.
Source: [LIB selection distribution](/public/home/shiys/0selection2/code/V4/runs/R76_R45A_single_reld_mt/lib_selection_distribution/RESULTS.md).

## 4. R78 Zero-Selection Goal
All six rounds failed acceptable net-gain certification; retain B. The overall goal is NOT achieved.
Historical round001/002 inconclusive execution labels remain; this is not a claim they fully converged.
Round006 actual candidate TRAIN: Top1=71.14247273%, CE=0.6982815944, cost=11.3662307454, regret=0.4025346470%.
Round006 actual candidate VAL: Top1=70.80159749%, CE=0.7074014793, cost=11.3959798696, regret=0.4123337566%.
Separate locked B/fallback VAL: Top1=70.89430894%, CE=0.7136416104, cost=11.3955583435, regret=0.4013267454%.
All three fixed gates failed;14904 updates completed, no TEST, no rescue or next configuration.
[Actual TRAIN/VAL and limits](/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round006/FINAL_RESULTS.md), [89 method counts/split](/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round006/native_method_counts.csv), [all18 task changes](/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round006/per_task_comparison.csv).
[GPU exit receipt](/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/round006/PROCESS_EXIT_RECEIPT.json): step1475.24 COMPLETED0:0; assigned4090 released,0%/2MiB/no processes.
[R76/R77 TEST cold counts](/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/final_delivery/R76_R77_TEST_COLD_COUNTS.md) are completed-TEST descriptive evidence received ONLY AFTER this trial closed, solely for delivery writing; no rule/budget/gate changes.

## 5. Original NSS / New Data
Original NSS setup with user-requested logical batch1024, seed2; TSP/CVRP each completed50 epochs.
The four already-delivered outcomes below are copied from the existing report; Top1/2/3 are fractions.

| Task/split | n | Top1 | Top2 | Top3 | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| TSP/TEST |3895|0.4395|0.6737|0.8290|7.634391|
| TSP/TSPLIB |49|0.2041|0.6122|0.7347|8.217907|
| CVRP/TEST |3895|0.4621|0.6508|0.7669|17.895818|
| CVRP/CVRPLIB |100|0.7000|0.8100|0.9000|67.101112|

Sources: [R79 report](/public/home/shiys/0selection2/code/V4/runs/R79_NSS_new_data_batch1024/REPORT.md), [full-precision summary](/public/home/shiys/0selection2/code/V4/runs/R79_NSS_new_data_batch1024/summary.csv).
Each TEST/LIB used its corresponding VALID-selected frozen checkpoint once; no values informed R78 rules.

## Ownership And Current State
All future non-paper work, including planning, budgets, protocols, gates, code, execution and result checks:6.1Sol max.
Astra's latest assignment was ONLY [PAPER_IDEAS_ONLY.md](/public/home/shiys/0selection2/code/V4/runs/R78_zero_selection_goal/review-stage/PAPER_IDEAS_ONLY.md), literature/high-level ideas; it is closed. Older Astra notes/plans retain original authorship, never relabeled Sol.
NSS delivery is complete. No R78 GPU run remains; file goal stays active/unresolved, awaiting a genuinely different mechanism, not another automatic trial.
No native/legacy goal operation or Git commit was made.
