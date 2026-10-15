# Selection: Current Research Snapshot

Release date: **2026-10-15**. The retained complete 18-problem model is **R77B**.
R78 rounds 001-006 did not yield an accepted replacement. R79 NSS has completed
its two task-specific new-data runs and their locked TEST/LIB evaluations.

## Main Deliverables

PDFs are the primary deliverables; CSV files provide machine-readable results.

| Deliverable | Files |
| --- | --- |
| R77B complete statistics, 22 pages | [PDF](code/V4/runs/R77_sample_size_batch1024/B_full_seed2/full_model_statistics/MODEL_STATISTICS.pdf), [summary](code/V4/runs/R77_sample_size_batch1024/B_full_seed2/full_model_statistics/RESULTS.md), [per-problem metrics](code/V4/runs/R77_sample_size_batch1024/B_full_seed2/full_model_statistics/metrics_by_problem.csv) |
| R77B method selection and true-winner distributions, 18 pages | [PDF](code/V4/runs/R77_sample_size_batch1024/B_full_seed2/full_model_statistics/selection_distributions.pdf), [CSV](code/V4/runs/R77_sample_size_batch1024/B_full_seed2/full_model_statistics/selection_distribution.csv) |
| R77B versus NSS on new TEST and original LIB | [CSV](code/V4/runs/R77_sample_size_batch1024/B_full_seed2/full_model_statistics/nss_comparison.csv), [protocol and exact candidate pools](code/V4/runs/R77_sample_size_batch1024/B_full_seed2/full_model_statistics/protocol.json) |
| R78 six-round retrospective, 3 pages | [PDF](code/V4/runs/R78_zero_selection_goal/final_delivery/zero_selection_retrospective/ZERO_SELECTION_RETROSPECTIVE.pdf), [summary](code/V4/runs/R78_zero_selection_goal/final_delivery/zero_selection_retrospective/RETROSPECTIVE.md), [CSV](code/V4/runs/R78_zero_selection_goal/final_delivery/zero_selection_retrospective/round_summary.csv) |
| R79 NSS selection distributions, 2 pages | [PDF](code/V4/runs/R79_NSS_new_data_batch1024/selection_distribution/selection_distributions.pdf), [CSV](code/V4/runs/R79_NSS_new_data_batch1024/selection_distribution/selection_distribution.csv) |
| R76 paired winner/selected probability and cost gaps | [PDF](code/V4/runs/R76_R45A_single_reld_mt/missed_winner_probabilities/winner_selected_gap_report/winner_selected_gap_report.pdf), [single-file offline HTML](code/V4/runs/R76_R45A_single_reld_mt/missed_winner_probabilities/winner_selected_gap_report/winner_selected_gap_report_standalone.html), [summary](code/V4/runs/R76_R45A_single_reld_mt/missed_winner_probabilities/winner_selected_gap_report/SUMMARY.md) |

The distribution PDFs distinguish predictions from true-winner frequencies.
TSP and CVRP each include LIB predicted, LIB true winner, TEST predicted and
TEST true winner panels; other tasks have the two TEST panels. Candidate names
and order are aligned. A predicted panel's Top1 is per-instance agreement with
the saved winner, not the most frequently selected method's share. True-winner
panels have no fabricated model accuracy.

## Locked Current Model

R77B uses the R43A/R45A handcrafted-feature plus solver-ID architecture,
ScoreDifference/maximin scores, supervised comparison/risk losses and legal-pool
R-Drop. It starts from fresh **seed2**, uses logical **batch1024**, and completes
**49,680 successful optimizer updates**. The current checkpoint is **epoch28 /
update34776**, selected by the predeclared strict minimum complete-VAL
`macro_vs_sbs_pct`, using FULL-TRAIN-fitted SBS identities.

Checkpoint SHA256:
`31c0c75d66b1bde6ad533909a9c04b0036e80af4c4a5775b01b957ce310e6d1b`.

| Split | ALL native Top1 (%) | Mean cost | SBS cost | Oracle cost | vs_SBS (%) | vs_Oracle (%) | Actual regret (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VAL | 70.8943089 | 11.3955583435 | 11.4064769309 | 11.3509101442 | -0.1921187 | 0.3911299 | 0.4013267 |
| TEST | 71.2252175 | 11.3457711712 | 11.3568857997 | 11.3021773178 | -0.1940478 | 0.3856015 | 0.3940143 |

ALL is the equal-weight mean over 18 problems, not a pooled raw-cost score.
Mixed-problem raw mean costs have heterogeneous units and are descriptive only.
The CSV/PDF reports each problem as well as ALL. `vs_SBS` and `vs_Oracle` are
per-problem ratios of mean costs (then macro-averaged); actual regret is the
mean of per-instance relative excess over the legal-pool Oracle. These are not
interchangeable. Oracle is the per-instance minimum over legal candidates, not
a proven optimal tour. SBS identities are fitted on FULL TRAIN and frozen for
VAL/TEST/LIB; any posthoc split/LIB SBS is separately labeled supplementary.

One checkpoint and inference rule are used across all four comparison datasets:

| Dataset | n | R77B native Top1 (%) | R77B mean_cost | R79 NSS native Top1 (%) | R79 NSS mean_cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| New TSP TEST | 3,895 | 44.8267009 | 7.6355463775 | 44.3388960 | 7.6343906345 |
| New CVRP TEST | 3,895 | 47.3940950 | 17.9024307169 | 46.2130937 | 17.8958182949 |
| Original TSPLIB | 49 | 20.4081633 | 8.2166893199 | 20.4081633 | 8.2179070743 |
| Original CVRPLIB | 100 | 60.0000000 | 67.3002676868 | 70.0000000 | 67.1011117172 |

These LIB results are the newly completed **same-lock R77B** evaluation, not
R76 LIB values. NSS predictions come from the existing R79 validation-selected
task-specific checkpoints, not another retraining for this report. TSP has 8
candidates and CVRP has 10; the comparison protocol records their names/order,
data versions, prediction alignment and checkpoint hashes.

Primary Top1 uses saved **native/FP64 `ind`** winners. Original NSS reports also
preserve **FP32 argmin** compatibility: its new TSP TEST Top1 is **43.9537869%**
under that rule, versus **44.3388960%** under the primary native rule. R77B TSP
TEST is 44.4929397% under FP32 argmin versus 44.8267009% native. Never silently
substitute one winner definition for the other. Raw costs remain FP64.

TRAIN updates parameters. VAL controls checkpoint selection under the fixed
protocol; TEST/LIB do not select checkpoints, temperatures or per-task rules.
Later research has partly used historical TEST diagnostics, so TEST is **not**
claimed to be an entirely untouched blind set. This release does not create a
new split, rerun a model or alter a checkpoint/goal.

## Data Version

Current R76/R77/R78/R79 new-data version:
`selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014`.
Each of 18 problems has **TRAIN 70,114 / VAL 3,895 / TEST 3,895** instances.

Only the eight B tasks remove `RouteFinder`, `MoSES_RF` and `MoSES_CaDA`.
The single-ReLD derivation removes `RELD_MTL` everywhere, retains the original
`RELD_MOEL` costs unchanged, and renames that candidate `reld_mt`; it does not
take a best-of-two cost. Historical global ID16 remains MOEL, and ID17 is
inactive. Use the scenario metadata for legal pools and column order.
These exclusions do not certify that every remaining solver has identical
constraint semantics. R75 uses an earlier complete-label TRAIN release and
the old VAL/TEST, so its scores are not a matched new-data comparison.

Release code: [split reader](code/V4/new_split_data.py),
[B-task filtering exporter](code/V4/export_new_split_data.py),
[single-ReLD exporter](code/V4/export_single_reld_data.py).

## Experiment Navigation

| Version | Entry | Final evidence and status |
| --- | --- | --- |
| R75 | [README](code/V4/R75_README.md) | [Final summary](code/V4/runs/R75_large_complete_labels/FINAL_RUN_SUMMARY.md); completed old-split experiment, negative versus saved R65B |
| R76 | [README](code/V4/R76_README.md) | [Results](code/V4/runs/R76_R45A_single_reld_mt/RESULTS.md); historical R45A/new-data baseline and paired-gap report above |
| R77 | [README](code/V4/R77_README.md) | [Paired A/B results](code/V4/runs/R77_sample_size_batch1024/RESULTS.md); B is the retained full model, A uses 10k TRAIN/task with the same update budget |
| R78 | [README](code/V4/R78_README.md) | [Six-round retrospective](code/V4/runs/R78_zero_selection_goal/final_delivery/zero_selection_retrospective/RETROSPECTIVE.md); no accepted candidate, retain whole R77B |
| R79 | [README](code/V4/R79_NSS_README.md) | [Completed NSS report](code/V4/runs/R79_NSS_new_data_batch1024/REPORT.md); original NSS architecture/loss, new data, seed2 and batch1024 overrides |

R58 remains deferred; its unified semantic cost table has not been released.
R74 was terminated by user request to release GPU memory;
all disk artifacts remain. [R74 termination record](code/V4/runs/R74_adaptive_incomplete_acquisition/TERMINATED.json)
is not a completed-model result or permission to resume. Historical goal/solver
issues are not claimed to be fixed. Older launch/resource notes are provenance,
not instructions to restart jobs. No Rxx best-score or per-task checkpoint mix
is used to describe the current model.

## Repository And Server Assets

The Git root is `/public/home/shiys`, not this subdirectory. Run commands only
after activating the existing environment:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
```

Git contains selected source, tests, protocols/configuration, aggregate CSV,
Markdown and PDF reports, plus the R76 self-contained HTML. It intentionally
does **not** distribute the complete runnable data/weight bundle. Server assets
remain at their original locations:

| Asset | Server location |
| --- | --- |
| Current exported data and release metadata | `/public/home/shiys/selection_data_new_only_B_filtered_single_reld_mt_seed2_20261014/` |
| Parent B-filtered export | `/public/home/shiys/selection_data_new_only_B_filtered_seed2_20261014/` |
| Earlier complete-label TRAIN data | `/public/home/shiys/训练数据_公共完整标签_20261013_1923/` |
| Original LIB data | `/public/home/shiys/0selection2/data/TSPLIB/`, `/public/home/shiys/0selection2/data/CVRPLIB/` |
| R77B weights and evaluation arrays | `/public/home/shiys/0selection2/code/V4/runs/R77_sample_size_batch1024/B_full_seed2/` (`best.pt`, `evaluation/`, `lib_evaluation/predictions/`) |
| Locked R77 TRAIN/VAL arrays | `/public/home/shiys/0selection2/code/V4/runs/R77_sample_size_batch1024/selected_train_val_predictions/` |
| NSS weights and per-instance predictions | `/public/home/shiys/0selection2/code/V4/runs/R79_NSS_new_data_batch1024/{TSP,CVRP}/` |
| Other run artifacts, caches and historical launch snapshots | `/public/home/shiys/0selection2/code/V4/runs/` |

Large NPZ/NPY arrays, model weights, feature/CPU/embedding/teacher caches,
environment directories, W&B state, large logs/update traces, redundant ZIPs
and source-launch copies are not newly uploaded. Existing staged removals of
tracked generated files remove them from Git only, not from server storage;
this is not a purge of historical Git objects. Root ignore rules also match
JSON/CSV/PNG/log files, so release configuration/report files are included via
an explicitly reviewed per-file allowlist, never a forced whole-directory add.
