# Observation Metrics

This directory reorganizes prior experiments around the metric checklist in `观测指标/观测指标.md`.

For each retained full run:
- `loss_curve.png`: train/val loss over epochs.
- `val_problem_metrics.csv` and `test_problem_metrics.csv`: per-problem metric tables.
- `val_problem_metrics.md` and `test_problem_metrics.md`: readable markdown summaries.
- `val_top1_vs_single_methods.png` and `test_top1_vs_single_methods.png`: one consolidated figure per split, containing all problems.
- `val_mean_cost_vs_single_methods.png` and `test_mean_cost_vs_single_methods.png`: one consolidated figure per split, containing all problems.
- `val_arm_distribution.png` and `test_arm_distribution.png`: one consolidated figure per split, containing all problems.

Cross-run comparison tables:
- `comparison/val_all_runs_problem_metrics.csv`
- `comparison/test_all_runs_problem_metrics.csv`

Runs included:
- `round2_full_balanced_stats-0415-024024`
- `round2_full_true_baseline-0415-040256`
