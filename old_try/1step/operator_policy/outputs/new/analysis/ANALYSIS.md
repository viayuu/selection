# Operator Policy Output Analysis

- Output dir: `1step/operator_policy/outputs/new`
- Train steps (latest run): `927`
- Eval checkpoints (latest run): `9`
- Train segments detected in log: `2`
- Detected appended history; analysis uses only the latest monotonic train segment after previous max step `80`.

## Charts
- `1step/operator_policy/outputs/new/analysis/train_lengths_improvement.png`
- `1step/operator_policy/outputs/new/analysis/action_distributions.png`
- `1step/operator_policy/outputs/new/analysis/timing_breakdown.png`
- `1step/operator_policy/outputs/new/analysis/optimization_signals.png`
- `1step/operator_policy/outputs/new/analysis/eval_overview.png`

## Key Findings
- Mean train init length = `7.8129`, mean final length = `7.7646`, mean improvement = `0.0484`.
- Best train improvement occurs at step `840` with value `0.0818`.
- Mean timing per step: total `42.60s`, reset `0.60s`, rollout `41.23s`, optimize `0.76s`.
- Last train initializer distribution: `{'lehd': 0.0, 'elg': 1.0, 'difusco': 0.0}`.
- Last train operator distribution: `{'two_opt': 0.0006249999860301614, 'lehd_rrc_step': 0.5571874976158142, 'dact_2opt_step': 0.44218748807907104}`.
- Last eval @ step `900`: init `5.2312` -> final `5.2232`, improvement `0.0080`.
- Best eval final length appears at step `800` with final `5.0042` and improvement `0.1581`.
- Last eval initializer distribution: `{'lehd': 0.0, 'elg': 1.0, 'difusco': 0.0}`.
- Last eval operator distribution: `{'two_opt': 0.0, 'lehd_rrc_step': 0.0, 'dact_2opt_step': 1.0}`.

## Interpretation
- `train_lengths_improvement.png` only provides an online training signal because each train step uses a different batch; use eval curves for strict performance comparison.
- `action_distributions.png` is the most informative for policy behavior: it shows whether the selector remains exploratory or collapses to a narrow subset of actions.
- `timing_breakdown.png` reveals whether training is bottlenecked by solver execution (`reset`/`rollout`) or by optimizer compute.
- `optimization_signals.png` should be read qualitatively: entropy decay indicates confidence increase, while REINFORCE loss is not expected to monotonically decrease.
- `eval_overview.png` is the most reliable view of actual progress because it compares checkpoints on the eval set rather than across different train batches.

## Caveat
- If `advantage_mean` stays near zero, that is normal here because the baseline is the batch mean; this field is not a strong diagnostic signal in the current implementation.