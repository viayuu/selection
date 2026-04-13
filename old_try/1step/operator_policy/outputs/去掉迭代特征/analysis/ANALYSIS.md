# Operator Policy Output Analysis

- Output dir: `1step/operator_policy/outputs/去掉迭代特征`
- Train steps (latest run): `100`
- Eval checkpoints (latest run): `10`
- Train segments detected in log: `2`
- Detected appended history; analysis uses only the latest monotonic train segment after previous max step `3`.

## Charts
- `1step/operator_policy/outputs/去掉迭代特征/analysis/train_lengths_improvement.png`
- `1step/operator_policy/outputs/去掉迭代特征/analysis/action_distributions.png`
- `1step/operator_policy/outputs/去掉迭代特征/analysis/timing_breakdown.png`
- `1step/operator_policy/outputs/去掉迭代特征/analysis/optimization_signals.png`
- `1step/operator_policy/outputs/去掉迭代特征/analysis/eval_overview.png`

## Key Findings
- Mean train init length = `7.8977`, mean final length = `7.7696`, mean improvement = `0.1281`.
- Best train improvement occurs at step `77` with value `0.2218`.
- Mean timing per step: total `39.57s`, reset `30.57s`, rollout `8.71s`, optimize `0.28s`.
- Last train initializer distribution: `{'lehd': 0.0, 'elg': 0.0, 'difusco': 1.0}`.
- Last train operator distribution: `{'two_opt': 0.3003906309604645, 'lehd_rrc_step': 0.3492187559604645, 'dact_2opt_step': 0.35039061307907104}`.
- Last eval @ step `100`: init `7.8955` -> final `7.8075`, improvement `0.0880`.
- Best eval final length appears at step `30` with final `7.7656` and improvement `0.0779`.
- Last eval initializer distribution: `{'lehd': 0.0, 'elg': 0.0, 'difusco': 1.0}`.
- Last eval operator distribution: `{'two_opt': 0.0, 'lehd_rrc_step': 0.0, 'dact_2opt_step': 1.0}`.

## Interpretation
- `train_lengths_improvement.png` only provides an online training signal because each train step uses a different batch; use eval curves for strict performance comparison.
- `action_distributions.png` is the most informative for policy behavior: it shows whether the selector remains exploratory or collapses to a narrow subset of actions.
- `timing_breakdown.png` reveals whether training is bottlenecked by solver execution (`reset`/`rollout`) or by optimizer compute.
- `optimization_signals.png` should be read qualitatively: entropy decay indicates confidence increase, while REINFORCE loss is not expected to monotonically decrease.
- `eval_overview.png` is the most reliable view of actual progress because it compares checkpoints on the eval set rather than across different train batches.

## Caveat
- If `advantage_mean` stays near zero, that is normal here because the baseline is the batch mean; this field is not a strong diagnostic signal in the current implementation.