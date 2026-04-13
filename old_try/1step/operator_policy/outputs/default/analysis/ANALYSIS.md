# Operator Policy Output Analysis

## Files
- `1step/operator_policy/outputs/default/analysis/train_lengths_improvement.png`
- `1step/operator_policy/outputs/default/analysis/action_distributions.png`
- `1step/operator_policy/outputs/default/analysis/timing_breakdown.png`
- `1step/operator_policy/outputs/default/analysis/optimization_signals.png`
- `1step/operator_policy/outputs/default/analysis/eval_summary.png`
- `1step/operator_policy/outputs/default/analysis/summary.json`

## Key Findings
- Training contains `80` steps and `1` evaluation record(s).
- Mean training init length is `7.8014`, mean final length is `7.7741`, so mean improvement is `0.0274`.
- Best single-step training improvement appears at step `25` with improvement `0.0925`.
- By the last step, initializer distribution is `{'lehd': 1.0, 'elg': 0.0, 'difusco': 0.0}`.
- By the last step, operator distribution is `{'two_opt': 0.27812498807907104, 'lehd_rrc_step': 0.4749999940395355, 'dact_2opt_step': 0.24687500298023224}`.
- Mean per-step time is `12.30s`, of which reset `1.09s`, rollout `11.02s`, optimize `0.17s`.
- Last eval at step `80`: init `5.2787` -> final `5.1491`, improvement `0.1297`.
- Eval uses initializer distribution `{'lehd': 1.0, 'elg': 0.0, 'difusco': 0.0}` and operator distribution `{'two_opt': 0.0, 'lehd_rrc_step': 1.0, 'dact_2opt_step': 0.0}`.

## Interpretation
- The policy has already collapsed strongly toward `lehd` as initializer by the end of this run.
- During training rollout, `lehd_rrc_step` becomes the dominant operator, while `two_opt` and `dact_2opt_step` remain secondary.
- In the only available eval record, the policy is fully deterministic: it chooses `lehd` for initialization and `lehd_rrc_step` for every operator step.
- Most wall-clock time is spent inside rollout rather than policy forward or optimizer update, so the current bottleneck is solver/operator execution.
- `advantage_mean` stays at zero because it is logged before normalization and the baseline is the batch mean; this field is therefore not informative in the current setup.

## Caveat
- There is only one evaluation record, so generalization trends cannot be judged yet. A longer run with multiple eval checkpoints would make the analysis much stronger.