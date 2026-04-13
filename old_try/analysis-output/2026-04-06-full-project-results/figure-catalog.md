# Figure Catalog

## figure-01-legacy-family-comparison.png
- Purpose: compare all comparable selector families on the same 3000-instance legacy split.
- Data source: `legacy_family_metrics.csv` plus per-instance traces from `2cmab2` and `2l2r` full runs.
- What to notice: `SingleBestPerProblem` remains the strongest overall baseline; among learnable methods, legacy `Neural-LinUCB` has the lowest mean cost, while `alpha2` regresses clearly.
- Caption requirements: state that CI bars are instance-level 95% CI on mean cost, not seed-level uncertainty.
- Interpretation checklist: explain that the figure supports “no learnable method beats single-best on this fixed split”, and that top1 differences do not automatically translate to mean-cost gains.
- Caveat: baseline and oracle here are re-derived from the same fixed test split using `2l2r` arm-level truth.

## figure-02-current-nss-comparison.png
- Purpose: compare current NSS-based `Neural-LinUCB` runs against `NeuralUCB-Diag` and summary-level baselines.
- Data source: `current_nss_metrics.csv` and current NSS summaries/traces.
- What to notice: both `Neural-LinUCB` runs beat `NeuralUCB-Diag`; `safe_bs64` and `server` are nearly tied.
- Caption requirements: state that baseline bars are summary-level only; only selector methods have per-instance traces.
- Interpretation checklist: this figure changes the decision by narrowing the current bandit branch to `Neural-LinUCB`, not `NeuralUCB-Diag`.
- Caveat: no per-instance `single_best_per_problem` trace on this protocol, so baseline significance is blocked.

## figure-03-stage2-top1-vs-cost.png
- Purpose: show whether tuning gains in `top1` translate to lower `mean_cost`.
- Data source: `stage2_tuning_metrics.csv` from eight `2cmab2` stage2 sweeps.
- What to notice: points spread widely on `top1`, but are almost flat on `mean_cost`.
- Caption requirements: explicitly state that each point is one run on the same 1500-instance split.
- Interpretation checklist: the figure should trigger the decision to stop using `top1` alone for model selection.
- Caveat: single-seed runs only.

## figure-04-benchmark-ood-comparison.png
- Purpose: inspect whether the current NSS best checkpoint transfers cleanly to TSPLIB/CVRPLIB benchmark.
- Data source: benchmark `summary.json` under `benchmark_eval_tsplib_cvrplib`.
- What to notice: selector improves `top1` over single-best, but not `mean_cost`.
- Caption requirements: emphasize this is OOD benchmark evaluation and only aggregated baseline metrics are available.
- Interpretation checklist: use this figure to argue that `top1` can be misleading under distribution shift.
- Caveat: no paired significance test available for benchmark baselines.

## figure-05-two-gate-collapse.png
- Purpose: document a concrete failure mode from the earlier RL branch.
- Data source: `1two_gate/outputs/two_gate_tsp_seed2024_n100.log.txt`.
- What to notice: `gate2=none` rapidly saturates near 1.0 and eval length plateaus early.
- Caption requirements: mention that this is a single training log, included as failure analysis rather than main evidence.
- Interpretation checklist: the decision change is to treat dual-gate RL as a useful prototype, not as the current primary line.
- Caveat: no repeated runs, no held-out statistical validation.
