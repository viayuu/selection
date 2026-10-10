# R53 Execution

All commands run from `/public/home/shiys/0selection2`, in the existing `easynco` environment. Only the idle RTX3090 with UUID `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d` is used. Other GPU processes and historical results are not modified.

## Prepare and Check

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
bash code/V4/run_v4_r53.sh prepare
```

This computes the original-FP64 training weight mean and the frozen validation gate budget. It does not open test or launch training. Focused unit tests run before the entry point.

## Actual Training and Evaluation Command

```bash
tmux new-session -d -s r53_pair_specialist \
  'source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && exec srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G env R53_GPU_UUID=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d bash /public/home/shiys/0selection2/code/V4/run_v4_r53.sh all'
```

The two arms run sequentially on that single GPU, sharing the cached data and the pre-generated training plan. Both are trained from scratch; no baseline encoder weights are copied. The common random initialization and training plan are saved locally as `shared_initialization.pt` and `sampling_plan.pt`; their hashes are in `runtime.json` and the arm configurations.

The `all` stage trains A/B, locks their validation-selected checkpoints, then performs one test evaluation per arm and produces the report. Existing completed arm results and test outputs are retained, not automatically retrained/reselected. Partial training can resume from the same arm's `last.pt`; unchanged source/data provenance is required.

## Artifacts

- `protocol.json`: fixed losses, gate, selection rule, stopping rules, source hashes and baseline identity.
- `gate_budget.csv/json`: pretraining validation coverage and diagnostic-only recovery bounds.
- `training_counts.csv`, `geometry_stats.json`, `sampling_plan.json`: all training populations and training-only normalization.
- `A_ce_seed2/`, `B_cost_sensitive_seed2/`: configurations, best/final checkpoints, histories, predictions and local offline W&B logs.
- `locked_checkpoints.json`: both best checkpoint identities, fixed before opening test.
- `comparison.csv`, `binary_comparison.csv`, `corrections_and_harms.csv`: system results, specialist results and cost accounting.
- `learning_curves.png/csv`, `RESULTS.md`: curves and interpretation.

Ranking outputs only swap the original two methods. They are saved as explicit rankings, including equal baseline logits; ordinal ranking representations are not treated as calibrated class probabilities. Complete-system percentages retain equal problem weights. Binary CE/accuracy/regret are separate diagnostics and are never substituted for seven-pool or 18-task Top1.

W&B runs offline under project `selector`, entity `yjkds-southern-university-of-science-technology`. Checkpoints/caches and W&B internals remain local under the repository's artifact policy; source, configurations, prediction archives and reports are committed.
