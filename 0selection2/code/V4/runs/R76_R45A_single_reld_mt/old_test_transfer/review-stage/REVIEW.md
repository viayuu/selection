# Scoped Evaluation Review

Requested reviewer: gpt-6-astra, reasoning effort max.
Agent: 01a13a6a-9770-7c62-89e3-bacd2f1e10f1.
Review independence: same-family. Acceptance: provisional.
Scope: candidate/label alignment and frozen-checkpoint evaluation correctness,
not research effectiveness or a new architecture experiment.

Initial finding: relative_gap was not excluded by the inherited inputs_only
helper. It was not present in the actual cache batches. The new evaluation
adapter now whitelists the 14 model input keys; all three focused CPU tests pass.

Final reviewer response:

> Confirmed: updated aligned_inputs() whitelists 14 model-input keys and excludes
> relative_gap. My sole P2 is resolved.
>
> Verdict: ready for correct evaluation delivery, same-family provisional,
> based on this fix and your reported completion evidence, not research effectiveness.
>
> Reported scores: Top1 11,481/18,000 = 63.7833%; cost regret 1.1136505%.
>
> Memory Update: Epoch18 and trained normalization remain fixed. Per your report:
> 3/3 CPU tests pass; four workers completed all 18,000 instances once with
> original-source hashes and FP64 costs matching; outputs remain isolated under
> old_test_transfer; report nesting is corrected and CPU analysis completed
> without repeated inference. No outstanding findings in this scoped review.
> Memory is response-only; no edits, GPU use, or extended audit.

The original formal R76 test_results.json nests metrics under result/macro.
A report-only KeyError was corrected after inference; the saved predictions
were reused for CPU analysis. No instance was inferred again for this fix.

Commands:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
python -m unittest code.V4.test_r76_old_test -v
bash code/V4/run_v4_r76_old_test.sh
bash code/V4/run_v4_r76_old_test.sh analysis
```

All four worker completion receipts cover disjoint canonical problem subsets.
The two RTX4090 and two RTX3090 devices were released after inference.
The 18 per-problem prediction files retain the logits and aligned FP64 targets.
