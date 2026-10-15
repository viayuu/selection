# R78 Round006 Final Results

Owner: Solmax, 6.1Sol max. Original fixed protocol; no additional configurations.

- Actual successful updates: 14904 (828 per task,18 tasks).
- Logical batch1024; seed2; one approved4090. Update1 is formal budget update1.
- Runtime: 4858.433007s (1.349565 GPUh), cap4h.
- Checkpoint SHA256: `29a5491b4fc984d8acb839d9bd72d91270161b0155e911e3106e62bb3c922e5b`; locked before candidate VAL.
- Gate: `{"actual_regret_not_worse": false, "raw_mean_cost_not_worse": false, "top1_strictly_improves": false}`; pass=False.
- Selected rule: `whole_B_fallback`. TEST completed=False.
- Actual candidate metrics below are NOT the fallback metrics.

| Split | Model | Top1 % | CE | Raw mean cost | Actual regret % | Macro VS % |
|---|---|---:|---:|---:|---:|---:|
| TRAIN | locked B | 71.1767027032 | 0.704272755318 | 11.365977443941617 | 0.393011700163 | -0.196819981305 |
| TRAIN | actual candidate | 71.1424727349 | 0.698281594449 | 11.366230745391302 | 0.402534646976 | -0.187844074466 |
| TRAIN | whole B fallback | 71.1767027032 | 0.704272755318 | 11.365977443941617 | 0.393011700163 | -0.196819981305 |
| VAL | locked B | 70.8943089431 | 0.713641610410 | 11.395558343544208 | 0.401326745448 | -0.192118669621 |
| VAL | actual candidate | 70.8015974897 | 0.707401479284 | 11.395979869603726 | 0.412333756565 | -0.182535728994 |
| VAL | whole B fallback | 70.8943089431 | 0.713641610410 | 11.395558343544208 | 0.401326745448 | -0.192118669621 |

## Choice Changes

| Split | Changed | Corrected | Harmed | Wrong-to-wrong | B zero winner-method pairs | Candidate zero pairs |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 33999 | 11629 | 12061 | 10309 | 25 | 25 |
| VAL | 1933 | 641 | 706 | 586 | 30 | 26 |

Full89 native method counts per split are in `native_method_counts.csv` (178 rows).
All18 task metrics and corrected/harmed/wrong-to-wrong signed FP64 cost changes
per split are in `per_task_comparison.csv` (36 rows); aggregate changes are in
`change_summary.json`. Cost sign is candidate minus B: negative is improvement.

## Historical Matched Control Limits

The previously completed actual final14904 control VAL has Top1=70.81443446013407%,
raw mean cost=11.395855343252101 and actual regret=0.41092598253845325%.
Its actual endpoint, not its selected B fallback, is the descriptive comparator.
No control TRAIN metrics were available or invented, and the baseline was not rerun.
This is one seed with reused VAL and a historical, not fresh randomized, control.
Comparison with B confounds extra training with the new objective; kernel/code drift
also limits causal attribution. No feature-starvation, universal task-safety,
statistical non-inferiority or paper-level mechanism claim follows from this screen.

## Scope And Receipts

No NSS TEST/LIB values, other TEST/LIB artifacts or new raw-data audits informed the rule.
A failed gate closes this fixed trial; no rescue, extra seed/configuration or budget.
The overall R78 file goal stays active; native/legacy application goals are untouched.
Authoritative complete outputs: `results.json`, `candidate_lock.json`,
`execution_receipt.json`, `update1_receipt.json`, `cpu_checks.json`,
`evaluation/{train,val}/final14904/` and `train.log`.
Actual worker/Slurm exit and assigned-card release evidence: `PROCESS_EXIT_RECEIPT.json`
(recorded after the GPU worker exits; allocation1475 is retained, not cancelled).

Final exit: Slurm1475.24 COMPLETED0:0,12:50:23-14:11:27 CST; tmux exited.
At14:12:57 CST the single assigned4090 was0%/2MiB with no processes, confirmed
by one read-only UUID-specific query. No other GPU queried or configuration started.
