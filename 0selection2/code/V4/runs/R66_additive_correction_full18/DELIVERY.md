# R66 completed delivery

The launch-time R66_README and source_launch snapshot remain unchanged. Their
future-tense TEST/handoff statements describe the original training launch,
not missing work in this completed delivery.

## Actual execution

- TRAIN: tmux r66_train, existing Slurm1465.94, COMPLETED0:0,6109s.
- Both arms40epochs/9600successfulupdates/7.2Mpresentations; four AMP retries each.
- Original integer validation counts froze A9 and B33; official A, before TEST.
- ONE original full18 TEST per complete bundle: tmux r66_final_test,
  Slurm1465.95, COMPLETED0:0,37s. Both selectors ran live on current raw inputs.
- CPU analysis ran once, exit0. No repeated inference or solver execution.
- Official A TEST8499/18000=47.216667%; B8569/18000=47.605556% does not reselect A.
- R58 remains fully deferred and legacy eight-B-task semantics unresolved.

Actual commands are retained by handoff.json, delivery_receipt.json and process
receipts. The completed delivery adds isolated r66_frozen_evaluation.py,
run_v4_r66_test.sh and r66_analysis.py; these were not retroactively inserted
into or used to change the frozen training-source snapshot.

The TEST launcher requires the parent validation lock and exclusive one-time
marker. It is a reproduction entrypoint, not authorization to replay TEST.

## Review and publication

reviews/ contains the actual GPT-6-Astra/max primary TRAIN/validation closure
and a fresh read-only repository delivery review. New R66 TEST was excluded from
the next-plan decision. Integrity readiness is same-family/provisional and is
not a performance acquittal: original full18 native70% remains active/unachieved.

Code, metrics, tabular curves, figures, config/lock/receipts and review records
are published in one completed-version commit. The concurrent repository
cleanup and code/.gitignore are left untouched. Weights, prediction arrays,
raw update traces, W&B and regenerable data/schedule caches remain local on
this server, not newly forced into Git. The existing final_test_lock.json
identifies the exact selected local bundles and their unchanged inference rule.
This is not a claim that Git alone contains all data needed to replay inference.

The complete selected bundles remain at the A_independent_sum_seed2 and
B_additive_correction_seed2 best.pt and selected best_checkpoints paths under
this run root. The next R67 experiment does not reuse them as initialization.
