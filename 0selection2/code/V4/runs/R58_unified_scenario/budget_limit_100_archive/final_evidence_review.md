# R58 Final Evidence Review

Both reviews concern the completed bounded measurements and the over-budget
skip, not a completed benchmark, trained model or performance improvement.
Requested reviewer: gpt-6-astra/max. Same-family/provisional; no cross-family
acceptance or new solver replay is claimed. Full raw responses follow.

## Primary Review

**Verdict**
The evidence supports a **genuine measured budget blocker under policy008**. Stop this fixed recipe. No material code, accounting or evidence defect was found in this bounded review.

- **Evidence integrity: 9/10, Ready** to report the negative budget result.
- **Whole-R58 readiness: 6/10, Not Ready.** Benchmark completion and performance claims remain unsupported.

Same-family; provisional.

**Verified Evidence**
- Verified all128 current preflight qualifications and source/profile bindings; all63 refined deployments and159 sessions completed with successful worker exits.
- All48 RF deployments passed exact serial1/normal4/MPS4 equivalence. Selected on-disk worker evidence matched the recorded sessions;352 sampled TRAIN route/cost observations passed independent rechecking. Sampled MPS client lists confirmed attachment to the designated UUID.
- MPS’s family projection was **31.529182h versus normal4’s31.414276h**. Zero MPS credit correctly follows the frozen adoption rule; this is not a claim of statistically significant slowdown.
- Verified the settled Slurm receipt and deduplicated ledger: **3.330399h actually spent**, with8088.902146 seconds charged against the10800-second pilot cap.

The [whole-budget artifact](/public/home/shiys/0selection2/code/V4/runs/R58_unified_scenario/budget.json) reproduces, rounded:

```text
3.3304 + 95.2812 + 1.1092 + 3 + 2 + 9.5281 = 114.2489 GPUh
```

Startup is counted once. Even excluding contingency, the reference is **104.720806h**. These are measured-data-based projections, not114 hours already spent or a proven runtime lower bound.

The actual [labels-stage gate](/public/home/shiys/0selection2/code/V4/runs/R58_unified_scenario/pipeline_state.json) now records `skipped_over_budget`, with no solve command. Execution approval, deployment lock, complete release, baseline directory and locked-test artifacts are absent.

**Minimum in-scope actions:** preserve this evidence and report the fixed recipe as budget-blocked. No corrective code change or additional measurement is required for that conclusion. Do not change allowances, reduce pools, regenerate labels, train, or start the post-R58 performance loop.

**Reviewer Memory Update**
- **Settled:** implementation, fresh128 preflight and the final63/159-session timing exercise are completed. Qualifications, sampled raw evidence, equivalence and fixed008 accounting support the blocker.
- **Measured conclusion:** normal4 selected; zero MPS credit; whole reference114.248924h, exceeding100h. Actual labels gate correctly skipped.
- **Boundary:** the precommitted measurement opportunity is exhausted despite unused pilot-cap time. R58’s full54-dataset benchmark, baseline/evaluation and later ≥70% objective remain unfinished. No rescue branch is authorized.

## Fresh Nightmare-Mode Verification

**Findings And Verdict**
No actionable findings. I found no error invalidating the budget skip or inflating the stated measurement qualifications. No correction is required within scope.

**Evidence integrity: 9/10, Ready for the bounded budget-skip conclusion only.** `review_independence=same-family; acceptance_status=provisional`. Requested Astra/max routing was not independently verifiable.

**Verified Evidence**
- **Coverage:** All 128 distinct preflights pass under the current source freeze. TRAIN hashes match; all 128 preflight timing estimates independently recompute. The refined63 and unchanged65 partition the roster exactly. [budget.json](/public/home/shiys/0selection2/code/V4/runs/R58_unified_scenario/budget.json:1932)
- **Raw measurements:** Cross-checked all 159 sessions, 1,134 groups and 2,862 worker shards. All timing estimates reproduce. All48 RF comparisons have identical literal routes and costs, covering both checkpoint buckets; all288 measured MPS groups have matching four-client attachment evidence. MPS’s aggregate projection is **413.664 seconds slower**, correctly receiving zero acceleration credit. [summary.json](/public/home/shiys/0selection2/code/V4/runs/R58_unified_scenario/timing_refinement/summary.json:4)
- **Spend and cap:** Actual spend is **11,989.436714 seconds**. Cumulative pilot/refinement spend is **8,088.902146 / 10,800 seconds**, without a cap restart. Failed bootstrap1465.77 contributes **33.436714 seconds once**, including its nested setup/readonly work. Receipt hashes and outer-wall reconciliation match. [Bootstrap Receipt](/public/home/shiys/0selection2/code/V4/runs/R58_unified_scenario/bootstrap_failure/1465_77/spend.json:4)
- **Whole budget:** Independently reproduced **114.248924 GPUh** using actual spend, **95.281179h** label reference, **1.109229h** H63, fixed **5h** allowances and **10%** label contingency. H63 is subtracted from refined references before being added exactly once. [Accounting Code](/public/home/shiys/0selection2/code/V4/r58_budget.py:376)
- **Stop enforced:** `pipeline_state.json` records `skipped_over_budget`; that branch returns3 before approval, locking or generation. No execution approval, deployment lock, release, model checkpoint, test result or label/training log exists in this R58 run. [Pipeline State](/public/home/shiys/0selection2/code/V4/runs/R58_unified_scenario/pipeline_state.json:7), [Gate](/public/home/shiys/0selection2/code/V4/r58_pipeline.py:77)

Residual limitations: this was read-only artifact and CPU arithmetic verification, not solver replay or statistical replication. Validation/test distributions remain TRAIN-based estimates; allowances and contingency are not confidence bounds. No validation/test data access, GPU execution or edits occurred.

**Memory Update**
- **Settled:** Completed128 preflights, final63/159 measurement evidence, RF48 equivalence, MPS attachment, zero MPS credit, deduplicated spend and the fixed114.248924h over-budget skip are supported.
- **Outstanding:** Full54 release, baseline retraining and locked evaluation remain absent. WholeR58 is unfinished; full18 single-solver Top1≥70% remains unachieved.
- **Boundary:** This verdict authorizes neither generation nor additional recipes, measurement rounds or branches.

