# R42 Experiment Audit

Date: 2026-10-01 (Asia/Shanghai).
Reviewer: fresh read-only Codex code-reviewer, inherited model, high reasoning.
Review independence: same-family. Acceptance: provisional for semantic judgment.

## Overall Verdict: PASS

No blocking integrity failure was found. The reviewer initially identified a missing forced-AMP-overflow test; the added actual CUDA test passes and verifies the retry behavior. The final reviewer verdict is A-F PASS. The intentionally inactive 640-parameter coord_dist embedding is shared legacy behavior under ignore_coord_dist=true, not a dead relation/query branch.

### A. Ground Truth Provenance: PASS

Dataset raw_label.pkl supplies native winners and costs. Labels and performance targets are stripped from forwards. Geometry normalization is fitted on training nodes only and restored from checkpoint buffers. All 108 input/label hashes match manifests. This verifies frozen offline-label provenance, not all original solver recipes or mathematical optimality.

Evidence: performance_targets.py:52; r42_experiment.py:37; r41_frozen_evaluation.py:48; protocol.json.

### B. Metrics and Normalization: PASS

Native ind remains the strict Top1 label. Full candidate order is retained; saved costs match original FP64 values. Oracle is the available pool minimum and SBS the split-specific best fixed solver. ALL percentages are means of per-problem percentages, not ratios of ALL costs or normalized model scores.

Evidence: performance_evaluation.py:18; r40_experiment.py:104; prediction_verification.json. The executor independently recomputed every best-val/test prediction's Top1-3, CE, cost and regret; the reviewer checked representative prediction files and all macro aggregates.

### C. Results and Selection: PASS

The schedule covers complete natural-distribution permutations, including tails. Successful updates reconcile at A 42,660 and B 54,036. Best epochs 22/30 are strict validation-cost minima and were locked before one test per model. All 36 test prediction files contain 1,000 ordered unique samples. No test-based checkpoint reselection was found.

All saved report rows and bootstrap point values match results after rounding. Best validation checkpoint replay produces zero prediction flips and zero logit differences.

Evidence: r42_experiment.py:339; locked_checkpoints.json; test_results.json; validation_replay.json; RESULTS.md; paired_test_bootstrap.json.

### D. Implemented Paths and Numerical Updates: PASS

Edge values enter sparse node messages, not merely attention weights. B has solver queries over node memories without old solver writeback/common-context decoder. Two independent dropout forwards contribute gradients to one optimizer update.

The post-run CUDA test injects infinite gradients once. The real GradScaler skips that update, lowers its scale, retries the identical indices/dropout streams, and calls AdamW once successfully. The first and second dropout outputs differ, while their retry counterparts match exactly. Forty-four core/legacy tests plus this CUDA test pass.

Evidence: relation_encoder.py:31; solver_query.py:39; r42_experiment.py:73; test_amp_retry.py; unit_tests.log; amp_retry_test.log.

### E. Claim Scope: PASS

Only seed2 is claimed; A/B share stopping rules and budget caps, not equal realized compute. Neither run met the preset breakthrough target. The report does not attribute all effects to R-Drop or Q/K/V, and bootstrap intervals are conditional on fixed weights rather than multi-seed robustness. R41's prior use of the same test set is disclosed.

Evidence: protocol.json; RESULTS.md:27; RESULTS.md:38; RESULTS.md:47.

### F. Evaluation Type: PASS / real_gt

Benchmarks use dataset-provided offline solver labels. Synthetic unit-test fixtures are separate correctness checks, not selector benchmark results.

## Source and Review Record

The original launch snapshot remains intact. After all training/test finished, the report renderer received a layout-only patch; its live training plotting function is unchanged. This is recorded in post_training_reporting_patch.json. Model, loss and evaluation sources remain unchanged; test inference was not repeated.

Reviewer ID: 01a0f425-d44a-7470-bad3-1e4908f64b73. The reviewer modified no files and launched no GPU workloads. It checked bootstrap saved-number correspondence, not an independent rerun of the bootstrap.

Supported: R42 implementation, full paired-protocol execution, frozen test evaluation and qualified comparisons.
Unsupported: robust architecture superiority, a +2pp/-10% breakthrough, isolated R-Drop causal gains, or universal label-stability conclusions.
