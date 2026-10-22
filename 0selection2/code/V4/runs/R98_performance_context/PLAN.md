# R98: Full Performance Context

Status: implementation, not a measured improvement. User resumed the research
goal on 2026-10-22 (Asia/Shanghai). No result-dependent setting is selected.

## Fixed Decisions

- Use the frozen NEW90k full18 release and the complete 81,000 TRAIN rows per
  task, with unchanged legal candidate pools and native FP64 evaluation labels.
- Reuse R97 input-only statistics, TRAIN-constant column removal, and source IDs.
  Do not supply current-query outcomes, VAL/TEST/LIB labels, or self-context.
- The public frozen TabICL v2 regressor predicts each legal method's expected
  relative gap `(cost - row_min) / row_min`. Its target is raw relative gap,
  not log gap, winner frequency, raw cost, or a post-hoc classification label.
- Fit scalar-target contexts sequentially, with one model resident at a time.
  Every nonconstant target uses every TRAIN row in natural order. Only exactly
  constant TRAIN targets permit a constant-output shortcut.
- Select the minimum predicted gap, with ascending global solver ID resolving
  exact ties. Do not clip predictions or add a learned decision head.
- Use one predeclared public checkpoint and inference configuration: seed 2,
  one estimator, no feature normalization, AMP, no FA3, GPU-only inference,
  no KV cache, and no SGD or fine-tuning. Batch 1024 is a neural-update rule;
  this frozen-context experiment has no neural optimizer updates.
- Complete all18 VAL before locking the single bank and evaluating TEST/LIB.
  No per-task model assembly, TEST-dependent rule change, or parameter grid.
- Report all18 equal-task metrics, paired model/native-winner distributions,
  historical NSS four-dataset comparison, and true fixed-target recovery/loss,
  false picks and net cost relative to R89B/R84A. R97 is a historical
  winner-context reference, not a new compute-matched control.

## Assumptions And Limits

Historical TRAIN cost vectors are authorized context under AGENTS.md. Public
synthetic pretraining is additional knowledge and compute. Expected-gap
minimization need not maximize Top1 or recover rare winners. Previous direct
regression and R92D cost trees were negative; a different public prior is a
hypothesis, not evidence of improvement. Retained old held-out instances have
informed development, so TEST is not a new blind evaluation.

Runtime measurement must include all per-method context construction and
inference, identify cached versus newly computed input features, and never
present cached-feature latency as end-to-end graph selection latency.

No solver execution, new instances, labels, R58 work, or environment-wide
package changes are authorized by this implementation.
