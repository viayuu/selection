# R97: Frozen Tabular In-Context Solver Selection

## Question

Can a classifier pretrained on synthetic tables extract transferable winner
information from the existing input statistics and labeled historical TRAIN
context, without fitting another selector by gradient descent?

This is a different adaptation regime, not another score head, class penalty,
output calibration, or ReLD binary specialist. It covers all18 frozen tasks.
The literature screen is `docs/research/post_r94_tabular_icl_screen.md`.

## Authorized Information Boundary

On 2026-10-22 the user explicitly allowed historical TRAIN labels as forward
context, but prohibited the current query's label. Context remains TRAIN-only;
VAL, TEST and LIB labels never enter fit/context/predict. Current-query costs,
winner, routes and solver outputs remain forbidden input. TRAIN diagnostics
must exclude the query and all copies of the same source/base instance; an
in-context self-prediction accuracy is not a training/generalization metric.

## Fixed Scientific Scope

- Reuse NEW90k frozen splits, legal candidates, native FP64 winner identities,
  TRAIN-fitted SBS, and the existing canonical R92 input-only357 statistics.
- Each task uses its own full81000-row TRAIN context. Preserve the natural
  class distribution; no winner quotas or synthetic class examples.
- Remove only columns constant on that task's TRAIN, preserving column order.
  No label-based selection, test-fitted transform, neural embedding, PCA or
  new solver execution. Current counts are100-217 nonconstant columns/task.
- Use one public, frozen TabICL-family checkpoint and one predeclared setting
  for all18 tasks. The exact source revision, checkpoint/license/hash and
  inference settings must be locked after source compatibility is established,
  before reading any R97 VAL result. No downstream SGD or fine-tuning.
- Seed2. This has no optimizer minibatch or successful-update budget: historical
  context is not a claim of matching scratch-model training compute. Query
  batching, context/cache preparation and inference cost must be reported.
- Return one legal solver by fixed probability argmax with ascending-global-ID
  ties. No output bias, temperature search, per-task model/setting selection,
  solver portfolio or restored deleted candidate.

## Resource Gate, Not Model Selection

R94A/B and R95A/B remain untouched while healthy. Target the first completed
slot only after its entire locked TEST/LIB/report pipeline finishes. Use the
existing allocation-owned container, exact GPU UUID and shared GPU mutex;
no new allocation or ordinary-SSH training.

A single bounded resource witness may use full TSP TRAIN context and32 VAL
input rows, without loading their labels. It checks actual inference, host/GPU
memory and time on the24GB GPU /9GiB CPU-slot budget. Import success alone
does not establish readiness. If full context does not fit, preserve the failure
and stop; do not silently substitute a smaller context or change settings after
observing accuracy. This plan currently authorizes preparation, not an unready
GPU launch.

## Lock And Delivery

Evaluate the frozen full18 bank on VAL, then lock one bank manifest containing
the checkpoint, context identities, column transforms and inference rule.
There is no task-wise VAL hyperparameter selection. Only then read/evaluate
TEST and LIB using that same bank. TEST has already influenced the broader
research and is not claimed to be blind.

Deliver full18 ALL/per-task Top1, mean cost, TRAIN-fitted SBS, Oracle, actual
regret and mean-ratio metrics; NSS four-dataset comparison; fixed36 winner
recovery, false picks and net selection cost; paired selected/native-winner
distribution PDFs with model Top1 titles. Separate full context/preprocessing
time, cold query cost, cached query cost and memory. Do not invent an epoch
learning curve for a frozen model. Any TRAIN diagnostic must disclose its
out-of-context protocol and different context membership.

R89B and R92C are historical references, not compute/pretraining-matched
controls. Imported synthetic pretraining and potentially lossy statistics are
explicit limitations. No claim of improvement without full18 Top1 gain and
non-worsening cost/regret plus genuine recovered winners. Publish one curated
GitHub commit only after the complete version, excluding weights/context arrays.
