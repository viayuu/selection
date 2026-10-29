# R123 learner decisions before implementation

Only new r123_model.py/r123_experiment.py and their tests plus learner* outputs
are owned. Existing meta_data artifacts are immutable. No GPU execution,
solver, Git command, parent objective/runner edits or training is authorized.

- The explicit astra_followup.md supersedes the earlier combined action-loss
  proposal. One primary model predicts each frozen expert's native correctness
  independently using sigmoid/Brier; multiple correct experts are all 1 and
  all-wrong rows are all 0. No cost-weighted primary or auxiliary loss.
- Bank order is fixed R77B, R80A, R83. The input-only gate uses a separate
  four-layer 128-wide InstanceEncoder plus learned expert-query attention.
  Experts do not provide logits, hidden states, actions or costs to the gate.
  Deterministic gate ties use bank order, then only the selected frozen model
  executes per instance. Cached evaluation selects that same single model's
  original full-pool scores, never averages incomparable expert logits.
- Historical models/normalizers are frozen. The new encoder copies only
  matching encoder parameters from R84's verified seed2 SCRATCH initialization;
  new query/readout parameters are seed2 scratch. R84 TRAIN-only geometry
  buffers are reused; no trained historical encoder initializes the gate.
- MetaTRAIN is exactly the immutable certified 10,886 current TRAIN81k local
  positions per task. All 18 current TRAIN input_metadata.npz headers were
  inspected cheaply: source_index, scale, coord_dist, distribution,
  capacity_type only; no base, parent, copy, identity or hash field. No stronger
  content/base-copy certificate or cohort representativeness is claimed.
- Cache schema is reusable for R124: atomically completed per-expert NPZ,
  shared load_scores returns float32 logits[N,E,K], current local row
  indices, declared release source references and source-record identities.
  It pins bank/checkpoint hashes, exact pools/IDs, input metadata and row order.
  TRAIN means metaTRAIN, VAL means all 4,500; TEST is forbidden before the
  unique completed-run VAL checkpoint lock. No raw inputs are regenerated.
- Bank cache and live bank use FP32 frozen-expert execution, with stable
  ascending solver-global-ID score ties. Cached scores are original FP32,
  not native costs. Original float64 routing costs/native winners are separate
  supervision and evaluation data; no FP32 re-argmin label reconstruction.
- Single fixed run: seed2, logical batch1024, 7,200 successful updates,
  warmup540, eval720, AdamW peak1e-4/end1e-6 cosine, WD1e-4, dropout0.1.
  Tasks alternate equally. Microbatches use true sample-count normalization;
  nonfinite gradients fail without an optimizer update. Resume pins RNG and
  shuffle-stream cursors. No TEST-driven checkpoint/hyperparameter choice.
- Parent bounded-cap amendment: 3090 defaults TSP/CVRP128, other tasks512;
  4090 defaults256/1024. No capacity sweep or automatic tiny initial caps.
  Only an actual CUDA OOM permits one same-batch/RNG retry at half cap, before
  any optimizer step; observed caps are saved/restored with runtime state.
- Checkpoint order is lowest full18 VAL mean instance actual regret, then
  highest native Top1, then earliest update. This quality selection rule is
  explicit even though training predicts native correctness using Brier.
  Historical held-outs were development-exposed, not fresh blind sets.
- Learner outputs default beneath preparation/learner_run. Shared predictions
  live at R123/prediction_cache, generated ONLY by frozen prediction_cache.py;
  CPU smoke outputs are distinctly incomplete and cannot train the formal run.
  A launcher must wait for parent's R120 TEST/LIB/report release before GPUs.

Scientific sources: docs/research/return18_major_routes_20261030/astra_followup.md,
HANDOFF.md section3; frozen post_r119_global_reassessment_20261028 remains the
primary research assessment. R123 is a separate historical-mapping route, not
an internal FFN-MoE and not a demonstrated improvement.
