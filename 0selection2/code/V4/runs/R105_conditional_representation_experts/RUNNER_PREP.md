# R105 Runner And Reporter Contract

Scope: r105_experiment.py, r105_reporting.py, their two CPU test files, and
this note only. Main owns PLAN.md, protocol_lock.json and all launch/goal
code. The parallel model worker owns r105_model.py and MODEL_PREP.md.
No GPU, scheduler, training, data/label reads, solver runs or commits are
performed by this preparation task. Historical/live modules are read only.

## Early CLI Contract

Root: `code/V4/runs/R105_conditional_representation_experts`.
Arms: `A_static_gate`, `B_input_gate`.
Frozen factory: `code.V4.r105_experiment.load_checkpoint(path, device='cpu')`.
Aliases: `load_model`, `from_checkpoint`. Main import: `required_lock_fields()`.

All R104 flags are retained. Commands below document the eventual pipeline;
they are not launches performed here. Use the other arm independently.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
python -m code.V4.r105_experiment describe --arm A_static_gate
python -m code.V4.r105_experiment train --arm A_static_gate --device cuda:0 --gpu-type 3090 --large-micro-cap 128 --eval-batch-size 128 --wandb-mode offline
python -m code.V4.r105_experiment test --arm A_static_gate --device cuda:0
python -m code.V4.r105_experiment timing --arm A_static_gate --device cuda:0
python -m code.V4.r105_reporting lib --arm A_static_gate --device cuda:0 --lib-batch-size 2
python -m code.V4.r105_reporting report --arm A_static_gate --device cpu
python -m code.V4.r105_reporting compare --device cpu
```

Runner flags: `--root`, `--arm` (required), `--prepared-root`, `--data-root`,
`--cache-root`, `--protocol-lock`, `--device`, `--gpu-type {3090,4090}`,
`--large-micro-cap 128`, `--eval-batch-size 128`, `--log-every`,
`--wandb-mode {auto,online,offline}`, `--resume`,
`--state-hash` / `--expected-state-hash`. Commands: train/test/timing/describe/status.
No budget, coefficient, temperature, quota, routing-calibration or OOM-search
flags are exposed. Authorization and GPU/device binding remain main-owned.

Reporter flags: `--root`, `--arm`, `--device`, `--lib-batch-size {1,2}` (default2),
`--nss-root`, `--historical-root`, `--r89-root`, `--r89a-root`.
Commands: lib/report/compare/status; compare needs no arm.

## Required Main Lock Fields

`required_lock_fields()` returns exactly this scientific subset. Main may add
descriptive/provenance/resource fields without rejection. Nested scientific
values below must match exactly; paths are checked separately by resolution.

```json
{
  "experiment": "R105", "locked": true,
  "arms": ["A_static_gate", "B_input_gate"], "seed": 2, "task_scope": "full18",
  "rows_per_problem": {"train": 81000, "val": 4500, "test": 4500},
  "batch_size": 1024, "logical_global_batch": 1024,
  "max_updates": 57600, "warmup_updates": 4320,
  "eval_every": 1440, "train_eval_every": 7200,
  "large_micro_cap": 128, "other_micro_cap": 512, "eval_batch_size": 128,
  "model_role_arm": "B_execution_anchor", "role_feature_slot": 3,
  "embedding_dim": 128, "encoder_layer_num": 4, "joint_layer_num": 2,
  "head_num": 4, "experts_per_layer": 4, "expert_hidden_dim": 512,
  "trainable_parameters": {"original": 1842953,
    "A_static_gate": 3425561, "B_input_gate": 3425561},
  "expert_init_seed": 105002,
  "gate": {
    "A_static_gate": "Linear(zeros_like(post_attention_nodes)) then softmax",
    "B_input_gate": "Linear(post_attention_nodes) then softmax",
    "initialization": "all router weights and biases zero",
    "all_four_experts_computed": true,
    "A_inactive_feature_weight_parameters": 2048
  },
  "identical_complete_fresh_parameters": true,
  "original_fresh_trainable_sha256": "dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93",
  "current_query_labels_in_forward": false,
  "historical_TRAIN_label_context_used": false, "external_pretraining": false,
  "new_targets": false, "auxiliary_loss": false,
  "optimizer": "AdamW", "peak_lr": 0.0001, "end_lr": 0.000001,
  "weight_decay": 0.0001, "grad_clip": 1.0, "dropout": 0.1,
  "loss": {"CE": 0.35, "comparison": 0.35, "relative_risk": 0.02, "RDrop_final": 0.35},
  "checkpoint_rule": "strict minimum whole18 VAL macro_vs_sbs_pct; first exact tie; common full-TRAIN SBS",
  "inference_rule": "original score MLP utility differences and maximin; no calibration or solver quotas",
  "routing_diagnostic": {"split": "val", "first_instances_per_problem": 128,
    "input_only": true, "selection_use": false, "grouping": "same problem and exact real n"},
  "R58_operations": false
}
```

Additional required paths: `data_root` and `prepared_root` equal CLI paths;
`primary_cache` equals CLI cache root. Defaults are the existing NEW90k data,
R84 prepared root and `/tmp/shiys_R84_new90k_cache`. Never build/copy caches.

## Integration

Source integration is ready for main-owned authorization/launch. No real-data
evaluation, GPU witness, timing, LIB inference, PDF delivery or commit has run.

The private R84 integration retains successful-update accounting, natural
task/row streams, original stateless two-forward R-Drop, exact-row microbatch
normalization, AdamW schedule and strict minimum full18 VAL selection. Train
includes the unique lock and full TEST; explicit TEST reuses saved predictions.
Timing then records the whole standalone model, first128 TEST inputs/task,
one warmup plus three synchronized forwards, excluding I/O and host transfer.
The input-only first128 VAL/task routing diagnostic runs only after model lock,
without selection/calibration and without reading cost/winner/target arrays.
It is executed after successful full TEST inside the test hook, is idempotent,
and does not introduce another CLI stage or modify the inference path.

Model capture follows MODEL_PREP.md exactly:
`routing, outputs = model.capture_routing(raw_inputs)` gives detached FP32
`[128,4,4]` valid-physical-node means. Only entropy of this graph-mean mixture
is reported, never mislabeled mean node entropy. Overall layer means plus
problem/exact-N/layer means, population std/range, argmax coverage and singleton
unavailability are saved in routing_diagnostic.json and CSV/PDF. No special
tokens/padding contribute to the captured mean. Small fixed-sample limits are
explicit. Timing uses a fresh ordinary-forward model, without capture hooks.

Final selected VAL binding must prove checkpoint file SHA, full model state
(parameters and buffers), protocol state and successful update, leaving original
capture identities/archives unchanged. Null selected TRAIN file SHA is allowed
only with the same proof; no administrative historical repair is applied.

Reporter stages consume the per-arm R105 frozen factory and saved evidence via
private adapters, never by mutating shared historical reporting globals.
Five PDFs, original NSS4, historical R89A/B/R84A, fixed36 recovery/false picks/
net costs, same-axis choice/native-winner plots and task-equal metrics remain
required. Equal total tensors and common fresh initialization are not equal
functional freedom: A has2048 inactive zero-gradient router feature weights.
Both arms compute all four experts; no sparse/equal-FLOPs or efficacy claim.
W&B defaults offline explicitly; no claim about an unverified online service.

## CPU Verification

The first focused runner/reporter suite passed14/14 in3.140s before any PDF
work. This covers main-lock compatibility, original engine isolation/caps,
strict per-arm trained-state restoration (including geometry buffers), complete
fresh cross-arm equality, successful-update row weighting, exact capture proof,
private reporter/factory/root isolation, timing/routing metadata and fixed-target
recovery arithmetic. An additional real capture/timing-adapter CPU check is
included in the final suite, which passed15/15 in3.926s. Its synthetic CPU timing is only a test of the
adapter, not a claim of measured experiment/GPU latency.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m unittest code.V4.test_r105_experiment code.V4.test_r105_reporting -v
```
