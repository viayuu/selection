# Principal R121 Learner, Fixed Before Training

Primary science is frozen post_r119 HANDOFF D1. This is one principal full18
model, not a Ridge/MLP response pilot. No model training/GPU is run by preparation.

- Retain the original ScoreDifferenceSelector raw coordinate/matrix encoder,
  candidate features, score MLP, pair differences and maximin inference. Add one
  centered log-advantage potential head to the SAME per-method features. Both
  endpoint and family-response losses update the shared raw-query representation.
  Responses are differences of this SAME potential, not a separate pair network.
- The immutable natural release still contains81000 TRAIN rows/task. Natural CE
  samples81000 minus every identifiable heldout source-record association, using
  each task's actual TRAIN `input_metadata.npz/source_index` mapping. Endpoint
  `local_index` is NOT treated as the current row number. Every matching row is
  excluded, including duplicate source records. Closure includes all known
  selected geometry/base/source associations across tasks; any related family
  TRAIN endpoint is rejected rather than given gradients. New full-pool family
  labels require a COMPLETE separate89-recipe collection. Old files/splits/labels
  are unchanged; no new TEST label or scenario relabeling occurs.
- Every logical update has1024 native input rows:820 natural CE rows and102
  (original,changed) family pairs =204 endpoint slots, approximately4:1. One task
  per update; every shuffled18-task cycle visits ALL18 once. Microbatches preserve
  whole pairs and independently normalize CE over820 rows, endpoint Huber over204
  slots and response Huber over102 pairs. No independent-microbatch objective.
- Loss is `1*natural_native_CE + 1*endpoint_Huber + 1*family_response_Huber`.
  Advantage units are `(log C - mean_legal(log C)) / 0.01`; Huber delta1 in these
  units. Endpoint loss averages legal candidates. Response loss uses paired
  differences of those same centered potentials. No R-Drop, risk, ranking,
  inverse-frequency or other extra term is silently added.
- Seed2; AdamW lr1e-4 ->1e-6, weight_decay1e-4, grad_clip1, dropout0.1.
  Logical batch1024; max14400 successful updates; warmup1080; full18 VAL every720.
  Unique checkpoint: minimum natural full18 VAL macro actual_regret_pct, then
  maximum Top1, then earliest update. No TEST/LIB rule or checkpoint selection.
- Reuse original R84 TRAIN-only geometry statistics and untrained fresh seed2
  initialization for ALL common parameters, never a trained selector checkpoint.
  The extra advantage head is fresh seed2 initialization. No special TSP role,
  solver trajectory, candidate probing, family ID, current cost or winner in forward.
- Only `family_train` endpoints/edges enter family losses. `family_val` is a
  TRAIN-derived diagnostic whose identifiable base/source records and known
  selected-geometry associations receive no supervised optimization gradients
  from natural CE or family losses. It is NOT an independent new benchmark or
  proof of exhaustive content-copy novelty. Existing full81000 TRAIN geometry
  normalization and TRAIN-fitted SBS are explicitly retained for history; this
  is not a claim that heldout inputs never affected normalization. Unknown
  content aliases are not exhaustively searched. Official VAL/TEST are unchanged.
- `learner_natural_isolation_v1.json` seals actual per-task metadata file/array
  hashes, excluded source IDs/local rows, eligible row hashes/counts and known
  cross-task associations. The learner protocol references its immutable seal.
  This is a source-record isolation correction before training, not a new model,
  architecture, extra ablation, or change to the fixed schedule/initialization.
- At14400 updates each task receives800 logical updates:656000 natural row
  exposures and163200 family endpoint slots (duplicates are counted honestly).
  This is a deliberately different population/exposure from historical models,
  not an update/data-matched control. No empirical benefit is presumed.
- Once locked, the same checkpoint performs all18 TEST and native TSPLIB/CVRPLIB.
  Existing saved NSS4500 references may be reused with identity checks. Report
  full VAL/TEST/LIB metrics and native winner vs model selection PDFs via existing
  reporters. No taskwise checkpoint assembly or TEST-guided rule changes.

Actual training is additionally gated by main's R120 final TEST/LIB/report
resource release and an explicit physical GPU/CPU binding. Labels must be complete
first. CPU fixture forwards/backwards in unit tests are not an actual train run.
