# R127: R83 model-only source selector without MVMOE

One fresh full18 seed2 model, concurrent with R125 and R126 without modifying
or stopping them. Start from the original frozen new90k B-filtered single-reld_mt
scenario, not either earlier candidate-deletion overlay. Remove only MVMOE in
CVRP and all15 MVRP variants. Keep BQ, LEHD and MoSES_RF wherever originally
legal; retain the existing B8 exclusions and the MOEL-only reld_mt alias.

Retain all81000 TRAIN /4500 VAL /4500 TEST instances per problem, including
former MVMOE winners. Mechanically rebuild FP64 winner, ranking, Oracle and
TRAIN-fitted baselines. Do not rerun solvers, regenerate Embeddings, change
input geometry, or overwrite any original data. TSP8 /CVRP9 /ATSP5 /
non-B MVRP5 /B MVRP2 candidates. Original loss/metrics support the two-candidate
case; do not add special targets, quota rules or per-task inference tuning.

Use the actual R83 model-only source representation: encoder and decision model
code only, no inference/config/train/environment code. Frozen real R83 vectors,
learned role projections/fusion and method IDs; GLOP has its recorded ID-only
exception. Four-layer128-d instance encoder, two synchronous dual-stream layers,
original QKV decoder/shared score MLP/score differences/maximin.

From-scratch original common seed2 initialization and frozen R84 TRAIN-only
normalization. Logical batch1024, TSP/CVRP micro256, other-task micro1024,
eval128. AdamW peak1e-4/end1e-6 cosine, warmup4320 successful updates, wd1e-4,
dropout.1, gradient clip1. Original CE.35/comparison.35/risk.02 and RDrop ramp.35.
Fixed57600 successful updates, full VAL every1440, full TRAIN diagnostics every7200.

Select strict minimum full18 VAL macro_vs_sbs_pct using this candidate scenario's
full-TRAIN SBS; first exact tie wins. Then freeze one checkpoint for full TEST,
TSPLIB and CVRPLIB. No test-based selection, per-task model mixing or calibration.
Use an idle allocation-owned3090 and original per-UUID mutex, persistent tmux,
W&B with explicit offline fallback, and an automatic TEST/timing/LIB/PDF/publish
pipeline. Deliver standard paired selection/true-winner PDFs, full18 macro and
per-task metrics, same-pool fixed NSS diagnostic where available, and timing.

Candidate deletion changes labels, Oracle, baseline strengths and task difficulty.
An increased Top1 alone cannot establish an architecture-family imbalance cause.
R126 uses the same architecture/input rows but removes a different method;
R125 also differs in source scope. Historical R83 used older70k data. Keep these
comparison boundaries explicit. Publish once the complete version is finished.
