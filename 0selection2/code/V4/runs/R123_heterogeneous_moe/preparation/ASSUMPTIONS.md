# R123 CPU preparation contract

Base commit: `04866414831d25330b0d2e575e4b63c68b4196d0` (HEAD/packed-refs read,
no Git command). No R120 files, GPU jobs, training, or scientific mechanism
are changed. Astra owns the final MoE/re-ranking mechanism.

- Provisional engineering trio: R77B (handcrafted solver features and original
  composite/R-Drop objective), R80A (same architecture, CE-only objective), and
  R83 (model-only source-code representation and original composite objective).
  R80C/D are inventoried alternatives, not chosen using held-out scores.
- Use each run's existing VAL-selected `best.pt`; do not reselect checkpoints.
  Only protocol, TRAIN membership, checkpoint loading/completion and source
  provenance are examined. No TEST metrics or prediction files are read.
- Current metaTRAIN is restricted to the frozen new90k TRAIN81k of every
  routing problem. Full historical TRAIN populations, including populations
  used to fit input normalization, conservatively count as exposed even if a
  selected checkpoint predates some optimizer visits.
- Source identity is `(problem, declared immutable generator-shard SHA256,
  row-within-shard)`. Bare split/global row numbers and file paths are not
  identities. Verify old subset lineage against the completed generator and
  validate hashed per-release TRAIN metadata/index mappings before joining.
- Certification is explicitly source-record-level. No new all-input content
  hashing or exhaustive semantic-base duplicate audit is performed. Missing
  explicit base-group/content identities remain a recorded limitation, never
  a stronger claim of unseen mathematical instances or all possible copies.
- Retain all legal candidates with explicit `reld_mt -> RELD_MOEL` aliasing;
  never combine two ReLD costs, restore removed candidates, or collapse expert
  checkpoints. Preserve each checkpoint's own TRAIN-fitted geometry buffers.
- Optional score centering/z-scoring is per query across candidate scores,
  label-free, and explicitly requested; default raw scores remain unchanged.
  No temperatures/calibration or normalization are fitted on VAL/TEST.
- R94/full81k experts are negative exposure controls, not unseen-data donors.
  Old VAL/TEST membership has already been used during research; it is not
  described as a fresh blind evaluation. Neither split becomes metaTRAIN.

Planned API: `prepare()`, `load_meta_indices(problem)`, `source_record_ids(...)`,
`align_candidate_scores(...)`, `normalize_scores(...)`, `load_expert(...)`, and
`ExpertAdapter.predict(...)`. Artifacts stay under this preparation directory.
