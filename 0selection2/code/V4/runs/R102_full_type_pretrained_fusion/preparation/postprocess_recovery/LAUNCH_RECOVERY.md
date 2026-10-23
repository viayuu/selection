# R102 POSTPROCESS RECOVERY Administrative Entry

This is a new, explicitly authorized postprocessing attempt, not a successful
original route or an automatic retry. Main alone launches execution. Do not run
the old launch/ready/start/watch commands or change old failure receipts.

## Paths And CLI

- `ROOT=code/V4/runs/R102_full_type_pretrained_fusion` (canonical absolute root).
- `BASE=ROOT/preparation/postprocess_recovery`; `VIEW=BASE/report_view`.
- Container entry: `bash code/V4/run_v4_r102_gpu_recovery.sh --arm A --stage gpu --evidence "<main evidence>"`; use B for B.
- CPU report entry: the same wrapper with `--stage report`; no GPU is exposed.
- Read-only status: `python -B -m code.V4.r102_goal_recovery`.
- New quiet watcher: `python -B -m code.V4.r102_goal_recovery --watch --interval 120 --evidence "<main evidence>"`.
- Add `--publish` only when main explicitly authorizes ONE curated whole-version publication. Without it there is no Git activity.
- `--compare --evidence "<main evidence>"` is the alternative explicit one-shot saved-evidence CPU comparison, not an inference command.

All commands must first source
`/public/home/shiys/miniconda3/etc/profile.d/conda.sh` and `conda activate easynco`.
Use tmux for authorized long-running work. This document does not launch anything
and deliberately supplies no Slurm, queue, ready-generation or Git commands.

## Readiness Contract

Main authorizes the one explicit attempt with `--evidence` after the science
helper's CPU checks and actual107-array proof pass. There is NO new ready file,
source-freeze command or publisher/PDF prerequisite for starting GPU recovery.
The GPU entry does not import the delivery gate. Main owns the original
allocation/container/CPU4/mem9G step and tmux; this wrapper never dispatches it.
The entry opens the existing R81 mutex without creating/replacing it, verifies
the inode across acquisition, and records its device/inode in the new receipt.

Entry source provenance includes the entry/wrapper, science helper,
`code/V4/r102_cache_identity_probe.py` and `run_v4_r102_cache_probe.sh`.
Final saved-evidence compare/publication records the completed administrative
helpers/tests/docs separately in `BASE/source_provenance.json`. This late
delivery snapshot is not a prerequisite for the first GPU attempt.

## Explicit UUID Format Correction

The original GPU preflight failed before origin/shadow/TEST creation with
`Actual CUDA UUID differs from the original assignment`. After the authorized
UUID normalization CPU fix, main may explicitly invoke the SAME GPU entry with
`--uuid-format-correction --stage gpu --arm A|B --evidence "<main authorization>"`.
This calls the same helper `gpu`, never TRAIN, and claims only
`BASE/{arm}/gpu_uuid_format_correction_entry.json`. It cannot replace/retry that
new receipt or overwrite the original failed `gpu_entry.json` or GPU log.

Main first writes `BASE/uuid_preflight_observation.json` with schema
`r102_uuid_preflight_observation_v1` and `arms.A`/`arms.B`. Each arm contains
`entry: {path, sha256}` for its exact original failed receipt,
`log: {path, sha256}` for `BASE/A_gpu.log` or `BASE/B_gpu.log`,
`error: "Actual CUDA UUID differs from the original assignment"`, and
`origin_absent: true`, `shadow_absent: true`, `test_outputs_absent: true`.
The top-level `main_authorizes_one_explicit_format_correction` must be true.
Paths are canonical absolute paths. Both old receipts must be failed/exit1 and
both logs must contain the exact error. Own-arm absence is checked before continuation;
delivery rechecks the historical observation and old receipt/log hashes, NOT
absence of the now legitimately created recovered outputs. The gate accepts a
completed correction only with these links verified, and discloses the original
failed preflights in `preserved_uuid_preflight_failures`; they never become
successful original attempts. A failure of the correction remains terminal.

## Explicit Post-Test Timing Continuation

Main has separately authorized `--stage post-test --arm A|B --evidence "<main authorization>"`
for the exact recovery-only timing exception
`R102 cached-vs-online drift; preserve first error, do not change inference/grouping`.
This maps to science helper `finish-gpu`, uses the SAME original GPU reservation,
and claims only `BASE/{arm}/post_test_entry.json`. It never invokes TRAIN or TEST.
The unique original-lock `VIEW/{arm}/test_results.json` must already be complete.
The new receipt preserves hashes of that saved first TEST, `first_error.json`,
the failed UUID-correction entry, the first GPU preflight entry/log and observation.
These files must remain unchanged across the explicit timing/LIB continuation.
The new entry remains one-shot even if it fails; no removal, renaming, old-error
rewrite or automatic retry is authorized.

Only the exact declared post-test chain can make its earlier timing exception a
disclosed historical failure rather than a fatal delivery error. Delivery still
requires real same-lock GPU completion, original all18 online/cache timing,
native LIB, CPU reports and all PDFs. `preserved_prior_recovery_failures` lists
all earlier failures without turning them into successful original attempts.
The original timing procedure and model/buffer/inference validation remain
unchanged. New all18 `timing_observations` disclose numerical differences;
sampled pick agreement or FP16 bits never certify whole-TEST online equivalence
or bitwise exactness. The original `timing_checks` and first error stay verbatim.
Science completion must declare `completion_kind=post_test_original_timing_observation`,
`test_rerun=false`, `online_efficacy_bitwise_certified=false`, `numerical_boundary`,
the SHA-bound `prior_timing_failure` and all18 `timing_observations` artifacts.
Every observation is explicitly limited to first128 TEST rows/task. Changed
picks or non-identical scores/FP16 bits are reported, not hidden or turned into
an extra rejection gate. A new `post_test_error.json` is always fatal.
CPU report metadata also binds `recovery_numerical_boundary`, the same all18
observations and `numerical_boundary_artifacts` (note/PDF), with
`online_efficacy_bitwise_certified=false`; these are curated with the five
required original model PDFs. No extra helper CLI flag is needed for `finish-gpu`.

## Exact Resource And Failure Gate

GPU work is allowed only in the existing job1475/gpu05 container
`76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6`,
CPU4/mem9G per arm. A: slot4090_0, CPU0,1,2,3,
UUID `GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be`.
B: slot4090_1, CPU4,5,72,73,
UUID `GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d`.
The entry reuses the existing R81 UUID mutex by inode; no new GPU lock namespace,
Slurm allocation, container, GPU probe, process termination or queue operation.
Healthy R1053090 work is unrelated and must not be touched.

Both original workers must remain failed in TRAIN phase, with their original
failed allocation controllers unchanged, because
the first TEST encountered `Full frozen R84 header identity differs`; TRAIN is
already57600 complete. A's unique VAL lock is update33120 / SHA
`10694b4d0fed80f1baec0d5e7de15cceec210eca5dfa846623783c4b9d6e7a2c`;
B's is update20160 / SHA
`7df434c762a0e0b7726ab67211f63f51b55d1f86d108ec307a0aede0060f921f`.
This is NOT R99's timing failure. No original full TEST prediction may exist.

## Receipt And Delivery Contract

The sibling `RECOVERY_CONTRACT.md` defines the science worker's final artifacts.
Administrative entries consume `BASE/{A,B}/gpu_complete.json` and
`report_complete.json`, and write separate `gpu_entry.json`/`report_entry.json`.
Each completed science stage carries the original `checkpoint_sha256`,
`state_hash`, `selected_update`, `origin` artifact, `report_view`, `complete=true`
and `exit_status=0`. Administrative entries additionally retain the full
original `checkpoint_lock`. No entry receipt, including a failed/partial one, is replaced
by a new attempt. Original worker/controller SHAs and all original failure/lock
SHAs are recorded and rechecked. CPU reports use only saved results.
Main may also invoke the science helper's `report --root <ROOT> --arm A|B
--device cpu` directly; the CPU report entry is optional. GPU administrative
entries and both science GPU/report completion receipts remain mandatory.

The new watcher writes its own receipts only under BASE, including
`BASE/watch_receipt.json` and `BASE/goal_progress.json`. It waits for BOTH new GPU entries and CPU reports,
actual same-lock full18 TEST, full TRAIN/VAL metrics, native LIB49/100, verified
online timing, fixed36 targets and all FIVE PDFs per arm. Then it runs compare
once (`BASE/compare_entry.json`, science receipt `BASE/compare_complete.json`)
against saved evidence in VIEW. Only complete
comparison PDFs/metadata unlock an explicitly authorized whole-version
publication at `BASE/github_publication.json`. Original status/error/worker/
controller/test_started and stopped goal watcher remain byte-for-byte unchanged.

Publication contains original scientific sources, first-failure JSON, proof and
manifests, new helpers/tests/docs, recovered aggregate PDFs/CSVs/metadata and
source provenance. It excludes weights, raw arrays, labels, per-instance
predictions, caches, credentials, unrelated dirty files and unsafe old launch
receipts containing tokens. No WIP commit; delivery does not claim the research
goal has been achieved or that the original route completed successfully.
