# R106 Administrative Handoff

Status: implementation / CPU evidence only, NOT main-ready, NOT started.
Main owns PLAN.md, the scientific lock, the single bounded Sol review, final
source freeze, readiness, hardware witnesses, actual launch and publication.
This worker owns only r106_launch.py, r106_goal.py, r106_wave.py,
run_v4_r106_launch.sh, test_r106_launch.py and this file. Other agents' changes
are preserved. No old Rxx source, source protocol, dataset or worker is edited.

## Resources

These bindings were supplied as verified by main, not probed by this worker.

| Arm / run ID | Slot | Complete UUID | Exact CPUs |
| --- | --- | --- | --- |
| A_frozen / R106_A_frozen | 4090_0 | GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be | 0,1,2,3 |
| B_finetune / R106_B_finetune | 4090_1 | GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d | 4,5,72,73 |

Reuse job1475/gpu05 and its allocation-owned container:
`76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6`.
Each process has four CPUs / nine GiB. No new allocation, ordinary SSH training,
GPU substitution, healthy-worker restart or R1053090 intervention is allowed.

Receipts/specs are independent at
`runs/R106_native_encoder_adaptation/preparation/dispatch`.
The private transport retains the existing R81 per-UUID GPU mutex paths and
`runs/R98_parallel_dispatch/gpu05.lock`; it reads both shared and new dispatch
CPU reservations. An occupied assigned quartet waits; it never selects other
CPUs or restarts another worker. The original mutex inodes are not replaced.

`normalize_gpu_uuid` accepts full canonical UUIDs with or without Torch's
`GPU-` prefix; partial strings are rejected. `verify_uuid_binding` requires both
the complete normalized Torch UUID and true NVML GPU-UUID to equal the assigned
identity. This is CPU validation logic, NOT an observed hardware witness.
NVML process IDs may be container namespace IDs, not host IDs. Main must retain
the `/proc/<hostPID>/status` NSpid mapping and container cgroup when attributing
a real worker; never compare a bare NVML PID to host PID by assumption.

Frozen inputs are reused in place:

- Data: `/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`.
- Prepared identities: `runs/R84_new90k_attention/prepared`.
- Source manifest: `runs/R84_new90k_attention/input_cache`.
- Existing node-local raw inputs: `/tmp/shiys_R84_new90k_cache`.

The reuse-only worker helper checks the existing cache identity, file sizes
and declared small-file hashes. Missing/inconsistent files fail without copy,
repair, feature extraction or generation. No cached native features are used.

## CLI And Science

Stable experiment CLI is aligned to RUNNER_PREP.md: positional `train`, `test`,
`timing`; reporter positional `lib`, `report`, `compare`. Exact stage order is
`train -> TEST -> timing -> LIB -> PDF`; PDF and saved-data compare run on CPU.
There is no input_features, routing diagnostic or predecessor-completion stage.
R105's healthy two3090 training is not a predecessor gate.

Experiment arguments: --root, --arm, --prepared-root, --data-root, --cache-root,
--protocol-lock, --gpu-type4090, --large-micro-cap64, --atsp-micro-cap128,
--other-micro-cap256, --eval-batch-size128, --device and --wandb-mode offline.
These caps are tentative until main's ONE necessary input-driven memory witness
fixes the final common caps in the lock. Specs derive caps from that lock;
runtime mismatch is rejected, not automatically shrunk or refrozen.

The launcher checks compact administrative lock fields. Before readiness it
also calls the real runner's CPU `protocol_lock` against its complete
`required_lock_fields(**final_caps)` subset, parses every actual arm-stage CLI
and the compare CLI, and requires all exact named team files. Missing reporting
code remains pending, not guessed ready. Detailed science stays runner-owned.

Both arms retain original full18 pools/targets, seed2/logical1024,
57600 successful updates, warm4320/VAL1440/TRAIN7200, ordinary AdamW core1e-4
with B native0.1 effective multiplier from update1, dropout0.1 and the original
loss. They start from the original fresh R84 selector, three identical seeded
projections and all three pinned R102B pretrained native encoders. Both use
the same masked native execution and same scoped deterministic native graph
shared by the two selector dropout forwards within one microbatch. No extra
freeze stage, objective, label, route, solver, retrieval or quota is introduced.

Donors stay online and eval-mode in both arms for TRAIN/VAL/TEST/LIB; only B
allows native parameter gradients. The actual model initialization counts are
used, not R105 counts. Expected core1941257 / total11965097 are not hardware or
efficacy evidence. Complete fresh tensor hashes match across arms; trainables
and training compute intentionally do not. Pretraining overlap stays unknown.
Native-online R106 has no relationship to R102 cache-to-online report recovery.

W&B remains enabled in explicit offline mode, project `selector`, entity
`yjkds-southern-university-of-science-technology`. Main supplied existing
no-credential evidence. This worker neither searches for credentials nor claims
an online W&B session, and never prints keys.

## Readiness And Completion

`r106_wave <arm>` writes only `dispatch/spec_drafts/R106_<arm>.json`.
`--print-only` writes nothing. `--final` requires the main science lock and
writes `dispatch/specs/...`; a final spec still says main_thread_ready=false.
Neither operation creates ready.json, a source freeze, a protocol or a worker.
Existing unequal specs are preserved and rejected, never overwritten.

Only main may call `prepare` with nonempty --evidence and --runner-tests-passed.
All exact team sources, native Python definitions, PLAN.md, RUNNER_PREP.md,
this handoff, main protocol and original prepared identities must exist.
The source closure is computed statically; test names are exact, not globbed.
Plan and ready bind exact source hashes, exact argv, evidence and cache-reuse
policy. Any source edit/removal/set change blocks later start/watch; no implicit
refreeze. `start` is one-shot and cannot silently resume/retry old output.

`r106_goal` default status reads saved JSON/CSV/PDF only. It imports no model,
torch or arrays and does no launch, notification, GPU or Git work. Watch is
quiet at default120s and stops on actual current failed/error states or nonzero
exit codes. Scientific stage strings, pending_report and historical-error
notes are not nested transport receipts or current failures. Malformed saved
evidence prevents completion but is not invented into a running-stage failure.

Whole-version requires BOTH complete fixed-budget arms, exact unique whole18
VAL cost locks, same-lock full TRAIN/VAL/TEST, all18 actual online donor+selector
timing, LIB49/100, fixed36, four same-model NSS rows and all five PDFs. Missing
saved NSS references may be explicit missing rows, never fabricated scores.
TRAIN and VAL capture file SHA may be null only through the runner's exact
parent model-state/protocol-state/selected-step proof; immutable archive
identity JSON is compared without rewriting it. VAL official identity binds
the final checkpoint SHA. CPU status never loads or repairs a checkpoint.

Each timing row must have real online native+selector measurements, one donor
call per measured forward and warmup, native output FP32, no cache-hit total,
and the selected full-model tensor hash. Timing is batch128 GPU-resident
amortized model time, not singleton or I/O-inclusive system/solver latency.

After both arm deliveries, one saved-data CPU `r106_reporting compare` produces
paired_changes_B_vs_A.csv, target_recovery_B_vs_A.csv and
target_recovery_summary_B_vs_A.csv under comparison_artifacts, with the exact
two locks. Pipeline completion never asserts the research goal is achieved.

Publication is opt-in only after the WHOLE version and comparison complete.
The private stable publisher retains the shared codex-publication.lock,
main-branch / empty-index gates, a single curated commit/push and permanent
failure receipts. Only exact frozen R106 source names and curated R106
JSON/CSV/PDF/Markdown are eligible. Arrays, weights, caches, predictions, raw
costs/labels, credentials, tokens, symlinks, logs and unrelated dirty work are
excluded. No WIP commit, index reset, old-file revert or automatic retry.

## CPU Evidence And Main Remainder

Initial base suite:13/13 passed in0.156s. It verifies pure imports/status,
resource command construction (mocked, not executed), exact bindings, UUID
normalization, immutable specs/source freeze, reuse-only cache and authorization.
Final expanded saved-evidence and real-parser suite:27/27 passed in3.634s, including
six actual runner CLIs, four actual arm reporter CLIs and compare CLI parsing.
Publication tests mock one whole-version commit/push and rejection of unrelated
staged work. No actual Git write, stage function, model construction or GPU API
was executed. The full runner scientific lock and mismatch path use tiny
temporary JSON only. bash -n passed for the wrapper.

The latest read-only named-file inventory found only main's protocol_lock.json
missing. This is not a final source freeze, hardware proof or main readiness.

Reproduce CPU checks only:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B -m unittest code.V4.test_r106_launch -v
bash -n code/V4/run_v4_r106_launch.sh
```

Main still must finish model/runner CPU evidence, run its ONE bounded Sol
review, retain the model's masked/exact-N CPU agreement evidence and perform
the input-only memory/gradient witness, fix caps and science lock, inspect source/CLI
agreement, then explicitly prepare and authorize actual launch. Actual Torch/
NVML UUID binding, namespace-to-host PID mapping, CPU affinity, native A-frozen/
B-update behavior, cache identity and offline W&B receipts need main's limited
startup proof. This worker has performed NONE of these actual GPU witnesses.

Main-only eventual administrative entrypoints, NOT executed here:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
python -B -m code.V4.r106_wave A_frozen --print-only
python -B -m code.V4.r106_wave A_frozen --final
python -B -m code.V4.r106_wave B_finetune --final
# Only after main's final lock, CPU evidence and single bounded review:
bash code/V4/run_v4_r106_launch.sh prepare --spec code/V4/runs/R106_native_encoder_adaptation/preparation/dispatch/specs/R106_A_frozen.json --evidence "$R106_READY_EVIDENCE" --runner-tests-passed
# Same prepare for B_finetune, then main alone may use start for each spec.
python -B -m code.V4.r106_goal --watch --evidence "$R106_READY_EVIDENCE"
# Optional, only after the complete whole-version gate:
python -B -m code.V4.r106_goal --publish --evidence "$R106_READY_EVIDENCE"
```
