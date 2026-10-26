# R109 Administrative Launch Plan

This worker owns only the R109 dispatch, wave CLI, completion watcher, their CPU
tests, shell launch entry and this note. Main owns the scientific lock, MODEL_PREP,
PLAN, kernel witness, final specs, readiness and actual start. Import, help,
draft-spec printing and default goal status never launch, freeze, probe, or commit.

## Fixed Transport

| Arm | Slot | Allocation / node | CPU affinity |
| --- | --- | --- | --- |
| A_tied_late | 3090_0 | original job1465 / gpu03 | 12,13,14,15 |
| B_private_late | 3090_1 | original job1465 / gpu03 | 16,17,64,65 |

The private R98/R84/R81 transport retains the exact R107 GPU UUIDs, physical-UUID
mutex inodes and original allocation-owned container. Worker entry is an existing
allocation srun step: --overlap --exact --gres=none --cpus-per-task=4 --mem=9G.
No SSH training, new allocation, new container, cache copy or preprocessing.
Reuse /tmp/shiys_R84_new90k_cache in place and fail on identity/integrity mismatch.
Controllers and CPU PDF stages stay on admin01. W&B stays explicitly offline
with project selector and the configured entity, as in the proven launcher.

The assigned predecessor is the matching original R107 arm, not R108 and not its
GitHub publication. Require that arm's same-lock full TRAIN/VAL/TEST, timing, LIB,
PDF evidence and ordered worker/controller rc0 receipts. The CPU status reader
does not inspect hardware or wait for the predecessor's version comparison/Git.
Resource inspection happens only in the explicitly started controller under the
existing physical GPU mutex; this worker never invokes it.

## Scientific Source Of Truth

Use r109_experiment.required_lock_fields(), including r109_model.SPEC. The
launcher does not create or independently derive the scientific record. Main
adds data_root, prepared_root and primary_cache. Both arms use the full18 frozen
NEW90k 81000/4500/4500 split, logical1024, seed2, 57600 updates, warmup4320,
VAL1440, TRAIN7200, original AdamW 1e-4 to 1e-6, WD1e-4, clip1, dropout.1 and
original R77 CE.35/comparison.35/risk.02/RDrop.35 with two forwards.

The original model is four shared node layers and two joint layers at width128,
with two additional method-conditioned graph layers. A ties late weights; B has
21 global-method-private banks. Original fresh hash is
dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93.
Counts are A=2305033 and B=10220553. Common original tensors and identical late
prototypes give the same initial evaluation function, NOT equal complete tensor
hashes, total/active parameter matching, or FLOP matching.

Latest pre-freeze requested caps are large128/other1024/eval128. They are not
witness approval. CLI generation and lock validation consume the runner CAPS,
so main may change that one configuration after its witness, before freeze.
No automatic OOM shrink or per-arm fallback is implemented.

## CLI Handoff

Run all commands after sourcing conda.sh and conda activate easynco, from
/public/home/shiys/0selection2. Long-running controllers/watchers belong in tmux.

```bash
python -B -m code.V4.r109_launch --help
python -B -m code.V4.r109_wave A_tied_late
python -B -m code.V4.r109_goal --help
python -B -m unittest code.V4.test_r109_launch code.V4.test_r109_goal
```

Default wave output is JSON only, spec_draft=true and main_thread_ready=false.
After main's witness, lock, required source closure and CPU tests are ready,
main explicitly performs the following for each ARM (no action here is executed
by this worker):

```bash
python -B -m code.V4.r109_wave ARM --write-final
python -B -m code.V4.r109_launch prepare --spec SPEC --evidence MAIN_EVIDENCE --runner-tests-passed
python -B -m code.V4.r109_launch start --spec SPEC --evidence MAIN_EVIDENCE --runner-tests-passed
```

SPEC is preparation/dispatch/specs/R109_ARM.json under this model root. Both
arms' evidence must be identical. Optional queue is a one-shot CPU-only wait for
the assigned R107 completion, followed by the same start gate. Final specs do
not themselves grant readiness. Source freeze is one-shot; after ready, do not
edit sources, overwrite specs, refreeze, resume or retry automatically.

Source closure includes r109_memory_witness.py and the now-required
run_v4_r109_witness.sh, without inventing another witness wrapper. The optional
docs/research/post_r108_learning_mechanisms_20261024.md is included only if
available at final-spec freeze. Its absence does not block the initial run;
late arrival cannot silently enter the frozen source set or publication.

## Completion And Publication

GPU stages are train, TEST, timing, LIB(batch2), in that order, followed by CPU
report/PDF. Require all stage and controller rc0 receipts, unique VAL selection
and the same locked checkpoint across full TRAIN/VAL/TEST/LIB/timing/report.
No TEST/LIB reselection, cross-arm checkpoint assembly or parameter-equality
claim is accepted. Timing includes the complete method-conditioned computation.
The four NSS rows retain version/pool/winner metadata and that arm's checkpoint;
CSV parsing supports a UTF-8 BOM. Missing historical evidence stays explicit.

After both arms complete, the watcher runs one CPU-only saved-evidence compare.
Version completion also requires paired/recovery CSVs, comparison provenance,
and a root comparison PDF. The reporter supplies MODEL_COMPARISON.pdf through
comparison protocol's comparison_pdf; R109_COMPARISON.pdf and COMPARISON.pdf
are also accepted root names. Per-arm report PDFs are also
required. Pipeline completion does not imply improved selection or completed
research goal.

```bash
python -B -m code.V4.r109_goal --watch --publish-github --evidence MAIN_EVIDENCE
```

Watch/compare/publication require both main-ready frozen plans. Watcher is quiet
while waiting, one-shot, and stops on saved errors. GitHub publication is only
after the whole version and comparison complete, using the existing
<git-dir>/codex-publication.lock and curated R109 source/docs/JSON/CSV/PDF paths.
No caches, model weights, raw predictions, W&B/credentials or unrelated work
are staged. No publication, commit or push is performed by this worker.

## Verification

CPU administrative contracts passed: launch 9/9 (2.94 seconds), goal 12/12
(0.64 seconds). Five stage CLIs were parsed against the actual runner/reporter,
and launch/goal --help plus shell syntax passed. No model audit was repeated.
No GPU probe, witness, science write, real preparation/start, commit or
publication was performed by this worker. Main reports the final128/1024/eval128
witness passed; scientific readiness and source freeze remain main actions.
