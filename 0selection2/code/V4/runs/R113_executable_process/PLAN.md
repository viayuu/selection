# R113 Executable Process Runner

Scope: implement major review Section 5, especially the three required arms in
Section 5.5. The user has authorized the code, training and new TRAIN process
supervision. This handoff does not launch jobs, call solvers, allocate resources
or publish Git; the main thread schedules formal execution after its bindings.

## Fixed Design

- A_scratch, B_snapshot, C_transition use the same complete selector and fresh
  seed2 backbone / seed113002 process initialization. Only B/C load their own
  newly pretrained processor. No old trained selector is initialization.
- Process pretraining extracts instance_encoder.processor from the complete
  initialize_shared result, never an independently seeded processor constructor.
- B independently predicts each snapshot from input/time and the initial state;
  C feeds back its own predicted state. Both supervise identical typed states
  and posthoc scalar deltas, with all seven loss components weighted equally.
- Deployment and all selector finetuning use transition mode, initialized from
  inputs. Teachers, true hints, costs and winners never enter model forward.
- Two raw graph-stem layers supply step context. Candidates read predicted
  state trajectory tokens only; no raw-pool or old-score shortcut.

## Budgets And Data

- Process: immutable process_train_v1/manifest.json, complete 18 x 81000 TRAIN
  only; batch512, 8640 successful updates (480/task; 245760 exposures/task),
  AdamW3e-4 to1e-5, warmup432, weight decay1e-4, clip1.
- Process order: seed2 shuffled task rounds; shuffle shard order and rows within
  each shard per task/cycle, carrying tails. This locality-aware process stream
  is shared by B/C; it is not claimed to be the selector's full-row shuffle.
- Process provenance retains TRAIN row, source/base index, raw input hash and
  input-derived sequence. The reader refuses building/subset manifests and
  verifies each immutable shard hash on first read. It never opens native labels.
- Selector: frozen R84 TRAIN81000/VAL4500/TEST4500, original candidate exclusions
  and aliases, original R77 two-forward composite objective; batch512, 57600
  successful updates, warmup4320, AdamW1e-4 to1e-6, dropout.1, RDrop.35.
- Select maximum full18 VAL macroTop1; exact ties use minimum macro actual
  regret, then the earliest checkpoint. TEST and LIB cannot select parameters.

## Main-Thread Bindings

The main thread must provide protocol_lock.json with the exact fields printed
by describe, plus complete-manifest SHA256, data/preparation/cache/pretrain
paths, per-problem pretraining and finetuning microbatch caps, and eval batch.
The draft is deliberately unlocked. No GPU-name inference or allocation occurs.
Changed bindings require a new output version, not overwritten prior evidence.

## Delivery And Checks

CLI: pretrain, train, test, timing, finalize, describe, status. train includes
locked TEST; finalize idempotently performs TEST, input-only timing, both LIBs,
and offline PDF/CSV. Each checkpoint embeds complete model weights; restoration
does not open external processor artifacts. NSS uses saved aligned references
only; absent references remain MISSING. ALL is problem-equal, SBS TRAIN-fitted,
and native FP64 winner labels remain distinct from selection frequencies.

Only bounded CPU fixtures are used during implementation. Formal pretraining,
selector training, TEST/LIB inference, timing, and final PDFs remain unrun.

## Current Integration Evidence

- The real complete manifest was read, not a root-level NPZ glob:1458000 rows,
  18 tasks x81000,5706 manifest-named shards. First/last rows for TSP, ATSP and
  VRPBLTW passed shard hashes, typed schemas, row/source IDs and sequence checks.
- Complete manifest SHA256:
  975282dea4458cfb0009d042a6e369b2afe415988929dfd1b29418926dafbcce.
- Current model includes global/node trajectories and final directed edge-state
  tokens. No token count or parameter count is hardcoded in the runner.
- B/C pretraining endpoint was exercised only with temporary synthetic data and
  reduced CPU budgets, including interruption/resume and processor-only export.
- All15 runner-specific CPU contracts passed in18.109 seconds. This includes
  complete checkpoint restore without external pretrain files and a temporary
  synthetic PDF smoke with paired TEST/LIB axes and explicit MISSING NSS values.
- The main thread still owns formal source freeze, assigned device, resource
  sizes and the final locked protocol. The unlocked draft is not launch approval.

## Freeze Handoff

1. Freeze the current model, processor, Torch input compiler, process schema/data
   generator, pretrain runner, selector runner and reporter together. describe
   prints current inference_source_sha256, pretrain_source_sha256 and
   reporting_source_sha256, covering the reused R84/R112 helpers without editing
   their original modules. Counts/layouts come from the complete model itself.
2. Bind protocol_lock.json from the unlocked draft: actual paths, all18
   microbatch_caps, all18 pretrain_microbatch_caps, eval_batch_size, and locked
   true. The complete process manifest SHA256 is already filled in the draft.
   The main launcher assigns --device and visible devices; no auto allocation.
3. B/C exports are pretrain/B_snapshot/processor.pt and
   pretrain/C_transition/processor.pt. They embed complete process schema/source,
   initialization and stream provenance and contain processor weights only.
   Finetuning best.pt/last.pt embed the entire selector; restoration and offline
   reporting never require opening those external process exports.

## Entry Points

Run from the repository root after conda activate easynco:

```bash
conda activate easynco
cd /public/home/shiys/0selection2
python -m code.V4.r113_experiment describe --arm A_scratch
python -m code.V4.r113_experiment pretrain --arm B_snapshot --device cuda:0
python -m code.V4.r113_experiment pretrain --arm C_transition --device cuda:0
python -m code.V4.r113_experiment train --arm A_scratch --device cuda:0
python -m code.V4.r113_experiment train --arm B_snapshot --device cuda:0
python -m code.V4.r113_experiment train --arm C_transition --device cuda:0
python -m code.V4.r113_experiment finalize --arm A_scratch --device cuda:0
python -m code.V4.r113_experiment finalize --arm B_snapshot --device cuda:0
python -m code.V4.r113_experiment finalize --arm C_transition --device cuda:0
```

train automatically finishes the fixed budget, locks the unique VAL checkpoint,
and performs full18 TEST. finalize reuses that saved TEST, then timing, both LIBs
and PDF/CSV. To resume individual post-training stages for an arm:

```bash
conda activate easynco
python -m code.V4.r113_experiment test --arm A_scratch --device cuda:0
python -m code.V4.r113_experiment timing --arm A_scratch --device cuda:0
python -m code.V4.r113_reporting lib --arm A_scratch --device cuda:0
python -m code.V4.r113_reporting report --arm A_scratch
```

These are entrypoint examples, not executed jobs. Finalize every completed arm
with the corresponding --arm. Add --resume after interruptions; source/stream
identity is verified at the last saved successful-update boundary. Standalone
processor entrypoint: python -m code.V4.r113_pretrain --arm B_snapshot.

## Main Execution Decision

The main thread has now written `protocol_lock.json`; `protocol_draft.json`
remains an intentionally unchanged historical draft. Logical batch is512,
TSP/CVRP microbatch384, other tasks512, process-pretraining microbatch512 and
evaluation batch512, shared across all three arms and both GPU types. The batch
and VAL rule changes are explicitly permitted by the user's latest instructions.
The complete process corpus contains1,458,000 TRAIN rows and1,496,914,803
compressed bytes; its synthetic programs did not call any native solver.

An assigned GPU first executes one disposable capacity check without saving
weights. Failure preserves its receipt and stops that branch; it does not
silently shrink the batch or restart training. Formal source approval is a
separate queue receipt after implementation is complete. The queue waits for
the four already authorized GPU slots; it cannot stop R109 or R112.

After the unique checkpoint and full TEST are locked, process predictions are
also measured on the same fixed64 input-only VAL rows per task. Teachers are
computed only in that diagnostic and never enter forward, finetuning or
checkpoint selection. The primary endpoint remains complete full18 selection,
not process accuracy alone. Final reports and publication wait for all three
arms, independently of the separate R114 solver-label job.
