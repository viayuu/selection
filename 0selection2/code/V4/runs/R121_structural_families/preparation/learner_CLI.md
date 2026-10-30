# Principal Learner Handoff

No actual training, solver or GPU launch was performed by preparation. Only CPU
unit fixtures and untrained-initialization identity checks are allowed here.

## Saved Labels To CPU Features

Input-only CPU isolation preparation is available BEFORE labels complete:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 46 python -B -u -m code.V4.r121_experiment prepare-isolation \
  --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/learner_v1 \
  --device cpu
```

The immutable `learner_natural_isolation_v1.json` records per-task actual source
mapping hashes, eligible/excluded counts/rows and known cross-task associations.
It does not require solver labels or load any model.

Export only AFTER all18/89 native labels genuinely complete:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m code.V4.r121_universal_collection export --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/collection_explicit_cpu_v2
```

Prepare a separate, bounded-memory read-only family feature cache. This reuses
original UnifiedProblemDataset/build_batch/local_geometry extraction; costs stay
FP64 and native winners int64. It never modifies natural caches/data/labels:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c "${R121_CPU_SET:?main CPUs}" python -B -u -m code.V4.r121_experiment prepare-family \
  --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/learner_v1 \
  --device cpu
```

The native89 export is mandatory even though only family_train supplies auxiliary
training. Default export path is now `collection_explicit_cpu_v2/family_labels.jsonl`:
the complete single-root113920 native labels, including byte-identical adopted
prefixes and new explicit-CPU/unchanged-session solves. No cache preparation is
started before this full export exists. Family-val
known source records are excluded from natural CE via actual metadata. The scope
is unseen supervised-gradient source records/known selected geometry associations,
not an independent benchmark or exhaustive content-copy novelty. Full81000
normalization/SBS history remains retained and disclosed.

## Fixed TRAIN Diagnostic Logging

This pre-training re-freeze adds diagnostic logging and scope-only plotting; model, three-term objective,
initialization, task/sample streams, family sampling, LR,14400-update budget and
unique natural-VAL checkpoint rule are unchanged. Prior learner source3104 is
superseded by the exact experiment/test hashes in `learner_READY.json`. The
later16b diagnostic version is superseded only for R121-local figure labeling;
training and diagnostic sampling code are unchanged since16b.

Before training, independently select one FIXED subset of <=1024 eligible
natural TRAIN rows/task (seed2, per-task seed from
`r114.seed_for(2, problem, 'r121_fixed_train_diagnostic_v1')`). Use NumPy
`default_rng(seed).choice(eligible, replace=False)`, then sort local indices.
There is no winner stratification and no holdout-source row is eligible. Sampling
is performed once; the exact same indices are evaluated at every720-update VAL
interval. Sampling
and diagnostic forward passes do not advance training RNG or sampling streams.
Forward runs in eval/no_grad; model mode and Python/NumPy/Torch/current-device
CUDA RNG are restored even on exceptions. CUDA restoration uses the existing
RNG helper; preparation regression tests are CPU-only, not GPU validation.

`<root>/principal/protocol.json:train_diagnostic` pins per-task count, seed and
SHA256 of selected/eligible little-endian int64 index sequences. Protocol builder
signature and all CLIs stay unchanged. `protocol(...)` reads the saved sealed
source-record isolation to derive the same diagnostic identity before training;
it does not require diagnostic model inference or materialized family features.
The authoritative immutable protocol still requires the complete native export
and main's existing CPU protocol builder, as previously handed off.

`<root>/principal/history.json[*].train` contains per-task and equal-task macro
CE, native Top1, actual regret, and vs fixed full81000-TRAIN-fitted SBS, plus the
subset identity. Existing `r84_reporting.learning_curves` reads TRAIN Top1,
regret and SBS directly; the structural objectives PDF additionally plots the
fixed-subset eval CE separately from the unchanged three optimization terms.
R121 `report()` locally redraws its final `learning_curves.pdf` and PNG from the
same saved points: metric-panel legends and the figure footer explicitly say
`TRAIN: fixed eligible subset (1024/task)` and `NOT full 81,000 TRAIN rows/task`.
Training-batch objective losses remain separately labeled. Shared
`r84_reporting.py` is not modified and no new model inference is performed.
All TRAIN diagnostic curves are a FIXED TRAIN SUBSET, NOT complete TRAIN and
never a checkpoint selection signal. Existing `family_val` diagnostics remain.
At most18432 diagnostic rows/interval and368640 at14400updates are additional
eval-only exposures, not gradient-training samples or an extra label budget.

No automatic full-TRAIN locked-checkpoint capture is added here. Main owns
`r121_followup.py`; the independent finisher owns full TRAIN capture. Continue
explicit `train -> test -> lib -> finish` stages, not automatic TEST from train.

## Main's GPU Resource Worker

The same entry point supports train/test/lib. Required release/binding checks run
BEFORE loading models or training caches. Use main's node-local natural caches;
the original labels/input identity and81000/4500/4500 membership remain fixed.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES="${R121_GPU_UUID:?main UUID}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c "${R121_CPU_SET:?main CPUs}" python -B -u -m code.V4.r121_experiment "${R121_STAGE:?train or test or lib}" \
  --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/learner_v1 \
  --device cuda:0 --gpu-type "${R121_GPU_TYPE:?3090 or 4090}" \
  --cache-root "${R121_NATURAL_CACHE:?node-local train/val cache root}" \
  --test-cache-root /public/home/shiys/0selection2/code/V4/runs/R120_multidomain_selector/existing_test_cache/test \
  --release "${R120_RELEASE_RECEIPT:?main receipt}" --binding "${R121_BINDING:?main binding JSON}"
```

The TEST path ending in `/test` is accepted directly and normalized to its parent
cache root. No TEST features/labels are collected anew. Existing prepared full18
source manifest is used to verify transported caches. If unavailable, fail with
the missing path rather than create/replace old data. Train ends with the unique
full18 natural-VAL actual-regret lock; TEST and LIB are explicit subsequent stages.
`--resume` only resumes an unchanged saved successful-interval/pending-VAL state.

Only after TEST+LIB, generate existing offline PDFs/CSVs, NSS4500 comparisons,
selection/native-winner panels plus new structural-loss curves on CPU:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m code.V4.r121_experiment finish --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/learner_v1
```

Basic CPU-only interface:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && cd /public/home/shiys/0selection2 && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m code.V4.r121_experiment describe --root /public/home/shiys/0selection2/code/V4/runs/R121_structural_families/preparation/learner_v1
```

Checkpoints expose `code.V4.r121_experiment.load_checkpoint/load_model/from_checkpoint`.
Every common trainable parameter is copied from the original untrained seed2
initialization, not from R119/R120; the additional potential head is freshly seeded.
The new population/exposure differs from historical models and must be disclosed.
No accuracy/regret improvement is claimed from initialization or fixture tests.
