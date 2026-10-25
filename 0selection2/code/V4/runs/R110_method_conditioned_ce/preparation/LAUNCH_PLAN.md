# R110 Administrative Launch Plan

Exclusive scope: R110 launch/wave/goal, shell launch entry, launch/goal CPU tests
and this note. R109 is FINAL and is not edited. Main owns the scientific lock,
PLAN, witness, final specs, readiness, preparation, actual start and publication.
The runner/objective and reporter are owned by other workers and remain read-only.

## Existing Resources

| Arm | Slot | Allocation / node | CPU affinity |
| --- | --- | --- | --- |
| A_tied_ce | 4090_0 | original job1475 / gpu05 | 0,1,2,3 |
| B_private_ce | 4090_1 | original job1475 / gpu05 | 4,5,72,73 |

Reuse exact R108 physical GPU UUIDs, physical-UUID mutex inodes, allocation-owned
container and existing srun transport: --overlap --exact --gres=none,
--cpus-per-task=4, --mem=9G. No new allocation/container, SSH training, additional
probe, cache copy, cache repair or preprocessing. Use the warm
/tmp/shiys_R84_new90k_cache in place. Existing controller binding/idle/headroom
checks remain unchanged. Controllers and CPU reports stay on admin01; W&B is
explicitly offline, project selector with the configured entity.

Resource predecessors are R108_A_prior_ce and R108_B_density_ratio, respectively.
Require their original same-checkpoint full TRAIN/VAL/TEST, timing, LIB, PDF
model/numeric evidence and original ordered worker/controller rc0 receipts.
The later same_model_NSS4 metadata requirement is excluded ONLY from resource
release: it is not an original training/stage completion requirement. Version
comparison, GitHub status or an unrelated NSS metadata repair does not authorize
an automatic rerun and does not hold an otherwise completed resource predecessor.
Failures in original saved pipeline evidence still stop the launch.

## Scientific Interface

r110_experiment.required_lock_fields() is authoritative; the launcher never
derives or writes another protocol. Main adds data_root, prepared_root and
primary_cache. Graph computation is the actual shared r109_model, with mapped
arms A_tied_ce -> A_tied_late and B_private_ce -> B_private_late. Include main's
R110 MODEL_PREP.md and the frozen shared R109 MODEL_PREP.md in the source set;
no fictitious r110_model.py is required.

Use one forward and unweighted raw-utility CE across the whole legal pool.
Comparison/risk/RDrop/KL are off, no density ratio, priors, class weights, extra
head, targets, features or objective state. Inference remains original score
MLP -> utility differences -> maximin; maximin evaluation CE is only diagnostic.
Both arms retain full18 frozen NEW90k 81000/4500/4500, seed2, logical1024,
57600 successful updates, warmup4320, VAL1440, TRAIN7200, AdamW1e-4 to1e-6,
WD1e-4, clip1, dropout.1 and unique whole18 VAL checkpoint selection.

Common original scratch tensors and identical late prototypes give the same
initial evaluation function. A=2305033 and B=10220553; total/active parameters,
complete tensor hashes and FLOPs are NOT matched. Original fresh SHA:
dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93.
Final caps are large256/other1024/eval256 per main's reported witness PASS.
Both arms read the runner CAPS, with no fallback/shrink. R110 timing separately
retains first128 TEST inputs per problem, one warmup and three measurements.

## Safe CLI And Main Handoff

Activate easynco after sourcing conda.sh, then work from
/public/home/shiys/0selection2. Default help/draft/status do not write real specs,
freeze sources, query GPUs or start jobs.

```bash
python -B -m code.V4.r110_launch --help
python -B -m code.V4.r110_wave A_tied_ce
python -B -m code.V4.r110_goal --help
python -B -m unittest code.V4.test_r110_launch code.V4.test_r110_goal
```

After main's scientific lock, full required source closure and CPU tests exist,
main may perform these explicit actions for each ARM; none is run by this worker:

```bash
python -B -m code.V4.r110_wave ARM --write-final
python -B -m code.V4.r110_launch prepare --spec SPEC --evidence MAIN_EVIDENCE --runner-tests-passed
python -B -m code.V4.r110_launch start --spec SPEC --evidence MAIN_EVIDENCE --runner-tests-passed
```

SPEC is preparation/dispatch/specs/R110_ARM.json beneath this root. Both arm
plans use the same explicit evidence. Controllers run in tmux. Source freeze is
one-shot; a spec is not readiness. Changed sources/specs after ready fail closed;
no refreeze, duplicate start, implicit resume or automatic retry.

Required sources include the shared r109_model, R110 objective/runner/reporter,
r110_memory_witness.py and run_v4_r110_witness.sh. No nonexistent R110 model or
extra witness wrapper is required. The post_r108_learning_mechanisms_20261024.md
research note is optional: only files available at final-spec freeze enter the
frozen source set and curated publication. Late arrival does not mutate it.

## Completion And Publication

Each arm runs GPU train -> TEST -> timing -> LIB(batch2), then CPU report/PDF.
All original stages and controller/worker must be rc0, with full TRAIN/VAL/TEST,
timing, both LIB datasets and reports bound to that arm's unique VAL-selected
checkpoint. Current-version NSS four-row metadata/BOM handling, all per-arm PDFs,
saved paired/recovery CSVs, comparison protocol and root comparison PDF are
required. comparison_pdf metadata supports reporter's MODEL_COMPARISON.pdf.
Never require equal complete A/B hashes or count matching. No TEST/LIB tuning.

```bash
python -B -m code.V4.r110_goal --watch --publish-github --evidence MAIN_EVIDENCE
```

The main-ready watcher is quiet while waiting, compares once on CPU saved
evidence, then optionally publishes only the complete version. Preserve the
existing <git-dir>/codex-publication.lock. Curated assets include R110 sources,
the shared frozen graph model, docs, JSON/CSV/PDF; never stage raw predictions,
weights, caches, W&B credentials or unrelated dirty work. No real commit/push or
publication is performed by this worker. Pipeline completion is not a selection
improvement claim or completed research goal.

## Verification

FINAL: administrative CPU contracts passed21/21: launch9 and goal12. The only
initial pending test required the not-yet-landed reporter; after it arrived,
that actual five-stage parser test passed (1.57 seconds), without weakening or
skipping it. Launch/goal help, draft CLI and shell syntax passed. Exact source
closure contains106 existing files; final caps256/1024/eval256 match the runner.
Both assigned original R108 resource predecessors passed the actual saved-JSON
CPU gate. Count validation maps science.model_arm_map into the shared R109
model_spec.parameter_counts; science.trainable_parameters uses R110 arm names.
Main additionally reports46 launch/goal/report and12 runner/objective tests PASS,
with the seven model tests reused, not repeated. No model audit, hardware probe,
witness, real science/spec write, preparation/start, training or Git action was
performed by this worker. R109 files were not edited after their FINAL handoff.
R110 admin files are now final for main's one-shot source freeze and start.
