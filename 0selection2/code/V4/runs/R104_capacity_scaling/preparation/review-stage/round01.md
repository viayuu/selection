# R104 Bounded Engineering Review: Round 01

Date: 2026-10-23 (+08:00). One pass, under five minutes.
Verdict: **ready (engineering only)**. Score: **9/10**.
Reviewer: Codex, **same-family / provisional**. Actual serving-model and
reasoning-effort attestation: **unknown**; no independently verified Sol/max
identity or cross-family review is claimed.

## Findings By Priority

- P1 / blocking correctness: **no findings** in the reviewed snapshot.
- P2 / actionable correctness: **no remaining findings** in the reviewed snapshot.
- Non-blocking evidence limits: author reports model 9/9 and runner/report 7/7
  CPU tests passed. These suites were read, not rerun. No GPU feasibility,
  dataset/weight identity, training outcome, or capacity efficacy was verified.

## Targeted Checks

- Actual capacities: A is d256, instance4/joint2, four heads/qkv64; B is d128,
  instance8/joint4, four heads/qkv32. The factory checks instantiated module
  dimensions, not nominal R84 settings. Counts are 7,288,969 and 3,426,057.
  See [r104_model.py](/public/home/shiys/0selection2/code/V4/r104_model.py:30),
  [module checks](/public/home/shiys/0selection2/code/V4/r104_model.py:144),
  and [tiny model tests](/public/home/shiys/0selection2/code/V4/test_r104_model.py:60).
- Fresh initialization: regenerated original seed2 trainables must match the
  supplied hashed R84 scratch record. Only same-name/same-shape parameters are
  copied; resized/new parameters use isolated seed104002. Geometry moments are
  validated, and restore is strict. No trained donor or identical-full-init
  claim is introduced. See [initializer](/public/home/shiys/0selection2/code/V4/r104_model.py:171).
- Execution/objective: logical1024, TSP/CVRP micro128, all other tasks512,
  eval128 resolve inside the private train/evaluate globals. The underlying
  update weights by actual micro rows, then clips/steps once. Full18 task
  streams and original CE/comparison/risk/two-view R-Drop remain unchanged.
  See [private engine](/public/home/shiys/0selection2/code/V4/r104_experiment.py:238),
  [accumulation](/public/home/shiys/0selection2/code/V4/r84_experiment.py:83),
  and [full18 training](/public/home/shiys/0selection2/code/V4/r84_experiment.py:413).
- Selection/identity: strict minimum complete18 VAL macro_vs_sbs_pct, first
  exact tie, one whole-model lock before TEST/LIB. Selected VAL binding checks
  checkpoint SHA, model state including buffers, protocol hash and update;
  capture identities remain immutable and reporting uses their capture view.
  See [selection](/public/home/shiys/0selection2/code/V4/r84_experiment.py:463),
  [binding](/public/home/shiys/0selection2/code/V4/r104_experiment.py:197),
  and [report capture](/public/home/shiys/0selection2/code/V4/r104_reporting.py:43).
- Arrival snapshot: an earlier goal version rejected periodic TRAIN captures
  with null checkpoint SHA. The arriving version now accepts them only with
  matching model/protocol/update and official selected-VAL binding. Three
  reviewer-executed in-memory CPU checks accepted the valid 7200-step capture
  and rejected wrong model and wrong update. No files/arrays/weights were
  loaded by these checks. This is resolved, not an outstanding finding.
  See [selected_train_ready](/public/home/shiys/0selection2/code/V4/r104_goal.py:82)
  and [new regression fixture](/public/home/shiys/0selection2/code/V4/test_r104_goal.py:135).
- Reporting explicitly disclaims parameter/FLOPs/full-initialization matching
  and distinguishes actual standalone GPU-resident-model latency from
  end-to-end cost. Launcher/wave source has no import-time dispatch and keeps
  assigned R99 predecessor/readiness gates. No cosmetic issue blocks launch.
  See [report boundary](/public/home/shiys/0selection2/code/V4/r104_reporting.py:28)
  and [dispatch gate](/public/home/shiys/0selection2/code/V4/r104_launch.py:195).

## Scope And Snapshot

Read current diff/status first: R104 sources are untracked additions; tracked
changes elsewhere were not altered. Reviewed PLAN/protocol_lock, frozen model,
runner/report and tiny tests; inspected the arriving launch/goal/wave sources.
Only this file and adjacent REVIEWER_MEMORY.md were written. No production
code/protocol changes, GPU/job/process operations, dataset/weight/TEST-result
reads, external research, agent messages, or Git mutations were performed.
Engineering readiness does not authorize bypassing main readiness/resource
gates and does not claim future capacity benefit. R99/R101/R102 stay untouched.

Snapshot SHA256 (2026-10-22 17:04:22 UTC):

```text
r104_model.py       950e0f03dd63f61de0e0cb9c6b62fd63cb04abf66844a145a8715828656cff76
r104_experiment.py  c3c8466a20c1fb7cc89469907df2a5d0bd74f52d3caafe21d5a4be4a8ed0cac7
r104_reporting.py   3608e9be3b218404d2e157a7d4922bc8ca6c3f1618ed9e612ca1fcb65dac2a07
r104_launch.py      df09f4927cd520bd7a18b447f3edb50f5b6b3c81fed985191a49024ca8310138
r104_goal.py        c55a33a33c7bb307a43307db70aec37dfe5356b18662447d36f071ab8e0a3f6f
r104_wave.py        59409c7eceba95695ec941c8637fc56828c49a2f726f48da9df77d2bec53c14f
protocol_lock.json  69c36af27e89ac1157dabd4abaea67683653e422fea613add70a596f4e018172
```
