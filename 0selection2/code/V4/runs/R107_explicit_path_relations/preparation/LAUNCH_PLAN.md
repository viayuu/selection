# R107 Administrative Launch Draft

Status: administrative integration only, NOT formal readiness. Main alone owns
the memory witness, final protocol lock, formal spec writes, prepare and launch.
This worker does not start training, publish, or modify any previous version.

## Assigned Resources

| R107 arm | Assigned R105 predecessor | Slot | GPU UUID | CPU quartet |
| --- | --- | --- | --- | --- |
| A_edge_factorized | A_static_gate | 3090_0 | GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 | 12,13,14,15 |
| B_connected_paths | B_input_gate | 3090_1 | GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d | 16,17,64,65 |

Use only existing allocation 1465 / Sothis1465_gpu03, container
`b8ced5d8a8cdafdc5006f827a0d36c319c61056a65316af4ff9937453fa0635c`.
Slot numbers are dispatch aliases, not host physical indices. Main reports
physical GPU3/GPU4 idle; dispatch must still check live physical availability
under the original UUID mutexes and gpu03 node lock. Other users' GPUs and the
healthy R106 gpu05/4090 arms remain untouched. No new allocation or direct-SSH
training outside the allocation/container cgroup is supported.

## Protocol And Readiness Boundary

- Full18 frozen NEW90k, seed2, TRAIN81000/VAL4500/TEST4500 per problem; retain
  the frozen legal pools and native FP64 ind winner convention.
- Logical1024; 57600 successful updates, warmup4320, full VAL every1440,
  TRAIN diagnostics every7200. Existing R84 AdamW recipe, peak1e-4;
  R88B execution-anchor role, method ID+34 inputs, original score/maximin.
- Scratch only: original fresh R84 seed2 tensors and TRAIN-only geometry,
  plus matched fresh path parameters. No trained donor, solver, current-query
  labels, route output, extra supervision, routing stage or feature cache.
- Main-confirmed common caps: TSP/CVRP256, all other tasks1024, eval128. Main owns
  `r107_memory_witness.py`: input-only largest TRAIN, two-view forward/backward
  before formal lock. Declare both the source and `run_v4_r107_witness.sh`.
  Main's B_larger256_1024 witness passed all four largest-TRAIN shapes:
  TSP/CVRP allocated18805/18856MiB, reserved21618/21070MiB on24GB;
  ATSP/VRPBLTW allocated12075/12141MiB. These are main-reported measurements,
  not a separate administrative hardware probe or a formal readiness receipt.
  Runner, lock and both specs must agree before prepare. No per-arm caps,
  runtime shrink, changed logical batch, implicit resume or automatic retry.
- Reuse `/tmp/shiys_R84_new90k_cache` in place with exact R84 cache identity;
  no copy, repair, preprocessing, route cache or derived feature cache.
- Each assigned original R105 arm must have complete same-lock TRAIN/VAL,
  TEST, timing, TSPLIB/CVRPLIB and PDF/report evidence, and original worker
  and controller stage rc0. R105 comparison/publication is NOT the resource
  predecessor: the GitHub pack/push failure does not block these slots.
- R107 order is GPU `train -> TEST -> timing -> LIB`, then CPU `PDF`.
  Any nonzero stage stops downstream work. Selected VAL model and rule remain
  fixed for all TEST/LIB/timing/report evidence. No routing diagnostic needed.
- `prepare` requires explicit main evidence, passed tests, final scientific
  lock, all named sources (including witness/model/runner/report/tests), exact
  real stage parsers and comparison CLI. Source closure and exact argv freeze
  once; edits after prepare invalidate readiness rather than auto-refreezing.
- Whole-version watcher waits for both arms and saved-result CPU comparison;
  only explicit `--publish-github` publishes one R107-only curated version
  using `<git-dir>/codex-publication.lock`. No weights, raw predictions,
  labels, secrets, unrelated dirty files, or previous-version sources are added.
  Completion is not a claim of model improvement or research-goal success.

## Draft API And Main CLI

All commands start from `/public/home/shiys/0selection2` in `easynco`.
`r107_wave.make_spec(arm)` and the default CLI only return/print a draft; they
do not read a protocol lock or write anything. Only main may use `--write-final`
after its witness and final lock. These are handoff commands, NOT executed here.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
export PYTHONDONTWRITEBYTECODE=1
python -B -m unittest code.V4.test_r107_launch code.V4.test_r107_goal
python -B -m code.V4.r107_wave A_edge_factorized
python -B -m code.V4.r107_wave B_connected_paths
```

After model/report integration, main's real memory witness and final
`protocol_lock.json`, use one evidence slug for both arms:

```bash
EVIDENCE=MAIN_R107_FINAL_TESTS_AND_MEMORY_WITNESS
DISPATCH=code/V4/runs/R98_parallel_dispatch
for arm in A_edge_factorized B_connected_paths; do
  python -B -m code.V4.r107_wave "$arm" --write-final
  bash code/V4/run_v4_r107_launch.sh prepare --spec "$DISPATCH/specs/R107_$arm.json" --evidence "$EVIDENCE" --runner-tests-passed
done
for arm in A_edge_factorized B_connected_paths; do
  bash code/V4/run_v4_r107_launch.sh start --spec "$DISPATCH/specs/R107_$arm.json" --evidence "$EVIDENCE" --runner-tests-passed
done
```

`start` creates the existing-style `goal_r107_<arm>` tmux controllers. If a
predecessor has not completed, main may use the identical `queue` CLI in a
main-owned CPU tmux session instead of `start`; queue does not bypass gates.
Run the whole-version watcher inside a main-owned CPU tmux session:

```bash
python -B -u -m code.V4.r107_goal --watch --publish-github --evidence "$EVIDENCE"
```

The default `python -B -m code.V4.r107_goal` is read-only CPU status. Fake tests
do not certify the real model, final scientific lock, memory headroom or real
runner/report parsers, and are never themselves readiness evidence.
