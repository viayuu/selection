# R108 Administrative Launch Handoff

Administrative port of R107's private working pattern only, not main readiness.
Main owns witness, final lock, final specs, prepare, queue/start and publication.
No old source edits, GPU operations, spec/ready writes or Git actions here.

## Resource Bindings

| Arm | Original R106 predecessor | Slot | CPUs |
| --- | --- | --- | --- |
| A_prior_ce | A_frozen | 4090_0 | 0,1,2,3 |
| B_density_ratio | B_finetune | 4090_1 | 4,5,72,73 |

Reuse job1475/gpu05 and container
`76045731f6b453443fe18a9bce671599bdbf17c2713b8edcdd090925eea14bf6`.
UUIDs are `GPU-71479e7e-f8bd-e430-66ed-f8db9f0d47be` and
`GPU-597c682a-4b7f-7609-80a6-8ced791f5c5d`, respectively. Preserve original
physical mutexes, node lock, allocation/cgroup and warm raw R84 cache
`/tmp/shiys_R84_new90k_cache`. No SSH training, new allocations, cache copying,
preprocessing, automatic retry, implicit resume or restarts.

Main reports R106A full TRAIN/VAL/TEST/LIB/timing/PDF complete and ready=true;
A may start only after R108 main readiness. Healthy R106B is preserved and B
queues until that original arm's full pipeline completes. The gate privately
reads `r106_goal.status` with R106's own `preparation/dispatch`, then checks
assigned-slot original worker/controller ordered rc0 receipts. Other-arm or
version comparison/Git publication state is not the resource predecessor.

## Lock And Sources

- R108 root: `code/V4/runs/R108_conditional_density_ratio`.
  R108 dispatch: `root/preparation/dispatch`, separate from R106/R107 receipts;
  original hardware mutex identities remain unchanged.
- Final common caps: large512 / other1024 / eval256. Main reports actual
  `A_single512_1024` largest-TRAIN witness PASS, peak allocated13,367MiB and
  reserved17,366MiB. This is main's witness, not an administrative GPU probe.
  Runner, lock and both specs must agree. No per-arm fallback or runtime shrink;
  logical batch remains1024. Same forward/parameters does not match active
  gradient freedom: CE cancels the shared last-bias data gradient through its
  common-shift gauge, while ratio anchors origin; active-matched metadata is false.
- Include `r108_memory_witness.py` and `run_v4_r108_witness.sh`, model,
  objective, prior builder, runner, report/admin and their named tests.
  Include the small `root/prepared/priors.json` as a frozen source, never
  original `n/winner` arrays or feature-cache payloads. No prior CLI argument.
- Final specs require the main scientific lock and matching prior file hash.
  `prepare` additionally requires explicit main evidence, passed tests, all
  named sources, exact runner/report parsers and one source/argv freeze.
  Subsequent changes invalidate readiness; no automatic refreeze.
- Ordered rc0 pipeline: GPU `train -> TEST -> timing -> LIB`, then CPU `PDF`.
  A nonzero stage stops downstream work. Full18 metrics/fixed36/NSS4 and
  unique-VAL-state timing/report evidence precede whole-version comparison.
- Whole-version watcher publishes only on explicit `--publish-github`, once,
  with the shared `<git-dir>/codex-publication.lock`; curated R108 assets only,
  no raw predictions, arrays, weights, credentials or unrelated dirty files.
  Completed pipelines do not imply improved selection or goal success.

## Main-Only CLI

Default wave CLI prints a draft without reading a lock or writing files.
The following finalization commands are documentation only, not executed here.

```bash
conda activate easynco
EVIDENCE=MAIN_R108_FINAL_TESTS_AND_MEMORY_WITNESS
DISPATCH=code/V4/runs/R108_conditional_density_ratio/preparation/dispatch
python -B -m code.V4.r108_wave A_prior_ce --write-final
python -B -m code.V4.r108_wave B_density_ratio --write-final
bash code/V4/run_v4_r108_launch.sh prepare --spec "$DISPATCH/specs/R108_A_prior_ce.json" --evidence "$EVIDENCE" --runner-tests-passed
bash code/V4/run_v4_r108_launch.sh prepare --spec "$DISPATCH/specs/R108_B_density_ratio.json" --evidence "$EVIDENCE" --runner-tests-passed
bash code/V4/run_v4_r108_launch.sh start --spec "$DISPATCH/specs/R108_A_prior_ce.json" --evidence "$EVIDENCE" --runner-tests-passed
bash code/V4/run_v4_r108_launch.sh queue --spec "$DISPATCH/specs/R108_B_density_ratio.json" --evidence "$EVIDENCE" --runner-tests-passed
python -B -u -m code.V4.r108_goal --watch --publish-github --evidence "$EVIDENCE"
```

Run queue/watch in main-owned CPU tmux sessions. `start` creates the existing
style per-arm tmux controller. Default goal/status is read-only CPU. No fake
test, draft or this document certifies real memory headroom or formal readiness.
