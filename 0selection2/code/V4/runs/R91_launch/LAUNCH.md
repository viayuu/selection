# R91 Launcher Sidecar

Implementation only. No prepare, start, tmux session, resource probe, GPU context,
training, or process stop was performed by the launcher worker. Main alone runs
prepare/start after declaring the objective, runner and protocol final.

## Ownership And Interfaces

- Launcher owns `r91_launch.py`, `run_v4_r91_launch.sh`, `test_r91_launch.py` and
  launcher manifests below this directory. Main owns the objective/protocol;
  the runner worker owns experiment/reporting implementations.
- Exact arms: `A_utility_ce`, `B_utility_probabilities`.
- Model root: `/public/home/shiys/0selection2/code/V4/runs/R91_probability_link_control`.
- The main-owned `protocol_lock.json` is read/frozen, never generated or edited.
- Interface decision: retain the R88 reporting positional `lib` and `report`
  commands in `code.V4.r91_reporting`; both run on the same GPU as train/TEST.
  The real parsers must accept all commands before preparation can succeed.
- Train is followed by the runner's idempotent `test`, then LIB and PDF report.
  Every stage requires the previous stage's rc0; no resume, retry, alternative
  checkpoint, GPU fallback, cache copy/rebuild, or array audit is supplied.

## Frozen Resource Route

| Arm | Slot | GPU UUID | Physical GPU |
| --- | --- | --- | --- |
| A_utility_ce | 3090_0 | GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 | 3 |
| B_utility_probabilities | 3090_1 | GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d | 4 |

Both slots use gpu03/job1465 and allocation-owned container
`b8ced5d8a8cdafdc5006f827a0d36c319c61056a65316af4ff9937453fa0635c`.
Allowed CPUs are `12-17,64-67`; each arm claims four disjoint CPUs and 9 GiB.
The established route is `srun --jobid=1465 --overlap --exact --gres=none
--immediate=5`, then `sudo -n /usr/bin/ai_docker exec`, with the established
cross-version GPU-UUID flock held throughout the worker pipeline. No new
allocation or direct SSH training route exists. R89/gpu05 is not a candidate.
The user supplied the current 2 MiB/GPU observation; it is not fabricated as a
new launcher probe. Existing live resource checks still run at claim/entry time.

## Gates And Cache

Predecessors are the actual R88 arms `A_canonical_anchor` and
`B_execution_anchor`, respectively. Receipts come from
`code/V4/runs/R88_launch/<arm>/{worker,controller}.json` and
`code/V4/runs/R88_execution_role_inputs/<arm>/postprocess_status.json`.
Train/TEST/LIB/PDF and controller must be complete with rc0, and the report
receipt must be complete. Train completion or an idle GPU alone is insufficient.
Failure is terminal; waiting for readiness/resources is not a training retry.

The main explicit readiness evidence and runner-test confirmation are required
before `start` can create tmux. Main readiness may arrive before predecessor PDF
completion: the controller waits for delivery without claiming resources.

Reuse R84 preparation/initialization and `/tmp/shiys_R84_new90k_cache` in place.
The existing TRAIN/VAL cache metadata is validated against the real frozen
`code/V4/runs/R85_launch/resources.json`, independently of the R88 predecessor
root. TEST arrays already exist; the launcher neither copies nor scans them.
It calls the captured `engine._prepare`, never the late-bound R87 prepare that
would incorrectly look for R88 `resources.json`.

Controls for both arms: seed2, logical batch1024, updates57600, warmup4320,
evaluation every1440, micro-cap352, evaluation batch256, GPU type3090.
W&B is enabled but fixed offline; no credential discovery/login is attempted.

## CPU Verification

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
python -B -m unittest code.V4.test_r91_launch
bash -n /public/home/shiys/0selection2/code/V4/run_v4_r91_launch.sh
bash /public/home/shiys/0selection2/code/V4/run_v4_r91_launch.sh specs
```

`specs` only writes launcher manifests and parses the real two-arm train/test
argv (four commands), plus LIB/report argv when the reporting parser is present.
It does not freeze sources, write the protocol, or arm/start anything.

Verified on 2026-10-20: launcher CPU tests **11 PASS**; wrapper `bash -n` passed;
real runner train/test argv **4/4 parsed**; real reporting lib/report argv
**4/4 parsed**. `spec_parse.json` records both actual R88 full-delivery gates as
complete with rc0. No new resource audit was run. PDF completion/report polish
is not an additional main-readiness gate; only the specified R88 predecessor
delivery receipts gate resource claiming. Sources remain unfrozen, and no
plan/ready/claim/controller receipts have been created by this worker.

## Main-Only Prepare And Start

After the runner/objective/reporting sources and the main protocol are final:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
bash /public/home/shiys/0selection2/code/V4/run_v4_r91_launch.sh specs
bash /public/home/shiys/0selection2/code/V4/run_v4_r91_launch.sh prepare --spec /public/home/shiys/0selection2/code/V4/runs/R91_launch/specs/A_utility_ce.json
bash /public/home/shiys/0selection2/code/V4/run_v4_r91_launch.sh prepare --spec /public/home/shiys/0selection2/code/V4/runs/R91_launch/specs/B_utility_probabilities.json
bash /public/home/shiys/0selection2/code/V4/run_v4_r91_launch.sh start --spec /public/home/shiys/0selection2/code/V4/runs/R91_launch/specs/A_utility_ce.json --main-ready-evidence 'MAIN: final objective/runner/protocol reviewed; cite final CPU-test evidence' --runner-tests-passed
bash /public/home/shiys/0selection2/code/V4/run_v4_r91_launch.sh start --spec /public/home/shiys/0selection2/code/V4/runs/R91_launch/specs/B_utility_probabilities.json --main-ready-evidence 'MAIN: final objective/runner/protocol reviewed; cite final CPU-test evidence' --runner-tests-passed
```

Preparation freezes the actual sources, main-owned protocol, R84 initialization
and real R85 cache-evidence hashes. Start requires matching specs/hashes and a
fresh output directory. Replace the evidence string with main's actual dated
readiness/test evidence; it is not a readiness declaration by this sidecar.
