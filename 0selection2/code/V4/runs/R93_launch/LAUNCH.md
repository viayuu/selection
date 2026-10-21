# R93 Two-GPU Launcher

Implementation scope: `r93_launch.py`, `run_v4_r93_launch.sh`,
`test_r93_launch.py`, and this file, all under `code/V4`.
Only main may authorize and start a run. Implementation and CPU tests do not
probe resources, claim GPUs, create allocations, start workers, or commit.

## Locked Contract

| Arm | Existing Slot | Predecessor | Control |
| --- | --- | --- | --- |
| A | gpu03 / job1465 / 3090_1 | R92C explicit CPU recovery | CE only, dropout 0.1 |
| B | gpu05 / job1475 / 4090_1 | R92D TEST/LIB/PDF rc0 | CE only, dropout 0 |

R92A on 3090_0 and R92B on 4090_0 are protected healthy work. No fallback,
stopping, retraining C, automatic retry, or automatic start is permitted.
Both R93 arms are neural controls, not architecture experiments or tree arms:
57,600 successful updates, logical batch 1,024, warmup 4,320, VAL every 1,440,
micro cap 352, evaluation batch 256, seed 2, full18 frozen NEW90k/R84 inputs.
Main owns `runs/R93_plain_classification/protocol_lock.json`.

## Interface Decisions

| Decision | Resolution | Source |
| --- | --- | --- |
| Launch namespace | Private R92 transport adapter; replace R92-bound entry points, never mutate live modules | User |
| Before predecessors finish | `template` and `parse` are read-only; no resource probe | User |
| Busy resources or failed stage | One-shot failure; no polling, fallback, repair, or retry | User |
| Source freezing | R93 Python files, local dependencies, launcher chain, frozen R84 prepared artifacts and protocol | User |
| Reports | LIB in claimed container; PDF on ADMIN01 CPU only after GPU pipeline and UUID lock release | User |
| Comparison | Explicit CPU-only reporter command, not an automatic retraining or model-selection step | Default |

W&B is explicitly offline in argv, host environment, and container environment.
Cache reuse reads prior R85 evidence and the identity stamp only; no array audit,
copy, rebuild, or preprocessing.

## Handoff

Verification completed in `easynco`: 24/24 focused CPU tests (0.734 seconds),
all nine real runner/reporter CLI parses, shell syntax, and both read-only
predecessor gates. No live resource probe or startup verification was performed.
The launcher deliberately does not write a `ready.json` during implementation.
Only main's explicit `start` writes readiness and creates its tmux controller.

All commands below run from the repository after activation:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
python -B -m unittest code.V4.test_r93_launch -v
bash /public/home/shiys/0selection2/code/V4/run_v4_r93_launch.sh parse
bash /public/home/shiys/0selection2/code/V4/run_v4_r93_launch.sh gates
```

After main confirms final runtime/report files and focused runner tests, main
can execute the following. These commands have NOT been run by the launcher
implementer. `specs` and `prepare` only freeze plans; they do not probe resources.
Both arms may be prepared and explicitly started independently; no dependency
on R92A/B finishing and no automatic wait for another R93 arm's PDF.

```bash
bash /public/home/shiys/0selection2/code/V4/run_v4_r93_launch.sh specs --main-ready-evidence 'Main reviewed final R93 runtime and CPU tests' --runner-tests-passed
bash /public/home/shiys/0selection2/code/V4/run_v4_r93_launch.sh prepare --spec /public/home/shiys/0selection2/code/V4/runs/R93_launch/specs/A.json --main-ready-evidence 'Main reviewed final R93 runtime and CPU tests' --runner-tests-passed
bash /public/home/shiys/0selection2/code/V4/run_v4_r93_launch.sh prepare --spec /public/home/shiys/0selection2/code/V4/runs/R93_launch/specs/B.json --main-ready-evidence 'Main reviewed final R93 runtime and CPU tests' --runner-tests-passed
bash /public/home/shiys/0selection2/code/V4/run_v4_r93_launch.sh start --spec /public/home/shiys/0selection2/code/V4/runs/R93_launch/specs/A.json --main-ready-evidence 'Main authorizes R93A on fixed 3090_1' --runner-tests-passed
bash /public/home/shiys/0selection2/code/V4/run_v4_r93_launch.sh start --spec /public/home/shiys/0selection2/code/V4/runs/R93_launch/specs/B.json --main-ready-evidence 'Main authorizes R93B on fixed 4090_1' --runner-tests-passed
```

Source edits after `prepare` invalidate readiness; finish edits before freezing.
Existing armed/claimed attempts are never replaced. Busy resources fail once,
without interrupting other work or starting later unattended. Inspect receipts
under `runs/R93_launch/A` and `B`, and tmux sessions `r93_A` / `r93_B`.

For either arm, exact stages are `python -B -u -m code.V4.r93_experiment
train/test` with `--root`, `--arm`, `--prepared-root`, `--data-root`,
`--cache-root`, `--protocol-lock`, `--gpu-type 3090/4090`,
`--large-micro-cap 352 --eval-batch-size 256 --device cuda:0
--wandb-mode offline`. LIB is `code.V4.r93_reporting lib --root ... --arm A/B
--device cuda:0 --lib-batch-size 2`; PDF is `report --root ... --arm A/B
--device cpu`. `parse` prints every complete argv without executing it.

After both full-model deliveries, main may explicitly compare on ADMIN01 CPU:

```bash
CUDA_VISIBLE_DEVICES='' WANDB_MODE=offline python -B -u -m code.V4.r93_reporting compare --root /public/home/shiys/0selection2/code/V4/runs/R93_plain_classification --arm A --device cpu
```

Implementation assurance here is deterministic CPU testing and parser checks,
not live-start proof or an independent reviewer verdict. No reviewer was spawned.
