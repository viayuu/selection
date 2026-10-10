# R58 Pilot Handoff

Completed all requested representative measurements. No further code changes, GPU work, MPS start, or9-anchor refinement by this owner after handoff. Volta owns subsequent production work; Pascal owns the CPU refinement preparation. This pilot does not authorize full generation/training or claim changed-budget quality equivalence.

## Scope and Completion

- Only `code/V4/r58_throughput.py`, `code/V4/test_r58_throughput.py` and artifacts in this directory were changed by this owner. Frozen scenario/environments/backends/labels and shared bridge were not edited.
- Only gpu03 / `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`, existing job1465. No other GPU queries or workloads. easynco and the original read-only compatibility overlay throughout.
- Original preflight finished128/128 before launch. `gate_evidence.json` retains completion and absent original process/handle proof. Later CPU budget gate overwrote main state with finished `skipped_over_budget`; pilot accepts only that finished, fully covered state with original evidence and unchanged frozen hashes. It never overwrites main state.
- Original pilot, diffusion retry, initial aug1 pilot and MoSES_RF supplement exited0, steps1465.69/.70/.71/.72 completed. Model-controller cumulative wall1486.465432s =24.774424min, well below original10800s cap. `slurm_accounting.txt` also retains whole-step elapsed times, including startup.
- Both aug1 repetitions use identical12 fixed TRAIN indices per problem, four per min/median/max size. Warmup once per scale per worker is excluded only from warm time, included in cold. VRPBLTW50/75/100 and CVRP50/277/499 cover both checkpoint buckets50/100.
- Aug1 remains a NEW protocol, not equivalent to frozen augmentation8. Original orientation at actual policy input, all physical N starts, same weights/checkpoints, FP32, seed2, singleton input, masks/clocks/decoding unchanged. No val/test or label/archive costs read; no cost ranking/quality selection.

## Direct Aug1 Measurements

Seconds below are means of BOTH full repetitions for the SAME12-item workload, including common release/completion barriers. Speedups are ratios of summed wall times, never the fastest repetition or multiplied prior aug8 speedups. Cold includes imports, checkpoint loading, all warmups, solving and process teardown.

| Problem | Model | Warm seconds W1/W2/W4 | Cold seconds W1/W2/W4 | Warm speedup W2/W4 | Cold speedup W2/W4 |
| --- | --- | --- | --- | --- | --- |
| VRPBLTW | RouteFinder | 2.7204 /1.6829 /1.5292 | 19.2531 /19.5872 /29.3069 | 1.6165 /1.7790 | 0.9829 /0.6569 |
| VRPBLTW | MoSES_CaDA | 5.0832 /2.9705 /2.6901 | 15.8260 /16.3553 /22.1040 | 1.7112 /1.8896 | 0.9676 /0.7160 |
| VRPBLTW | MoSES_RF | 4.5056 /2.6162 /2.4643 | 14.2119 /13.7850 /19.0652 | 1.7222 /1.8283 | 1.0310 /0.7454 |
| CVRP | RouteFinder | 8.3914 /6.0745 /5.6591 | 26.8042 /26.8914 /37.6033 | 1.3814 /1.4828 | 0.9968 /0.7128 |
| CVRP | MoSES_CaDA | 13.1951 /8.6156 /8.6548 | 26.9254 /23.8693 /34.7082 | 1.5315 /1.5246 | 1.1280 /0.7758 |
| CVRP | MoSES_RF | 13.5455 /8.6034 /9.1112 | 25.7867 /23.0890 /31.2578 | 1.5744 /1.4867 | 1.1168 /0.8250 |

All36 aug1 phases succeeded;432 measured routes and252 warmups independently rechecked. Every concurrent repetition exactly matches its own aug1 serial repetition: paid FP64 cost delta0, physical-tour differences0, raw-tour differences0. Only duplicate trailing depot-zero padding could be ignored by protocol; none was needed here. Same checkpoints/source hashes as qualified preflight. Observed W4 whole-GPU memory peaks: VRPBLTW RF/CaDA/MoSES_RF1394/1666/1498MiB; CVRP1538/1882/1634MiB. Samples~0.5s are not exact hardware peaks. All1308 sampled UUID contexts belong to phase workers.

`aug1_summary.json` and `aug1_moses_rf_summary.json` retain both repetitions, per-size synchronized times and phase paths; `aug1_combined_summary.json` merges only the completed aggregates. Initial manifest and results are unchanged. Supplement manifest records original manifest SHA and identical inputs. `aug1_verification.json` and reproducible CPU `audit_aug1.py` retain audit evidence.

## Revised Diffusion

Each mean uses four genuine TRAIN routes at that true size, plus one disjoint warmup per size. All48 measured routes and12 warmups feasible/cost-matched under independent checker. These changed budgets make NO original-recipe quality/equivalence claim.

| Model | Mean seconds N50/N275/N499 | Warm12-item wall | Cold wall | Whole-GPU sampled peak MiB |
| --- | --- | --- | --- | --- |
| DIFUSCO | 0.120829 /1.108090 /3.564006 | 19.191043 | 30.453903 | 2298 |
| DIFUSCO500 | 0.123155 /1.128077 /3.577192 | 19.332239 | 30.158267 | 2298 |
| T2T | 0.258113 /2.330717 /7.841515 | 41.741109 | 58.521775 | 21934 |
| T2T500 | 0.254496 /2.374242 /7.882602 | 42.064543 | 58.641606 | 21934 |

Actual per-call witnesses: DIFUSCO20 denoising calls, zero guided calls, 2-opt cap[100]; T2T20 denoising plus10 original guided calls, 2-opt caps[100,100]. Both `backend.parameters` and `model.args` overridden before execution. T2T rewrite count1,10 guided steps, original respective ratios/checkpoints retained (.4/.25). Original `T2TCO/diffusion/pl_tsp_model.py:59` guided function executes `backward()` and asserts `xt.grad is not None`; observed calls returned actual checked routes. No initialization-only or unguided substitute. Frozen50-step/3-rewrite recipe unchanged.

Successful outputs are `diffusion20__METHOD__retry1/`. Initial failed `diffusion20__METHOD/` attempts, source snapshots and raw logs remain intact. `summary.json`, `verification.json`, `diffusion_retry_state.json` and `diffusion_retry_source/` preserve these separate attempts.

## Original Concurrency and Exact Failures

| Original method/problem | W2 warm/cold speedup | W4 warm/cold speedup |
| --- | --- | --- |
| RouteFinder VRPBLTW | 1.53108 /1.10605 | 1.72400 /0.76324 |
| MoSES_CaDA VRPBLTW | 1.72494 /1.10221 | 1.64328 /0.76559 |
| BQ CVRP | 1.16473 /1.02327 | FAILED, no credit |
| LEHD CVRP | 1.18764 /1.03332 | FAILED, no credit |

All complete original phases exactly match their24-item serial paid costs and tours. Failed W4 cases were NOT rerun or silently credited:

- BQ W4: returncodes[-15,-15,0,0], states ready/ready/complete/complete,19/24 outputs, five missing indices. Executed `attempt1/r58_throughput.py:394` treated ANY exited process as failure while waiting for other workers, including normal completed exit0. Parent then SIGTERM'd the two remaining owned workers. This is a pilot completion-barrier bug, not a demonstrated semantic failure or OOM. Fixed `premature_worker_exit` accepts clean exits only with a complete state at the completion barrier; CPU regression test passes and all new aug1 W4 phases completed.
- LEHD W4: returncodes[1,-15,-15,-15], zero outputs,0.115597s,2MiB. `worker_0.log` traceback ends in the old `preflight_gate`: main CPU labels budget check had changed mutable pipeline state to `skipped_over_budget`. Exit occurred before model initialization. Initial four diffusion failures have the same entrypoint cause. Fixed completed-evidence gate allowed actual retry; no model budget/batch change hid the failure.
- Slurm original step completed0:0, MaxRSS4501332K; raw logs/accounting show no OOM evidence. Do not interpret allocation ReqMem20Gn as a changed launch request: every srun explicitly requested16G. Preserve raw observations; no inferred W4 speedup.

`failed_attempts_preserved_sha256.json` fingerprints every retained file in the six failed original directories.

## Budget Assessment

`representative_projection.json` is explicitly a TRANSFER MODEL, not all128 qualification: each model's measured fixed-mixture VRPBLTW throughput is transferred to15 MVRP variants, and its separate measured CVRP throughput is used for CVRP.12,000 projected labels per problem use counts only; no actual val/test read. Equal min/median/max mixture need not match actual distributions, and variant runtimes remain unmeasured. No proven upper/lower bound is asserted.

| Workers | RF-family label hours | Whole label hours, adding parent31.33495 diffusion +37.019 other |
| --- | --- | --- |
| 1 | 61.04717 | 129.40112 |
| 2 | 36.76065 | 105.11460 |
| 4 | 34.35517 | 102.70912 |

Even this warm-only representative model does not establish <100h; overhead/contingency/training are not included. RF allowance for parent85h label target is16.64605h. Stop at BUDGET BLOCK. No full generation, lock, training, new recipe branch or9-anchor run executed. Subsequent refinement requires its own parent-controlled ownership/authorization and must retain this raw evidence.

## MPS Read-Only Result

After both aug1 runs ended, step1465.73 only queried the designated UUID and inspected current-user processes. `/usr/bin/nvidia-cuda-mps-control` and `/usr/bin/nvidia-cuda-mps-server` exist and are executable. Driver560.35.05, compute modeDefault; GPU2MiB/0%, no compute processes. No current-user MPS control/server daemon; MPS pipe/log environment variables unset. Control `-v` printed the NVIDIA banner (copyright2003-2024), not a numeric tool version; package manager could not associate either binary with a package. Server binary was NOT executed. Installation presence is not proof of isolation or working MPS. No daemon started, mode changed, installation or new pilot. Exact read-only command and output: `mps_readonly_check.sh`, `mps_readonly_check.log`.

## Commands and Handoff Files

Executed tmux handles and corresponding launcher scripts (all ended):

```bash
tmux new-session -d -s r58_throughput_pilot 'bash /public/home/shiys/0selection2/code/V4/runs/R58_unified_scenario/throughput_pilot/launch.sh'
tmux new-session -d -s r58_diffusion_retry 'bash /public/home/shiys/0selection2/code/V4/runs/R58_unified_scenario/throughput_pilot/diffusion_retry_launch.sh'
tmux new-session -d -s r58_aug1_pilot 'bash /public/home/shiys/0selection2/code/V4/runs/R58_unified_scenario/throughput_pilot/aug1_launch.sh'
tmux new-session -d -s r58_aug1_moses_rf 'bash /public/home/shiys/0selection2/code/V4/runs/R58_unified_scenario/throughput_pilot/aug1_moses_rf_launch.sh'
```

Every model launch uses exactly:

```bash
srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
```

Full argv including UUID, thread1 variables, unchanged PYTHONPATH overlay, GNU timeout and CLI stages is retained in `launch.json`, `diffusion_retry_launch.json`, `aug1_launch.json`, `aug1_moses_rf_launch.json`. Do not restart completed phases: output directories deliberately refuse overwrite.

CPU command: `python -m unittest code.V4.test_r58_throughput -v`;12 tests passed in1.240s, including alignment/failures, exact equivalence, actual overrides/witnesses, clean-worker-exit barrier, completed-preflight evidence, original orientation/all starts and supplement immutability. Log: `cpu_tests.log`. CPU post-run audit: `python code/V4/runs/R58_unified_scenario/throughput_pilot/audit_aug1.py`.

Executed-source snapshots: `attempt1/`, `diffusion_retry_source/`, `aug1_source/`; final supplement source SHA `d64567d73512b0c4e5e2e6a6e853f9f39ef499897c0feff30cae0ea47fd0c32f`. `handoff.json` hashes both owned sources and records completed handles/steps, cumulative wall and ownership boundary. Parent may unlock production core now that all GPU steps have exited. No subsequent edits/GPU activity by this pilot owner without parent request.
