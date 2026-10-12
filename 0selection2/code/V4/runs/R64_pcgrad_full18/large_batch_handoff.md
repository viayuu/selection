# R64 larger real-batch formal handoff

Verified at2026-10-12T14:30:15+08:00: tmux `r64_train`, pane PID1513395,
Slurm `1465.90`, formal Python PID450449 on gpu03/index1,
UUID `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d` (RTX3090).
Actual assigned-UUID idle receipt: `launch_receipt_20261012T062808017586Z.json`,
14:28:08CST,2MiB/no clients. Fresh formal A began14:29:18CST; checkpoint180
exists. B starts fresh automatically after A and replays the exact new LR/horizon.

Selected per-task256 / effective768;40 cycles and240 updates per complete epoch,
180000 presentations/epoch; final16/task / effective48. Warmup and RDrop ramp
720successful updates; LR1e-4 unchanged, no LR scaling. Max40epochs/9600updates/
7.2Mpresentations per arm. No accumulation or microbatching.

One authorized TRAIN-only sizing pass, CVRP/TSP/OVRP largest padded-node rows:

| Per-Task | A Queries/s | B Queries/s | B Physical Headroom | Decision |
|---|---:|---:|---:|---|
|256|893.548|892.870|25.7971%|eligible, selected|
|512|CUDA OOM|CUDA OOM|0.2945% bound|ineligible|
|384|900.008|899.947|2.9441%|ineligible, requires15%|

Two warmups+three timed complete real updates per arm/size, with initialized
AdamW, both dropout forwards and normal resident inputs. Physical evidence is
20ms sampled CUDA mem_get_info plus conservative external/context baseline and
peak reserved bound; B256 allocated15.304293GiB/reserved17.179688GiB/physical
bound17.503174GiB. Predeclared3% throughput tie tolerance toward smaller batch.
Calibration25.706883s,20 discarded successful updates/19200 presentations,
two genuine OOM attempts/zero AMP retries. All calibration state/RNG discarded;
calibration does not count as formal updates. See capacity receipts and cost JSON.

Four changed-code focused CPU tests passed1.026s. First fresh formal A update:
768rows, loss1.9512447119, grad norm.9690369368, finite, zero retries,
.1838609362s,4177.069996queries/s, peak4.400811GiB, first warmup LR1.388888889e-7.
This is a smaller-node MVRP triple; do not compare it directly to sizing throughput.

Changed source paths: code/V4/r64_experiment.py, test_r64.py, run_v4_r64.sh,
R64_README.md. Fresh root records locked decision/PLAN/protocol/source snapshot,
verbatim adopted Astra response, calibration overhead and normal formal artifacts.
REVIEW_STATE.json and REVIEWER_MEMORY.md were updated for the actual restart.
Astra/max amendment was text/context planning, not direct local-code audit.

Old root `../R64_pcgrad_full18_aborted_batch42_20261012T140928` stays immutable:
old A14epochs+partial15/21383updates, Bneverstarted, aborted by explicit USER.
Neither trained A nor its optimizer/cursor/dropout progress was reused. R63 fully
complete/pushed15363d0ec. No TEST/analysis/git, extra audit/review gate or other
GPU/process/queue/environment changes. R58 deferred, unresolved B semantics
preserved; full18/fullpools/native winners/single-solver70% goal remains active.
After both new arms finish, parent freezes integer-val-correct bests, inference
and official arm(tieA), then authorizes one complete original TEST/model.
