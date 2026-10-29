# Main-owned MKP native AS dispatch

CPU data is COMPLETE. No GPU job has been started by this worker. Four-card
ownership and UUID mapping remain with the main thread.

## Current handoff,2026-10-29

Main has ALREADY started formal AS coverage in `tmux r120_mkp_full_pool`, using
`code/V4/runs/R120_multidomain_selector/resources/as_full_pool_plan.json`.
Four physical GPUs each run four independent single-thread processes. Do NOT
launch the historical commands below again, change their ranges, or edit
`as_dispatch.py`/`nonrouting_supplement.py` while those jobs are running.
The27 initial TRAIN cases covered all27 cells and completed rc0; the main's
initial mean was about8.8s/instance. Concurrent throughput is a different
measurement, not an exclusive-GPU speed. Main owns completion and any GPU ETA.

Pinned hashes:

- `nonrouting_supplement.py`: `7355064443385b9cf6adae481a12584325d009b099325b982b32c391695b8cda`
- `as_dispatch.py`: `2bf0d404f200dd7b3ec8fd55b788f2b80150070cbfbd914dca702186b527c557`

The CPU timings and initial GPU recipes below are retained as HISTORY, not
current pending work or instructions to start a second dispatch.

## Invariants and current CPU evidence

`as_dispatch.py` imports the pinned upstream class. Only its20 independent
ant-capacity tests are vectorized; native sampling, heuristic, pheromone update,
best-solution selection,20 ants and100 iterations are unchanged. Each instance
has its original integer certificate and its fixed per-input RNG seed. There
is no short-time cutoff, new prior, epsilon, repair or early quality stop.

Full100-iteration CPU witnesses so far:

| Instance | Vectorized seconds | Unmodified native seconds | Full native path |
| --- | ---: | ---: | --- |
| n100,m5,alpha.25 | 1.1371 | 4.3105 | exactly identical |
| n250,m10,alpha.50 | 7.5775 | 23.4789 | exactly identical |
| n500,m30,alpha.75 | 58.3505 | 103.9403 | exactly identical |

The three fresh second instances took1.1887/7.4897/58.0731 seconds. Their mean
is22.2505s, implying about80.16 CPU-hours for12970 instances IF those three
represented layers were representative. They confound size/dimension/tightness,
so this is only a rough warning, NOT a27-cell estimate or GPU reservation.
Complete receipts/certificates: `as_vectorized_cpu_witness.json` and
`as_cpu_witness.log`. One real external benchmark instance also completed its
full100 iterations in1.1758s; it remains external, not training data.

GPU speed is UNMEASURED. None of these CPU timings justifies a precise GPU ETA.
AS is explicitly pending for the formal pool; first measure the authorized GPU
batch below rather than blocking all-six integration on an unbounded dispatch.

## Exact small GPU batch

Run under a main-assigned physical UUID. This command intentionally fails if
`MAIN_ASSIGNED_GPU_UUID` has not been set by the main thread. Use a distinct
non-overlapping interval for a second GPU, e.g.10..20 rather than0..10.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
cd /public/home/shiys/0selection2
CUDA_VISIBLE_DEVICES="${MAIN_ASSIGNED_GPU_UUID:?main must assign UUID}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -B -u /public/home/shiys/nonrouting_supplement_20261029/as_dispatch.py label --root /public/home/shiys/nonrouting_supplement_20261029 --split train --start 0 --stop 10 --device cuda:0 --main-authorized-gpu
```

Each completed label is saved immutably immediately, so this real budgeted
batch is reusable. `native_as_receipts/train_000000_000010.json` contains actual
mean/wall times and every label SHA. A failure preserves its first error and
certificate in `native_as_failures`; no retry or repair is automatic.

## Full coverage after the main accepts actual runtime

Same command and fixed candidate, with half-open intervals:

- TRAIN10000: `--split train --start 0 --stop 5000` on one GPU,5000..10000 on another.
- VAL1350: `--split val --start 0 --stop 675`,675..1350 on another.
- TEST1350: `--split test --start 0 --stop 675`,675..1350 on another.
- External270: `--split benchmark --start 0 --stop 135`,135..270 on another.

Benchmark input is read from the existing immutable OR-Library records and
stays in the `benchmark` namespace. It is never copied into formal TRAIN/VAL/TEST.
Total formal+external coverage is12970, NOT270 new training inputs.

## Reader/export

Base raw manifest: `/public/home/shiys/nonrouting_supplement_20261029/manifest.json`.
The original fast CPU export is `prepared_cpu/manifest.json`: SMTWTP and BPP
are ready, MKP retains its unobserved AS column and is not yet ready.

After all native formal AS labels actually complete, create a NEW immutable
export (never overwrite the CPU-only version):

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
cd /public/home/shiys/0selection2
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -u -m code.V4.nonrouting_supplement export --include-as --root /public/home/shiys/nonrouting_supplement_20261029
```

Runner argument: `--supplement-root /public/home/shiys/nonrouting_supplement_20261029/prepared_with_as`.
All3 entries retain native directions, fixed solver IDs and the existing
`nonrouting_raw_v1` schema. MKP ready=True only after all formal AS cells exist.

## Independent Ready LPPrimal Fast-pool Option

Preferred fallback to prepare, not silently substitute: `LPPrimalGreedy`.
Solve one bounded LP relaxation, sort items by decreasing primal x value with
stable original item-ID ties, then scan once, adding only under all ORIGINAL
integer capacities. No randomized restart, MIP, current-instance oracle, native
objective change or repair. This is the feasible LP-guided greedy construction
described in Puchinger-Raidl-Pferschy, section2.1/equation9:
https://optimization-online.org/wp-content/uploads/2009/03/2258.pdf .

Implemented separately in `lp_primal_option.py`, candidate ID
`MKP/LPPrimalGreedy/primal_desc_stable_id_v1`, pool ID
`mkp_lp_primal_fast_pool_v1_20261029`. This implements ONLY the cited LP-feasible
construction, not the paper's core/memetic algorithms. Both LP candidates are
LP-guided classical constructions; do not call their variants new independent
algorithm families. Raw floating LP values are ordered without epsilon/rounding;
only original integer weights/capacities decide feasibility.

Three real TRAIN witnesses passed, saved in `lp_primal_witness_v1.json`:

| TRAIN index | n,m,alpha | Integer profit | Solver seconds |
| --- | --- | ---: | ---: |
| 0 | 500,10,.50 | 220955 | .0155284 |
| 13 | 100,30,.75 | 58115 | .0105475 |
| 26 | 250,30,.25 | 56892 | .0091186 |

Timing covers one LP, construction and certificate, not import/I/O or a full
27-cell workload. It is not a full production ETA or a claim of selector gains.
Eight dedicated sidecar tests also passed; no AS solve/profile was rerun.

READY command, ONLY if main explicitly chooses this alternative protocol.
Set `FAST_POOL_CPU_SET` to ONE free main-assigned CPU core first. Do NOT run
this automatically just because AS is still live:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco
tmux new-session -d -s r120_lp_primal_fast_pool "source /public/home/shiys/miniconda3/etc/profile.d/conda.sh && conda activate easynco && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c ${FAST_POOL_CPU_SET:?main must assign one free CPU core} nice -n 10 python -B -u /public/home/shiys/nonrouting_supplement_20261029/lp_primal_option.py prepare --explicit-fast-pool --include-benchmark > /public/home/shiys/nonrouting_supplement_20261029/fast_pool_v1_production.log 2>&1"
```

One single-thread CPU process,2GiB address-space bound, no GPU or installation.
Same10000/1350/1350 synthetic inputs and original split identities. Reuses
SurrogateGreedy/LPDual labels and read-only BPP/SMTWTP prepared entries. New270
external labels/export are under `external_evaluation`, never formal TRAIN.
It writes ONLY `fast_pool_v1/`, preserving all AS and CPU exports.

If main chooses and actually completes it, the runner argument would be
`--supplement-root /public/home/shiys/nonrouting_supplement_20261029/fast_pool_v1/prepared`.
Manifest schema remains `nonrouting_raw_v1`. No full fast-pool production has
been run, no fallback is installed in the current manifest, and this sidecar
does not cancel or replace the active native AS plan.
