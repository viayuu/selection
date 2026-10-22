# R94 I/O Pipeline Recovery

2026-10-22 | Accepted B startup throughput restored; not evidence of model-quality improvement.

## Evidence and Startup Scope

The authoritative archive is `startup_slow_io_attempt1/ARCHIVE.json` under `/public/home/shiys/0selection2/code/V4/runs/R94_pretrained_node_fusion`. Main stopped only the very slow first R94B startup and the unstarted R94A queue. R92A/B and R93A remained healthy and untouched per the handoff; this report did not poll them.

The archived B log records 108 successful updates; actual executed updates are bounded at 108-125, not known exactly. Durable `last.pt` has 0 successful updates and an empty optimizer state. There was no VAL, TEST or LIB evaluation. The startup is discarded, not resumed. Original controller/worker SIGTERM failures remain in `/public/home/shiys/0selection2/code/V4/runs/R94_launch/startup_slow_io_attempt1`.

Archived steps 18/36/54/72 report 7.489/7.407/7.408/7.335 seconds/update; step 1 is 6.988 and step 108 is 7.647. The handoff reports an idle GPU with about 19 GB allocated. These are before-fix observations.

## Bounded CPU Input Microbenchmark

Supplied Sol measurements used the job-1475-owned container, unused CPU 74, CPU input only, no GPU or labels. Each batch requested `1024 x 101 x 128` FP16 values: **26,476,544 bytes**. Disjoint fixed-random TRAIN rows were read while other jobs ran: not paired, controlled tests or training-throughput measurements. This worker did not rerun them.

| I/O policy | Record | Seconds | Physical read bytes | Physical/requested |
| --- | --- | ---: | ---: | ---: |
| NORMAL | 1 | 12.626988 | 2,680,324,096 | 101.234x |
| NORMAL | 2 | 6.585618 | 1,203,400,704 | 45.452x |
| MADV_RANDOM only | 1 | 29.093819 | 18,288,640 | 0.691x |
| MADV_RANDOM only | 2 | 16.189236 | 9,932,800 | 0.375x |
| RANDOM + span WILLNEED | 1 | 0.102639 | 30,584,832 | 1.155x |
| RANDOM + span WILLNEED | 2 | 0.112926 | 30,564,352 | 1.154x |

Physical/requested = physical read bytes / 26,476,544. Ratios below one can reflect cache-served pages, not missing values. Exact times and bytes remain in `io_readaround_evidence.json`.

**MADV_RANDOM alone was rejected:** fewer bytes but slower than NORMAL in both records. Byte reduction alone is not a performance win. CPU probe latency is separate from the actual-run observation below.

## Observed End-to-End Startup Recovery

Root `io_runtime_recovery.json` records accepted B at 270 successful updates, `phase=training`, all 18 tasks finite, checked at `2026-10-21T21:14:58.827089+00:00`. Its complete contents are embedded in the published evidence JSON, so this report does not depend on that sidecar being published separately.

| Observation | Original B startup | Accepted fresh B |
| --- | ---: | ---: |
| Logged timing window, successful updates | 54 | 270 |
| Seconds per successful update | 7.4084640651 | 0.4247711723 |
| Feature-cache preparation, seconds | 747.88 | 0.5115 (reused) |

**Operational throughput restored:** observed old/new ratio **17.4411x**, approximately 17.44x, from exact JSON times. Old54 predates the extra CPU I/O probe; new270 is a different window. Co-resident jobs were active: this is not a controlled benchmark, a precisely established causal factor, or model-quality evidence. Old cache-prep time is supplied handoff evidence; the reused time is in the runtime JSON. No cache regeneration occurred.

<!-- PAGE_BREAK -->

## Adopted I/O Policy and Scientific Boundary

The sidecar combines mapping-level `MADV_RANDOM`, descriptor-level `POSIX_FADV_RANDOM`, and asynchronous `POSIX_FADV_WILLNEED` for explicit requested-row spans before unchanged gather. It limits read-around while allowing queued parallel I/O, rather than random-only demand faults.

Feature content, sample ordering, label/padding rules, RNG setup, logical batch and loss/gradient definitions are unchanged. This does not assert identical floating-point GPU values. No native online encoder fallback is introduced. The approximately 41 GB TRAIN/VAL cache at `/tmp/shiys_R94_B_node_features` is preserved, not regenerated.

Main reports exact equality of scientific source hashes/metadata for model, loss, initialization, native encoder and sample order; launcher, inference and report sources also remain unchanged. Only sidecar I/O and publication files changed. This worker did not rehash sources, weights or arrays; queue/source freeze remain with main. Original protocol-lock SHA256: `54ea04fbdf06d6d63d38550b6f313e4b3b10648dc9f29f307064e245a6846d85`.

## Fresh Restart and Budget Boundary

Accepted A and B use fresh starts from the same original locked seed-2 initialization under the unchanged protocol. Neither resumes discarded trained state: this is resource correction, not numerical-failure recovery, scientific reselection or automatic retry. Old A had not started; the actual-GPU recovery observation above is B only.

B's archived configured budget is 57,600 successful updates. The discarded 108-125 updates are separate recovery overhead (0.1875%-0.2170% of that B budget), not accepted progress, a checkpoint candidate, or a deduction from the fresh accepted budget. At logical batch 1,024, this bounds discarded sample exposures at 110,592-128,000. No accepted-run budget or selection rule is changed by this report.

## CPU/Metadata Checks and GPU Numerical Agreement

The handoff reports byte-for-byte CPU input parity, row order and file-descriptor cleanup checks passing. Full-suite counts remain owned by main in root `cpu_checks.json`; they were not read, inferred or awaited for this update. CPU input parity does not imply bitwise GPU arithmetic.

The initial trainable-parameter SHA256 is identical: `9db3df55a91ca0f331880cfcc4688ffab82b83bb66e80f9cebe9d6978e903cb4`. At common logged steps 1/18/36/54/72/90/108, maximum absolute loss difference is `3.1661518196e-7` (<=3.17e-7), and grad-norm difference is `1.0379652182e-4` (<=1.04e-4). These are close, **not bitwise-equal GPU results**; all objective-part differences are retained in the embedded runtime JSON.

No VAL or TEST was read/used for the fix. Startup throughput is restored at the recorded observation, not a model-quality comparison, completed-model result, VAL/TEST/LIB result or native-online performance conclusion.

## Report-Only Scope and Provenance

Only this Markdown, the two-page `IO_PIPELINE_RECOVERY.pdf` and `io_readaround_evidence.json` are written. Commands use `easynco`, empty `CUDA_VISIBLE_DEVICES`, and threads <=2. No frozen-runtime/model edits, GPU, SSH, raw arrays, weights, forward passes, profiling, training, launch, commit or subagents; no other changes reverted.

Archive/log facts and the small runtime JSON were read locally. CPU probes, checks, peer health, source equality and cache provenance remain main-controller evidence, not independently re-executed here. Actual GPU observations come from the recorded accepted run, not a new report-side run. PDF uses the existing ADMIN CJK font and Agg/embedded-font setup.
