# Versioned nonrouting supplement

Scope owner: `nonrouting_supplement.py`, `test_nonrouting_supplement.py`, this
note, and `/public/home/shiys/nonrouting_supplement_20261029/` only. The latest
user authorization supersedes the preparation-only flag in the Oct28 plan.
No old data, other workers' data/model/runner code, Git state, or GPU jobs are
changed. R119 is outside this task.

## Assumptions fixed before implementation

| ID | Choice | Source |
| --- | --- | --- |
| S01 | New namespace `nonrouting_supplement_v1_20261029`; data seed 20261029. Independent split and attempt substreams. | default, before code |
| S02 | SMTWTP 10000/3750/3750 TRAIN/VAL/TEST, balanced over 75 cells. Integer p=1..100 and w=1..10. Exact ceil/floor signed due interval, sample first, then clip at zero. | user and frozen preparation |
| S03 | MKP 10000/1350/1350 over 27 cells. Integer weights 1..999, noise 0..499, floor-mean profits and rational floor capacities. Record deterministic rejection attempts; require every item individually feasible and each capacity below total load. | user and frozen preparation |
| S04 | BPP reuses exactly 10000 accepted v2 TRAIN records by read-only source references, including original IDs, identities, seeds and labels. Only 1000 VAL and 1000 TEST are new. | user |
| S05 | No public benchmark enters TRAIN/VAL/TEST; no new pilot population or expansion to 100k is authorized by this implementation. | user, bounded scope |
| S06 | EDD/WSPT/ATC, FFD/BFD and SurrogateGreedy call the exact existing function bodies without importing their unavailable OR-Tools dependency. Stable IDs and ATC k=2/remaining-job mean are unchanged. | user, inspected source |
| S07 | LP is one SciPy HiGHS dual-simplex relaxation followed by descending profit / resource-shadow-price construction, with stable item-ID ties and exact integer feasibility. Zero denominator means +infinity priority; no epsilon objective modification or repair. | preparation and Raidl-Gottlieb section 3.5 |
| S08 | MBS_prime is the Fleszar-Hindi section 3.1 algorithm: fix largest remaining item, exact one-bin subset enumeration in descending-size/stable-ID order, retain first strictly best packing, stop enumeration at zero slack. No VNS, random restart, greedy subset substitute or shortened search. | original paper, pending implementation checks |
| S09 | MKP AntSystem stays a separate 20-ant/100-iteration native configuration. Only adapter/CLI and a CPU witness are run here; the main thread owns GPU dispatch. FP64 normalized capacity mapping and original integer certificates are both recorded. | user, explicit mapping configuration |
| S10 | Native integer objectives and min/max direction are retained. Exact ties use all_best_mask; unavailable/failed/deferred methods are unobserved, never zero-filled. SMTWTP zero-oracle relative regret is undefined, not epsilon-normalized. | user and Averroes schema |
| S11 | CPU production uses at most three single-thread domain workers, six-core affinity, low priority, and an aggregate RSS guard below 8 GiB. No dependency upgrades. Source and protocol hashes are frozen before production. | user and default |

## Early API for the data owner

Module: `code.V4.nonrouting_supplement` (no dependency on `nonrouting_data`).

- `make_instance(problem, split, index, seed=20261029)` returns an accepted-style
  `instance` dictionary with native `data`, immutable identity and generation
  provenance. BPP reused TRAIN is obtained through `iter_records`, not regenerated.
- `iter_records(root, problem, split)` streams completed immutable JSONL shards.
- `as_sample(record, solver_ids=None)` returns exactly `inputs`, `labels`, and
  `metadata`, with NumPy arrays and no labels/IDs/source/certificates in inputs.
- `solve_instance(instance, candidate, device='cpu')` returns a validated native
  label including full certificate and candidate protocol identity.
- CLI `prepare`, `audit`, `profile`, and `mkp-as`; exact commands and measured
  times will be appended after the first real certificate checks.

Input fields align with Averroes' `data_preparation.md`: BPP `sizes, capacity`;
MKP `profits, resources[N,R], capacities`; SMTWTP `p, w, d`. Labels expose
`native_objective[K]` float64, `observed_mask[K]`, `all_best_mask[K]`,
`canonical_winner_index` (first exact best in fixed sorted solver-ID order, not
a unique native winner), and native direction. Metadata retains all source IDs,
input identities, base groups, version, parameters, seeds and certificates.

Storage: `<root>/manifest.json`, `protocol.json`, and
`records/<problem>/<split>/<start>_<stop>.jsonl`. Each JSONL record has
`instance`, `labels`, explicit missing-candidate statuses and source references.
BPP TRAIN is the explicit exception: `instance_ref` binds the original shard
SHA256 and 1-based line, and only the NEW third-method label is stored locally.
`iter_records` resolves that immutable view and merges original labels in memory;
it does not duplicate or rewrite old inputs/labels. Consumers should use this API.
The manifest lists each shard with SHA256/count; incomplete shards are never
advertised as complete. BPP source rows are referenced and never rewritten.
`full_pool_complete=false` is the broad legacy deferred-method inventory flag.
The bounded declared pool's readiness is `ready_for_selector_training` and
`measured_pool_complete`; the optional receipt gate adds an explicit
`declared_main_pool_complete` receipt only after all fixed candidates complete.

Candidate additions are `BPP/MBS_prime/fleszar_hindi2002_v1`,
`MKP/LPDualGreedy/highs_ds_v1`, and
`MKP/AntSystem/deepaco_native20x100_fp64_v1`. Existing constructive IDs stay
unchanged. All candidate columns sort by full solver ID.

MBS_prime source: Fleszar and Hindi, Computers & Operations Research 29 (2002),
sections 2 and 3.1, DOI https://doi.org/10.1016/S0305-0548(00)00082-4.
The original paper text was inspected, including duplicate-size pruning,
remaining-sum bound, the smallest-item stopping condition and the fixed seed;
the implementation will be checked against exhaustive small-item enumeration.
LP source: https://www.ac.tuwien.ac.at/files/pub/raidl-04a.pdf, section 3.5,
equation 7; no profit biasing or evolutionary algorithm is being added.

## Initial verification ledger (superseded by delivery updates below)

The implementation followed the assumption ledger above. Initial preparation,
small fixtures and runtime estimates are historical stages, not current pending
work; final evidence and readiness are recorded in the delivery updates below.

Direct `send_input` is not exposed in this session. This owned note is the early
coordination contract for Averroes (01a1834a-85dd-7581-9592-bc8fb1a2d2e3).

## Runner integration contract, confirmed against the current reader

`export_prepared` independently writes the existing `nonrouting_raw_v1` binary
schema (ragged input bins/offsets/shapes, metadata offsets and five label arrays).
No data-worker module is imported or modified. For the CPU-only export pass
`--supplement-root /public/home/shiys/nonrouting_supplement_20261029/prepared_cpu`
to the runner. Its `manifest.json['nonrouting'][problem]` entries have fixed
`candidate_solver_ids`, native `objective_direction`, and formal splits.
SMTWTP/BPP become ready only after all their three candidates complete. MKP
remains incomplete until separately dispatched AS labels finish. A later
`export --include-as` writes a NEW `prepared_with_as` export; it never mutates
the existing CPU export. Do not silently use a two-candidate MKP as complete.

Implementation now exists; eleven small deterministic/real-certificate checks
passed, including full signed-interval and27-cell generation checks. A BPP
double-resolution bug was caught before production and fixed with an
idempotence regression; `smoke_v1` failure and source are preserved, and the
new `smoke_v2` completed all three real six-row domain fixtures.

Formal CPU production completed at 2026-10-29 14:47:22 CST, rc0, in2m51.59s:
SMTWTP10000/3750/3750, MKP10000/1350/1350, BPP10000 old read-only plus1000/1000
new. All executed candidate failure counts are zero. Parent maximum RSS
reported by time was80972KiB (NOT aggregate child peak); each of three CPU
workers had a2GiB address-space bound and all shared only cores42..47.
Compatible CPU export and read-only audit subsequently completed (see below).

GPU entry point and exact main-owned dispatch commands are in
`/public/home/shiys/nonrouting_supplement_20261029/GPU_COMMANDS.md`.
The independent `as_dispatch.py` leaves the production-frozen module unchanged;
it vectorizes native capacity checks only. Exact full100-iteration path
equivalence and the full CPU witness batch subsequently completed (see below).

## Delivery update,2026-10-29 14:55 CST

CPU export and audit COMPLETE. `audit.json` passed42,200 inputs/113,900 label
rows, with zero source changes, failed labels, cross-split duplicates or overlap
with645 public benchmark and240 legacy BPP pilot inputs. New label executions
are93,900;20,000 original BPP FFD/BFD labels are reused, not falsely counted as
new solver work. `NonRoutingDataset` and collator compatibility passed directly.

Runner can consume `prepared_cpu/manifest.json` immediately for ready SMTWTP
and BPP. MKP correctly remains ready=False with the fixed unobserved AS column.
The largest actual MBS_prime call across12,000 instances was0.04787 seconds;
there is no deferred MBS tail. This is not a model-benefit claim.

AS full native path equivalence passed at n100/m5, n250/m10, n500/m30, all at
20 ants/100 iterations. Complete saved CPU times, certificates and the very
rough80 CPU-hour total warning are in `as_vectorized_cpu_witness.json`.
GPU commands include benchmark270 as a separate external namespace. No GPU
was used here. The main thread chooses actual GPU throughput/budget or the
documented, source-backed LPPrimalGreedy fast-pool alternative; no shortened
AS, fabricated third candidate, or silent pool replacement is released.

## Final CPU handoff,2026-10-29 15:45 CST

No changes to the two main-pinned active sources after full AS dispatch:

- `nonrouting_supplement.py`: `7355064443385b9cf6adae481a12584325d009b099325b982b32c391695b8cda`
- `as_dispatch.py`: `2bf0d404f200dd7b3ec8fd55b788f2b80150070cbfbd914dca702186b527c557`

MKP external270 LPDualGreedy labels completed rc0 in2.82746s, using the SAME
frozen CPU solver and original benchmark identities. Receipt:
`/public/home/shiys/nonrouting_supplement_20261029/benchmark_cpu_receipts/MKP_LPDualGreedy.json`.
Original SMTWTP375 already has EDD/WSPT/ATC and is reused, never copied to TRAIN.

Main has started full AS in `tmux r120_mkp_full_pool`, using its owned
`resources/as_full_pool_plan.json`: four physical GPUs, four single-thread
processes/GPU. Main owns actual ongoing coverage and concurrent throughput.
This handoff does not re-profile AS, poll its processes, or equate concurrent
runtime with exclusive GPU speed. CPU generation/labels are complete; AS is the
only formal-pool gap. Do not start another dispatch from old example commands.

Independent postprocessing `full_pool.py` imports receipt-backed labels,
allows identical overlapping reuse, rejects conflicting evidence, validates
full native20/100/seed/path/integer certificates, and refuses a ready export
while any formal row lacks a completed receipt. Its benchmark stream stays
external. Main may keep its existing `export --include-as` sequence after real
completion; the gate is an optional strict wrapper, not an altered AS chain.

Independent optional fast pool is implemented in `lp_primal_option.py`,
candidate `MKP/LPPrimalGreedy/primal_desc_stable_id_v1`. Source: Puchinger,
Raidl and Pferschy, section2.1, text preceding equation9, primary paper
https://optimization-online.org/wp-content/uploads/2009/03/2258.pdf . It solves
one bounded LP, orders raw primal values descending with stable exact-ID ties,
and greedy-fills under original integer capacities. Only this constructive rule
is reproduced, not the full paper's core/memetic methods. Three real TRAIN
witnesses certified15.53/10.55/9.12ms; no full-grid ETA or selector gain inferred.
Eight new sidecar checks passed; no AS solve was rerun. Main may explicitly run
`prepare --explicit-fast-pool --include-benchmark` to create a NEW alternative
`fast_pool_v1/prepared`; full fast production has NOT been started here.
Exact one-CPU/tmux ready command and output schema: `GPU_COMMANDS.md`.

The3-page data/candidate PDF and4 CSVs are under
`/public/home/shiys/nonrouting_supplement_20261029/production_results_v1/`.
The report preserves ties and native max/min directions. The fixed initial27 AS
snapshot has LP26/SG1/AS0 unique wins, not a full-pool conclusion. BPP MBS improves
8646/877/874 rows and worsens167/19/9; FFD/BFD tie on all12,000. Oracle saves only
9 bins over fixed MBS on1000 TEST rows: genuine candidate diversity but limited
remaining selector headroom. No model evaluation/empirical selector improvement
claim is made. `HANDOFF.md` and `publication_manifest.json` list exact small
files for main's later curation; no Git operation or new watcher is scheduled.
