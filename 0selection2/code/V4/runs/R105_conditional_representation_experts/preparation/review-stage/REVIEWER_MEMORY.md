# R105 Reviewer Memory

Updated: 2026-10-23 (Asia/Shanghai). Scope: this R105 engineering review only.

## Current Decision

- Review record: `round01.md` in this directory.
- Engineering readiness: **9/10; ready, provisional**.
- Findings: no actual P1, P2 or P3 finding identified; no requested code fix.
- `review_independence`: `same-family`.
- `acceptance_status`: `provisional`.
- Requested reviewer: `gpt-6.1-sol`, reasoning `max`. Exact backend/effort
  attestation is unavailable; do not claim it was independently verified.
- No Astra or secondary/spawned reviewer was used. No efficacy claim.
- Budget: one initial round used; zero fix-checks used. At most one fix-check
  if an actual fix needs checking, not an open-ended review loop.

## Stable Contract And Evidence Anchors

- Main owns `PLAN.md`, `protocol_lock.json`, final source freeze/readiness and
  launch. The launcher owner notified final launch/goal/wave/tests freeze before
  review. A reviewer verdict is not a ready/start receipt or launch authorization.
- Read all six actual `r105_{model,experiment,reporting,launch,goal,wave}.py`
  modules and all five `test_r105_*.py` files. They were untracked additions at
  initial diff/status inspection, so reading tracked diffs alone is insufficient.
- `r105_experiment.private_engine()` binds cap lookup and final-VAL JSON binding
  in a private R84 module. Its train/evaluate globals use R105 caps, while the
  R84 loop resolves the new model/objective/checkpoint through integration hooks.
- Four dense FF experts per each of four instance layers, width128/joint2.
  A uses zero router inputs and learned biases, not fixed-uniform gates; B uses
  current post-attention nodes. Both execute all experts and begin with equal
  complete fresh tensors/functions. Each has3,425,561 parameters; A's2,048 router
  feature weights have zero gradients. Never call this equal active freedom or
  sparse/equal-FLOPs compute.
- Scratch seed2 original reference and expert0 mapping plus isolated seed105002
  extra experts are validated before training. Geometry buffers come from frozen
  R84 TRAIN preparation. Trained restoration is strict, includes buffers, and is
  distinguished from fresh initialization. No trained donor initializes a run.
- Original full18 CE/comparison/risk/stateless two-forward R-Drop, seed2,
  logical1024, updates57600, warm4320, VAL1440, TRAIN7200, micro128 for TSP/CVRP,
  other512 and eval128. Exact-row accumulation and original same-batch/RNG AMP
  replay remain; no terminal job retry, OOM shrink, extra loss, teacher or quota.
- Unique strict-minimum whole18 VAL checkpoint, first exact tie; fixed original
  R88B physical TSP node0, methodID+34, score MLP and maximin. Labels/costs are
  supervision only; routing RAW reader opens input arrays only.
- `selected_capture_proof` checks checkpoint file, complete state, protocol,
  arm/split, selected update and archived identity. Official selected VAL is
  bound without archive rewrites. Selected TRAIN null file SHA is allowed only
  under exact-state proof, not accepted as a generic missing-identity exception.
- Reporter uses the actual R105 factory for checkpoint/LIB with private historical
  adapters. First128 VAL routing groups by exact real N, excludes special tokens
  and padding, and reports entropy of graph-mean mixtures, not mean node entropy.
- Exact pipeline: train (includes locked full TEST), idempotent test/routing,
  standalone timing, LIB batch2, CPU PDFs. Saved postprocessing never reselects.
  Five PDFs/NSS4/full metrics/fixed36 and whole-version saved comparison remain.
- Existing job1465/gpu03, original two3090 UUIDs, CPU quartets/mem9GiB and mutexes;
  both R1024090 workers remain untouched. Missing cache fails without copy/repair.
  Main reports R104 completed/pushed `cb7afd2a6`; this review did not audit Git or
  remote state. Actual predecessor-acceptance test result was supplied, not rerun.
- Explicit main authorization/source freeze precedes dispatch/watch; one-shot
  failure receipts preserve first errors. One curated whole-version publication
  is explicit; engineering completion does not mean research-goal success.

## Evidence Status And Boundaries

- Supplied passes: model14 plus runner/reporter15 =29/29; launcher/goal23/23
  final in2.314s; total52/52. Test code and call paths were inspected, not rerun.
- No reviewer GPU/Slurm, job launch, realdata/solver execution, production timing,
  TEST/LIB/PDF run, commit/push or subagent spawn occurred. Shells sourced conda
  and activated `easynco`. Only this memory and `round01.md` were written.
- Real CUDA FP16/max-size memory fit, throughput and full production pipeline
  outputs remain unverified in this review. Main's existing bounded TRAIN-only
  witness and authorized pipeline supply that evidence; do not add duplicate
  preflights or reopen parameter/loss/model searches on this review's behalf.
- No efficacy inference from CPU tests, gate diversity, fewer zero selections,
  or identical fresh tensors. Final TEST is development-exposed, not blind.
- Do not widen scope into a new hash framework, speculative security issues,
  unrelated historical audits, or manufactured low-priority findings.
