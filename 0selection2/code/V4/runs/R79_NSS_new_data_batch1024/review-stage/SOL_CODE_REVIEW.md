# R79 NSS Solmax Critical-Path Code Review

- Reviewer role: Solmax; **provisional / same-family**.
- The current Codex session performed this review. No independent Solmax backend was available/invoked; do not treat this as independent cross-family certification.
- Date: 2026-10-15. Reviewed only after explicit confirmation that NSS code was ready/running.
- Scope: `r79_nss_experiment.py`, directly used original NSS model/loss/dataset/augmentation/configuration, and the existing four synthetic CPU parity tests.
- Decision: **No reproducible P0/P1 error found in the requested active paths. Do not pause training.** No worker script/model/loss, checkpoint, training configuration or diagnostic output was edited. No GPU/profile/full cost evaluation was run.
- Snapshot boundary: the reviewed script was `7553646c738b7e0d2210bc623c71ccda3efb72d00c78d8c6607869b92833bd94`. The final inventory found a concurrent worker revision, `6374c42e5d4866f483a4c73405b235f62b2bff2dd14b7e6e5eb93b43ac153119`. Line references, tests and conclusions in the original review sections apply to that reviewed snapshot, **not the new optimization revision**. A separately authorized, code-only resident-path supplement follows below; it does not repeat the original full review or tests.

## Findings First

No actionable major correctness regression was found in the reviewed new-data recipe snapshot. The performance observations below are static nonblocking candidates, not measured attribution of the417-478 samples/s throughput. The worker's limited micro128 optimization was not run, changed or certified by this reviewer; the new revision may already address some of these observations.

## Recipe and Semantics

| Path | Evidence and conclusion |
| --- | --- |
| Original architecture / loss | `r79_nss_experiment.py:47` imports original modules by path; `:273` constructs the original Selection_model; `:437` uses original RankingLoss(K,K). The only model adapter binds the original pooling padding operation to the embedding device (`:257`). It does not replace the encoder, head, parameters or loss. Original trainer `trainer.py:133` also passes raw logits and FP32 costs into RankingLoss. |
| 50 epochs / eight transformations | `r79_nss_experiment.py:125` checks50 epochs, rank loss, data_aug=True and original disabled auxiliary features. `:470` runs exactly50 trained epochs. `NSSDataset.__len__` at`:227` returns8N; `:231` maps an index to `(source_row, transform)`. This is **eight-fold coordinate augmentation, not eight-fold cross-validation**. Original `utils.py:23` implements `(x,y)`, `(1-x,y)`, `(x,1-y)`, `(1-x,1-y)`, `(y,x)`, `(1-y,x)`, `(y,1-x)`, `(1-y,1-x)`; CVRP demand is retained unchanged. TRAIN70114 becomes560912 augmented examples/epoch;547 full1024 batches plus one784 tail =>548 updates/epoch,27400/model. VAL/TEST/LIB use unaugmented N, not augmentation-time ensembles. |
| Effective1024 / accumulated64 | `r79_nss_experiment.py:431` loads logical batch1024; `:477` preserves the exact final784 tail. `:291` weights each micro loss by micro_rows/logical_rows; `:300` zeros once and `:317` steps Adam once. Full batch1024 with micro64 is16 forwards/backwards, not16 Adam updates; micro128 would be8 while retaining the denominator1024. Tail784 with micro64 is12*64+16 and still has one step. |
| No batch-pair loss coupling | Original `loss.py:16` ranks solver columns independently per row; `:19` adds per-row NLL losses with mean reduction. There is no cross-instance pair construction or batch-global winner denominator. The sum of micro_rows/N weighted means equals the full logical-batch mean gradient. |
| No BatchNorm inequivalence in this recipe | Original model has optional BN branches, but **both actual configs use `norm: rezero`**, `model.py:371`. Attention operates over nodes within each sample; masked mean/max and pooling use each sample's valid mask/length (`model.py:160`,`:257`). No BN instance is present in either current model. This conclusion must not be generalized to hypothetical BN-configured NSS models. |
| FP64 costs and pool order | `r79_nss_experiment.py:168` gets the original solver_order_ours pool;`:172` verifies new-data metadata order and row count;`:175` verifies LIB result-column names. Costs/gaps/times are extracted in the same row and column order;`:190` keeps source costs FP64 and native `ind`. Raw arrays are not reordered by predicted rank. NSS collate `dataset.py:108` converts the loss costs to FP32, matching original training. |
| Native vs FP32 labels | `r79_nss_experiment.py:193` retains native ind and`:196` checks source FP64 argmin consistency. `summarize` at`:329` deliberately reports original NSS FP32-cost-argmin top-k, while`:349` separately reports native-ind top-k and`:344` counts disagreements. `:331` uses FP64 selected costs for mean cost/regret. The FP32 training-rank convention is not silently substituted for native winners in saved NPZ (`:379`). |
| Fixed new-VAL best-gap rule | `summarize` at`:334` uses the saved gap table, falling back to cost only if that entire table is zero; the current new-data VAL gap table is nonzero. `:511` compares strict improvement of that same mean top-1 gap across trained epochs, with ties retaining the earlier checkpoint. Untrained initial VAL at`:467` is diagnostic, not a checkpoint candidate. First trained epoch can be persisted, fixing the original trainer's first-epoch-save omission. This is retained-pool Oracle regret, not optimal-tour gap. |
| Single best for TEST and LIB | `freeze` at`:558` requires both task-specific models to complete the original fixed update budget and freezes their VAL-best paths/hashes. `evaluate` at`:578` checks the boundary before opening data;`:582` loads each task's best model once, then reuses it for new TEST and that task's original LIB. Exclusive per-split receipts at`:596` prohibit automatic repeated inference after started/failed/completed attempts. No original TEST loading, TEST/LIB-driven retraining, best reselection or mixture of epochs was found. |

Candidate names inspected:

```text
TSP: BQ, DIFUSCO, DIFUSCO500, ELG, LEHD, OMNI, T2T, T2T500
CVRP: BQ, ELG, ICAM, LEHD, MVMOE, MoSES_CaDA, MoSES_RF, OMNI, RELD_CVRP, RouteFinder
```

## Bounded CPU Evidence

All four existing `test_r79_nss.OriginalNSSParity` tests passed: only requested seed/batch overrides, lazy augmentation parity, original-model padding/microgradient parity, and VAL-gap/zero-gap fallback behavior.

An additional tiny TSP/CVRP witness set ReZero coefficients to0.2 (not the trivial zero initialization), used node lengths7/11/5/9/8, and compared full5-row original-model RankingLoss gradients with accumulated2+2+1 adapted-model gradients. No BN modules existed. Maximum absolute gradient difference was **1.1920928955078125e-7** for each task. This supports sample independence beyond zero-initialized residual branches; it is not a real GPU1024/64 or1024/128 memory/performance test.

## Nonblocking Pipeline Notes

1. Original `model.py:259` calls `int(lengths[i] * downsample_ratio)` inside a per-instance GPU loop. Two pooling blocks imply about2048 GPU-scalar-to-Python reads per logical1024-row update, plus many small per-instance topk/gather operations. `model.py:349` also converts a GPU predicate to Python for every attention call. These are concrete synchronization candidates; no timing percentage or dominant-bottleneck claim is made. Micro128 reduces repeated micro-level calls but does not remove the per-instance pooling loop. Do not rewrite original model internals casually to address this.
2. `transfer` at `r79_nss_experiment.py:279` performs four slice transfers per micro. Collate first pads the whole logical1024 batch; slicing rows does not crop short micros to their own maximum valid length. This can carry avoidable padding/transfer/attention work. Any adapter optimization must preserve masks, true scales, CVRP demand, row order, output parity and original loss; no changes were made here.
3. Every formal update computes gradient norms, synchronizes `.item()` values, appends JSON and calls W&B (`:313`,`:490`,`:505`). This is additional host work, but its share of elapsed time is unmeasured. Existing `seconds` starts after DataLoader has yielded a batch (`:474`), so the quoted samples/s is update-body throughput, not a full wall-clock pipeline measurement including time waiting for the loader.

These observations do not justify stopping/restarting the current train, changing the original model/loss, or changing logical batch1024. A worker-owned bounded micro128 comparison is separate from this review. On resume, the inspected code restores the saved micro size (`:450`), so an optimization should not assume that merely passing a new CLI micro value overrides a checkpointed value.

## Limits and Snapshot

No complete real-data label/cost audit, final checkpoint/TEST execution, full50-epoch run, GPU bit-exact parity, memory test or profiling was performed. Exact softmax ties/FP32 near-ties and CUDA kernel rounding remain implementation-level residual risks; this review does not redefine original tie policies. Original model optional BN branches are outside the active ReZero recipe.

Captured source identities of the reviewed snapshot:

```text
7553646c738b7e0d2210bc623c71ccda3efb72d00c78d8c6607869b92833bd94  r79_nss_experiment.py
18e35f07bf6f582ed5a37a38e5fc028862ed1c05f89100f386eff089609066c8  original model.py
8c059f94a8fa72d1f4d72e1b56c1921703f7fd714031e5c9709e082ae57e6677  original loss.py
7dcba07900939e91c6864434c9e72146181e6de1d5ebf19d39f763c113492cdd  config_TSP_ours.yml
61b3a0a2fe463e16f23a99f8ac3377a0b5163e61b44175c090b1436941515cf8  config_CVRP_ours.yml
```

Final inventory of the concurrently revised script:

```text
6374c42e5d4866f483a4c73405b235f62b2bff2dd14b7e6e5eb93b43ac153119  r79_nss_experiment.py (new worker revision; not covered)
```

No Git commit. No TEST statistics were used to select rules or assess the training implementation.

## Supplement: GPU-Resident Input Path

Date2026-10-15; **provisional / same-family**. User authorized only a quick comparison of the resident input/augmentation changes against the previously inspected7553646c snapshot. The actual current file was `905e88538eb5d928f12ed12f3e3ebd48c031e53a36f206bd0febea6e5ff6c66b`, unchanged across these bounded reads, rather than the earlier6374c42e revision. No archived7553646c/6374c42e source file was present in the run directory; this is a semantic comparison with the earlier inspected functions, not a claimed byte-for-byte Git diff. No tests, prep, training, GPU operations or profiling were rerun. Only this review record was edited.

### Finding: Nonblocking Fresh-Start Guard

**[P2] GPU-resident inputs fail the legacy first-update pinning check.** At `r79_nss_experiment.py:693` the first formal update still unconditionally requires `batch[0].is_pinned()` (`:696`). Resident loader output is a CUDA tensor, not pinned CPU memory, so a fresh `input_path=gpu_resident` run would raise after its first optimizer step. The currently resumed run has updates>1 and does not enter this guard; this is **not a reason to stop/restart current training**. Worker should make the smoke assertion input-path/device aware before a future fresh resident launch. No worker code was modified by the reviewer.

### Resident-Path Equivalence

- `ResidentNSSLoader` (`:260`) retains the CPU seed2 generator and the same RandomSampler over8N augmented indices. `row=id//8` and `fold=id%8` preserve source row/transform identity; batch membership, full1024 batches and final784 tail are unchanged.
- Packed gathering (`:280`) uses the maximum **true length of this logical batch** for width, indexes points by offsets plus node column, and masks columns beyond each row's true length. Clamping protects invalid tail gathers; valid source points retain their correct offsets.
- Vectorized swap/flip (`:291`) gives the same eight ordered XY transformations as original augmentation. CVRP demand is concatenated without reflection. Crucially, `:297` zeroes **all input channels** of invalid nodes after reflecting coordinates, so a reflected padded zero cannot become a fake `(1,1)` node.
- Winner, cost, gap and time all use the same `rows` gather (`:299`), without permuting candidate columns. GPU loss costs are FP32, matching the original collate conversion. Original CPU cache costs stay FP64 for reports/native labels; the resident dictionary does not mutate the source cache or pool order.
- `accumulate` (`:454`) still weights every micro mean by `micro_rows/logical_rows`; Adam still steps once per logical batch. The optional TRAIN-only probe restores model, optimizer and CPU/CUDA RNG in its `finally` block (`:424`); probe sampling uses a cloned shuffle generator, not the formal stream, and probe steps do not increment formal updates.
- Fixed50 epochs,27400 updates/model, the same strict VAL mean-top1-gap best rule, and the same immutable per-task best reused for TEST/LIB remain in place (`:664`,`:715`,`:744`,`:755`,`:779`). New timing/telemetry and resident inputs do not select a new checkpoint criterion.

### Mixed True Lengths and Within-Batch Grouping

Original `model.py` and `loss.py` hashes match the first review. There is no fixed-N reshape requirement in the inspected active encoder: attention masks padded **keys** (`model.py:343`), masked mean divides by each sample's valid-node count (`:165`,`:225`), masked max excludes padding (`:171`,`:231`), and pooling selects `floor(true_length*0.8)` nodes per sample (`:257`). Padded queries may be computed, but their outputs are excluded from valid-node summaries/selection. The current ReZero normalization (`:393`), row-local attention/pooling and row-wise RankingLoss have no cross-sample statistics; other samples' padding width is a tensor-shape/performance context, not a mathematical batch-context dependency. This statement does not apply to the optional BN configurations.

**Grouping by true N inside the already sampled global1024 is mathematically equivalent** under this current deterministic, no-dropout/no-BN recipe, provided that:

1. Keep exactly the same logical-batch membership and apply the same row permutation to inputs, true scales, targets and IDs. Group using actual cached lengths, including the CVRP depot, not padded width or a substituted scale label.
2. Crop each group's node axis and mask to its true N; preserve masks, demands and per-instance pooling counts. Equal-N groups need no input padding, including after equal-count downsampling.
3. Weight every group/micro mean by its row count divided by the **original logical batch size1024, or784 for the existing tail**. Zero gradients once and make one Adam step after all groups; do not divide by group size as the global weight or step Adam per group.
4. Do not move samples between logical batches, drop bucket tails, change epoch/update/VAL budgets, alter the original model/loss, or use TEST/LIB to select this change.

This is a real-arithmetic equivalence argument from code, not proof of identical FP32 reductions/topk tie decisions or faster execution. Shapes and accumulation order can change CUDA rounding. More/smaller buckets can also increase launch overhead, so the micro128/192 slowdown does not by itself prove grouping will help.

Reported micro128/192 peaks3.56/4.47GiB and host fetch/prep0.05% do not support treating micro64 as a VRAM limit or continuing CPU-loader optimization as the primary remedy. Original per-instance pooling/scalar synchronization remains a code-evidenced model-side cost candidate (`model.py:259`); its measured share was not established here. No new optimization or profile was initiated. **No P0/P1 regression found in the resident-path comparison; current resume may continue.**
