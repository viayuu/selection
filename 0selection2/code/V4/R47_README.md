# R47B: instance-conditioned performance memory

Only R47B is implemented/run in this round. No R47A training, no API, no source
embedding, no extra seed, no relabeling and no new loss. Existing R45A results
are a historical reference, not a newly matched control.

## Model

The R45A handcrafted34-feature + solver-ID backbone is initialized from scratch.
Its four-layer instance Encoder runs once per query. Masked mean/max before
node/solver interaction plus log1p(n)/6 form a257-dimensional retrieval summary.
The original two joint layers, pooling, decoder and maximin decision remain.

Training memory contains only training summaries, original FP64 relative gap
vectors, native winners as global solver IDs, candidate masks, sizes and
input-derived base-instance IDs. `log1p(gap/.01)` is used as behavior input;
performance evaluation uses original costs. No solver-wise normalization.

Retrieval uses one shared LayerNorm/Linear key map, unscaled negative squared
L2 and32 neighbors from the same problem. Index selection is discrete; selected
reference keys are recomputed with gradients. Both sides train the shared key
map; query gradients also reach the current instance Encoder. References' graph
summaries are detached epoch snapshots, refreshed with the current Encoder in
eval/no_grad mode at epoch start and before every formal evaluation.

Each solver reads its own historical gap/winner values plus an instance/neighbor
key-difference correction, through shared networks. The128-dimensional result
is concatenated to its original384-dimensional scoring features. The old first
score-layer block and all other shared parameters retain the original scratch
initialization; new behavior weights are normally initialized, not zero-gated.
R43A comparison objective and two-forward/one-update R-Drop are unchanged.

## Leakage boundary

All query splits require input-derived `base_id`; every retrieval excludes the
same group, including train_eval and input duplicates in val/test. Coordinate
fingerprints conservatively group node permutations and D4 copies at1e-6
precision, including demand/time/depot attributes. This is only an exclusion
rule, not a claim that solver labels are transformation-invariant. Ordered
ATSP matrices use exact FP32 content. The exported data contain no external
lineage IDs; unmarked nonidentical ATSP permutations cannot be inferred.

Neighbor selection sees summaries and base IDs, never query labels or reference
performance. Only after selecting neighbors are their gap/winner rows read.
All train data still participate as supervised queries, with full coverage and
tail batches retained. Val/test never become references. Checkpoints contain
the training memory and can infer offline; before final evaluation memory is
rebuilt using the selected checkpoint, never reused from another weight state.

## Protocol and outputs

Seed2, batch128, AdamW1e-4/weight_decay1e-4, dropout.1,40 epochs maximum,
3-epoch warmup/R-Drop ramp, CostController LRpatience3/early-stop8/min_epochs15,
min_delta.001 percentage points. Strict lowest val macro_vs_sbs_pct selects best.
Full val every epoch; full train_eval every5 epochs and final. No test until
selection is locked. A single post-training validation control permutes whole
gap/winner rows within exact problem/size, leaving keys and weights unchanged.
A train-only size-quartile average-gap prior is also reported without training.

Outputs: `runs/R47_behavior_memory/`, including config, schedule, launch source,
train log/history, best/last checkpoints, train/val curves, mechanism predictions,
full test predictions, per-problem/family tables and paired changes versus R45A.

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
R47_GPU_UUID=<authorized-idle-3090-uuid> bash code/V4/run_v4_r47.sh --stage all
```

## Paper basis and differences

- [TabR, section3.2 Steps2-4/equation5 and appendixA](https://arxiv.org/html/2307.14338v2):
  shared key-only squared-L2 retrieval and label values plus key-difference
  correction. Its official [implementation](https://github.com/yandex-research/tabular-dl-tabr/blob/main/bin/tabr.py)
  also separates no-grad neighbor search from gradient-bearing selected keys.
  Its table inputs,96-neighbor default and optional context freezing are NOT
  copied. Graph-summary epoch caching,32-neighbor context, per-solver gap/winner
  values and maximin are project-specific adaptations.
- [Alors, Artificial Intelligence2017](https://www.sciencedirect.com/science/article/pii/S0004370216301436):
  algorithm selection as recommendation using instance/algorithm performance
  relations. This provides motivation, not the neural retrieval implementation.
  Publisher abstract was accessible; the HAL full-text mirror denied access.

These sources motivate the mechanism, not a guarantee of improvement. Without a
fresh R47A run, improvements over R45A are historical comparisons only. A single
memory shuffle is a sensitivity check, not a complete causal test.
