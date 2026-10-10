# R50: frozen solver-encoder transfer

Scope: OVRPTW, RELD_MOEL (0) vs RELD_MTL (1), seed2. Original FP64 cost labels; exact ties excluded. No test, new instances, new costs, selector continuation, solver decoding, or extra experiment branches.

## Fixed protocol

The exact R49 split and R49B 4000x128 sample plan are reused. All 8000 training instances appear 64 times each. Development: 999 non-ties of 1000; internal: 1000; original validation: 1000. All 21 evaluation points (u0,200,...,4000) evaluate complete sets in eval mode, with tails retained. Strict minimum development CE alone locks a single checkpoint per model.
A freezes original randomly initialized MOEL/MTL architectures; B freezes R48-pinned epoch5000 Train_ALL solver weights. Encoders stay eval/FP32/TF32-off and never enter the optimizer. Both groups copy identical fresh readout parameters and use paired dropout/sample streams.
Each node retains both 128-channel representations. Separate trainable LayerNorms precede concatenation with 7 original node fields (xy, demand, service, start/end TW, depot flag). Shared263->128->128 MLP, customer-only masked mean/max, separate depot vector and log1p(n)/6, then385->128->2 MLP. No embedding subtraction or early global pooling. Readout parameters: 100482; AdamW LR1e-3 WD1e-4, dropout0.1, clip1.0; unweighted CE only; no schedule/early stop.
Native encoder input is depot xy and customer xy/demand/start/end TW. Service time is not used by the original Encoder; it is retained identically as a readout input in A/B. Native capacity1 and OVRPTW constraints are unchanged. Graphs are encoded in equal-size batches without padding; padding is introduced only after encoding and masked in readout.

## Development-selected results

| Model | Update | Train accuracy | Development CE | Internal accuracy / regret | Original validation accuracy / regret |
|---|---:|---:|---:|---:|---:|
| R49B_8000 | 1800 | 59.13% | 0.679775 | 51.00% / 0.863333% | 51.90% / 0.795806% |
| A_random_seed2 | 1200 | 52.86% | 0.686883 | 51.80% / 0.885483% | 52.40% / 0.825390% |
| B_pretrained_seed2 | 200 | 58.41% | 0.665895 | 51.40% / 0.831422% | 56.90% / 0.709315% |

All following values use the same development-selected checkpoint:

| Model | Set | N | Accuracy | CE | Mean cost | Pair regret |
|---|---|---:|---:|---:|---:|---:|
| R49B_8000 | train | 8000 | 59.13% | 0.663984 | 10.0876991 | 0.691059% |
| R49B_8000 | development | 999 | 53.45% | 0.679775 | 10.0770721 | 0.820159% |
| R49B_8000 | internal | 1000 | 51.00% | 0.685293 | 10.1477532 | 0.863333% |
| R49B_8000 | original_val | 1000 | 51.90% | 0.688559 | 10.0459543 | 0.795806% |
| A_random_seed2 | train | 8000 | 52.86% | 0.686790 | 10.1024129 | 0.848806% |
| A_random_seed2 | development | 999 | 54.05% | 0.686883 | 10.0711244 | 0.776527% |
| A_random_seed2 | internal | 1000 | 51.80% | 0.686156 | 10.1491531 | 0.885483% |
| A_random_seed2 | original_val | 1000 | 52.40% | 0.688450 | 10.0452548 | 0.825390% |
| B_pretrained_seed2 | train | 8000 | 58.41% | 0.660328 | 10.0851224 | 0.663982% |
| B_pretrained_seed2 | development | 999 | 56.36% | 0.665895 | 10.0658001 | 0.700994% |
| B_pretrained_seed2 | internal | 1000 | 51.40% | 0.672479 | 10.1454566 | 0.831422% |
| B_pretrained_seed2 | original_val | 1000 | 56.90% | 0.668430 | 10.0379147 | 0.709315% |

## Final-update observations

Update4000 is not substituted for the development-selected checkpoint:

| Model | Train accuracy / CE | Development accuracy / CE | Internal accuracy / CE | Original validation accuracy / CE |
|---|---:|---:|---:|---:|
| A_random_seed2 | 54.16% / 0.683604 | 54.25% / 0.687760 | 51.40% / 0.689297 | 53.10% / 0.692701 |
| B_pretrained_seed2 | 99.56% / 0.016337 | 51.45% / 2.563613 | 52.70% / 2.487242 | 55.00% / 2.360072 |

## Interpretation

Predeclared screen: B gains at least5pp accuracy over R49B on BOTH held-out sets, reduces BOTH pair regrets, and is better than A. This is an engineering continuation screen, not a guaranteed outcome or significance test.
Screen passed: **False**.

- internal: B minus R49B accuracy +0.40pp; regret -0.031911pp (+3.70% relative regret reduction). B minus random A accuracy -0.40pp; regret -0.054061pp.
- original_val: B minus R49B accuracy +5.00pp; regret -0.086492pp (+10.87% relative regret reduction). B minus random A accuracy +4.50pp; regret -0.116075pp.

This configuration did not achieve the predeclared useful-transfer screen. There is a useful-looking original-validation signal, but no corresponding internal-holdout accuracy gain. It is therefore a mixed transfer result, not evidence of uniformly absent transfer or a robust breakthrough. The pretrained readout also reaches near-perfect training accuracy while held-out CE rises sharply, showing that the frozen features make memorization easy without delivering comparable held-out gains. This does not prove random labels, an accuracy ceiling, or that every form of solver-representation transfer fails. No automatic fine-tuning, pooling redesign, extra seed, or18-task experiment was started.

Paired 95% intervals below use 2000 resamples of the same held-out instances, conditional on fixed models. They do not quantify training-seed or split uncertainty, and the small multiple comparisons are descriptive.

| Contrast | Set | Metric | Difference | Paired 95% interval |
|---|---|---|---:|---|
| B_pretrained_seed2 minus R49B_8000 | internal | accuracy | +0.400000 | [-3.800000, +4.402500] |
| B_pretrained_seed2 minus R49B_8000 | internal | ce | -0.012813 | [-0.024955, -0.000121] |
| B_pretrained_seed2 minus R49B_8000 | internal | pair_regret_pct | -0.031911 | [-0.129799, +0.070329] |
| B_pretrained_seed2 minus A_random_seed2 | internal | accuracy | -0.400000 | [-4.600000, +3.700000] |
| B_pretrained_seed2 minus A_random_seed2 | internal | ce | -0.013677 | [-0.024811, -0.002855] |
| B_pretrained_seed2 minus A_random_seed2 | internal | pair_regret_pct | -0.054061 | [-0.172138, +0.057838] |
| B_pretrained_seed2 minus R49B_8000 | original_val | accuracy | +5.000000 | [+0.800000, +9.000000] |
| B_pretrained_seed2 minus R49B_8000 | original_val | ce | -0.020130 | [-0.032316, -0.007906] |
| B_pretrained_seed2 minus R49B_8000 | original_val | pair_regret_pct | -0.086492 | [-0.182725, +0.012461] |
| B_pretrained_seed2 minus A_random_seed2 | original_val | accuracy | +4.500000 | [+0.700000, +8.300000] |
| B_pretrained_seed2 minus A_random_seed2 | original_val | ce | -0.020020 | [-0.031261, -0.009197] |
| B_pretrained_seed2 minus A_random_seed2 | original_val | pair_regret_pct | -0.116075 | [-0.209454, -0.021883] |

Decision changes are in decision_changes.csv; corrected and newly harmed instances are reported together. Regret always refers to the two-method Oracle, not the original seven-candidate Oracle.

## Full inference cost

Measured on the one allocated idle RTX3090, with five warmups and30 synchronized repetitions. Input-derived median customer count is fixed before timing. Full pipeline includes native CPU collation, host-to-GPU transfer, BOTH encoders and readout; excludes disk reads/model loading/route solving. Cached readout is a separate lower-bound component, not deployment cost. Latency may be affected by another process on the other GPU/shared CPU.

| Model | Component | Batch | Customers | Mean ms / batch | Mean ms / instance | p95 ms / batch |
|---|---|---:|---:|---:|---:|---:|
| A_random_seed2 | cached_readout | 1 | 75 | 0.550 | 0.550 | 0.559 |
| A_random_seed2 | both_encoders_and_readout | 1 | 75 | 9.710 | 9.710 | 9.775 |
| A_random_seed2 | native_collation_transfer_both_encoders_readout | 1 | 75 | 9.926 | 9.926 | 9.987 |
| A_random_seed2 | cached_readout | 128 | 75 | 0.541 | 0.004 | 0.550 |
| A_random_seed2 | both_encoders_and_readout | 128 | 75 | 16.501 | 0.129 | 17.255 |
| A_random_seed2 | native_collation_transfer_both_encoders_readout | 128 | 75 | 21.225 | 0.166 | 21.279 |
| B_pretrained_seed2 | cached_readout | 1 | 75 | 0.546 | 0.546 | 0.554 |
| B_pretrained_seed2 | both_encoders_and_readout | 1 | 75 | 10.144 | 10.144 | 10.266 |
| B_pretrained_seed2 | native_collation_transfer_both_encoders_readout | 1 | 75 | 10.336 | 10.336 | 10.626 |
| B_pretrained_seed2 | cached_readout | 128 | 75 | 0.544 | 0.004 | 0.552 |
| B_pretrained_seed2 | both_encoders_and_readout | 128 | 75 | 16.402 | 0.128 | 16.460 |
| B_pretrained_seed2 | native_collation_transfer_both_encoders_readout | 128 | 75 | 21.410 | 0.167 | 21.452 |

## Provenance and boundaries

[ReLD, ICLR2025](https://arxiv.org/html/2503.00753v1), sections2.1-2.3, analyzes the information represented by static node embeddings during route construction. R50 borrows the trained instance encoders, not its decoder modification; the paper does not establish that these representations predict solver-vs-solver outcomes.
Exact local weight/source hashes, native parameters and limitations are in pretraining_provenance.json. Both current epoch5000 checkpoints identify Train_ALL. Available Trainer source generates fresh online routing instances and trains using POMO/REINFORCE; the task list includes OVRPTW. Historical complete CLI, generator seeds and training-instance manifest are not available, so no certified no-overlap claim is made. This introduces external solver pretraining and is not equal total pretraining-data budget.
Necessary witnesses: both encoder outputs match original pre_forward exactly; no routes decoded; frozen state hashes unchanged; readout blocks receive finite nonzero gradients; masks/labels/initialization/sample order and prediction replay checked. See preflight.json, cache_manifest.json, execution_checks.json and each group gradient_witness.json.
Cache generation seconds: A=9.48, B=9.51. Large FP32 node caches and weight files remain local/ignored by Git; hashes and source/config/predictions are versioned.

## Reproduction

```bash
R50_GPU_UUID=<currently-idle-3090-uuid> bash code/V4/run_v4_r50.sh all
bash code/V4/run_v4_r50.sh analysis
```
The all stage intentionally refuses to overwrite existing runs. Only train/val are permitted by the underlying data loader.
