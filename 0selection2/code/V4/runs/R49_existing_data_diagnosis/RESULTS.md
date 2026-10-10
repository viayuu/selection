# R49: existing-data coverage and sample-count diagnosis

Scope: OVRPTW RELD_MOEL (class0) vs RELD_MTL (class1), seed2. Existing train/val only; no test, new instances or solver reruns.
Both clean R48 classifiers start from the same fresh trainable initialization. No solver encoding, retrieval, pair/risk/R-Drop or auxiliary loss.

## Locked protocol

Split original 10,000 training rows into 8,000 training candidates, 1,000 development and 1,000 internal holdout. A uses a fixed size-stratified 2,000-row subset of B; original validation is unchanged. Splitting is by input-derived base group and exact customer count, never label/winner or index-derived distribution.
Original training unique base groups: 10000; original validation: 1000. No detected related-group overlap between train/development/internal/original validation under the input-canonicalization rule. No reliable per-instance distribution or historical base-ID metadata is present.
Each group uses 4,000 successful updates of exactly128 examples: 512,000 presentations, shuffled complete passes with tails carried forward. There is no early stopping or schedule; LR1e-4, AdamW WD1e-4, dropout0.1, gradient clip1.0, FP32/TF32 off match R48. A and B each fit geometry normalization on their own actual non-tie training nodes only.
Evaluate all four sets at update0 and every200 updates. Strict minimum development CE chooses best; internal and original validation are observations only. All metrics exclude exact FP64 cost ties; regret uses the two-method Oracle, not the seven-method pool.

| Set | Nominal rows | Non-ties | Exact ties |
|---|---:|---:|---:|
| training_pool | 8000 | 8000 | 0 |
| development | 1000 | 999 | 1 |
| internal | 1000 | 1000 | 0 |
| original_val | 1000 | 1000 | 0 |
| A_2000 | 2000 | 2000 | 0 |

## Main results

| Model | Best update | Train accuracy | Development CE | Internal accuracy / regret | Original validation accuracy / regret |
|---|---:|---:|---:|---:|---:|
| A_2000_seed2 | 200 | 54.550% | 0.690082 | 51.400% / 0.848128% | 52.500% / 0.848809% |
| B_8000_seed2 | 1800 | 59.125% | 0.679775 | 51.000% / 0.863333% | 51.900% / 0.795806% |

Full same-checkpoint metrics:

| Model | Set | N | Accuracy | CE | Mean cost | Two-method regret |
|---|---|---:|---:|---:|---:|---:|
| A_2000_seed2 | train | 2000 | 54.550% | 0.683003 | 10.184313 | 0.836118% |
| A_2000_seed2 | development | 999 | 54.054% | 0.690082 | 10.076717 | 0.819160% |
| A_2000_seed2 | internal | 1000 | 51.400% | 0.691137 | 10.147447 | 0.848128% |
| A_2000_seed2 | original_val | 1000 | 52.500% | 0.690257 | 10.047915 | 0.848809% |
| B_8000_seed2 | train | 8000 | 59.125% | 0.663984 | 10.087699 | 0.691059% |
| B_8000_seed2 | development | 999 | 53.453% | 0.679775 | 10.077072 | 0.820159% |
| B_8000_seed2 | internal | 1000 | 51.000% | 0.685293 | 10.147753 | 0.863333% |
| B_8000_seed2 | original_val | 1000 | 51.900% | 0.688559 | 10.045954 | 0.795806% |

## Diagnostic differences

Intervals below resample held-out instances conditional on these fixed models; they do not measure training-seed or split uncertainty.

| Contrast | Set / model | Metric | Difference | 95% interval |
|---|---|---|---:|---|
| B minus A | internal | accuracy | -0.400000 | [-4.300000, +3.900000] |
| B minus A | internal | ce | -0.005845 | [-0.014792, +0.002866] |
| B minus A | internal | pair_regret_pct | +0.015206 | [-0.086242, +0.108405] |
| B minus A | original_val | accuracy | -0.600000 | [-4.600000, +3.500000] |
| B minus A | original_val | ce | -0.001697 | [-0.011015, +0.008193] |
| B minus A | original_val | pair_regret_pct | -0.053002 | [-0.162587, +0.048553] |
| internal minus original validation | A_2000_seed2 | accuracy | -1.100000 | [-5.500000, +3.400000] |
| internal minus original validation | A_2000_seed2 | ce | +0.000881 | [-0.009299, +0.010445] |
| internal minus original validation | A_2000_seed2 | pair_regret_pct | -0.000681 | [-0.122879, +0.136856] |
| internal minus original validation | B_8000_seed2 | accuracy | -0.900000 | [-5.500000, +3.402500] |
| internal minus original validation | B_8000_seed2 | ce | -0.003267 | [-0.018987, +0.012786] |
| internal minus original validation | B_8000_seed2 | pair_regret_pct | +0.067527 | [-0.047972, +0.192692] |

Final update4,000 is reported separately, never used instead of the development-selected checkpoint:

| Model | Train accuracy / CE | Development accuracy / CE | Internal accuracy / CE | Original validation accuracy / CE |
|---|---:|---:|---:|---:|
| A_2000_seed2 | 100.000% / 0.000439 | 52.553% / 2.865999 | 52.100% / 2.814353 | 50.300% / 2.738509 |
| B_8000_seed2 | 83.025% / 0.411980 | 53.654% / 0.831568 | 50.500% / 0.857404 | 49.800% / 0.873115 |

## Input distribution comparison

All R48 hand-computed input statistics are in split_comparison.csv and input_statistics.npz. They are not added to the network. Standardized mean differences use pooled per-feature standard deviations against the 8,000-row training pool.

| Set | Mean customers | MOEL win fraction | Mean demand total | Mean TW width | Mean pair distance |
|---|---:|---:|---:|---:|---:|
| training_pool | 74.998375 | 0.496500 | 8.261045 | 1.079768 | 0.485610 |
| development | 74.979000 | 0.482482 | 8.256145 | 1.078682 | 0.483893 |
| internal | 74.940000 | 0.465000 | 8.244935 | 1.080703 | 0.484574 |
| original_val | 74.690000 | 0.491000 | 8.232830 | 1.080210 | 0.485481 |
| A_2000 | 74.988500 | 0.493500 | 8.265577 | 1.079000 | 0.486976 |

Largest absolute input-statistic standardized mean differences:

development: tw_start_std=-0.064, tw_width_q90=-0.064, directed_slack_std=-0.064, tw_width_std=-0.058, depot_tw_slack_std=-0.056.
internal: tw_end_mean=+0.082, bbox_y=+0.077, tw_start_mean=+0.069, tw_start_q50=+0.067, depot_distance_max=+0.062.
original_val: tw_end_std=-0.063, nn1_std=-0.054, depot_tw_slack_std=-0.051, nn1_max=-0.046, nn8_mean_max=-0.043.
A_2000: depot_tw_slack_q50=-0.043, centroid_depot_distance=+0.041, directed_slack_q90=-0.038, depot_distance_mean=+0.037, depot_tw_slack_mean=-0.036.

## Observations and interpretation

- A_2000_seed2: internal accuracy 51.400% versus original validation 52.500% (-1.100pp; conditional95% interval [-5.500, +3.400]). Internal/original regret is 0.848128% / 0.848809%.
- B_8000_seed2: internal accuracy 51.000% versus original validation 51.900% (-0.900pp; conditional95% interval [-5.500, +3.402]). Internal/original regret is 0.863333% / 0.795806%.
- Increasing distinct training rows2,000->8,000 changes internal accuracy by -0.400pp (conditional95% interval [-4.300, +3.900]), CE by -0.005845, and regret by +0.015206pp. Negative CE/regret changes are better.
- Increasing distinct training rows2,000->8,000 changes original_val accuracy by -0.600pp (conditional95% interval [-4.600, +3.500]), CE by -0.001697, and regret by -0.053002pp. Negative CE/regret changes are better.
- Neither fixed model provides clear accuracy evidence that same-source internal holdout is easier than original validation. A large original-validation source mismatch is not supported by this probe; this does not exclude a smaller, conditional or unobserved distribution shift.
- Neither accuracy gain from2,000->8,000 excludes zero in the conditional instance-bootstrap interval. This experiment does not establish a large sample-count benefit; any cost/CE improvements are reported rather than discarded. Do not extrapolate these two points to an irreducible ceiling or to much larger datasets.
- A_2000_seed2 first reaches>=99% training accuracy at update1800; the corresponding internal/original accuracy is 53.100% / 50.000%. Strong training fit is not itself evidence of transferable solver-win signal.

## Limits and next step

1. If the two held-out sets remain similarly difficult and larger subsets do not provide a clear joint benefit, the next targeted question is whether the instance Encoder captures transferable relationships governing relative solver performance, rather than another solver embedding or decoder replacement. This run does not train any additional architecture or use held-out observations to tune the two runs.
2. This is two sample counts at one optimization budget and one split/seed, not an asymptotic learning curve. Normalization changes with the actual training subset as predeclared, and may contribute to A/B differences. A repeatedly sees fewer unique instances; B has fewer passes per instance despite equal updates and presentations. Development selection can choose different update counts; equal budgets describe the complete runs, not necessarily the selected checkpoints.
3. Similar observed input statistics do not establish equal joint distributions or equal solver behavior; a lack of improvement within 2,000-to-8,000 cannot establish a predictability ceiling or rule out much larger independent datasets.
4. The internal and original validation results never choose checkpoints, stop training or change hyperparameters. No historical learned weights are used, because they would have seen the new internal holdout.

## Execution

Only the idle RTX3090 selected by R49_GPU_UUID was used; the busy other3090 and4090 were not used.
```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/0selection2
R49_GPU_UUID=GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d bash code/V4/run_v4_r49.sh all
```
On this launch the GPU is reached within the existing allocation with srun --jobid=1465 --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=10G; long tasks use tmux.
Checkpoints, per-run history/normalization/sampling plans and W&B offline logs are saved. Prediction artifacts reproduce the reported metrics. No API or embedding generation is needed.

Learning-curve context: [The Shape of Learning Curves: a Review](https://arxiv.org/abs/2103.10948). This experiment studies distinct training sample count, not only repeated optimization epochs; it does not assume more data always helps.
