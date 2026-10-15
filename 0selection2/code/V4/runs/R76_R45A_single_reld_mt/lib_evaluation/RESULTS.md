# R76 TSPLIB / CVRPLIB

Frozen validation-selected checkpoint: epoch 18; `5ba5777e437235e22682c4d54cfd8cdefca6fb53c27204a8e1e3f753ea9043c9`.
No retraining, solver runs, input augmentation, or selection using LIB results.
The original NSS comparison instances, preprocessing, pools, and FP64 costs are unchanged.

| Dataset | Instances | Top1 native | Top2 native | Top3 native | Mean cost | vs SBS | vs Oracle | Actual regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TSPLIB | 49 | 28.5714% | 51.0204% | 73.4694% | 8.209819 | -0.2870% | +0.8309% | 0.8754% |
| CVRPLIB | 100 | 33.0000% | 61.0000% | 87.0000% | 68.280504 | +1.5393% | +1.9948% | 1.4498% |

## Historical NSS Comparison
Top1 in this table uses the historical NSS FP32-argmin convention for both models.
TSPLIB has one row where that winner differs from native ind; CVRPLIB has no differences.
NSS numbers are reused, not rerun. Cost metrics use common raw FP64 SBS/Oracle references.

| Dataset | R76 Top1 NSS convention | NSS Top1 | R76 mean cost | NSS mean cost | R76 minus NSS cost |
|---|---:|---:|---:|---:|---:|
| TSPLIB | 28.5714% | 38.7755% | 8.209819 | 8.180314 | +0.029504 |
| CVRPLIB | 33.0000% | 68.0000% | 68.280504 | 67.048798 | +1.231706 |

SBS above is each LIB dataset post-hoc best fixed method, matching previous NSS reports.
The separate train_sbs_reference object records R76 TRAIN-fixed SBS on the same LIB costs.
These are cross-scenario evaluations of the new-data-trained R76 model, not matched-training-budget NSS experiments.
Frozen R76 TRAIN normalization is retained. Predictions retain native winners, FP64 costs, and all candidate scores.

## Command
```bash
bash code/V4/run_v4_r76_lib.sh
bash code/V4/run_v4_r76_lib.sh analysis  # saved predictions only
```
