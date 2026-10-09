# R46A: source views interact before aggregation

Only R46A was trained, seed=2, from scratch on all18 tasks. R46B is NOT run.
R45A/B are frozen historical references. ALL percentages are per-problem macro averages.
Original raw-label FP64 costs and native winners; no API calls, new embeddings, test-driven selection or augmentation.

## Model

Four-layer instance Encoder -> two synchronous node/source-token layers -> instance-conditioned per-solver source pooling.
No trainable solver ID, solver-specific parameter or handcrafted solver branch. Original shared decoder, score head, maximin and R43A objective are retained.
Source views/deployment slots: up to 8 per solver; no pre-interaction averaging.
Legacy role summaries are saved only for embedding provenance validation; forward reads the complete-view component bank.

## Fixed-input limitations

Identical full source packages: DIFUSCO / DIFUSCO500; T2T / T2T500.
Without an ID these methods cannot be distinguished by the model. They remain separate candidates; exact score ties use ascending global solver ID.
Previously mixed content inside one external vector cannot be recovered. The original26 assumed historical deployment records remain unverified.
Without R46B, this run cannot establish that correct source-to-solver correspondence is better than a fixed mismapping.

## Test

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R45A | ALL | 0.4791 | 0.8174 | 0.9243 | 11.168366 | -1.0320% | 0.9432% |
| R45B | ALL | 0.4740 | 0.8139 | 0.9231 | 11.170249 | -1.0149% | 0.9634% |
| R46A | ALL | 0.4776 | 0.8158 | 0.9209 | 11.169184 | -1.0286% | 0.9496% |

## Training and validation

Parameters: 1,984,137. Stop epoch: 40; validation-best epoch: 35.
Successful updates: 56,880; 3160 per task; full coverage, retained tail batches.
| Point | Top1 | CE | mean_cost | vs_SBS | actual regret |
|---|---:|---:|---:|---:|---:|
| Best val | 0.4899 | 1.0930 | 11.169060 | -1.0683% | 0.9327% |
| Final train | 0.5045 | 1.0437 | 11.169068 | -1.1280% | 0.8879% |
| Final val | 0.4884 | 1.0919 | 11.171041 | -1.0537% | 0.9481% |
| Last5 val | 0.4890 | 1.0948 | 11.170463 | -1.0593% | 0.9419% |

## Test families

| Model | Family | Top1 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|
| R45A | TSP | 0.3590 | 8.044507 | -0.4746% | 0.6504% |
| R45A | CVRP | 0.4810 | 18.247945 | -0.4603% | 0.6138% |
| R45A | ATSP | 0.7750 | 1.629876 | -2.3839% | 0.3274% |
| R45A | MVRP | 0.4673 | 11.540551 | -1.0171% | 1.0258% |
| R45B | TSP | 0.3680 | 8.045435 | -0.4632% | 0.6612% |
| R45B | CVRP | 0.4720 | 18.252667 | -0.4346% | 0.6380% |
| R45B | ATSP | 0.7730 | 1.630674 | -2.3361% | 0.3803% |
| R45B | MVRP | 0.4613 | 11.542380 | -1.0023% | 1.0441% |
| R46A | TSP | 0.3330 | 8.049228 | -0.4162% | 0.7048% |
| R46A | CVRP | 0.4720 | 18.255320 | -0.4201% | 0.6612% |
| R46A | ATSP | 0.7680 | 1.630038 | -2.3742% | 0.3368% |
| R46A | MVRP | 0.4682 | 11.540716 | -1.0203% | 1.0259% |

## Contrasts

- R46A minus R45A: Top1 -0.156 percentage points; mean cost +0.000818; actual regret +0.00633 percentage points.
- R46A minus R45B: Top1 +0.356 percentage points; mean cost -0.001065; actual regret -0.01382 percentage points.

## Decision changes

| Reference | Change | Instances | Delta mean cost | Delta regret |
|---|---|---:|---:|---:|
| R45A | corrected | 1034 | -0.014030 | -0.11807% |
| R45A | harmed | 1062 | +0.014398 | +0.11912% |
| R45A | both_wrong_cheaper | 370 | -0.003758 | -0.02673% |
| R45A | both_wrong_dearer | 426 | +0.004208 | +0.03201% |
| R45A | both_wrong_equal | 0 | +0.000000 | +0.00000% |
| R45A | all | 18000 | +0.000818 | +0.00633% |
| R45B | corrected | 1149 | -0.016727 | -0.13997% |
| R45B | harmed | 1085 | +0.014896 | +0.12023% |
| R45B | both_wrong_cheaper | 423 | -0.003959 | -0.02878% |
| R45B | both_wrong_dearer | 437 | +0.004725 | +0.03471% |
| R45B | both_wrong_equal | 0 | +0.000000 | +0.00000% |
| R45B | all | 18000 | -0.001065 | -0.01382% |

## Per problem

| Model | Problem | Top1 | Top2 | Top3 | mean_cost | vs_SBS | actual regret |
|---|---|---:|---:|---:|---:|---:|---:|
| R45A | TSP | 0.3590 | 0.6170 | 0.7520 | 8.044507 | -0.4746% | 0.6504% |
| R45A | CVRP | 0.4810 | 0.6920 | 0.8220 | 18.247945 | -0.4603% | 0.6138% |
| R45A | ATSP | 0.7750 | 0.9620 | 0.9950 | 1.629876 | -2.3839% | 0.3274% |
| R45A | OVRP | 0.3740 | 0.6490 | 0.8310 | 7.461105 | -0.1125% | 0.7130% |
| R45A | VRPB | 0.5170 | 0.9180 | 0.9880 | 9.287440 | -0.0820% | 0.7681% |
| R45A | VRPL | 0.3020 | 0.5280 | 0.6960 | 12.354608 | -0.0492% | 0.4926% |
| R45A | VRPTW | 0.4350 | 0.7710 | 0.9240 | 17.657704 | -4.2199% | 1.4119% |
| R45A | OVRPTW | 0.4910 | 0.9080 | 0.9990 | 10.114022 | -0.0916% | 0.8190% |
| R45A | OVRPB | 0.5510 | 0.9640 | 0.9920 | 6.295778 | -0.0085% | 1.0279% |
| R45A | OVRPL | 0.3700 | 0.6860 | 0.8570 | 7.444869 | -0.1178% | 0.6943% |
| R45A | VRPBL | 0.5470 | 0.9400 | 0.9910 | 9.245516 | -0.0396% | 0.7462% |
| R45A | VRPBTW | 0.4430 | 0.7850 | 0.9350 | 18.972974 | -2.8338% | 1.5118% |
| R45A | VRPLTW | 0.4450 | 0.7850 | 0.9280 | 17.603499 | -4.4683% | 1.3870% |
| R45A | OVRPBL | 0.5740 | 0.9710 | 0.9910 | 6.301951 | -0.0773% | 0.9841% |
| R45A | OVRPBTW | 0.5140 | 0.9030 | 0.9970 | 10.608161 | -0.0078% | 1.2650% |
| R45A | OVRPLTW | 0.5020 | 0.9210 | 1.0000 | 10.097594 | -0.0595% | 0.8210% |
| R45A | VRPBLTW | 0.4490 | 0.7890 | 0.9410 | 19.071665 | -3.0761% | 1.5746% |
| R45A | OVRPBLTW | 0.4950 | 0.9250 | 0.9990 | 10.591372 | -0.0132% | 1.1698% |
| R45B | TSP | 0.3680 | 0.6090 | 0.7390 | 8.045435 | -0.4632% | 0.6612% |
| R45B | CVRP | 0.4720 | 0.6950 | 0.8220 | 18.252667 | -0.4346% | 0.6380% |
| R45B | ATSP | 0.7730 | 0.9620 | 0.9950 | 1.630674 | -2.3361% | 0.3803% |
| R45B | OVRP | 0.3640 | 0.6400 | 0.8340 | 7.462366 | -0.0956% | 0.7327% |
| R45B | VRPB | 0.4860 | 0.9190 | 0.9890 | 9.296205 | +0.0123% | 0.8679% |
| R45B | VRPL | 0.3050 | 0.5260 | 0.7090 | 12.355738 | -0.0401% | 0.5005% |
| R45B | VRPTW | 0.4250 | 0.7450 | 0.9160 | 17.659679 | -4.2092% | 1.4221% |
| R45B | OVRPTW | 0.4600 | 0.9080 | 0.9990 | 10.123045 | -0.0025% | 0.9094% |
| R45B | OVRPB | 0.5600 | 0.9640 | 0.9910 | 6.293458 | -0.0453% | 0.9835% |
| R45B | OVRPL | 0.3740 | 0.6740 | 0.8600 | 7.443894 | -0.1309% | 0.6852% |
| R45B | VRPBL | 0.5520 | 0.9380 | 0.9900 | 9.245562 | -0.0391% | 0.7505% |
| R45B | VRPBTW | 0.4450 | 0.7820 | 0.9350 | 18.983222 | -2.7813% | 1.5528% |
| R45B | VRPLTW | 0.4430 | 0.7740 | 0.9240 | 17.599657 | -4.4891% | 1.3811% |
| R45B | OVRPBL | 0.5750 | 0.9710 | 0.9930 | 6.299660 | -0.1136% | 0.9415% |
| R45B | OVRPBTW | 0.5020 | 0.9040 | 0.9980 | 10.608499 | -0.0046% | 1.2788% |
| R45B | OVRPLTW | 0.4820 | 0.9210 | 1.0000 | 10.101600 | -0.0198% | 0.8670% |
| R45B | VRPBLTW | 0.4630 | 0.7930 | 0.9240 | 19.068798 | -3.0907% | 1.5635% |
| R45B | OVRPBLTW | 0.4830 | 0.9250 | 0.9980 | 10.594321 | +0.0146% | 1.2246% |
| R46A | TSP | 0.3330 | 0.5930 | 0.6900 | 8.049228 | -0.4162% | 0.7048% |
| R46A | CVRP | 0.4720 | 0.7000 | 0.8270 | 18.255320 | -0.4201% | 0.6612% |
| R46A | ATSP | 0.7680 | 0.9610 | 0.9940 | 1.630038 | -2.3742% | 0.3368% |
| R46A | OVRP | 0.3710 | 0.6520 | 0.8310 | 7.461139 | -0.1121% | 0.7202% |
| R46A | VRPB | 0.5090 | 0.9180 | 0.9870 | 9.287216 | -0.0844% | 0.7666% |
| R46A | VRPL | 0.2910 | 0.5190 | 0.6990 | 12.357442 | -0.0263% | 0.5167% |
| R46A | VRPTW | 0.4500 | 0.7710 | 0.9190 | 17.646128 | -4.2827% | 1.3423% |
| R46A | OVRPTW | 0.4840 | 0.9080 | 0.9990 | 10.118147 | -0.0509% | 0.8572% |
| R46A | OVRPB | 0.5800 | 0.9640 | 0.9900 | 6.290334 | -0.0949% | 0.9406% |
| R46A | OVRPL | 0.3650 | 0.6810 | 0.8600 | 7.445091 | -0.1148% | 0.6986% |
| R46A | VRPBL | 0.5440 | 0.9390 | 0.9890 | 9.248654 | -0.0057% | 0.7778% |
| R46A | VRPBTW | 0.4170 | 0.7790 | 0.9420 | 18.996201 | -2.7148% | 1.6392% |
| R46A | VRPLTW | 0.4570 | 0.7850 | 0.9290 | 17.601848 | -4.4772% | 1.3957% |
| R46A | OVRPBL | 0.5760 | 0.9710 | 0.9950 | 6.299024 | -0.1237% | 0.9265% |
| R46A | OVRPBTW | 0.5200 | 0.9020 | 0.9970 | 10.604639 | -0.0410% | 1.2454% |
| R46A | OVRPLTW | 0.5080 | 0.9210 | 1.0000 | 10.096859 | -0.0667% | 0.8127% |
| R46A | VRPBLTW | 0.4610 | 0.7970 | 0.9310 | 19.065222 | -3.1089% | 1.5575% |
| R46A | OVRPBLTW | 0.4900 | 0.9240 | 0.9980 | 10.592789 | +0.0002% | 1.1922% |
