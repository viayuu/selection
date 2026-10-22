# R101: TRAIN Hint Pretraining, Original Student Deployment

Matched A/B: random R94A versus pinned pretrained R94B TRAIN node representations; both12800 hint updates +57600 original full18 R77 updates, seed2/logical batch1024. Same original fresh R84 initialization and TRAIN geometry; transfer instance_encoder only. Original R88B physical TSP node0, hand34/methodID, joint2, score MLP/maximin,1842953 parameters. No added head/adapter, teacher input or hint MSE in Stage II or deployment. Original CE=.35, comparison=.35, risk=.02, R-Drop=.35, dropout=.1. Stage I uses TRAIN81000/task over CVRP+15MVRP, no outcome labels or teacher VAL/TEST; fixed endpoint, no VAL selection. Stage II resets optimizer/scaler/RNG/task/sample streams. R89B main / R84A secondary are historical same-data references, not total-budget-matched controls. R94B has additional external donor pretraining history of partly unknown provenance; 16-task interface coverage is not16-task source pretraining. The paper memo is conditional motivation, not measured success or an algorithmic novelty claim. Fixed R84 target36; selection proportions and zero-count movement are not winner accuracy or true recovery. Native/FP64 ind winners, original legal pools, TRAIN-fitted SBS; ALL is task-equal. Development-exposed TEST; one unique minimum full18 VAL macro_vs_sbs checkpoint; no TEST/LIB tuning or per-task assembly. Latency measures only the original deployed student on GPU-resident original inputs; excludes I/O/host transfer; no speed claim from R94 cached latency.

| Model | Split | Top1 (%) | mean_cost | SBS cost | Oracle cost | vs_SBS (%) | vs_Oracle (%) | actual regret (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R101/A | TRAIN | 71.368793 | 11.388948178 | 11.400702391 | 11.345073027 | -0.200424 | 0.384419 | 0.392607 |
| R101/A | VAL | 71.151852 | 11.416571917 | 11.427978473 | 11.371691902 | -0.194929 | 0.392695 | 0.402609 |
| R101/A | TEST | 71.322222 | 11.373838538 | 11.385299246 | 11.329932369 | -0.197855 | 0.385312 | 0.393029 |
| R101/B | TRAIN | 71.485117 | 11.388223258 | 11.400702391 | 11.345073027 | -0.206065 | 0.378755 | 0.386514 |
| R101/B | VAL | 71.201235 | 11.416299608 | 11.427978473 | 11.371691902 | -0.196477 | 0.391145 | 0.400001 |
| R101/B | TEST | 71.434568 | 11.373450462 | 11.385299246 | 11.329932369 | -0.200016 | 0.383136 | 0.391020 |
