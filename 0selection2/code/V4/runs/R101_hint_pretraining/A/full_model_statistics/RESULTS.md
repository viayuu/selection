# R101/A

Unique VAL checkpoint: update 41760

| Split | Top1 (%) | mean_cost | actual regret (%) |
| --- | ---: | ---: | ---: |
| VAL | 71.151852 | 11.416571917 | 0.402609 |
| TEST | 71.322222 | 11.373838538 | 0.393029 |

SBS was fitted on new81000 TRAIN. Native FP64 winner labels; equal task weights.
NSS uses historical R79 weights, not a matched new90k training control.
No checkpoint, temperature, or per-problem assembly was selected using TEST/LIB.
PDF: MODEL_STATISTICS.pdf; distributions: selection_distributions.pdf.

Matched A/B: random R94A versus pinned pretrained R94B TRAIN node representations; both12800 hint updates +57600 original full18 R77 updates, seed2/logical batch1024. Same original fresh R84 initialization and TRAIN geometry; transfer instance_encoder only. Original R88B physical TSP node0, hand34/methodID, joint2, score MLP/maximin,1842953 parameters. No added head/adapter, teacher input or hint MSE in Stage II or deployment. Original CE=.35, comparison=.35, risk=.02, R-Drop=.35, dropout=.1. Stage I uses TRAIN81000/task over CVRP+15MVRP, no outcome labels or teacher VAL/TEST; fixed endpoint, no VAL selection. Stage II resets optimizer/scaler/RNG/task/sample streams. R89B main / R84A secondary are historical same-data references, not total-budget-matched controls. R94B has additional external donor pretraining history of partly unknown provenance; 16-task interface coverage is not16-task source pretraining. The paper memo is conditional motivation, not measured success or an algorithmic novelty claim. Fixed R84 target36; selection proportions and zero-count movement are not winner accuracy or true recovery. Native/FP64 ind winners, original legal pools, TRAIN-fitted SBS; ALL is task-equal. Development-exposed TEST; one unique minimum full18 VAL macro_vs_sbs checkpoint; no TEST/LIB tuning or per-task assembly. Latency measures only the original deployed student on GPU-resident original inputs; excludes I/O/host transfer; no speed claim from R94 cached latency.
