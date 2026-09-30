# R32d LIB Evaluation with Gap

- Checkpoint: `code/V4/runs/R32d_g0main_seqtopk_seed2_gpu0_b448/best_top3_safe.pt`
- Checkpoint epoch: `26`

| Problem | top1 | top2 | top3 | mean_cost | Gap | SBS cost (vs_sbs) | Oracle (vs_Oracle) |
| -------- | ---------: | ---------: | ---------: | ----------: | -----: | --------------------: | --------------------: |
| **LIB-ALL** | **0.3995** | **0.7021** | **0.8336** | **38.0170** | **0.000%** | **37.7394 (+0.182%)** | **37.5436 (+0.966%)** |
| TSPLIB | 0.4490 | 0.7143 | 0.8571 | 8.1902 | 0.000% | 8.2334 (-0.525%) | 8.1422 (+0.590%) |
| CVRPLIB | 0.3500 | 0.6900 | 0.8100 | 67.8438 | 0.000% | 67.2454 (+0.890%) | 66.9451 (+1.342%) |

## Arm Distribution

- **TSPLIB**: BQ:53.1% | ELG:30.6% | T2T500:12.2% | DIFUSCO500:2.0% | T2T:2.0%
- **CVRPLIB**: ICAM:62.0% | OMNI:24.0% | MoSES_RF:7.0% | RELD_CVRP:5.0% | MoSES_CaDA:2.0%