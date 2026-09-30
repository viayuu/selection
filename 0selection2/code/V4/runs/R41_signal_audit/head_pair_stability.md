# OVRPTW RELD head-pair audit (partial pool)

This is a measured RELD_MOEL/RELD_MTL comparison, not seven-solver winner stability.
Original decoder: argmax, N starts, no augmentation, one sample; two independent processes per split/method.
R39A accuracy/regret below uses the two-solver minimum only, and is conditional on its choice being in this pair.

| Split | N | Sign stability | Historical sign reproduced | Max repeat cost difference | Max historical cost difference | R39A pair Top1 | R39A pair regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 64 | 1.0000 | 1.0000 | 0 | 4.96e-11 | 0.6562 | 0.7984% |
| val | 64 | 1.0000 | 1.0000 | 0 | 4.94e-11 | 0.5312 | 0.6766% |

R39A chose within the pair on 128 measured instances; 76 choices were pair-correct.
Errors remain against repeat-stable pair costs in this sample. This supports representation/generalization analysis for this particular pair, not a claim that all project labels are stable.
