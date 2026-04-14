# Experiment Tracker

| Block | Goal | Status | Output | Notes |
|------|------|--------|--------|-------|
| A1 | Freeze exact 16 CVRP variants | TODO | variant list | must happen before large label run |
| A2 | Build problem-method coverage matrix from EasyNCO | TODO | coverage table | identify supported solvers and checkpoints |
| A3 | Generate tiny mixed-problem label smoke dataset | TODO | pilot label json/csv/pkl | use 4-6 problems first |
| B1 | Build dataset conversion script | TODO | joint dataset file | create feasible mask, ranks, regrets |
| B2 | Run non-learning baselines | TODO | baseline metrics | random / single-best-global / single-best-per-problem / oracle |
| C1 | Train pooled naive selector | TODO | model + metrics | problem id only |
| C2 | Train pooled masked ranker | TODO | model + metrics | no solver metadata |
| D1 | Train full compatibility selector | TODO | model + metrics | main proposed method |
| E1 | Add shared/private adapters if needed | CONDITIONAL | ablation metrics | only if negative transfer appears |
| X1 | Solver insertion experiment | OPTIONAL | transfer metrics | second-stage novelty booster |
| X2 | Partial-label training experiment | OPTIONAL | cost vs quality curves | backup / extension |
