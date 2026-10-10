# R53: Independent MOEL/MTL specialists

## Protocol
Frozen R45A supplies the full ranking. Independent four-layer, d=128 encoders were trained from scratch on all 15 MVRP tasks (seed=2). Only original Top2={MOEL,MTL} decisions can change. No solver features/ID, retrieval, solver execution, risk, pairwise objective or R-Drop enters the expert.

A uses unweighted BCE; B uses actual pair misselection cost divided by a fixed training-only mean. Both include all non-tied instances, including those with another full-pool winner.
Fixed global mean weight: 0.0222730884314. A/B initialization hashes match; sampling plans match. Schedules follow the same rules, not necessarily the same actual LR trajectory or stop epoch.

Selection uses the strict minimum **integrated full18 validation actual regret**; plateau/early-stop min_delta=0.001 percentage points, warmup=3, min15/max40 epochs. Both best checkpoints were locked before test. Test never selects an epoch.

The [SATzilla2012 solver description](https://www.cs.ubc.ca/~kevinlb/papers/2012-SATzilla2012-Solver-Description.pdf) motivates independent cost-sensitive pair classification. This experiment does not reproduce its forests, portfolio voting or runtime prediction.

## Pretraining Gate Budget (Validation)
Gate: 10639/18000 instances. Recoverable native-winner errors: 4177; all MOEL/MTL mutual errors: 4207, gate-covered: 4177.
Diagnostic upper bound (not a deployable score): +23.2056 Top1 points and -0.514250 regret points (55.53% relative reduction).

## Integrated Results
| Split | Model | Best epoch | Top1 % | Mean cost | Actual regret % | Delta Top1 pp | Regret reduction % |
|---|---|---:|---:|---:|---:|---:|---:|
| val | R45A | 38 | 49.1167 | 11.168517 | 0.926048 | +0.0000 | +0.000 |
| val | R53A | 18 | 49.2667 | 11.167375 | 0.922663 | +0.1500 | +0.366 |
| val | R53B | 12 | 49.0389 | 11.167869 | 0.921408 | -0.0778 | +0.501 |
| test | R45A | 38 | 47.9111 | 11.168366 | 0.943221 | +0.0000 | +0.000 |
| test | R53A | 18 | 47.7611 | 11.168996 | 0.949729 | -0.1500 | -0.690 |
| test | R53B | 12 | 47.7111 | 11.168827 | 0.950101 | -0.2000 | -0.730 |

## Corrections and Harms
- val R53A: corrected 1070, harmed 1043; both-wrong cheaper/more expensive 161/167; net mean-cost delta -0.00114195, regret delta -0.003386 pp; cost improved on 8/15 MVRP tasks.
- val R53B: corrected 982, harmed 996; both-wrong cheaper/more expensive 165/149; net mean-cost delta -0.00064804, regret delta -0.004641 pp; cost improved on 9/15 MVRP tasks.
- test R53A: corrected 1009, harmed 1036; both-wrong cheaper/more expensive 157/193; net mean-cost delta +0.00063015, regret delta +0.006508 pp; cost improved on 5/15 MVRP tasks.
- test R53B: corrected 940, harmed 976; both-wrong cheaper/more expensive 124/153; net mean-cost delta +0.00046091, regret delta +0.006881 pp; cost improved on 6/15 MVRP tasks.

## Stopping and Generalization
- R53A: best epoch 18, stop 26, successful updates 30810. At best: expert train/val accuracy 57.333%/56.457%, unweighted BCE 0.663829/0.670242. Last5 integrated val: Top1 49.1211%, regret 0.934479%.
- R53B: best epoch 12, stop 20, successful updates 23700. At best: expert train/val accuracy 56.510%/56.670%, unweighted BCE 0.681775/0.686411. Last5 integrated val: Top1 49.0267%, regret 0.928328%.

## Conclusion
Neither expert meets the predeclared meaningful-improvement screen. Independent cross-problem pair learning and this cost-weighted objective did not produce the intended system-level breakthrough; do not expand this configuration into further seeds or threshold searches.

This is a fixed-seed direction screen, not proof that the pair is unpredictable or that labels are noisy. The gate limits what can improve. Top2/Top3 coverage is unchanged by construction; ranking-only combined outputs are not treated as calibrated probabilities or assigned a fabricated full-system CE.

## Artifacts and Runtime
See comparison.csv (ALL/MVRP/all18), binary_comparison.csv, corrections_and_harms.csv, gate_budget.csv, learning_curves.csv/png, instance_decisions.csv, locked_checkpoints.json and each arm predictions/{val,test}. Checkpoint and sampling-plan .pt files remain local, following existing repository rules.
Inference adds a second instance encoder only where the fixed gate triggers. Recorded specialist-only timings do not pretend that cached baseline predictions are a free end-to-end deployment. No underlying solver is executed.

W&B logs are retained locally in offline mode (project selector); no unrelated historical artifacts were changed.
