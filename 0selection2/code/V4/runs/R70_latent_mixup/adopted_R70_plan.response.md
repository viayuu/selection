**R69 TRAIN/validation integrity: 9/10, ready within the inspected scope. No actionable defect found.**

**Performance: 4/10, not ready for the 70% goal.** Task-specific B loses **40 validation decisions, -0.222222pp**, against fixed shared A. The predeclared rule therefore retains A.

`review_independence=same-family; acceptance_status=provisional`

The [terminal receipt](/public/home/shiys/0selection2/code/V4/runs/R69_task_specific_finetuning/training_terminal_receipt.json:4) records Slurm1465.101 `COMPLETED/0:0`, 3,604 seconds, and the training tmux session gone. Initialization records contain eighteen identical pre-fine-tuning states. The inspected [gradient path](/public/home/shiys/0selection2/code/V4/runs/R69_task_specific_finetuning/source_launch/code/V4/r69_gradients.py:59) correctly divides each disjoint task gradient by three, clips once and performs one AdamW step. Inactive parameters retain `grad=None`. Recorded active parameter counters reach1,600; unused paths remain at zero. [Selection](/public/home/shiys/0selection2/code/V4/runs/R69_task_specific_finetuning/source_launch/code/V4/r69_experiment.py:382) saves the complete bank from one epoch. All40 validation records uniquely select epoch20. No repair or replay is warranted.

Results from the saved [shared A](/public/home/shiys/0selection2/code/V4/runs/R68_sharpness_aware_finetuning/A_plain_adamw_seed2/result.json), [task-specific B](/public/home/shiys/0selection2/code/V4/runs/R69_task_specific_finetuning/B_task_specific_seed2/result.json) and their histories:

| Record | TRAIN Top1% / CE | VAL correct / Top1% | VAL CE | Regret% |
|---|---:|---:|---:|---:|
| A selected28 | 49.584444 / 1.065074 | 8644 / 48.022222 | 1.119215 | 0.974169 |
| B selected20 | 50.331111 / 1.050462 | 8604 / 47.800000 | 1.119310 | 1.042269 |
| A final40 | 53.568333 / 0.991854 | 8469 / 47.050000 | 1.148645 | 1.064938 |
| B final40 | 59.985556 / 0.871883 | 8369 / 46.494444 | 1.273051 | 1.102857 |
| A epochs36–40 mean | unavailable | 47.431111 | 1.156541 | 1.030852 |
| B epochs36–40 mean | unavailable | 46.181111 | 1.256336 | 1.104494 |

Last-five values are arithmetic history means, not averaged checkpoints. Only epoch40 has a complete TRAIN evaluation within that interval, so a five-epoch TRAIN mean is unavailable.

At common epoch20, A’s TRAIN/VAL Top1 was **47.862222%/47.261111%**, versus B’s **50.331111%/47.800000%**: B initially gains97 validation decisions. At epoch28, B falls to8548 correct versus A’s8644. At epoch40, B fits TRAIN **6.417222pp** better but scores100 fewer validation decisions. Its final TRAIN/VAL gap is **13.491111pp**, versus A’s6.518333pp.

| Selected-checkpoint family | A28 Top1% | B20 Top1% | B−A pp |
|---|---:|---:|---:|
| TSP | 36.0 | 36.7 | +0.7 |
| CVRP | 48.8 | 49.2 | +0.4 |
| ATSP | 77.6 | 78.5 | +0.9 |
| MVRP | 46.8 | 46.4 | -0.4 |

Eight tasks improve: TSP+0.7, CVRP+0.4, ATSP+0.9, OVRP+2.1, VRPL+0.5, OVRPTW+0.5, OVRPB+0.1 and OVRPBL+0.2pp. Ten regress: VRPB−1.7, VRPTW−0.7, OVRPL−0.7, VRPBL−0.8, VRPBTW−0.6, VRPLTW−0.9, OVRPBTW−2.2, OVRPLTW−1.0, VRPBLTW−0.6 and OVRPBLTW−0.2pp.

New B completed **9,600 updates/7.2M selected rows**,14.4M successful view rows,14,405,152 attempted view rows and five whole-update retries. Run wall was **3,554.574466s**, with15.661316GiB peak allocation. Reused A’s9,600 updates/7.2M rows/3,002.024237s remain historical expenditure; inherited stage1’s9,600 updates/7.2M rows count separately once. B has33,173,154 parameters versus A’s1,842,953, with1,600 active updates/400,000 rows per branch. This is protocol-equivalent control reuse, not replication.

R69 establishes additional native TRAIN fitting, unlike several R59–62 transfer attempts, but that capacity does not improve selected validation performance. Its late divergence resembles R40’s generalization problem. R64 and R68 retained shared parameters; R69 removed sharing, while also changing capacity and per-parameter optimizer histories. It therefore rejects this bounded unsharing recipe without isolating interference, establishing label noise or proving an accuracy ceiling. B remains **281 decisions,1.561111pp below R60B**.

**Recommend R70: clean-plus-latent-mixup fine-tuning. Next-plan score: 8/10, ready, PLAN ONLY.**

The focused primary-source search supports testing interpolation between training representations. [Manifold Mixup, Section2/Eq.1](https://proceedings.mlr.press/v97/verma19a/verma19a.pdf), mixes hidden states and corresponding targets while backpropagating through the encoder. [RegMixup, Section2/Eq.3](https://proceedings.neurips.cc/paper_files/paper/2022/file/5ddcfaad1cb72ce6f1a365e8f1ecf791-Paper-Conference.pdf), retains clean CE alongside mixed-example CE. Their evidence concerns vision. Applying a fixed mixture to candidate-aligned selector features is an unproven project adaptation.

1. **One new arm, reusable control.** R70 A is the fixed **R69B20 bank**,8604 validation correct, with its complete40-epoch history. Reuse its authorized one-time TEST when delivery completes. Train only R70 B. Both protocols use the same18-branch architecture and original pre-fine-tuning initialization: R67B terminal trunk plus entire fresh head670003. B starts from that initial state, not trained B20. R69’s official arm remains unchanged.

2. **Change the training objective.** For each task and dropout view, obtain candidate-aligned features `H = encode_pairs(inputs)` immediately before the existing scalar head. Within that task batch, draw one random row permutation `pi` and `lambda ~ Uniform(0,1)`, shared across both views. Form `Hmix_i = lambda*H_i + (1-lambda)*H_pi(i)`. Preserve solver-ID alignment and legal masks. Use mixed CE `lambda*CE(head(Hmix_i), y_i) + (1-lambda)*CE(head(Hmix_i), y_pi(i))`, averaged over rows and views. Set **`L_B = 0.5*CE_clean + 0.5*CE_mix + 0.35*min(u/720,1)*KL_clean`**. Keep gradients through both mixed partners. No mixed-view KL, layer search or coefficient scan. These are auxiliary latent constraints, not asserted winners for synthetic routing instances.

3. **Preserve clean dropout and target isolation.** Compute both ordinary clean forwards first, retaining their features. Use a private auxiliary RNG seeded by `700002 + 18*(u-1) + problem_id` for permutation, mixing and auxiliary-head dropout; restore the ordinary RNG afterward. Reuse the same draws on retries. Labels enter only the loss. Calculate interpolation/loss reductions in FP32, gathering the two legal target log-probabilities rather than multiplying zero targets by masked `-inf`.

4. **Keep optimizer and update semantics fixed.** Fresh ordinary AdamW: LR`3e-4*min(u/720,1)`, betas`(.9,.999)`, epsilon`1e-8`, WD`1e-4`, `foreach=False`. Use each active branch’s complete task gradient divided by three, one combined clip at1 and one optimizer step. Inactive parameters retain `grad=None`. Any nonfinite clean/mixed output or gradient retries the entire triple with restored RNG, reduced AMP scale and no optimizer, cursor or successful-clock advancement; retain the20-attempt limit.

5. **Fixed data and budget.** Preserve original18 tasks, pools, winners and splits; reuse exact seed2 global41–80 task/row/dropout order. Run40 epochs, real256/task×3, tails16×3: **9,600 new updates/7.2M selected rows**. B retains14.4M encoder-view rows and adds14.4M mixed-head rows. Thus example/update budgets match the reused control, while head computation differs. Report new runtime separately from reused R69 training and inherited pretraining. Log clean CE, mixed CE and total objective separately.

6. **Memory and implementation.** Keep sequential task processing and reuse the two existing encoder graphs; add only candidate-feature interpolation and two head evaluations. Use isolated R70 files. One focused synthetic CPU check should cover masked mixed-loss gradients, gradient flow to both partners and whole-update retry preserving auxiliary draws. Because auxiliary heads add activation storage, permit **one** discarded TRAIN-only capacity update at real256 on the existing worst-padded triple, accounting for the complete bank’s optimizer storage. No sweep or silent microbatching; an actual capacity failure requires reporting.

7. **Selection and delivery.** Evaluate native full18 validation every epoch, FP32 batch128; retain normal five-epoch/final/selected TRAIN evaluations. Select maximum integer validation correct count, earliest tie, saving all18 branches from one common epoch. Official R70 B requires **more than8604 correct**; otherwise fixed control A. Parent freezes weights, official arm and unchanged full-pool utility/global-ID-tie rule before **one full18 TEST for new B, regardless of validation improvement**. Reuse historical tests without inference. Deployment activates one selector and completes one solver. Sol/max executes only on the designated UUID/allocation1465/gpu03; publish one completed version.

A validation gain supports this objective package; remaining below R60B remains limited progress. Improved CE or reduced TRAIN/VAL gap without improved native Top1 rejects the selection hypothesis. Mixing could smooth genuine discontinuous winner boundaries, so failure remains plausible. This differs from R67’s same-winner clustering and R68’s parameter perturbations; no70% gain is promised or vetoed by regret.

**Memory update:** R69 completed successfully with no actionable TRAIN/VAL integrity defect. New task-specific B20 achieved8604/18000 validation correct,47.8%, versus fixed shared A28’s8644/48.022222%; official A remains. B initially gains97 decisions at common epoch20, then develops substantial divergence: final TRAIN59.985556%, validation46.494444%, gap13.491111pp. Eight tasks improve and ten regress at selected checkpoints. New B expenditure is9,600 updates/7.2M rows/3,554.574466s, separate from historical A and inherited stage1. Recommend one R70 clean-plus-latent-mixup objective contrast using exact R69B as reused control, identical initial bank/schedule, normalized equal clean/mixed CE and unchanged clean RDrop. Only the new model trains and receives a new frozen TEST. Capacity is matched within R70; extra head computation is disclosed. R58 and legacy B semantics remain deferred. The70% goal remains active and unsupported by this review.

**NEW R69 TEST and all prohibited delivery outputs were excluded from every finding and planning decision.** Historical TEST exposure remains disclosed. No inference, replay, checksum verification, GPU operation or local edit was performed.