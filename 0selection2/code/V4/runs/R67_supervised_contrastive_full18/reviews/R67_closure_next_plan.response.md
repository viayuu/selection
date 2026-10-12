`review_independence=same-family; acceptance_status=provisional`

**R67 performance: 4/10, not ready for70%. TRAIN/validation integrity: 9/10, ready within the inspected scope.** No actionable defect found. New R67 TEST and delivery outputs were excluded.

| Selected model | Stage2 epoch | TRAIN Top1 | Validation correct / Top1 | Validation regret |
|---|---:|---:|---:|---:|
| [A: classification pretraining](/public/home/shiys/0selection2/code/V4/runs/R67_supervised_contrastive_full18/A_classification_pretrain_seed2/result.json) | 16 | 50.375000% | 8767/18000 /48.705556% | 0.951395% |
| [B: contrastive pretraining](/public/home/shiys/0selection2/code/V4/runs/R67_supervised_contrastive_full18/B_contrastive_pretrain_seed2/result.json) | 36 | 51.277778% | 8872/18000 /49.288889% | 0.935440% |

B gains **105 decisions, +0.583333pp**, over matched A. The matched comparison is positive, so the scout’s conditional failure trigger did not occur. B nevertheless remains **13 decisions,0.072222pp below R60B**. The broad validation incumbent is unchanged.

The benefit spans **13 improved tasks and five regressions**. Family gains over A are TSP+0.4pp, CVRP+0.5pp, ATSP+1.7pp and MVRP+0.526667pp. Against R60B, B improves the first three families but loses0.18pp across MVRP. This supports a modest, distributed benefit from this pretraining recipe; one seed does not establish a reproducible breakthrough.

The complete stage2 curves show late fitting/generalization divergence:

- A’s selected-to-final TRAIN Top1 rises50.3750%→52.9983%, while validation falls48.7056%→48.1444% and validation CE rises1.09381→1.13451.
- B’s epochs20→40 TRAIN Top1 rises49.9189%→51.6439%, while validation stays approximately49.06%→49.08% and CE rises1.08725→1.10309.

These are comparable native-classification evaluations after full fine-tuning. Stage1 B’s contrastive loss is not CE; its0.1 multiplier remains a project choice. The evidence supports late generalization loss, especially in A, without establishing sharp minima, label noise or an accuracy ceiling.

Both arms completed19,200 updates/14.4M selected-row presentations, with8/6 AMP retries. Combined recorded arm wall time was11,506.55seconds; peak allocated memory15.3311GiB. The [objective](/public/home/shiys/0selection2/code/V4/r67_objective.py:23), [stage transition](/public/home/shiys/0selection2/code/V4/r67_model.py:72) and [selection path](/public/home/shiys/0selection2/code/V4/r67_experiment.py:318) support the intended loss, terminal averaged-x transfer, identical complete head reset, fully trainable stage2 and integer selection. Both selected validation recounts agree.

**Recommend R68: SAM fine-tuning from the shared R67B contrastive-pretrained trunk. Score: 8/10, ready, PLAN ONLY.**

Among the three scouted directions, SAM directly tests the newly observed late fine-tuning divergence. Joint cost-distribution learning changes supervision but introduces stronger distributional assumptions. RouteFinder transfer changes the backbone and requires additional adaptation; R59 already tested trainable ReLD transfer. Neither currently has stronger project evidence.

[SAM’s Algorithm1 and Sections3.2/C.3](https://arxiv.org/html/2010.01412v3) support neighborhood optimization and its use during fine-tuning, including a fixed radius0.05. That evidence concerns image models. AdamW, native CE/RDrop and the shared three-task perturbation below are project adaptations.

1. **Shared starting model.** Load R67B’s `stage1_terminal_x.pt`, using its terminal averaged-x trunk. Replace the entire utility head with the identical fresh `stage2_head` from R67’s `shared_initialization.pt`, seed670003. Copy this complete starting state into both arms. Discard the projector; train every selector parameter. Preserve the R64 d128/four-encoder/two-joint-layer architecture, handcrafted34+ID, all original inputs and directed ATSP path. The trunk is inherited pretraining, explicitly disclosed.

2. **One contrast.** `A_plain_adamw_seed2` uses ordinary AdamW; `B_sam_adamw_seed2` adds one SAM perturbation. Both use the existing two-forward native CE plus symmetric legal-pool RDrop, weight`0.35×min(u/720,1)`. Reuse the R64 model/objective and R67 input/evaluation machinery in isolated R68 runner/gradient files. Both optimizers/scalers start fresh. Future weights are ordinary AdamW weights, with no schedule-free transitions or averaging.

3. **Matched data and optimizer.** Reuse R67’s stage2 task/row/dropout schedule, global epochs41–80, for both arms. Keep real256/task, three distinct tasks/update, effective768; tails16/task/effective48. AdamW settings: betas`(.9,.999)`, epsilon`1e-8`, weight decay`1e-4`, `foreach=False`; LR`3e-4×min(u/720,1)` using successful-update counter`u=1…9600`, then constant. Run40 fixed epochs with no early stopping or plateau controller.

4. **Exact SAM update.** Compute the unscaled FP32 mean gradient`g` across all three task losses at unchanged weights`w`. Before clipping, form`epsilon=0.05*g/max(||g||₂,1e-12)`. Perturb all participating parameters once. Restore the pre-pass RNG state and recompute the three-task mean gradient on the **same rows and same two dropout masks** at`w+epsilon`. Copy the exact original weights back, clip the second mean gradient once at1, and perform one AdamW step. A clips its ordinary mean gradient once at1. Weight decay occurs only in that single optimizer step.

5. **Numerical recovery.** If either pass or reduction is nonfinite, restore exact unperturbed weights and RNG, clear both passes’ gradients, reduce AMP scale and retry the entire same update. Optimizer state, LR counter, row cursor and presentation counters advance only after success. Retain the existing20-attempt failure limit; checkpoint only restored, unperturbed weights. One focused CPU check covering perturbation/restoration, repeated dropout masks and whole-update retry is sufficient before formal training.

6. **Budget and feasibility.** Each arm performs9,600 new updates/7.2M selected-row presentations; the pair performs19,200/14.4M. A processes14.4M dropout-view rows and B28.8M, so updates/examples match while computation does not. Count the shared inherited stage1’s9,600 updates/7.2M rows separately once. Sequential passes should retain batch256’s activation footprint; extra weight/gradient buffers are small relative to the measured15.33GiB peak. Approximately2.5hours for the new pair is a planning estimate, not measured throughput or a cutoff. Inference remains one selector forward.

7. **Selection and delivery.** Evaluate all18,000 validation rows each epoch in FP32 batch128. Select maximum integer native-correct count, earliest exact tie; official arm by validation, tieA. Freeze complete weights and the full-pool utility-argmax/global-ID-tie rule. Parent then authorizes one original full18 TEST per frozen model, with historical results reused. Retain normal TRAIN evaluations, task/family reporting and cost disclosure. Sol uses only the assigned UUID; one commit follows completed training, locked TEST and reporting.

The SAM hypothesis is unsupported if B fails to beat matched A on validation. Beating A while remaining below R60B is limited progress. Comparisons with R67 also change the fine-tuning optimizer, so only the new A/B contrast isolates SAM. Stronger regularization could harm fitting; no70% jump is promised.

Memory update, proposed without editing files: R67 completed matched80-epoch arms. Contrastive B36 achieved49.288889% validation, +0.583333pp over A16 across13 tasks and all four families, but−0.072222pp versus R60B. Both stage2 curves show late fitting/generalization divergence; the matched contrastive result is positive. Inspected training/selection paths are correct. R68 proposes one SAM-versus-AdamW fine-tuning comparison sharing R67B’s terminal contrastive trunk and fresh identical head. New R67 TEST did not inform planning; historical TEST remains exposed. R58 and eight legacy B-task semantics remain deferred; the full18 single-solver70% goal remains active and unachieved.