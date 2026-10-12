# Conditional mechanisms after R63

Planning note, 2026-10-12. Same-family/provisional; not a production plan,
performance acceptance, experiment authorization or R64 adoption. R63 is still
running; neither its outcome nor a failure is assumed. This is one focused
primary-source scan, not a new review or preflight gate.

## Updated constraints

- Future rounds may change major training regimes, optimization, losses and
  selector inference, not only the encoder. Choose ONE substantive mechanism
  with its minimum informative control after the completed R63 train/validation
  outcome; this shortlist is not a queue of three authorized experiments.
- Keep original full18 tasks, all legal pools/128 deployments and native winners.
  Primary target is equal-task-macro native Top1 >=70%. Report cost/regret and
  mechanism contrasts, but do not invent regret or B-minus-A vetoes on that goal.
- Formal delivery now uses validation-only selection of a unique checkpoint and
  decision rule, followed by one complete original test evaluation. No test-based
  arm, epoch, rule, pool or per-task reselection. The parent's existing R63
  TEST_PROTOCOL.md specifies its predeclared controls/references, each evaluated
  once; the recommended arm is fixed by validation before test. This note does
  not change that protocol or live training. Report validation AND test; historical
  test exposure is disclosed, not relabeled as a pristine new holdout.
- Subsequent design uses train/validation evidence, not the newly reported test
  errors. R59--62's historical no-test outcomes remain unchanged.
- R58 is fully deferred; legacy B semantics remain unresolved. No solver calls,
  new costs, annotations, relabeling, queue changes or current-query answers.
- Astra/max planning/review and Sol/max implementation/execution remain requested.
  No extra review campaign; only necessary brief checks, then formal training.
  One commit per completed whole version. Sole assigned RTX3090 UUID
  `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`; no GPU-hour cutoff.

## Evidence that changes the priorities

R59 native-encoder transfer, R61 route pretraining and R62 different-source
portfolio pretraining did not advance. R62 B achieved 48.85% validation Top1,
below its A's 49.0278% and incumbent R60B's 49.3611%. R60's real expanded pair
supervision produced a small gain: +0.6056pp over replay and +0.2444pp over R45A,
not evidence of a ceiling or a route to 70% by simple scaling.

R40 fitted TRAIN to roughly 70--74% while validation stayed around 45--46%;
memorization and more updates alone are insufficient. Conversely R62 A's selected
TRAIN/validation Top1 were only 50.7061%/49.0278%, so severe overfitting cannot be
assumed for the current regime. R48/R49's pure-CE failures were single-problem
pair experiments, not a full18 native-classification impossibility result.
R57 established instance signal beyond task/size priors, not its easy learnability.

R63 already replaces the old cost/comparison/risk objective and cost-based
controller with native CE, consistency and Top1 selection. Historical checkpoint
Top1-versus-cost differences were small; checkpoint choice alone cannot explain
the approximately 20pp shortfall. Its gradient effect is not isolated, and R63
changes architecture too. Even a positive R63 cannot attribute all gain to CE.

Available reliable targets remain original 10k complete full-pool TRAIN rows per
task. Expansion assets exist, but R62's 11.4M observed cells mostly describe
different deployments; 39/128 small historical calibration matches do not make
the rest equivalent legacy targets. None of the candidates below needs future
queue output or another source-provenance campaign.

Local evidence: [R40](/public/home/shiys/0selection2/code/V4/runs/R40_discriminative_training/RESULTS.md),
[R48](/public/home/shiys/0selection2/code/V4/runs/R48_core_diagnosis/RESULTS.md),
[R49](/public/home/shiys/0selection2/code/V4/runs/R49_existing_data_diagnosis/RESULTS.md),
[R57](/public/home/shiys/0selection2/code/V4/runs/R57_task_signal_audit/RESULTS.md),
[R59](/public/home/shiys/0selection2/code/V4/runs/R59_trainable_solver_encoder/RESULTS.md),
[R60](/public/home/shiys/0selection2/code/V4/runs/R60_expanded_partial_supervision/RESULTS.md),
[R61](/public/home/shiys/0selection2/code/V4/runs/R61_behavior_pretraining/RESULTS.md),
[R62](/public/home/shiys/0selection2/code/V4/runs/R62_portfolio_performance_pretraining/RESULTS.md).

## 1. SAM: change the optimization geometry

The primary paper minimizes loss in a parameter neighborhood rather than only
at the current weights. Its empirical results are not evidence about this
solver-selection task. Sources: [Foret et al., ICLR 2021](https://arxiv.org/abs/2010.01412),
[official Google Research implementation](https://github.com/google-research/sam).

Project hypothesis: the same original winner information may yield a less brittle
decision boundary under neighborhood-robust training. This directly addresses
R40's fitting/generalization separation without another pretrained source or
pair expert. First priority ONLY if completed R63 shows meaningful TRAIN fitting
with a persistent validation gap; weak fitting and a small gap weaken this case.

Minimum comparison: same full18 classifier, initialization, original batches,
native objective, update count and schedule. A uses a zero-radius two-pass
control; B uses one fixed, predeclared SAM radius, no radius/loss/batch grid.
Both compute the same two complete gradient evaluations per update; A's second
evaluation is the applied gradient at unchanged weights. Microbatch accumulation
must form one effective-batch perturbation, not unrelated per-microbatch SAM steps.
Match stochastic draws across arms and disclose this control's redundant compute.

Cost: roughly twice ordinary training gradient work for each arm; inference is
unchanged. This is an algorithmic work estimate, not measured GPU-hours. SAM can
hurt fitting and does not supply missing winner information. A negative result
rejects this fixed optimization recipe, not all batch/LR regimes.

## 2. PCGrad: coordinate full18 task updates

The primary method projects conflicting task gradients before updating shared
parameters. Negative gradient alignment alone does NOT establish harmful
interference; the paper gives additional conditions. Source:
[Yu et al., NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).

Project hypothesis: sequential single-problem updates may impede a shared
classifier across heterogeneous tasks. Transfer pretraining does not resolve this
optimization issue. Prioritize if R63 shows uneven task fitting or progress that
does not translate into balanced macro accuracy. Existing reports do not measure
the proposed gradient conflict; per-task accuracy differences are not proof.
R48's weak single-task result also argues against treating sharing as the sole cause.

Minimum comparison: the same predeclared groups of three distinct task batches,
equal full18 coverage and original full-pool CE. A averages their gradients;
B applies PCGrad on shared parameters then averages. Same initialization,
presentations, macro-updates and schedule in BOTH arms. Do not compare grouped
PCGrad against an ungrouped historical run and attribute the entire difference to
projection. Grouping changes effective batching/update counts versus R63; disclose
that change. No per-task experts, candidate pruning or new task weights.

Cost: retain three task-gradient buffers and perform dot products/projections;
no additional solver or selector inference. Log conflict statistics during normal
training, not a separate diagnostic launch. Projection could suppress genuinely
useful gradients; there is no measured conflict or promised gain here.

## 3. A fixed deep ensemble of selectors, not a solver portfolio

The primary paper combines independently initialized networks by averaging
class probabilities; its uncertainty gains do not guarantee accuracy gains.
Source: [Lakshminarayanan et al., NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html).

Project hypothesis: some winner errors may be estimation variance rather than
missing input information. Use three independent full18/full-pool native-CE
selectors trained on ALL original TRAIN, no bagging or extra source. Uniformly
average aligned legal-candidate probabilities and complete exactly ONE selected
solver. This is input-only selector inference, not minimum cost over solver runs.

Minimum control: predeclared member0 versus the fixed three-member mean, not a
validation/test-selected subset or learned per-task gate. Each member has the
same training budget and validation-selected checkpoint; the ensemble bundle and
rule are frozen before its single full test. Training costs three member runs,
not a compute-matched single-model comparison. Selector inference has three
network evaluations; report its real overhead in ordinary evaluation.

Lowest priority: existing corrected/harmed counts do not establish independent
errors, and oracle unions are not predictions. Correlated biases may leave all
members wrong. This increases parameters and computation explicitly; the causal
hypothesis is variance reduction, not additional memory capacity. No test-time
augmentation, geometry invariance assumption, query-cost access or hidden solvers.

## Considered but not shortlisted

[The Road Less Scheduled, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/136b9a13861308c8948cd308ccd02658-Abstract-Conference.html)
offers a substantial momentum/iterate-averaging change without a prescribed
decay horizon. The [official implementation](https://github.com/facebookresearch/schedule_free)
requires different training/evaluation parameter points and explicitly still
requires LR/regularization tuning. It is not constant-LR AdamW or tuning-free.
Current evidence does not identify decay timing as the main blocker, so it is
below the shortlist, not an extra authorized optimizer branch.

New loss coefficients, focal/margin sweeps, temperature calibration and more
generic pretraining are not prioritized. Positive scalar temperature alone cannot
change argmax. R63's native objective must finish before another loss redesign;
class reweighting would change the instance-accuracy target unless justified.
Input permutations/rotations cannot be assumed to preserve deployed winners.

## Decision boundary

Wait for completed R63 and its ordinary train/validation report. If it reaches
the target, finish the locked test delivery rather than launching a rescue. If
it does not, choose at most ONE direction using train/validation evidence and
write that version's production plan then. Do not use R63 test errors to choose
the next mechanism, require extra probe runs, or execute this shortlist as a grid.
None of the papers or current assets justifies promising a jump to 70%.
