**Conditional ranking: first, fixed heterogeneous selector consensus; second, teacher consistency on existing expansion inputs. Neither is adopted.** R71 must finish its approved TRAIN/validation experiment before the next decision.

`review_independence=same-family; acceptance_status=provisional; scope=research only`

I examined four primary papers:

| Paper and Sections Read | Relevant Evidence and Limitation |
|---|---|
| [Tornede et al., *Machine Learning*, 2022](https://link.springer.com/article/10.1007/s10994-022-06161-4), §§5.1, 5.5, 6.3, 6.6 | Heterogeneous selector voting and rank aggregation improved algorithm-selection performance. The target was penalized runtime, not native Top1. Learned stacking did not outperform the best individual selector overall. |
| [Lakshminarayanan et al., NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf), §2.4 and Figure 2 | Independently trained networks combine predictions through probability averaging. Classification and uncertainty results support aggregation, but do not establish useful diversity among our saved selectors. |
| [Xie et al., CVPR 2020, Noisy Student](https://arxiv.org/pdf/1911.04252), §2, §4.1/Table 6, Appendix A.2 | Joint training on true labels and teacher predictions can improve the student. Their ablation retains some benefit from model noise without input augmentation. Evidence concerns image classification with strong teachers and substantial additional computation. |
| [Wei et al., ICLR 2021, revised manuscript](https://arxiv.org/pdf/2010.03622v5), §§3.1–4, especially Theorem 4.3 | Self-training can correct teacher errors under class expansion, separation and teacher-quality assumptions. These conditions are unestablished here; the theorem does not guarantee improvement from dropout consistency alone. |

1. **First: Fixed heterogeneous consensus across complete full18 policies.**

A concrete conditional candidate would reuse **R60B39, R67B36 and R70B23**, representing expanded partial supervision, contrastive pretraining and task-specific latent mixup. Each remains its original, globally selected complete policy. Every query runs the same bundle.

One fixed full-pool Borda rule would select the candidate with the smallest sum of within-model ranks, breaking final ties by global solver ID. This avoids fitting temperatures or weights across differently trained score functions. Member selection, aggregation and tie handling would be declared before evaluating the bundle, with no subsequent subset search.

**The R66 distinction needs correcting:** its [objective](/public/home/shiys/0selection2/code/V4/r66_objective.py:1) compared native CE/RDrop on standalone `g` against native CE/RDrop on `F+g`; both deployed `F+g`. It was not a native-CE versus regret-routing contrast. The proposed difference is heterogeneous complete policies and ordinal aggregation, rather than another corrective member trained beside the same frozen base.

This remains an **extension of an already tested ensemble family**. R66’s independent sum improved its base by only eight validation decisions, and corrective training lost against that sum. Furthermore, R67 and R70 share inherited pretraining. Different objectives, family gains or selected epochs do **not** prove complementary errors on individual instances.

The plausible benefit is combining already learned decision signals when their errors differ. It adds no query information and can dilute the strongest member.

**Minimal decisive comparison:** evaluate exactly one frozen bundle against its unchanged components and the standing validation incumbent on all18,000 native validation cases. Failure to beat the strongest component rejects that bundle’s selection benefit. Disagreement, hypothetical oracle coverage and better calibration are insufficient.

**Compute:** zero new training; three live selector evaluations per query, followed by **one completed solver**. Sequential evaluation limits activation memory, but latency must include every member. No reference retraining or new standalone reference evaluation is needed.

2. **Second: One-stage teacher consistency using the existing expansion inputs.**

The [R62 preparation receipt](/public/home/shiys/0selection2/code/V4/runs/R62_portfolio_performance_pretraining/data_cache/preparation.json) records **1.62 million expansion input rows**. Those inputs offer a different opportunity from their auxiliary deployment costs.

The mechanism would freeze an existing complete native selector, concretely R60B39, and obtain its full-pool probability targets on unchanged expansion inputs. A student jointly learns from original native winner labels and consistency with those teacher probabilities under model dropout. Original winners remain immutable; teacher probabilities are auxiliary predictions, not measured winners or new solver labels.

Reuse the existing raw-input cache, native input whitelist, complete legal pools and full-information backbone. Discard auxiliary deployment costs from this objective. Preserve coordinates, orientation, customer order and physical start roles. **No rotation, reflection or coordinate perturbation is justified by presumed winner invariance.**

This differs from R62’s comparisons between auxiliary solver deployments and R71’s likelihood training on measured original cost vectors. It also differs from RDrop on the original labelled examples: the additional constraint is applied across many more existing input locations.

The potential source of improvement is broader input coverage plus regularized learning from true and predicted targets together. Exact imitation alone cannot create better labels: without useful inductive bias and the original supervised anchor, the student can simply reproduce the teacher’s errors.

**The counter-evidence is substantial.** [R49](/public/home/shiys/0selection2/code/V4/runs/R49_existing_data_diagnosis/RESULTS.md:25) increased distinct training examples from2,000 to8,000 at equal4,000-update budgets; original validation accuracy changed52.5%→51.9%. That narrow binary experiment establishes neither a benefit from more samples nor a ceiling. R60’s genuine additional pair labels produced only a modest gain, while R62’s broader auxiliary supervision failed to advance the incumbent. Pseudo-targets are weaker evidence than those measured labels.

The expansion rows provide additional inputs, **not additional independently observed native winners**. Repeating their optimization increases presentations, not sample count. Teacher confidence also cannot certify correctness, and ImageNet-style uniform class balancing would change the native winner distribution.

**Minimal decisive comparison:** two otherwise matched students could share initialization, frozen teacher, original supervised stream, optimizer updates and forward-row budgets. The auxiliary stream uses original-TRAIN replay in A and prefix-excluded expansion inputs in B. This is a new scientific control because no saved run used that objective; historical references remain reused. No B-over-A native validation gain rejects the proposed coverage benefit under that recipe. Better pseudo-target agreement alone does not count.

**Compute:** one teacher pass over the chosen existing input bank, followed by explicitly budgeted student training. Teacher inference, distinct rows and repeated presentations must be counted separately. Deployment returns to one selector forward and **one completed solver**. Runtime is unmeasured; no new solve or ground-truth acquisition is required.

Consensus ranks first because it directly tests whether completed mechanisms combine usefully with little additional expenditure. Self-training is a more substantial data-organization change, but its teacher-error risk makes it the weaker recommendation. Neither paper set supports forecasting the remaining gap to70%.

**Memory update:** This scout read four primary papers and retained two conditional directions: fixed heterogeneous full18 consensus first, expansion-input teacher consistency second. R66 was native CE/RDrop standalone-versus-composite training, not regret routing; its weak ensemble gain remains material counter-evidence. Saved task summaries do not establish instance-level complementarity. R62 supplies an existing expansion input bank, but teacher targets would add no observed native winners. R49’s2,000/8,000 comparison showed no clear accuracy benefit and cannot establish a ceiling. Self-training requires assumptions about teacher quality and class structure that remain unproved. No successor, member bundle, loss coefficient or execution budget is adopted. R71 must complete before closure and adoption. R58 and legacy eight-B semantics remain deferred; full18 native single-completed-solver70% remains active and unachieved.

**R71 partial TRAIN/VAL and NEW R71 TEST evidence did not inform this research.** Its current result, status, history, report, analysis-output, delivery and parent-state artifacts were not inspected. Historical TEST exposure remains disclosed. No experiment, model inference, prediction replay, cache creation, GPU/job query, solver execution, edit, Git action or subordinate-agent delegation occurred.