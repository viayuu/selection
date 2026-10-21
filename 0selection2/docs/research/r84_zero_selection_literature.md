# Zero-selection literature handoff

Date: 2026-10-18. Scope: literature and next-improvement ideas only.

No experiment code, project results, resource state, or training implementation
was inspected. No experiment, hyperparameter search, or evaluation protocol is
specified here.

Provenance clarification: the orchestrator reports that its native
`spawn_agent` request with `model=gpt-6-astra` and `reasoning=max` was accepted
for this research worker. No separate nested model or backend-review call was
made inside this worker. The earlier backend-availability wording referred to
the absence of such a nested call and must not be read as evidence that the
orchestrator's Astra routing was unavailable. The worker has not independently
verified the actual serving-model identity; requested/accepted routing and
independently verified runtime identity are distinct claims.

## Context supplied by the user

- NEW90k means 81,000 TRAIN / 4,500 VAL / 4,500 TEST instances per task, across
  all 18 tasks. These counts are supplied context, not independently checked.
- Winner frequencies are naturally imbalanced. Illustrative TSP observations
  are LEHD about 16% winners, about 15% mean predicted probability, but zero
  selections; DIFUSCO about 8% winners and zero selections.
- Supplied R77 objective: `0.35 CE + 0.35 within-instance pairwise +
  0.02 relative risk + 0.35 RDrop`, with a static-ID/features dual-stream
  representation and a maximin score head.
- Already explored: global AUC and condition-matched cross-instance regret
  ranking; fixed candidate intercepts; cost-sensitive offsets; calibration and
  naive reranking; risk replacement; a frozen-score cost head;
  equal-probability ensembles; pool-performance-image contrastive warmstart.
- The user reports no evidence that focal or LDAM was executed in the R78
  scan. This is not an independently verified claim that they were never run.
- Historical R78 remains active-unresolved in persistent context, while the old
  app goal was paused. This research handoff changes neither status.

## First recommendation: CE plus confidence-adaptive focal supervision

**Prioritize this low-complexity objective for NEW90k. Do not impose a target
selection frequency, inverse-frequency class weights, or inference offsets.**

For instance i, let A_i be its legal candidate set, y_i its genuine winner under
the existing label convention, and p_iy the masked-softmax probability of that
winner. Use the following proposed supervised loss:

\[
\ell_i^{\mathrm{adapt}}
=-[1+\lambda(1-p_{iy})^\gamma]\log p_{iy}
=\mathrm{CE}_i+\lambda\mathrm{FL}_i,
\qquad \lambda>0,\ \gamma>0.
\]

The focal component comes from Lin et al., ICCV 2017, Section 3.2, Eq. (4).
Their original loss is `FL = -(1-p_y)^gamma log(p_y)`: it suppresses easy
examples rather than uniformly increasing the absolute loss on mistakes.
**Keeping CE and adding FL is the adaptation proposed here, not their original
formula or a previously validated solver-selection result.**
[Original paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf).

The conceptual change to the supplied objective is only
`L_proposed = L_R77 + 0.35 * lambda * mean(FL)`; this is not a proposal to replace
the risk objective, introduce another head, or change inference. With two
dropout passes, the supervised term means the average of this loss evaluated
on each pass, not focal loss evaluated on the averaged distribution.

Why this targets a different mechanism:

- Low true-winner confidence gives extra supervised weight regardless of the
  method's population frequency. Easy high-confidence correct cases retain
  their original CE floor. The loss multiplier approaches 1 + lambda and 1,
  respectively.
- This is example-dependent supervision through the representation, not a
  fixed candidate intercept or a transformation of frozen predictions.
- Differentiate the entire focal expression. Detaching its modulation gives a
  different weighted-CE gradient, not the stated objective.
- Low p_y is a smooth difficulty proxy, not exactly the event `argmax != y`.
  A low-confidence correct Top1 also receives extra weight. Candidate-pool size
  and intrinsic task difficulty can therefore affect its emphasis.

For clarity, the logit gradient is

\[
\nabla_z\ell_i^{\mathrm{adapt}}
=\left[1+\lambda\left((1-p)^\gamma
-\gamma p(1-p)^{\gamma-1}\log p\right)\right](p_i-e_{y_i}),
\quad p=p_{iy}.
\]

Thus each sample retains the CE gradient direction but has a confidence-
dependent magnitude; the aggregate gradient is generally not a constant
rescaling. The bracket is not just the loss multiplier and need not vary
monotonically over the entire probability interval.

### Why this differs from RDrop

RDrop minimizes bidirectional KL between two dropout predictions, whereas the
proposed term uses the true winner. Equal predictions can have zero consistency
penalty even when both are confidently wrong. RDrop is not an entropy-maximizing
loss and does not inherently force flat probabilities. Its presence alone is
not evidence that it caused zero selection.
[Liang et al., NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html).

Replacing CE by pure focal would reduce supervision on easy cases relative to
an unchanged consistency term. The proposed CE floor avoids that particular
coupling, while increasing supervised pressure relative to consistency on hard
cases. It also increases that pressure relative to the existing cost term;
therefore preserving the cost term is not a guarantee against cost regression.

### Evidence boundary and risks

Lin et al. establish hard-example focusing in dense detection, not recovery of
LEHD or DIFUSCO on these tasks. Persistent hard examples can be noisy or
near-tied winners rather than learnable niches. Stronger winner supervision
can improve Top1 yet increase mean cost by inducing expensive false positives.
No selection-coverage or cost guarantee follows from this objective.

Charoenphakdee et al. prove that standard focal loss is classification-calibrated
but not strictly proper: its population minimizer's confidence is not generally
the class posterior. Those results do not constitute a theorem for the proposed
CE-plus-focal, pairwise, risk, and RDrop combination. In particular, agreement
between mean probability and winner frequency is not its success criterion.
[Focal theory, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Charoenphakdee_On_Focal_Loss_for_Class-Posterior_Probability_Estimation_A_Theoretical_Perspective_CVPR_2021_paper.html).

## Two distinct alternatives, not a proposed stack

| Mechanism | Mathematical change | Difference from previous attempts | Main limitation |
| --- | --- | --- | --- |
| Cost-sensitive Top1 margin | Penalize the most violated within-instance winner-versus-rival margin, with the rival's actual cost gap inside the maximization. | Changes which decision-boundary violation drives the gradient; not cross-instance AUC or a fixed cost offset. | Can overlap existing pairwise behavior; large-cost, low-score rivals can dominate, and near ties offer little cost margin. |
| Conditional recall with a cost objective | Constrain recovery on instances where a candidate genuinely wins, while minimizing selection cost. | Controls conditional misses rather than predicted method shares or unconditional probability averages. | More complex constrained optimization; infeasible recall demands can force costly false positives. |

### Alternative 1: loss-augmented Top1 margin

For costs to be minimized, define

\[
r_{ij}=\frac{c_{ij}-c_{iy_i}}{\max(|c_{iy_i}|,\epsilon)},\qquad
\ell_i^{\mathrm{margin}}=\max_{j\in A_i}
\{s_{ij}-s_{iy_i}+r_{ij}\}.
\]

The winner is included in the maximization, so the loss is nonnegative. For
`hat_y = argmax_j s_ij`, the loss upper-bounds `r_i,hat_y`, since
`s_i,hat_y >= s_i,y`. This elementary pointwise bound is not a guarantee of
cost consistency or neural-network generalization. It is an instance-cost
adaptation of structured margin rescaling, not the original LDAM objective.

[Tsochantaridis et al., JMLR 2005, Section 2.2.4](https://jmlr.org/papers/volume6/tsochantaridis05a/tsochantaridis05a.pdf)
provides the underlying general-loss margin formulation and explicitly notes
that high-loss outputs can dominate despite having low predicted scores.

LDAM is relevant evidence that training margins are a separate lever from
post-hoc calibration: its loss subtracts a margin from the true-class logit,
with `Delta_y` proportional to `n_y^(-1/4)`. However, its motivating bound targets
balanced class error. That is not automatically the objective for naturally
imbalanced solver costs, so inverse-frequency margins are not the first
recommendation here.
[Cao et al., NeurIPS 2019, Eqs. (10)-(13)](https://proceedings.neurips.cc/paper_files/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Paper.pdf).

A more recent adjacent paper, Son et al.'s DBM, adds instance difficulty margins
to misclassified positives. Its actual formulation uses cosine classifiers and
class-frequency margins; borrowing its intuition does not justify calling a
score-head-only adaptation DBM. The checked source is a December 2024 preprint;
no venue acceptance is asserted here.
[DBM primary manuscript](https://arxiv.org/html/2412.15477v1).

If existing within-instance pairwise supervision already emphasizes exactly
the same most-violated cost-sensitive rival, this alternative would not be a
genuinely new mechanism. No implementation inspection was performed to decide
that question.

### Alternative 2: conditional recall, not a selection quota

Let `R_tk = P(hat_y = k | y = k, task = t)`. A conceptual objective is

\[
\min_\theta\ \mathbb E[r_{i,\hat y_i}]
\quad\text{subject to}\quad R_{tk}\geq\rho_{tk}.
\]

This does not require `P(hat_y=k)` to match winner frequencies. Recovery must
occur on the candidate's own genuine-winning instances. Constraints refer only
to legal task-candidate combinations with positive winner support. The symbols
describe a research objective, not proposed numerical targets or a protocol.
Here `E[r]` is mean per-instance actual regret, not `mean_cost` or a ratio of
mean costs; these objectives are not interchangeable.

Narasimhan et al. give multiclass confusion-matrix optimization with precision
and recall constraints. Cotter et al. address nondifferentiable rates via a
proxy-Lagrangian. Their assumptions and possible randomized classifier
solutions cannot be transferred as guarantees for one deterministic neural
checkpoint. A differentiable rate proxy alone also does not guarantee actual
hard Top1 recall.
[Narasimhan et al., JMLR 2024](https://www.jmlr.org/papers/v25/22-1137.html);
[Cotter et al., JMLR 2019](https://jmlr.org/papers/v20/18-616.html).

## Interpretation guardrails

Uniform scaling is not targeted correction. For an entire objective L,
`grad(aL) = a grad(L)` and positive scaling preserves its minimizers; for plain
SGD this is equivalent to scaling the learning rate. Adam, clipping, and
unscaled regularizers complicate the update equivalence. Multiplying only R77's
CE term changes its balance against pairwise, risk, and RDrop, so it is not
literally a global learning-rate change. Weighting only misclassified examples
is also not uniform scaling.

Mean probability is not instance discrimination. With a free softmax bias,
the mean CE bias gradient for method k is `mean(p_k) - frequency(y=k)`.
Matching those means can happen without learning any winning niche. A
feature-independent predictor can match class frequencies perfectly while
always choosing the most frequent class. Conversely, zero selection need not
be irrational if available features cannot identify when the method wins, or
its false-positive costs outweigh recoverable gains.

All proposed uses of y and costs assume the same legal pool, precision, and
tie convention; a numerical near-tie is not evidence of a robust niche. None
of these proposals aims to reproduce a winner-frequency histogram. Increasing
the number of selected methods is not, by itself, evidence of improvement.

## Source coverage

The local `research-lit` skill was read first. Local PDF title triage and the
existing Seesaw manuscript were consulted; arXiv metadata search and original
CVF, NeurIPS, JMLR, and author-uploaded arXiv papers supplied the external
evidence. Zotero and Obsidian interfaces were not available. No paper PDFs were
downloaded and no project code or result files were opened.

Seesaw remains relevant background: it combines negative-gradient mitigation
with instance-dependent compensation for false positives. It is not merely a
fixed logit bias, but its count-based mitigation is less directly aligned with
the present request than confidence-adaptive supervision. Its LVIS evidence
does not establish that the supplied 16%/8% cases suffer the same mechanism.
[Wang et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Seesaw_Loss_for_Long-Tailed_Instance_Segmentation_CVPR_2021_paper.html).
