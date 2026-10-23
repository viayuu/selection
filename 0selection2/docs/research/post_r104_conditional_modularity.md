# Post-R104: Conditional Representation Modularity

Date: 2026-10-23. Requested model: `gpt-6-astra`; requested effort: `max`.
These are the requested settings, not independent attestation of the execution backend.
Workflow: `research-lit`, `sources:web`; three primary papers, relevant body sections read.
This is a bounded mechanism review, not a current-state literature census or experiment plan.

## Judgment And Evidence Boundary

**Retain one conceptual direction: jointly learned, input-conditional representation experts inside one selector.** The distinctive object is the intermediate feature transformation and its allocation of training gradients, not an expert's final solver vote. Experts are latent feature modules, not solver classes. Same-task instances can use different modules, and different tasks can reuse modules. All legal candidates remain eligible without selection quotas. This is a different sharing/learning bias, not proof of a strictly larger function class or of the cause of today's missed winners. The architectural basis is [Shazeer et al., Sec. 2](https://arxiv.org/pdf/1701.06538#page=3) and [Ma et al., Sec. 4.2](https://dl.acm.org/doi/10.1145/3219819.3220007).

**Supplied project context, not independently audited:** full18 NEW90k has 81,000 TRAIN examples/task; the shared approximately 1.84M model reaches approximately 71%, while some methods have thousands of TRAIN winners but zero picks. The user reports completed R104-width256/7.29M TEST Top1 71.126%, regret 0.40049 versus historical R89B 71.420%/0.38965, with stronger TRAIN fitting not accompanied by corresponding VAL improvement. These historical comparisons do not isolate a causal mechanism. No inference is made about unfinished R104-depth or R102.

Overlap was checked only in [R60-R79 index](experiment_mechanism_index.md) and [R80-R102 index](experiment_mechanism_index_R80_R102.md), without following their code/result links or refreshing their historical status. CE, balance/ranking, recurrence, ICL, input competition and pretraining are not treated as new. The relevant distinctions are:

| Already represented in the supplied history/indexes | What conditional representation specialization would change |
| --- | --- |
| R69 complete task-private bank, older scenario | Task identity fixes the branch; conditional modules can differentiate instances within a task while retaining cross-task sharing. R69 is not a matched NEW90k exclusion of this hypothesis. |
| R87 method-private residual parameters/heads | Candidate identity fixes private parameters; latent experts would specialize intermediate features by input, not allocate one private scorer per candidate. Merely gating final scores is outside this recommendation. |
| R104 width increase | Width adds shared capacity; conditional modules change which inputs train/use which parameters. Dense networks can also learn specialization, so this is an inductive-bias distinction, not an impossibility claim about width. |

## Primary Papers

All three are conference papers. Sections below identify the body evidence actually read; their results are external evidence, not project measurements.

| Paper And Body Sections | Mechanism, Evidence And Limit |
| --- | --- |
| **Shazeer et al. (ICLR 2017), [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/pdf/1701.06538).** Read author arXiv v1: Secs. 2-2.1, 4, 5.1, pp. 3-6. | End-to-end hidden-layer experts receive input-dependent sparse activation. Sec. 4 reports self-reinforcing expert imbalance; noisy routing and importance/load penalties address utilization. Sec. 5.1 reports improved language-model perplexity at comparable computation but substantially different total capacity. This supports conditional computation, not a parameter-matched diagnosis of solver-selection errors. |
| **Ma et al. (KDD 2018), [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007).** Sec. 4.2, Eqs. 6-8; Secs. 5.1-5.2; Sec. 6.4.2/Fig. 6; Sec. 7, pp. 1934-1935, 1938. | Input-dependent task gates mix shared bottom representations. Synthetic results show better average loss/robustness under lower task relatedness, while best losses can be comparable. This argues for trainability/sharing, not just expressivity. The demonstrated mixture is soft/dense; sparse routing is a prospective extension. One recommendation objective concentrates on one expert, so task-specific gates do not ensure distributed use. |
| **Chi et al. (NeurIPS 2022), [On the Representation Collapse of Sparse Mixture of Experts](https://proceedings.neurips.cc/paper_files/paper/2022/file/df4f371f1f89ec8ba5014b3310578048-Paper-Conference.pdf).** Sec. 2.2; Secs. 3.1-3.3; Secs. 4.2, 4.4-4.5, pp. 2-9. | A router-gradient component lies in the expert-embedding span, motivating a projected, normalized routing space rather than direct routing in the full feature space. X-MoE reports less collapse and downstream gains: softmax XTREME average 65.3 versus 63.8 for its SMoE baseline, not solver Top1. It still uses load balancing; downstream adaptation freezes both router and expert networks. This is not evidence for an unregularized, from-scratch selector recipe. |

Access note: ACM full text returned HTTP 403 and OpenReview a browser challenge. MMoE body was read from this [KDD paper PDF mirror](https://raw.githubusercontent.com/wzhe06/Reco-papers/master/Multi-Task/%5BMMoE%5D%20Modeling%20Task%20Relationships%20in%20Multi-task%20Learning%20with%20Multi-gate%20Mixture-of-Experts.pdf), streamed without saving a local PDF; venue/authorship were cross-checked on the [official KDD entry](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-). No secondary explanation is used as evidence.

## Main Objections And Remaining Gap

The following are conceptual transfer judgments, not observed project failures:

- **Problem/size priors can masquerade as specialization.** A route depending only on task and node count remains coarse conditional sharing; it does not establish useful within-task/within-size feature differentiation. Removing an explicit task ID would not remove information inferable from constraints or input shape. None of the three papers guarantees freedom from such shortcuts. MMoE's per-task gates explicitly permit task-dependent preferences; its same-input multi-objective setting also differs from this project's task domains and categorical solver choice.
- **Expert utilization is not class recovery.** Shazeer's balancing addresses expert starvation, not whether suppressed solver winners occupy learnable regions. Balanced expert traffic can coexist with identical experts, task-only partitioning, or zero picks for a solver. Internal load balance is not a solver-class quota, but its presence cannot be concealed or credited as natural winner recovery. No such auxiliary objective is prescribed here.
- **Representation collapse is a separate risk.** X-MoE addresses routing geometry and norm-driven dominance, not task/size invariance or guaranteed complementary experts. Its gradient argument concerns one component, not inevitable collapse of every model. Stable routes can still encode the wrong partition. A router operating after relevant instance information has been lost cannot recover it merely by selecting another downstream module.
- **More winners do not establish separability.** Thousands of TRAIN winners demonstrate support, not that available inputs distinguish those winners from nonwinners. A useful conditional partition remains unestablished. Sparse specialization can fragment that support further. The width observation alone neither proves interference nor selects modularity as the remedy.

**Bottom line:** the genuine research gap is useful *within-task instance-conditioned feature learning with reusable modules*, not more private heads or a better-looking selection histogram. Retain that one mechanism as conceptually distinct, but defer any efficacy claim. Do not reinterpret it as output calibration, forced uniform predictions, or an ensemble of already-trained models. Sparsity is an implementation choice rather than the conceptual contribution. No configurations, budgets, arm counts, dimensions, coefficients, optimizer, router specification, acceptance gates or launch plan are set here; those remain with main Sol.
