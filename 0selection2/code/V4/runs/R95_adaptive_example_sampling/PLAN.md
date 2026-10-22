# R95: Adaptive Training Exposure With Inverse-Probability Correction

## Question

Does allocating repeated training exposures using each example's current
gradient signal help the unchanged full18 selector, without changing the
natural empirical loss into a class-balanced or cost-reweighted objective?
This tests data/optimization dynamics, not another score head or penalty.

## Fixed Design

- NEW90k frozen split and candidates; fresh R84 seed2 parameters and TRAIN-only
  geometry; original full R88B/R77 selector, method ID/features, MLP/maximin.
- Both arms retain the original CE .35, comparison .35, relative risk .02,
  two independent Dropout passes, RDrop ramp .35, AdamW and gradient clip1.
- Logical batch1024, micro352 TSP/CVRP, other tasks1024, eval256; 57600
  successful updates, warm4320, VAL1440. No extra seed or coefficient search.
- First7200 updates use identical original natural shuffled rows. This visits
  every TRAIN row before the exposure experiment begins. Thereafter A uses
  uniform sampling with replacement; B uses `q_i=.5/N+.5*s_i/sum(s)`.
  All-zero proxy norms fall back to uniform. The .5 mixture is fixed.
- Each occurrence's ENTIRE row loss, including both Dropout passes and KL,
  is multiplied by `1/(N*q_i)` and averaged over actual batch rows. No
  normalization by summed weights, inverse-weight clipping, deduplication,
  winner quotas or label-driven model input. All rows retain positive support.
- Both arms use the same per-task uniform RNG stream after the natural phase;
  inverse CDFs differ intentionally. Do not claim identical sampled instances.
- Refresh own TRAIN proposals every7200 updates, reusing already-scheduled
  full TRAIN evaluation. The scalar proxy is the exact final affine weight
  and bias gradient norm of deterministic eval CE+comparison+risk, including
  component cancellation. RDrop is zero in eval. No extra network forward.
- The proxy is NOT the stochastic two-pass full-network gradient norm.
  Positive support and matching correction protect the expected pre-clip
  gradient, not proxy quality, gradient variance or nonlinear AdamW updates.
  Staleness, batch1024 and the limited proxy may yield no useful effect.

## Evaluation And Execution

One unique strict full18 VAL macro_vs_sbs_pct minimum using frozen TRAIN SBS;
first wins ties. Then the locked model gets full TEST/LIB, PDF and CSV,
fixed36 real-winner recovery/false picks and cost/regret, and NSS4. No TEST
or per-task selection. Existing TEST has informed development, not blind.

B uses the released R92B slot4090_0. A queues after R92A's completed
TEST/LIB/PDF on3090_0. Healthy R94A/B are not stopped. Existing allocation
containers and local R84 caches are reused; W&B explicitly offline. Publish
one curated GitHub commit after the complete version, not on each launch.

## Literature Boundary

- [Katharopoulos and Fleuret, ICML2018](https://proceedings.mlr.press/v80/katharopoulos18a.html)
  motivates affordable gradient-based importance sampling and accounting for
  scoring cost; this is not its exact bound or implementation.
- [Zhao and Zhang, ICML2015](https://proceedings.mlr.press/v37/zhaoa15.html)
  provides the corrected finite-sum sampling identity, not a deep-selector
  generalization guarantee.
- [Alain et al., arXiv1511.06481v7](https://arxiv.org/abs/1511.06481)
  studies stale proposals and per-example gradient norms. No claim of
  reproducing its distributed system or convergence results.

Original literature screen: `docs/research/unbiased_importance_sampling_screen.md`.
Acceptance remains joint full18 Top1 improvement with mean cost/regret not
increasing and genuine recovery rather than merely fewer zero-count groups.
