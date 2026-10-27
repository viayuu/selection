# Scientific Follow-Up: Execution Conditions And Symmetry

Original-paper research and scientific ideas ONLY, gpt-6-astra/max. Sol owns
all code inspection, numerical analysis, experiment planning, protocols,
budgets, execution and publication. No code review, numeric recertification,
GPU work, new solving, child agents or Git. Own REPORT.md,sources.csv,directions.csv
in this directory only. Main renders PDF unchanged after completion.

This is ONE bounded scientific question, not another comprehensive inventory
or automatic recommendation to launch a new model. Aim4-6 PDF pages and a few
directly relevant primary papers, with exact methods/experiments actually read.
If no actionable idea follows, say so; do not repackage a positional embedding
or argue that all71% errors must arise from an omitted random seed.

## New Operational Evidence Supplied By Sol

R118 is completed and published as861438cac5d4c4e0b43fe80f2a2e516c6dd92e45.
It captured144 frozen TRAIN instances,8/problem across18 problems, with genuine
node-resolved native states/actions/probabilities and all actual start columns.
Only18/89 deployments were observed: TSP/LEHD, CVRP/RELD_CVRP, ATSP/MATNET,
and retained reld_mt on15MVRP. No selector trained or old label changed.
These are973743 nonterminal trajectory probability rows, NOT that many
independent training instances. Native masks are not independent certificates.

Report: code/V4/runs/R118_native_trace_assets/RESULTS.md and coverage.csv.
17 problem samples,136/144 costs, matched frozen TRAIN scalars within fixed
5e-5+2e-6*abs(cost). ATSP/MATNET matched0/8, max absolute difference0.461335897.
Fresh captures use the locked production recipe but newly specified per-source
RNG states, not recovered historical process bytes. This does not isolate RNG
from lifecycle or numeric causes, diagnose label error, or quantify any global
accuracy limit. Do not certify these numbers yourself or turn the mismatch
into an already established causal explanation.

Sol checked the actual MATNET initialization implementation:
EasyNCO/neural_solvers/methods/matnet/policy.py::ATSPPolicy.pre_forward creates
zero row embeddings and col embeddings using torch.rand(batch_size,node_cnt),
argsort, then one-hot assignment; this occurs before encoding the matrix.
Original-orientation actual role inputs were captured in all8 cases. Production
executes128 augmentations but selects best start on orientation0. No claim that
the roles corresponding to the frozen old labels have already been recovered.

Sol checked actual CVRP/RELD_CVRP execution: N in the selected cases is102-472,
native starts=100, budget=min(N,100). All actual start columns, second-action
physical nodes and complete trajectories were retained. A start-budget cap is
not automatically a physical-role omission: one must inspect the actual start
selection and the selector before making a project-wide assertion.

Sol checked TSP/LEHD's misleading first_mode='random' path: the helper actually
uses arange(P); P=1 starts at physical node0. The observed8 cases do not support
a random-start claim from that configuration word. Do not lump deterministic
anchors and stochastic role assignments together.

The current selectors use raw graph/matrix inputs and learned feature routes.
Question to reason about, NOT a code fact for you to audit: if their inference
symmetry omits some candidate's physical execution role, how should the
learning task and representation be formulated? R57 previously raised this
issue, so it is not a newly discovered research direction. R117 added some
input-defined native reference activations, including first0 for TSP, but did
not recover full historical execution conditions. Its complete matched B-A
TEST Top1 difference was only+0.065432pp, regret worse0.001511pp. TSP LEHD still
had1 correct of762 true winners. Do not ignore that negative evidence.

## Scientific Questions

### Additional Sol Code Facts,2026-10-27

The parent completed two targeted code checks after this request. These are
counterevidence, not a request for you to verify code:

- R94 already inherits R88's B_execution_anchor. r94_experiment.MODEL_ROLE_ARM
  and input_roles explicitly mark physical TSP node0 in raw node slot3;
  r94_model imports prepare_role_inputs from r88_model, which preserves an
  explicit physical role on synchronous relabeling. Thus the current R94B
  baseline is NOT an unmarked permutation-invariant TSP input. Do not propose
  adding that same anchor as a new fix.
- CVRP/RELD_CVRP production profile has forcing_first_step=False. Its original
  CVRPModel.one_step_rollout computes decoder probabilities and takes TopK at
  selected_count1; the random.sample branch is only for forcing_first_step=True.
  The100-start cap is therefore NOT evidence that the method uses the first100
  physical customers or a hidden random subset. Do not infer such a missing
  input from the start count. An input-dependent native decision is computation,
  not automatically an exogenous condition omitted from the selector.

These findings further weaken a blanket execution-role explanation of the
remaining platform. R118 still establishes actual fresh MATNET role draws and
the cost-mismatch observation, but not their causal contribution or a new
whole18 model direction. Keep your scientific scope and recommend nothing
merely to keep GPUs occupied.

1. Distinguish fixed known deployment settings, inference-known physical
   start/end/start-set roles, exogenous random initialization, action sampling
   and unknown historical lifecycle state. A seed scalar is not automatically
   a useful model feature, nor a proof that one can reconstruct all draws.
2. Formulate the target as a fixed recipe's F_m(X,Z_m) where appropriate.
   Separate representation loss, unobserved execution variables and finite
   optimization. Under X-only prediction, when can a positive-frequency winner
   legitimately never be argmax? This is a logical point, not a Bayes-limit
   estimate for our data. Do not confuse conditional probability with mean cost.
3. Explain the symmetry requirement: reindexing while carrying physical roles
   versus changing which physical nodes receive the roles; stochastic draws
   versus their distributional invariance. Problem optimum invariance need not
   imply finite-budget solver-output invariance. Some reorderings may violate
   the original adapter's input conventions; do not call them valid tests by fiat.
4. Is a major 'instance + candidate execution conditions' learning object
   justified, as opposed to another method description, global position tag,
   native reference probe or output bias? Specify only scientific computation
   and knowledge boundaries, not an operational architecture/batch/launch plan.
5. If pre-execution roles/noise can legitimately be supplied, how must they
   remain bound to the actual selected deployment? If historical variables
   cannot be recovered, say why this prevents a claimed same-label comparison;
   do not quietly substitute newly sampled variables for the old realization.
6. Give at most ONE candidate scientific direction, its strongest objection,
   closest failed precedent and ONE decisive scientific contrast. It must
   address whole18 competition, not create a MATNET-only or ReLD binary goal.
   A limited annotation/condition coverage must remain explicitly limited.

## Relevant History And Nonclaims

Read post_r117_scientific_decision_20261027/REPORT.md and the needed parts of
the two earlier deep_direction_review_20261023 reports as scientific context.
Latest Astra decision was NO_READY_MAIN_DIRECTION, with one native-process
reserve. R118 provides real assets, not proof of a cheap sufficient state,
learnability, process transfer or a ready NN experiment. Do not promote the
reserve just because a recorder now exists. Full joint cost distributions,
ranking posteriors, actual query probing and label-bound context already have
substantial precedents; avoid erasing them or reverting to small loss searches.

Frozen new90k/whole18 pools,81000/4500/4500, native/FP64 labels unchanged.
R58 semantics/cost-table repair stays deferred. No deleted B candidate or
second ReLD restored. No query route/cost/winner or holdout answers in forward.
Historical TRAIN context exception remains self/base excluded. TEST has
participated in R&D, not a fresh blind test. Only two24GB RTX3090s are authorized
for future execution; this is not a request to plan your own experiments.

Use original sources where helpful, avoid mathematical or empirical guarantees
the papers do not give. No Zotero connection should be invented. Record exact
versions, reading scope, primary URLs and transfer limitations in sources.csv.
Report explicit model mismatch and stop rather than silently substitute Sol.
