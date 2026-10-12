# R63: native-winner classification with recurrent node workspaces

Date: 2026-10-11. Performance round 5/100. Locked production design.
Plan readiness: 8/10, READY PLAN ONLY; same-family/provisional.
Not implementation verification, performance acceptance or attainment of70%.
One candidate and one matched control. No additional review/preflight gate.

## R62 completed; reference retained

R62 completed rc0 at22:37:34 CST; parent analysis ran once, rc0. Both arms
completed25,344 pretraining updates and40 fine-tuning epochs/56,880 updates.
Its RESULTS.md confirms the locked screen was false:

| Arm | Cost-selected epoch | Native macro Top1 | Actual macro regret % |
|---|---:|---:|---:|
| R62 A |39 |0.49027777777777776 |0.9252248081470262 |
| R62 B |24 |0.48850000000000005 |0.9312329232715721 |
| R60 B incumbent |39 |0.4936111111111112 |0.9227890652257744 |

B-A is-.1778pp and B-incumbent-.5111pp, with worse regret. No partial gain
over R60B is established. B pretraining/fine-tuning walls were1,984.334543s/
3,772.505203s. These are completed transfer results, not a generalization
ceiling. Retain R60B as the historical reference; also report R45A49.116667%/
.9260483583% regret. Neither reference initializes R63.

## User objective and information boundary

PRIMARY target: original full18 equal-problem-macro NATIVE-winner Top1 >=.70,
selecting ONE solver from each original legal pool. Regret/cost and B-minus-A
are disclosed outcomes, NOT additional veto conditions on this user target.

Use original10k complete fullpool TRAIN/task, original validation,22 solver
identities and128 legal deployments:180k labeled training queries. Use unchanged
native ind, including its existing tie convention; no tie-aware relabeling,
task/pool reduction, cost-table replacement or guessed complete-pool winner.
No current query's target cost, rank, winner, route or row identity enters forward.

Reuse original R45A input adapters, solver features and original-TRAIN geometry
normalization. Preserve all original node/task/constraint information: coordinates,
signed demand, depot identity, service, customer windows, route limit and task
conditions. ATSP retains its ORIGINAL full directed input travel matrix and
the original dense encoder path; it is not replaced by Euclidean coordinates
or only undirected/summarized graph data. Input travel distances are not solver
target costs. No order augmentation or index-derived execution metadata.

No expanded inputs/costs, saved routes, external pretrained/native solver weights,
new annotations or solver runs. Do not wait for or touch user queues/assets.
R58 remains fully deferred; legacy B semantic mismatch remains unresolved.
NO TEST ACCESS in this round, including on validation advancement.

## Mechanism and paths

ROOT: `code/V4/runs/R63_recurrent_workspaces` from the repository root, or
`runs/R63_recurrent_workspaces` relative to `code/V4`.
A: `A_static_memory_seed2`; B: `B_communicating_memory_seed2`.
Each holds best/last checkpoints, history, result and validation predictions;
parent writes one completed RESULTS.md after both arms.

The original DualStreamSelector maintains one node bank shared by all candidates.
R63 instead maintains a candidate-specific node workspace and repeatedly passes
computed information between nodes. It asks whether distributed candidate-
conditioned computation can improve native winner discrimination, not whether
another route-pretraining target, solver embedding or local Top2 expert helps.

Both arms start from identical fresh seed2 original R45A instance/solver encoder
parameters (`r45_experiment.parameters('A')`), d128. Retain the original complete
InstanceEncoder and its three special tokens. Replace common joint layers with
one shared recurrent cell applied to `H[batch, candidate, node, 128]` for exactly
EIGHT rounds. Cell weights are tied across rounds, tasks, nodes and candidates;
there are no per-method experts or candidate-specific parameters beyond the
original shared SolverFeatureEncoder identity inputs.

Let Z be encoded nodes and S be the common encoded solver bank for the legal
pool. Initialize `H0[m,v] = LayerNorm(Z[v] + S[m])` with shared normalization.
S is the same bank for all candidate workspaces/readout and stays fixed within
the recurrent rounds. Its encoder remains trainable across optimizer updates.
All legal candidates receive a workspace, not only predicted strong candidates.

## Graph and recurrent computation

Build a fixed INPUT-ONLY graph for each query: union of each physical node's16
smallest outgoing and16 smallest incoming off-diagonal travel entries, using
fewer neighbors when necessary. Add bidirectional connections between every
valid node and the depot/three special tokens. Preserve directed edge values,
deduplicate graph edges and mask padding. No guessed TW/backhaul feasibility
filter or solver-dependent graph selection. The original dense encoder still
reads full input relations before this sparse recurrent stage.

Each round uses a shared learned additive scalar gate over projected target
state, projected neighbor source state, candidate vector and input edge distance
information; normalize valid neighbor gates by their sum. Aggregate32-dimensional
projected source messages, project the aggregate to d128, and update each node
with a shared d128 GRU cell. Mask padded states. No attention/QKV variant,
edge labels, auxiliary loss, route decoder or adaptive depth/halting search.

Project source/target NODES to message/gate width32 BEFORE gathering edges.
Do not materialize concatenated source/target full-d128 vectors on every edge.
Both arms use the same per-round projection/gather/update path; do not create
an A-only cached-projection optimization that changes the nominal work contrast.
The recurrence is synchronous: all updates read the previous round's states.

ONLY A/B difference:
- A reads each neighbor's candidate-conditioned INITIAL H0 as the message
  source in EVERY round. Its own target-node state still evolves, so it retains
  iterative local processing and rich encoded fixed-memory access.
- B reads that neighbor's PREVIOUS-round Ht, including special/global nodes,
  allowing newly computed information to propagate between locations.

Same parameters/initial values, workspace dimensions, graph, eight rounds,
projection/message counts, supervised data and effective update budget. This
tests evolving neighbor communication, not eight calls versus one, more solver
executions, a larger training memory or independently parameterized deeper stacks.
Actual wall times may differ and must be reported, not asserted equal.

## Exact common readout and classification objective

After round8, use the original shared mean/max pool per candidate to obtain
`h_m = pool(concat(masked_mean(H_m), masked_max(H_m)))`.
Keep the original shared decoder, explicitly:
`c_m = decoder(Q=h_m, K=S, V=S, key_mask=legal_solver_mask)`.
The unchanged shared3d-input score-head form reads `concat(h_m, c_m, S_m)`
and outputs ONE direct class logit `u_m`. Candidate-specific h_m is not replaced
by a common pooled h. Readout weights/dimensions are identical between arms.

Apply the complete legal-pool masked softmax to u. Native full-class argmax,
with original stable global-ID tie breaking, selects ONE solver. Do not construct
maximin logits or call the old comparison/risk objective. Latent workspaces
produce no feasible route, current-query solver outcome or execution trace.

For each original batch take two independent dropout forwards:
`loss = .5*(CE(u1, native_ind) + CE(u2, native_ind)) + lambda_e*RDrop(u1,u2)`.
Use the existing symmetric KL consistency convention on the direct class logits;
lambda_e ramps linearly to.35 over three epochs and then remains.35, identical
in both arms. This is native-winner-only supervised learning with label-free
consistency, not a bare-CE reproduction of R48. No cost/gap weighting, cmp,
risk, regression, auxiliary loss, smoothing or class balancing.

## Matched training, accumulation and resume

Both FROM SCRATCH, seed2, identical original task/instance batch order and tails.
Keep sampling RNG separate from common model initialization. All parameters are
trainable; no R59--62 transfer, frozen branch or extra head/source branch.

Effective batch128, fixed microbatch16 in BOTH arms. For each microbatch average
its two dropout CE losses and combine its KL, then accumulate weighted by
`microbatch_rows / actual_original_batch_rows`. Tail batches use their actual
row count. Accumulate across the entire ORIGINAL batch, then unscale/check,
clip the complete accumulated gradient and take ONE optimizer update. Do not
step per microbatch or separately for the two dropout forwards.

Save RNG at the original-batch boundary. On AMP failure discard ALL accumulated
gradients and retry the ENTIRE original batch with restored CPU/CUDA/dropout RNG
and the same data order, using the existing scaler-backoff convention. Do not
retry only the failed microbatch, advance cursors or count a successful update.
Use existing epoch-atomic resume patterns, including model/optimizer/scaler/RNG,
common schedule and sampling state. No source-fingerprint machinery is needed.

AdamW LR1e-4, WD1e-4, dropout.1, clip1, warmup3. Controller driven by full18
validation native macro Top1: effective improvement.001 percentage point,
LR patience3/factor.5, stop patience8, min15/max40 epochs. Run A first; A defines
the COMMON learning-rate trace and horizon E. B follows exactly E, without
independent stopping or extension. Each arm has1,422*E successful updates,
at most56,880, with complete original coverage each epoch.

Each arm saves its STRICT MAXIMUM full18 validation native Top1 checkpoint.
Equal Top1 retains the earlier epoch, not a lower-cost substitute. Historical
cost-selected checkpoints/results stay unchanged; no retroactive selection.

## Decision and interpretation

Either arm reaching native macro Top1 >=.70 is a validation target hit. Do NOT
reject it because B-minus-A is small/negative or regret worsens. Preserve that
arm's validation-locked checkpoint and disclose all cost/regret/task outcomes.
Validation alone is not held-out proof of the user goal: any subsequent held-out
evaluation must follow a locked model/inference policy without test-guided
selection. This plan performs no test access or automatic test search.

B-minus-A measures the communication mechanism under the common classification
protocol. +2pp is a materially useful mechanism contrast; reference+3pp is an
interim substantial-progress milestone (currently.5236111111111112 against R60B).
These are research interpretation aids, NOT replacement user goals or vetoes
on attaining70%. No regret-no-worse gate applies to this round.

Report all18/family/native macro Top1, CE, actual regret, raw mean cost,
macro_vs_sbs_pct and complete curves. Explain accuracy/cost trade-offs plainly.
If both improve without B-over-A, do not attribute gains to evolving messages.
Comparison with historical cost-trained models changes the learning protocol
as well as computation and does not isolate an objective-change effect.

The common A-controlled E/LR and fixed8-round/d128/message32 recipe define THIS
round's fair budget. A negative result rejects this instantiated recipe under
that budget, not every depth/capacity, full18 classification or instance signal.
No loss/head/round-count/seed rescue grid follows the result.

## Evidence limits, cost and execution workflow

R57 establishes genuine instance-dependent baseline signal; R52 locates much
error in competitive method ordering. Neither demonstrates easily recoverable
20pp. R42 already supplied negative WHOLE18 relational-model evidence. R53--56
were local specialists, and R48/R49 pure CE was one problem/pair, not a strict
refutation of this full18 classifier. R40's strong TRAIN fit warns of memorization.
R59/R61/R62 transfer failed; R60's independent pair-data gain was small.

Native Top1 and minimum expected cost are different objectives, but saved-history
cost-versus-highest-Top1 gaps were only0--.1667pp in the five previously summarized
arms (R45A/R60B/R61A/R61B/R62A). That does not explain the70% gap or isolate loss
gradients. [NSS Table8](https://arxiv.org/html/2410.09693v2#A9) also reports only
36% TSP/61% CVRP greedy classification accuracy in different pools. No promise
that CE or more computation alone will reach70%.

Risks: the propagation bias may not capture solver differences; dense encoding
may already contain sufficient relations; repeated messages may oversmooth;
10k/task may permit memorization rather than generalization; and extra selector
latency may buy no useful gain. Candidate sharing/tied weights do not remove
these risks. Legacy B semantics remain disclosed, not repaired or blamed by fiat.

Planning estimate12--24 GPUh TOTAL plus ordinary data preparation, not a measured
runtime or cap. No GPU-hour limit. Only assigned available RTX3090 on gpu03,
UUID `GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`; easynco, tmux, existing offline
W&B/resume. No other GPU, installs, foreign-process changes or queue modifications.

Record actual time/peak memory ONLY during normal formal training and final
validation, not a standalone timing pilot. Include input graph construction,
selector GPU time and total selection wall; report batch throughput separately.
Within the final normal validation pass, use the fixed first16 queries/task
individually for batch1 timing, then ordinary batches for the rest, counting each
query once. Disclose first-call/subsequent latency and covered actual sizes;
these are selector timings, not measured full-solve overhead. No solver run.

Sol implements and executes. Only necessary seconds-scale interface/CPU tests,
then the FIRST actual training batch is FORMAL training; no disposable GPU
preflight, separate timing pilot, audit or further review campaign. Run A then B,
parent analysis once after both complete, ONE commit/push for the completed
whole version. This planning action writes documents only, starts no GPU/code
execution and makes no performance claim.
