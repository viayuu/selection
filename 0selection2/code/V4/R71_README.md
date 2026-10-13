# R71 joint performance-mixture learning

The scientific contract is `runs/R71_joint_performance_mixture/approved_plan.md`.
This is one new likelihood/readout/inference package, compared with the fixed
complete R70B23/global63 control (8709/18000 VAL correct), not covariance alone.
No R70 TEST or delivery analysis informs this implementation. A's complete
TRAIN/VAL history/result are reused unchanged; there is no new A computation.
Control reuse is protocol equivalence, not replication.

All18 branches retain the original full-information d128 encode_pairs backbone.
Every retained trunk starts from the ORIGINAL R67B terminal stage1 averaged-x
state in R69's pre-fine-tuning container. The old scalar head is deleted. Each
branch flattens all canonical ascending-global-ID legal Kx384 features into
Linear(384K,128), GELU, Linear(128,8*(1+D+D*(D+1)/2)), D=K-1, with no head
dropout. Eight full-covariance Gaussian components use the installed PyTorch
MixtureSameFamily/Categorical/MultivariateNormal(scale_tril) APIs.

Original FP64 TRAIN10000 cost differences against the first global-ID candidate
fit per-coordinate means/population standard deviations, with scale floor1e-6.
Standardization is FP64, then targets enter the loss in FP32. Costs and native
winners never enter the model forward. The mean of two backbone-dropout joint
NLLs divided by D is the only training objective. Density NLL may be negative
and is distinct from native winner-probability CE. Head/factor/likelihood work
is FP32 outside encoder AMP. Private head seed710002+problem_id initializes
ordinary first-layer weights, zero output weights, uniform mixture weights,
unit covariance and independent N(0,.1^2) component means while preserving the
outer RNG. Initial zero backbone/first-head-layer gradients are intentional.

Persistent buffers contain canonical pool IDs, FP64 TRAIN normalization and
scrambled Sobol2048 standard normals/task, seed710003+problem_id; inverse-normal
uniforms are clipped to [2^-24,1-2^-24]. One eval encoder forward predicts all
components. Each component uses the SAME table for EVERY query, transforms its
normals, undoes normalization, inserts reference zero and counts full-pool
minima. Mixture-weighted frequencies select maximum win probability, with
global-ID ties. Only one component and256 vectors are streamed at a time,
vectorized over queries. These16384 predicted vectors/query execute zero solvers.
The reporting CE alone floors winner probability at1e-12. All legal candidates
remain; input candidate reordering is canonicalized and output columns restored.

Seed2/global41-80 task/row/dropoutseed schedules, real256/task x3 and16/task
tails remain fixed:40E/9600 successful updates/7.2M selected rows/14.4M encoder
view rows. Replacing the head prevents a claim of bitwise-matched A/B dropout
tensors. Fresh AdamW uses LR3e-4*min(u/720,1), WD1e-4, betas(.9,.999), eps1e-8,
foreach=False. Each disjoint active task gradient is divided by3, clipped once
jointly at1 and stepped once. Inactive gradients remain None. Whole-triple AMP
recovery restores rows/RNG, lowers scale, and advances no optimizer/LR/cursor
or successful clock;20 attempts, no automatic job restart.

Full original18 VAL is evaluated every epoch in FP32batch128/tails using the
fixed probability rule; full TRAIN every5/final/selected best. Max INTEGER
VAL correct selects the earliest tied common epoch, saving the entire bank
and inference buffers. Official B requires strictly more than8709 correct.
This launch task performs no TEST load, freeze, TEST or delivery analysis.
Parent later freezes the selected complete bank/official arm and authorizes
one original full18 new B TEST regardless of improvement; A TEST is reused.

Exactly one focused synthetic CPU invocation checks density likelihood,
normalization/order/masks, target isolation, both-view connectivity (nonzero
output-weight TEST FIXTURE only), fixed probabilities and whole-update recovery.
Exactly one discarded TRAIN-only real256x3 capacity update uses the known
worst-padded VRPLTW/CVRP/TSP triple101/500/499. Both Adam moments are actually
materialized for the entire18-bank, including inactive/unused parameters.
Capacity failure is preserved and stops execution, with no fallback/sweep.
The mutated model/optimizer/scaler/RNG are discarded; formal state is recreated.
Actual parameters/memory/wall are measured. B is approximately38.2M parameters
versus A33173154; capacity and head/inference FLOPs are NOT matched.

Launch commands are `bash code/V4/run_v4_r71.sh cpu-check`, then one owned tmux
`r71_train` running `bash code/V4/run_v4_r71.sh train`. The launcher uses only
existing Slurm1465/gpu03/index1 UUID GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d,
with one assigned-UUID idle check immediately before GPU work, waiting if busy.
Conda easynco, existing python_compat bridge, thread1 and W&B offline selector /
yjkds-southern-university-of-science-technology are fixed.

Receipts include source snapshot/identities, protocol/args, CPU/capacity results,
actual parameter counts, normalization, initialization, schedule, raw updates,
AMP attempts, active counters, histories, whole-bank checkpoints/last optimizer,
W&B, status, pipeline and first-formal-update handoff. These assets stay local.
New B's formal expenditure excludes the discarded extra768-row update. Reused
A9600/7.2M/3668.194599s/5 retries and inherited stage1 once9600/7.2M/2764.217798s
are separate historical costs. No from-scratch encoder or two-new-arm claim.

Likelihood may fit irrelevant continuous variation, overfit or approximate
exact cost-tie atoms poorly; lower NLL alone does not establish selection
benefit. Native winner CE already targets winner probabilities. Beating A
while below descriptive R60B is limited progress. R58 and eight legacy B
semantic mismatches remain deferred/unfixed and do not veto native70 numerics.
The single-completed-solver70% goal remains ACTIVE and unachieved. Parent owns
reviews, full objective audit, state/memory and the sole completed publication.
