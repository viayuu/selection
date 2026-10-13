# R71 Conditional Delivery Protocol

Parent delegates CPU-only preparation in `r71_delivery_prep` and an independent
controller in DIFFERENT owned tmux `r71_delivery`. Neither existing session nor
the original trainer may be killed/replaced. Preparation adds isolated delivery
sources only, performs ONE combined Python/shell syntax invocation, and starts
the independent controller. Future execution is UNVERIFIED at preparation.
No preparation GPU query/job/checkpoint load/forward/metric replay/freeze/TEST/
analysis or repeat CPU/capacity/pilot occurs. The six training implementation
files, approved plan, protocol/launch snapshots, data and historical files stay
unchanged. This executor performs no Git, agents, reviews or successor design.

Original TRAIN: Slurm1465.105/gpu03, `r71_train`/pane%96/panePID1287378,
trainerPID1476336; physical GPUindex1 UUID
`GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d`. The CPU controller observes only
this original step with sacct, its owned tmux existence, and small published
status/pipeline records every240 seconds, without time cap/restart or GPU/queue
survey. RUNNING0:0 is not terminal success. Require actual COMPLETED0:0 on gpu03
AND `r71_train` absent. Record command/stdout/stderr/returncode/time/elapsed and
separate tmux-absence evidence in `training_terminal_receipt.json`. Preserve any
actual failure and stop without automatic retry or altering/stopping training.

After original terminal success, require published training_complete and
40E/global41-80 histories: full18x1000 native VAL each epoch,9600 successful
updates/7.2M selected rows/14.4M encoder views, branch1600 updates/400000 rows.
Read actual saved records only. Select earliest MAX INTEGER original full18 VAL
correct; all18 complete branches from ONE common epoch. Official B strictly
greater than8709; otherwise fixed A. TEST, NLL, costs and task gains never select.

The actual selected checkpoint schema includes `model`, `args`, zero-based
`epoch`, `common_epoch`, `global_epoch`, `branch_epochs`, `problem_names`,
`native_correct`, `native_total`, `macro`, `parameter_mode`, `checkpoint_mode`,
`projector_included`, `inference_rule`, `successful_updates`, `presentations`,
`model_state_sha256`, `all18_complete_branches`, `persistent_inference_buffers`.
The model has `branches.0.*` through `branches.17.*`, including every retained
backbone/new density parameter and persistent canonical IDs, FP64 TRAIN means/
scales, Sobol normal vectors and triangular indices. Head shapes vary by legal
pool size. No scalar score head remains. The future loader uses actual
`r71_model.make_bank` as the factory and strictly restores the ENTIRE saved bank
and buffers. A retained B trunk-shaped factory template initializes allocation
only; strict whole-bank loading replaces every branch before any forward.
No head transplant/averaging/taskwise-best assembly or A weight load occurs.
Inference requires the selected checkpoint and saved arguments/buffers, not
label access, normalizer fitting or an external API.

Freeze the unique selected B checkpoint identity, its buffers and official arm
in exclusive `final_test_lock.json` BEFORE any new TEST. Fixed A is EXACT
R70B23/global63/8709 VAL correct; reuse R70 final_test_lock.json/models/B metadata
and test_results.json/models/B object unchanged. No A weights/load/inference/
training/recount/reselection/retest. New TEST counts A0/B1; original A already1.
Control reuse is protocol equivalence, not replication.

Only after the gates/lock, dispatch ONE owned `r71_final_test` using
`bash code/V4/run_v4_r71_test.sh`, within existing allocation1465/gpu03 and ONLY
the assigned UUID. Check assigned-UUID idle at actual launch; wait if occupied,
never switch. Exclusive `final_test_started.json` permits ONE new B TEST,
regardless of VAL improvement. No automatic TEST retry/reselection. Preserve
bootstrap/final-test/process/scheduler failures and original logs.

TEST uses the unchanged actual R71 validation evaluation path: FP32batch128,
all original18x1000 rows,104-row tails,144 wrapper/active-branch/encoder batches.
One encoder pass/query predicts eight full-covariance Gaussian components.
The SAME2048 persistent scrambled Sobol standard normals/task are reused for
every query/component, streaming ONE component and256 vectors, vectorized over
query rows. Undo TRAIN-only normalization, insert reference difference zero,
count full-legal-pool minima, weight frequencies by mixture probabilities,
select max winner probability/global-ID ties. Reporting CE alone uses floor1e-12.
These16384 predicted vectors/query are not route samples or solver executions.
Optional density NLL uses the SAME encoded parameters with an explicit external
target interface; costs/winners never enter inference. No mean-argmin,
alternative sampling, augmentation, pruning, ensemble, tuning, solver or
singleton benchmark. Deployment completes one selected solver.

Save live probabilities/selection scores/predictions/native winners/rawFP64
costs/indices/nodes/global solver IDs/names locally as NPZ, plus official TEST
JSON. Total evaluation time includes mixture integration and is not singleton
latency or solver runtime. Require actual new TEST step COMPLETED0:0 on gpu03
AND owned test tmux gone, with independent `test_terminal_receipt.json`.
Process0 is only `test_process_receipt.json`, not scheduler evidence.

Only after actual TEST terminal0, invoke `r71_analysis` ONCE with an exclusive
marker. Read saved JSON/CSV/updates only, no NPZ replay/forward/representation
audit. Reuse existing metric/summary/plot helpers. Report full18 equal-weight
macro, task and TSP/CVRP/ATSP/MVRP TRAIN/VAL/TEST Top1/2/3, native-winner CE,
mean_cost, vsSBS and actual regret; best/final/last-five VAL and common ACTUAL
epochs. TRAIN every5/final/selected stays sparse; never interpolate or invent
five-epoch TRAIN means or seed uncertainty. Historical references use existing
locked official JSON only, with missing historical measurements left missing.

R71 optimization jointNLL and NLL/D are separate from native-winner probability
CE and may be negative. Curves preserve actual NLL/objective/LR/gradient/clip/
AMP/retries/successful-vs-attempted encoder views/time/memory. No cleanCE/mixCE/KL
is invented for R71. B38231040 versus A33173154 parameters: capacity and density
head/probability compute are unmatched. This tests the full likelihood/readout/
inference package, not covariance alone; full target supervision also differs
from R70. Lower likelihood loss alone does not establish selection benefit.

Account formal newB9600/7.2M/14.4M successful encoder views, actual attempted
views/retries/wall/memory, zero mixed-head rows. Fixed A's historical R70B
9600/7.2M/3668.194598888047s/5 retries and inherited stage1 ONCE
9600/7.2M/2764.217798s remain separate. ONE discarded capacity update5.965898s
total/3.752883s update/768 rows/full-bank Adam moments305848320bytes is EXTRA,
outside formal9600; its mutated state was discarded before formal recreation.

Write RESULTS/REPORT/comparison CSVs/curves/reference provenance/analysis JSON,
analysis receipt and actual delivery receipt including locks/terminal records/
counts. Weights, NPZ, rawupdates, W&B/cache/regenerable schedules stay LOCAL;
Git alone is not all inference assets. Historical TEST exposure is disclosed.
NEW R71 TEST never changes model/selection/budget/rules or reaches the successor
planner. Native70 threshold remains12600/18000; goal70 stays ACTIVE/unachieved
pending parent whole-objective audit and actual reviews/publication. R58 and
eight legacy B feasible-domain semantic inconsistencies remain fully deferred,
disclosed/unfixed and do not veto native70 numerics.

Conda easynco, existing python_compat bridge, thread1 and offline W&B selector /
yjkds-southern-university-of-science-technology are fixed. No environment/
package/network probes, other GPU/allocation/client, solver/new labels, agent
or Git. Parent owns Astra reviews, memory, whole-objective audit and publication.
