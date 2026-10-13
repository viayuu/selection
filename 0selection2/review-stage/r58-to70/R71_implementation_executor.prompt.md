You are the user's actual GPT-6.1-Sol/max coding/execution/monitor agent.
Resume SAME actual Sol thread after R70 conditionaldelivery CLI0/turn.completed
at2026-10-12T23:07:29UTC. R70 is COMPLETE and must never be changed/restarted.
Primary and fresh actual Astra/max reviews ended0 with no actionable defects;
all fullrawresponses+Memoryupdates are preserved. Parent owns R70's ONE final
commit/push now. Do not run Git or edit parent state/memory/review/R70 files.

Implement and launch ONLY the actually approved R71 plan, not a proposal.
Read the FULL approved source (including its counterarguments and boundaries):
code/V4/runs/R71_joint_performance_mixture/approved_plan.md
This is the actual completed-R70 TRAIN/validation-only Astra/max response,
not a TEST-informed choice. Do not reinterpret it as a covariance-only claim.
No paper/backend/loss/component-count/sample-count/seed/hyperparameter search.

Scope: add isolated r71_model.py, r71_gradients.py, r71_experiment.py,
r71_cpu_check.py, run_v4_r71.sh, R71_README.md in code/V4 and generated artifacts
under code/V4/runs/R71_joint_performance_mixture/. Reuse actual existing
R69/R70 native data, schedule, initialization and whole-update helpers when
appropriate without modifying them. Read those actual sources first. Do not
alter old results, .gitignore, AGENTS, tools, data, solver code or cost labels.
R58/eight legacy B semantic issues remain fully deferred/unfixed. Original
full18/native fullpools/one completed solver and target70 remain unchanged.

ONE new B, exact fixed A:
- A is complete globally selected R70B23/global63 bank/8709of18000VALcorrect.
  Reuse its complete40E TRAIN/VAL history/result unchanged as fixedcontrol;
  no A weights/load/inference/recount/reselection/training/newTEST.
  Later authorized delivery reuses existing R70 TEST object unchanged, but
  do NOT read newTEST/analysis/RESULTS/REPORT to design or launch this model.
- B retains18 complete task branches, existing encode_pairs full-information
  backbone/d128/4instance/2joint/handcrafted34+ID/input geometry/legalpools.
  Initialize every retained trunk from ORIGINAL R67B terminalstage1
  averaged-x PREFINE-TUNING state, not trained R70/R69. Replace/delete old
  scalar utility head, not add a bypass. All retained backbone/newhead train.
- Each legal pool is canonical ascending GLOBALsolverID; r=firstcandidate,
  D=K-1. Flatten candidate-aligned Kx384 encode_pairs features. Preserve all
  original legal candidates; padded/invalid rows do not enter flattening.
  Candidate input reordering must canonicalize then align output correctly,
  not change cost-column/ID/nativewinner semantics or hard-prune a candidate.

Targets and head exactly as approved:
- Original raw costs inFP64; target differences c_j-c_r. Fit mean/population
  std independently for each nonreference coordinate using ONLY original
  taskTRAIN10000rows, scale=max(std,1e-6). TargetstandardizationFP64 thenFP32
  loss interface. No perbatch stats or heldout fitting. Preserve native ind.
- Head Linear(384K,128), GELU, Linear(128,8*(1+D+D*(D+1)/2)), NO headDropout.
  Shared over rows, one head per existing complete taskbranch;8 mixture
  components/full covariance. Mixture logits, mean vectors, lower triangular
  factor; diagonal softplus(raw)+1e-3. Use torch.distributions actual installed
  MixtureSameFamily(Categorical,MultivariateNormal(scale_tril=...)), not a
  handwritten density/flow or a diagonal approximation.
- Private head seed710002+problem_id: ordinary firstLinear initialization,
  finalweightszero, logitsbiaszero/uniformweights, unitinitialcovariance
  (strictlowerbiaszero and diagonalinverse-softplus appropriate to floor),
  independent componentmean biases N(0,.1^2). Preserve outer initialization
  RNG. Record actual parameter counts; expectedapproximately38.2M vsA33.17M,
  NOT capacity matched. No old scalarhead/performanceargmin/CE shortcut.
- Two ordinary backbone-dropout views; mean jointNLL/D only. All density
  heads/factors/likelihood/reductionsFP32 outsideAMP, encoderAMP allowed.
  No cleanCE/mixCE/RDrop/KL/risk/pair/auxiliary term or new labels. NLL may
  be negative and is NOT winner-probabilityCE; report separately. Do not
  incorrectly label this the old objective or matched FLOPs/capacity.

Frozen inference from beginning, including every validation:
- Before anyVAL, generate/save scrambledSobol2048 standardnormalvectors/task,
  seed710003+problem_id, dimensionD, inverse-normal with U clipped
  [2^-24,1-2^-24]. Same fixed table for EVERYquery and component; not current
  queryindex-dependent/randomsampling. Tables, normalizers, ordering are
  persistent nontrainable buffers saved with complete checkpoint.
- One eval encoderforward produces mixtureparams. For each8component, affine
  transform its SAME2048 normals with mean/scale_tril, undo TRAIN normalization,
  insert reference0, count minima across ALLlegalcandidates. Weight component
  frequencies by predicted mixture probabilities. StreamONEcomponent and256
  vectors at a time; vectorize over queryrows, do not Python-loop perquery.
  This is16384 PREDICTEDvectors/query, not solver executions or route samples.
- Choose max winprobability/global-IDtie; use probabilityfloor1e-12 ONLY for
  reporting nativewinner CE. Do not use mean-argmin, sampledcostlabels,
  validation-fitted temperature, probabilitycalibration or countbudget scan.
  Actual current costs/winner/gap are never in eval forward. Inference works
  from saved checkpoint/buffers without data-label access or external API.

Training budget/control EXACT:
- Seed2 originalR67/R69/R70 global41-80 task/row order, real256/task x3 per
  logicalupdate/effective768; tails16x3/effective48. Full18 originalnative
  TRAIN10000/task,40E/9600successfulupdates/7.2Mselectedrows/14.4Mencoderviewrows.
  No earlystop/microbatching/gradientaccumulation/newsizebucket/drop-tail.
- FreshAdamW LR3e-4*min(u/720,1), WD1e-4, betas(.9,.999), eps1e-8,
  foreachFalse. Per-active-disjoint-task FP32gradient/3; ONE combinedclip1,
  ONEoptimizerstep; inactivegradNone/no decay/moment/step advance.
- Nonfiniteoutputs/likelihood/gradients retryWHOLEtriple same rows/RNG with
  loweredAMPscale, no optimizer/LR/cursor/successfulclockadvance;20attemptlimit.
  Record successful/attempted rows/views/retries/actualwall/grad/clip/LR/memory
  and activeparameter counters. Different replacedhead means do not pretend
  all A/B dropout tensors bitwise match; sample/task/dropoutseed schedule does.
- Full18VAL FP32batch128/tails eachE with EXACT fixedprobabilityrule; actual
  completeTRAIN every5/final/selectedbest, same rule/evalmode. MaxINTEGER
  originalfull18VALcorrect, earliesttie, all18branchesONEcommonepochcheckpoint.
  OfficialB iff>8709, otherwise fixedA. No cost/TEST veto/pertaskstitch.
  Do NOT loadTEST, freeze or executeTEST in this implementation/launch task.
  Later parent authorizes ONEoriginal18newBTEST regardlessofimprovement.

Engineering cautions, not protocol changes:
- Zero finalheadweights imply initially zero gradients into firstheadlayer
  and backbone, an intended analytic consequence of approvedinit. Do not
  falsely fail the connectivity check or change productioninit. The ONE
  synthetic CPUcheck can verify connectivity after a synthetic optimizer
  update, or with nonzero outputweights in a TESTFIXTURE only. Graph/optimizer
  paths must then carry finite nonzero gradients normally.
- Canonical pool order is a known input role, not costsorted/truewinner order.
  Fitnormalizers only originalTRAIN; query cost only explicitNLL target path.
- Optimizerwholeupdate and checkpoint factories must accommodate newhead
  shapes and persistent buffers; don't cache trainable projected features.
  Training computes density parameters/NLL, not expensive MCwinprobs every
  batch. Full requiredVAL/TRAINEVAL DO use the approved probabilityrule.
- Reuse current ancestors-aware repository metadata discovery and data facade;
  do not repeat old .git path mistakes or launch another provenance audit.
  Report source/identity by existing conventions, no new hash framework.

Minimum checks and actual formal launch, promptly:
1. ONE focused syntheticCPUinvocation covering actual densitylikelihood,
   normalization/order/mask/restoration, targetisolation, bothviewgradient
   path (respecting zero-init above), fixedprobabilitytable/global-IDdecision
   and one whole-update recovery. Do not repeat historical checks/audits.
2. ONE discarded TRAIN-only capacity update atreal256x3 on same scheduled
   worstpaddedfulltriple selected by known inputshapes only (R70 used
   VRPLTW/CVRP/TSP padded101/500/499), including materialized ENTIRE18bank
   Adam first/secondmomentGPUstorage. NoGPUbatchsweep. Record peakallocated/
   reserved/physical/wall and actualfullbankbytes. IfOOM, STOP/report; no
   unauthorizedmicrobatch/fallback. This extra768rows not formal9600budget.
3. Discard mutatedcapacity model/optimizer/RNG, recreate exact approvedformal
   initialstate. Use existingallocation1465/gpu03 and ONLY assignedGPUUUID
   GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d (index1 RTX3090). No SSH/new
   allocation/otherGPU/client/queue, no CPU-on-GPUprobes beyond minimumabove.
   Start ONEBtrainer in ownedtmuxr71_train. R70 TRAIN/TEST are alreadyended.
4. Capture actualschedulerstep/node/trainerPID/pane/PID and firstfinite formal
   successfulupdate at256x3 (or actualscheduledtailiffirstisone), loss/NLL,
   gradient/LR/retries/memory. Return handoff promptly withfixed40EtrainerLIVE.
   Settle all your foreground execsessions, not stop tmuxtraining. Parent then
   handles conditionalCPUdeliveryprep and actualAstra reviews/Git later.

No second arm/reference rerun/extra seed/solver/newcost/route labels/R58,
no giant preflight, no model/API change, no subordinate agent. Model's actual
runtime and memory are measured, not assumed. Preserve inheritedstage1
9600/7.2M/2764.217798s ONCE and reusedA9600/7.2M/3668.194599s/5retries
separately; newB has different headcompute/capacity. Goal70 remains ACTIVE/
unachieved. Report actual failure accurately; do not use synthetic vectors,
partial histories or intended commands as completed experimental evidence.
