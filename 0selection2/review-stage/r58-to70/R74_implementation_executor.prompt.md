You are the SAME actual GPT-6.1-Sol/max execution agent, thread
01a13212-f99c-72a3-9279-db14f0073262. Your bounded R74 feasibility CLI ended0
07:45:50UTC; actual SAME Astra/max completed0 08:16:30UTC with one full18 plan8/
ready FOR ADOPTION ONLY. Parent now adopts it. R73 is fully closed/pushed
2a1fd0629fafd7ab3f2aefa23b9907daa2f3a3f1; no R73 rerun/edit/review/extra commit.

Read FULL code/V4/runs/R74_adaptive_incomplete_acquisition/approved_plan.md.
Implement EXACT one bounded adaptive-acquisition B with reused frozen R73B23
global63 A, VAL8959/18000. Plan is a new information-acquisition PACKAGE, not
a matched proof of adaptivity. Native70 goal remains ACTIVE/unachieved.
All original18/full legal pools/native winners retained; no R58 semantic repair,
new instances/final-solver-cost labels/completed solver execution/source embedding/
ReLD-only system/recipe grid. Only one new B; no A standalone inference.

Allowed changes: isolated R74 modules/runner/readme, R74 root records/caches and
the MINIMUM factual corrections to your acquisition_feasibility notes for the
two source errors accepted by actual primary. MATNET applicable class is
ATSPPolicy at policy.py157, not MatNetGLOPPolicy. Your cited ICAM lines151/176
consume CVRP coordinates/demands and do NOT certify ATSP; mark this unverified
rather than inventing a source. R74 ATSP uses verified MATNET only. Do not reopen
a broad source audit. No edits to historical experiment/core/native solver source,
registry/data/AGENTS/skills/user changes, no Git/reviewer/subagent/delegation.

Implementation:
1. Use isolated r74_acquisition.py, r74_model.py, r74_data.py/r74_experiment.py,
   r74_cpu_check.py, run_v4_r74.sh and R74_README.md as actually needed. Reuse
   existing model/schedule/official optimizer helpers, not whole duplicated
   frameworks. Full actual recipe/source snapshots before formal generation.
2. Frozen FP32/eval original R73B23 averaged-x COMPLETE selector F incl head and
   normalization buffers. Expose existing encode_state and reconstruct original
   utilities through same decoder/head in ONE base forward/input. No new base
   TRAIN/VAL reference pass. Cache as required new B component only. All original
   input fields, orientation/order/physicalroles/constraints/directed ATSP intact.
   Only new token adapter, two crossattention blocks, gates and correction head
   train; zero-initialized correction finalhead as approved, no extra head/loss.
3. Fixed frozen greedy FP32/eval augmentation1 probe recipes exactly in plan:
   BQ TSP actual greedy_decoding_step (not decode/fullwrapper),
   ReLD_CVRP learned min(n,100) starts/one_step_rollout,
   MATNET ATSPPolicy/all n starts/private seed2 permutation PER n broadcast/query,
   BOTH Train_ALL MTL/MOEL on actual all15 native task envs and true positive-
   demand B start subsets. No NN/tour initializer/postprocess/fullsolve call.
   Extractor is separately locked NEW protocol, not historical reproduction.
   Source-path checkpoint/runtime failure MUST stop and preserve evidence;
   not permission to substitute model/weight/default/constraint/recipe.
4. h=floor((floor(n/4)-1)/2), same two-increment state path; first initialized
   customer/node then h and h further selections. Depot consumes selections.
   Before+after every step guard zero finished/done and required-node visited
   <=1+2h<=floor(n/4). Never finish/reward-reconstruct/finalroute/finalcost.
   For tiny n outside admissible boundary stop rather than invent fallback.
5. Keep EVERY start and shallow/deep checkpoint separate. Token403 = F start,
   current and DISTINCT visited-node mean embeddings (3*128), exact12scalars,
   five probe IDs and two checkpoint IDs. Distance/time fields normalized by
   input n*mean_offdiagdistance only; unavailable fields0. Preserve directedATSP
   and real normalized loads, waiting, route lengths, entropy/legal fractions.
   No current measured winner/cost/gap/rank, complete tour or TEST data in features.
   Labels only in objective. Source IDs for alignment, never network feature.
   Do not pool away trajectory/start identity. Exact observations may be saved
   LOSSLESSLY factorized (base nodeembeddings + integer start/current/visited
   identities + FP32 scalar/visitedmean) instead of repeating massive F copies;
   reconstruct mathematically same inputs, no compressed numerical approximation
   or change in protocol. Stream/cache CPU, only current batches resident GPU.
6. MVRP root gate sees h ONLY; second sees h + mean of CURRENTLY OBSERVED
   shallow adapted tokens + two-way first-probe ID ONLY. Future other/deep
   observations inaccessible to gating, not merely detached. Retain shallow
   observations when deepening. Leaf sets MTLshallow+MTLdeep, MOELshallow+MOELdeep,
   BOTHshallow. NonMVRP one two-checkpoint leaf. Learned token128, candidate
   projection384->128, two candidate-query blocks heads4/FF256/dropout.1,
   shared scalar correction Futility+delta; all candidates compete.
7. Expected native CE weighted w=(a*b,(1-a)*c,a*(1-b)+(1-a)*(1-c)).
   TWO independent adapter dropout views perleaf. FP32 legalpool meanCE plus
   .35*min(u/720,1)*weighted symmetricKL. DETACH w forKL only, gates get CE grads.
   No sampling/RLtargets/teacher/auxcost/risk/classrebalance. Gate no dropout,
   zero finallayers/root tieMTL/secondtiedeepen. Newseed2.
8. Sole formal generation TRAIN330000trajectoryinitializations/180000distinctrows,
   VAL33000/18000rows, both checkpoints perprobe, baseF198000queryrows once.
   Fixed batch16 within task/trueN/actualstartcount groups, natural tails, exact
   original row alignment. Record actual groups/batches/forward counts and
   counterfactual acquisition overhead, not a selected-path deployment timing.
   After featuregen release all probe states/models before optimization.
   No TEST featurebank. Do not load TEST split anywhere in this stage.
9. R67seed2 global41..80 exacttask/roworder, real256/task/triple sameparameters,
   16rowtails,40epochs/240updatesperE/9600success/7.2Mnativepresentations/
   38.4Msuccessfulleaf-viewrows. Fresh official installed ScheduleFree LR3e-4,
   internalwarmup720,betas(.9,.999),eps1e-8,WD1e-4,foreachFalse,r0,
   weight_lr_power2,inner_momentum0; yTRAIN/xEVALandSAVE. ONLYnewparams optimizer.
   SAMEstate taskgradients mean3 FP32/clip1 ONCE/stepONCE. Release leaf graphs as
   appropriate without changing weightedobjective. Wholetriple AMP nonfinite
   restores RNG/reduces scale/retries SAMErows <=20; no counteradvance/skips.
   No LRsearch/controller/earlystop/capacitygrid beyond40E.
10.Full hardpolicyVAL everyepoch, FP32 batch128/tails104, cachedobservations but
   hardgatechosenpath only. All18000rows/originallegalpools/globalIDutilityties.
   Integercountbest/earliesttie wholepolicy checkpoint, officialB>8959elseA.
   Full nativeTRAINEVALevery5/final/selectedbest. Log losses/nativeCE/KL/gate
   probabilities+hardpathcounts, Top1/2/3/cost/regret, LR/grad/attempts/rows/time/
   memory. Do NOT use soft expected probabilities to selectcheckpoint.
   No TEST/freeze/deliveryanalysis now. Parent will authorize uniqueVALlock then
   ONEfull18BTEST with actual hard selected acquisitions live; ATESTreuse only.

Minimal check/launch:
11.ONE combined SYNTHETIC CPU invocation covers quarterboundary/no completed
   trajectory, label and futureobservation isolation, masks/ties, exactexpected
   loss/policygrad and wholeupdateRNGrecovery. No real checkpoint/source-data
   preflight/referenceinfer/sizepilot/GPUcapacityupdate/multiplechecks.
   Then DIRECT formal generation+training in ONE owned tmux r74_train.
12.Existing Slurmalloc1465/nodegpu03 ONLY assigned index1 UUID
   GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d. Check ONLY this assignedUUID at actual
   launch; if busy wait same device. Never inspect/use another GPU/queue/client
   or allocation. conda easynco, python_compat bridge,threads1,W&Boffline project
   selector/entityyjkds-southern-university-of-science-technology.
13.Record first ACTUAL formal batches' trueN/startcounts/selections/visitedmax/
   zerocompletions/finitefeature counters as partof formal generation, NOT a
   separatepilot. Data/interface/OOM/weight failure preserve exact traceback
   and stop unchanged, notify parent; no silentrecipe/batch fallback or restart.
   No time/GPUhourcap. Save own exact launch Slurmstep/tmux/pane/PIDs as observed.
14.Return PROMPTLY after ONECPUcheck and actual first finite formal extraction
   batch, with handoff.json, full model/trainableparamcounts, storage design,
   counters/time/memory/currentstage. Do not wait40E in LLMturn. Original formal
   job intentionally lives independently; all needed foregroundexecsessions
   settled before return. Parent monitors and can later delegate CPUconditional
   delivery while same original job lives. Do not claim first optimization
   update if only cachegeneration observed.
15.Full selecteddeployment compute includes F+all actuallyacquired prefixes/
   gates+finalutility+ONE complete restarted chosen solver. No prefixdeduction,
   fullprobesolve, retrospective completiontiming, speedup or endtoendlatency
   claim. Counterfactual featuregeneration separately counted. No new completed
   solverlabels in this screen. Existing timings may be reused only exactmatch.
16.No Git/extra reviews/nextplan by executor. Parent publishes ONE completed
   meaningful version after actual primary+fresh closure and objectiveaudit.
   Known R58/eightB discrepancies deferred/unfixed/notnative70numericveto.
   Actual original12600/18000 nativeTEST still required for goalcompletion.
