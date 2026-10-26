# Sol Only: Authenticate The Actual Saved-Results Evidence Pack

Required model: gpt-6.1-sol, reasoning_effort=max. This is evidence verification,
not literature research, a new direction proposal, model execution, or training.
Do not use Astra as a results/code reviewer. Do not modify original artifacts.

Read source_snapshot.json, actual final reports/CSV/PDFs and native lock
receipts. Authenticate R109 full delivery/publication or linked CPU-only
report recovery, R112 recovered reports + saved analysis/publication, R113
new two-3090 queue and private adapter completion/publication, R114 terminal
pilot analysis and complete OR explicit partial-negative closeout/publication.
Distinguish scientific/native-contract limitations from implementation faults:
the first CVRP/ICAM cold attempt in queue_v1 failed because CUDA input tensors
were passed directly to NumPy at the common native audit boundary. This is an
interface error, not evidence against candidate capability or response learning.
An additive host-view correction is being prepared under
R114_paired_responses/preparation/tensor_boundary_recovery_20261026; preserve
the original failure and distinguish preparation from actual corrected labels.
Verify original failed receipts remain failed and hash-identical. Verify the
actual published commit includes the claimed artifact bytes; don't accept
directory existence or optimistic LLM messages as evidence. No gigantic
reaudit of raw instances/weights/solvers, no validation forwards and no GPU.

Save in evidence_dir:

1. evidence_pack.md: concise Chinese conclusions with source paths and actual
   negative/partial facts, original vs implemented hypothesis, matched controls,
   all18 and task-wise results, selection vs native true-winner frequency,
   zero-selection/Top1 plateau evidence and limits. Use actual reports, including
   historical R57 known limits and running-condition metadata, without restarting
   a gigantic audit. Do not invent missing measurements. Include NSS comparisons
   for TSP TEST/CVRP TEST/TSPLIB/CVRPLIB when available, mark missing explicitly.
2. checkpoints.csv: exactly seven rows, experiment,arm,checkpoint_sha256,
   state_hash,selected_update,selection_split,inference_rule. selection_split=VAL.
   One whole-model checkpoint per arm, not per-dataset/per-task best stitching.
3. metrics.csv: columns experiment,arm,split,problem,n,checkpoint_sha256,
   top1_pct,mean_cost,sbs_cost,oracle_cost,vs_sbs_pct,vs_oracle_pct,
   actual_regret_pct,data_version,candidate_pool_sha256,winner_rule,
   sbs_fit_split,all_aggregation. For EACH of seven models: VAL and TEST all18
   plus ALL (19 rows/split), TSPLIB/TSP and CVRPLIB/CVRP (2 rows). Exactly 280
   rows. Counts per problem VAL/TEST=4500; ALL=81000. ALL fields are per-problem
   equal means, all_aggregation=equal_problem, never mixed raw costs. SBS fitted
   on TRAIN, sbs_fit_split=TRAIN. Percentages are percentages, ratios are not
   substituted for per-instance actual regret. data_version is exactly
   selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018 for VAL/TEST;
   native_TSPLIB or native_CVRPLIB for LIB. candidate_pool_sha256 = SHA256 of
   json.dumps(protocol.candidate_pools,sort_keys=True,separators=(',',':'),
   allow_nan=False). winner_rule must preserve the actual native/FP64 ind
   rule; never silently replace it with FP32 argmin. No invented R114 full18
   model row. If mandatory actual evidence is missing, block instead of making
   a partial pack look complete.
4. verification.json: complete=true, model=gpt-6.1-sol, reasoning_effort=max,
   saved_results_only=true, unique_val_checkpoints=true,
   same_version_metrics=true, test_development_exposed=true,
   scientific_completion_gate_sha256=context.source_snapshot_sha256,
   verified_inputs=the EXACT source_snapshot.json object,
   outputs={evidence_pack.md:sha256,checkpoints.csv:sha256,metrics.csv:sha256},
   r114_closeout_kind=actual authoritative complete_labels or
   partial_negative_closeout; negative_partial_facts=nonempty sourced list;
   implemented_hypothesis_limits={R112:nonempty explanation,R113:nonempty
   explanation,R114:nonempty explanation}. Attest only what you authenticated.

Fidelity limits MUST be explicit: R112 uses four regions / 36 latent interface
conditions, not certified resource DP; R113 uses synthetic processes of at most
12 fragments, not a native complete program; R114 covers only 16 applicable
demand/TW tasks, 32 TRAIN bases/problem, no TSP/ATSP new responses. Approximation
failure cannot reject a whole paradigm; pilot results cannot prove universal
full18 data improvement. These are scope constraints/questions, not new Sol
scientific directions. TEST/LIB have participated in development and are not a
fresh blind confirmation. Distinguish prior-SBS risk, task-wise decision cost,
posterior-argmax-zero legitimately possible and learnable misses. Don't infer
unmeasured Bayes optimality or absence of information from zero frequency.

Name the CURRENT EXECUTION PHASE: R114 is ACQUISITION/response pilot, NOT a
trained function learner. Even complete labels cannot establish downstream
response-learning success or failure. This implementation gap must enter the
pack and its scope limits. Do not claim all three paradigms fully validated,
and do not spawn additional NN arms; finish the existing authorized jobs first.
R112 is already published at 446ac3c046b8c3dfb5722b529a4356b1ead80a61; reuse
its actual complete artifacts and receipt, no duplicate repair/publication.
