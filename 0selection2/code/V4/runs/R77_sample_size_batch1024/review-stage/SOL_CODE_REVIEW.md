# R77 Solmax Critical-Path Code Review

- Reviewer role: Solmax; **provisional / same-family**.
- This review was performed in the current Codex session. No independently routed Solmax backend was exposed or invoked; this is not an independent cross-family certification.
- Date: 2026-10-15. Code was reviewed only after the user confirmed both workers ready/running.
- Scope: `r77_data.py`, `r77_experiment.py`, `run_v4_r77.sh`, and the directly called R76 update / per-instance loss helpers.
- Decision: **No reproducible P0/P1 error found in the requested paths. No training stop requested.** No worker code, checkpoint, training configuration or existing diagnostic output was modified.

## Findings First

No actionable correctness regression was found in the inspected current R77 paths. This is a limited code review, not an experiment-integrity certificate or validation of final results. No full dataset / cost recomputation, GPU run, profiling, training replay or full checkpoint-resume experiment was performed.

## Core Checks

| Path | Evidence and conclusion |
| --- | --- |
| Effective batch / micro gradients | `r77_experiment.py:315` forms exact1024 row requests; `:323` checks rank/global counts; `:329` calls the shared updater with world_size1. `r76_experiment.py:245` scales each micro mean by `count/1024`, including the final320 rows with cap352. Gradients are zeroed once, unscaled/clipped once, then stepped once. `pairwise_objective.py:58`, `:65`, `:67` and `r42_experiment.py:64` reduce comparisons, CE, risk and KL per instance, then take the batch mean; this is not a variable-valid-pair global reduction mistakenly weighted by row count. |
| Stratified subset | `r77_data.py:37` samples only scale, optional distribution and capacity_type, with largest-remainder allocation per input-only level and uniform leaf sampling without replacement. `:63` requires exact distinct size; `:193` uses task_seed(2, problem) from all70114 eligible TRAIN inputs. A is10000/problem; B is the full70114/problem, not a second10000-row arm. Cost/winner access occurs after input-only selection and is used for SBS reporting, not sampling. |
| Subset geometry | `r77_data.py:140` gathers only A indices; `:145` uses true node lengths to exclude padding; sums/squares are FP64. ATSP/matrix inputs are excluded. A fits its own subset TRAIN nodes; B reuses full TRAIN stats. `r77_experiment.py:108` explicitly installs each arm's geometry buffers. No VAL/TEST geometry fitting occurs. |
| Common update axis / LR | `r77_experiment.py:35` fixes eval_every1242, max49680, warmup3726, peak1e-4, end1e-6. `:51` constructs the same local-seeded task schedule for both arms. `:325` uses successful_update+1 for LR, independent of VAL, and increments only after a successful logical step. Final checks enforce2760 successful updates/problem. |
| Identical initialization | `r77_data.py:229` persists only trainable parameters from seed2 scratch initialization; `r77_experiment.py:99` copies and hashes them for both arms. Different TRAIN-fitted normalization buffers are intentional, so this is identical trainable initialization, not identical full state_dict or identical initial predictions. No R76 trained weights are loaded. |
| Own checkpoint resume | `r77_experiment.py:256` rejects changed protocol; `:265` compares checkpoint args to that immutable protocol. `:281` restores AdamW state, `:286` scaler, `:288` successful update / per-task counts, `:289` stream draws, and `:293` Python/NumPy/Torch/CUDA RNG. Saved stream cursors must equal successful counts*1024 at an evaluation boundary. The schedule is deterministically regenerated from seed2; LR resumes from the saved successful update. Prefetched future cursors are not checkpointed mid-interval. |
| VAL best / TEST lock | `r77_experiment.py:357` evaluates full VAL and uses strict minimum macro_vs_sbs_pct against the same full-TRAIN SBS for both arms. Own-subset SBS is reporting-only. `:370` saves the complete best state; `:402` locks it only after the fixed training budget. `:442` checks completion, arm, data root, checkpoint path/hash and the one-shot claim before opening TEST; `:465` claims exclusively; `:470` rechecks the hash after evaluation. No TEST-dependent LR, stopping, checkpoint reselection or per-problem assembly was found. |
| Wrapper | `run_v4_r77.sh` preserves env activation/cwd, explicit A/B ownership, existing cache usage, and --resume opt-in. `bash -n` passed. Actual Slurm placement/transition orchestration was outside this review; no scheduler/GPU changes were made. |

Read-only prepared receipts show both arms have the same trainable hash `dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93` and task-order hash `2ecb2e834e8f1a0a5ceb01e82ac975d261657cd2061a42fb6a9eba2d612bb183`. Recorded geometry node counts are A16,577,803 and B116,237,943; these receipts were read, not independently recomputed on the full cache.

## Small CPU Witnesses

Six selected existing `test_r77.R77Tests` passed: input-only hierarchical subset, tail stream/resume, fixed LR, identical parameters/separate geometry, accumulation mean/single clip, and common SBS/prelock TEST refusal. Together with four NSS synthetic tests, the combined10 tests completed successfully in1.585s.

Additional bounded witnesses passed:

- One scalar synthetic batch with1024 rows: cap352 =>352+352+320 gives the same mean-gradient update as a single1024 micro; exactly one optimizer step, no retry.
- A tiny temporary geometry cache: only the selected three valid nodes contribute; huge padded values and unselected rows do not affect mean/std. Matrix/ATSP is skipped.

These tests do not establish bit-exact equality of stochastic GPU R-Drop forwards under different microbatch layouts. The review establishes the gradient **scale**, retry/state structure and intended invariants, not identical dropout masks across differently split GPU forwards or measured optimizer-resume parity on real GPUs.

## Snapshot

The following source hashes were unchanged across the review. Ordinary improvements must not be used to interrupt the running experiment; any later worker revision is outside this snapshot.

```text
eecdc66e18d42f9b77fb571e323d01baa5859c646050c999804c67ffc4592d7d  r77_data.py
0c6061c2cf04f51c4a52f72ed1da969bf4329768f7073d2ea97cb12bed58f68f  r77_experiment.py
67a515bb2b3840990b6a018a6c92ad990ab63203af7eb4d5507747f8c621cfbb  run_v4_r77.sh
```

No Git commit. The existing R76 TEST report remains a label diagnostic, never a basis for R77 tuning or review acceptance.
