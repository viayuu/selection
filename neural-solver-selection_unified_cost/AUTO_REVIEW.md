# Auto Review Loop

Started: 2026-04-15 01:15 CST
Project root: `/public/home/zhoucl/shiys/9nss论文/neural-solver-selection_unified_cost`

## Round 1 (2026-04-15 01:15 CST)

### Assessment (Summary)
- Score: 4.0/10
- Verdict: not ready
- Key criticisms:
  - The experiment design was still confounded: the supposed `baseline` and `stats_film` runs used effectively the same model, so the comparison was not interpretable.
  - Cross-problem imbalance was not handled explicitly in training or checkpoint selection, even though the project goal is per-problem robustness.
  - The evaluation outputs were not yet strong enough for paper-style claims: selector vs best single vs oracle comparisons needed to be more explicit per problem.
  - Solver naming was inconsistent across exports (`MVMoE` vs `MVMOE`, `bq` vs `BQ`, `Omni` vs `OMNI`), which polluted the global solver space.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Score: 4/10

Verdict: Not ready.

This project has a promising direction and a sensible starting implementation, but it is not yet in a submission-ready state because the current evidence is still too fragile and partially confounded.

The most severe issue is experimental identifiability. The current "baseline" and "stats/FiLM" configurations are not actually separated at the model level, so any observed difference between them would be uninterpretable. Until the code exposes explicit toggles and reruns a clean comparison, the reported gains cannot support a scientific claim.

The second major issue is mismatch between objective and intended evaluation. The project cares about per-problem quality, yet the training pipeline still lacks explicit problem-balanced sampling and a macro-balanced checkpoint criterion. This risks optimizing for the dominant mix of problem families rather than for robust cross-problem performance.

The third issue is that the analysis, while already better than before, is still missing the clearest tables needed for decision-making and later paper writing: per-problem selector mean cost, top-1 accuracy, best single-solver mean cost, oracle mean cost, and the selector's gap to the best single baseline. These numbers must be emitted automatically.

The fourth issue is data hygiene. A unified selector is extremely sensitive to solver identity consistency, and the current export names are not canonicalized. This can silently create duplicated solver entries and undermine both learning and evaluation.

Minimum fixes:
1. Canonicalize solver names globally and rerun.
2. Add explicit model toggles for instance-stats, FiLM conditioning, and solver interaction so that baseline vs improved comparisons are scientifically valid.
3. Add problem-balanced sampling plus a macro-normalized checkpoint metric.
4. Expand per-problem analysis outputs to include selector vs best-single vs oracle.
5. Run a fast but real medium-scale comparison first, then promote the winning setting to the full export dataset.

The work is not ready now, but it is close enough that a disciplined review-and-rerun loop could materially improve it within a few rounds.

</details>

### Actions Taken
- Canonicalized solver names to avoid duplicated solver identities across exports.
- Wired the existing instance-statistics branch end-to-end and cleaned up the model interface.
- Added explicit model toggles:
  - `use_instance_stats`
  - `use_film`
  - `use_solver_interaction`
- Added problem-balanced sampling support.
- Added macro-balanced validation metrics:
  - `macro_problem_top1_accuracy`
  - `macro_problem_oracle_ratio`
  - `macro_problem_gap_vs_best_single`
- Standardized future config files on `val_macro_problem_oracle_ratio` as the checkpoint-selection metric for fairer cross-problem comparison.
- Added richer per-problem analysis outputs and explicit baseline comparison CSVs.
- Stopped the redundant `round1_medium_baseline_real` and `round1_medium_stats_film_real` pair after confirming that their epoch-0 metrics were identical, hence not a valid ablation.
- Launched a clean replacement pair on the medium export:
  - `arl_r2_med_base` -> `config_round2_medium_true_baseline.yml`
  - `arl_r2_med_balanced` -> `config_round2_medium_balanced_stats.yml`
- Queued the next ablation pair so GPU utilization stays high after the current pair ends:
  - `arl_r2_med_balanced_base` -> `config_round2_medium_balanced_baseline.yml`
  - `arl_r2_med_stats_nobalance` -> `config_round2_medium_stats_no_balance.yml`

### Results
- Redundant medium pair produced identical epoch-0 validation metrics, confirming the comparison was confounded:
  - `val_acc = 0.695714`
  - `val_top_1_cost = 12.996359`
  - `val_macro_problem_oracle_ratio = 1.006334`
- Clean replacement pair has produced the first meaningful comparison at epoch 0:
  - `round2_medium_true_baseline`: `val_acc = 0.692857`, `val_top_1_cost = 12.995544`, `val_macro_problem_oracle_ratio = 1.006220`
  - `round2_medium_balanced_stats`: `val_acc = 0.683810`, `val_top_1_cost = 13.001271`, `val_macro_problem_oracle_ratio = 1.007161`
- Interim finding: at the first comparable checkpoint, the plain true baseline is ahead of the heavier `balanced+stats` variant, so the next queued ablations split these factors apart instead of assuming the bundle is helpful.
- Updated comparison after epoch 1 reverses that first impression:
  - `round2_medium_true_baseline`: `val_acc = 0.693095`, `val_top_1_cost = 12.998212`, `val_macro_problem_oracle_ratio = 1.006580`
  - `round2_medium_balanced_stats`: `val_acc = 0.700714`, `val_top_1_cost = 12.995359`, `val_macro_problem_oracle_ratio = 1.006301`
- New interim finding: the heavier variant appears to learn more slowly but may generalize better after the first epoch, so it remains a live candidate rather than an immediate reject.

### Status
- Continuing after `arl_r2_med_base` and `arl_r2_med_balanced` finish.

## Round 2 (2026-04-15 01:35 CST)

### Assessment (Summary)
- Score: 5.0/10
- Verdict: not ready
- Key criticisms:
  - The current signal is promising but still small-margin and not yet stable across epochs.
  - The bundled `balanced+stats` improvement still needs factorized ablations to separate balancing from richer conditioning.
  - The evidence is still validation-only; the same direction needs to appear on `test` and ideally benchmark / LIB.
  - The project still needs per-problem breakdowns in the actual review narrative, not just as generated files.
  - Checkpoint selection fairness must remain identical across variants.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

1. Score: `5/10`

2. Verdict: `not ready`

3. Remaining critical weaknesses:
- The evidence is still too early and too small-margin. The gains at `epoch 1` are real-looking but modest, and they reversed the `epoch 0` ordering. That means the current signal is promising, not stable.
- The improvement is still bundled. `balanced+stats` beats the plain baseline at `epoch 1`, but you still cannot say whether the gain comes from balancing, richer conditioning, or their interaction.
- You only showed validation. For a selector paper, the claim is not credible until the same direction appears on `test` and preferably benchmark / LIB data.
- The claim is “cross-problem unified selection helps,” but you have not yet shown where it helps and where it hurts. Per-problem tables exist now, but the review evidence still needs those breakdowns.
- Checkpoint fairness must be identical across variants. If different runs select checkpoints with different criteria, small gains become hard to trust.

4. Minimum next fixes / experiments:
- Let the current clean pair finish and compare `best-val` under the same selection rule, then report `test` and benchmark with per-problem tables.
- Run the queued factorized ablations exactly as planned:
  - `balance only`
  - `stats/FiLM/interaction only`
- If one factorized variant explains most of the gain, promote only that winner against the plain baseline on the full `export_v1` data.
- If the final medium-scale gain remains this small, add one more seed for just the top two variants. Not for everything.

5. Should you keep the current ablation plan?
- `Yes`, this is the right next step.
- One amendment: enforce the same checkpoint-selection metric for all variants and make the comparison on `best val -> test/benchmark`, not just early-epoch snapshots.

On the main claim: the current evidence is `starting` to support it, but only weakly. Right now the strongest honest statement is:

- Richer conditioning plus balancing is a plausible improvement for unified cross-problem selection.
- It is not yet established.
- The queued factorized ablations are exactly what you need to turn this from “suggestive” into “convincing.”

</details>

### Actions Taken
- Kept the current medium pair running because it already provides a useful learning-curve comparison.
- Patched `trainer.py` so future runs save every epoch checkpoint plus `best_checkpoint_info.json`, which prevents checkpoint-selection ambiguity from forcing full reruns.
- Upgraded loader settings for queued runs:
  - `persistent_workers`
  - `prefetch_factor`
  - `zero_grad(set_to_none=True)`
- Standardized future experiment configs on:
  - `selection_metric: val_macro_problem_oracle_ratio`
  - `save_every_epoch: true`
- Prepared full-data configs for the factorized winners:
  - `config_round3_full_balanced_baseline.yml`
  - `config_round3_full_stats_no_balance.yml`
- Ran the factorized follow-ups and diagnosed the actual systems bottleneck:
  - `num_workers > 0` can trigger container memory OOM on the medium export when two runs are active in parallel.
- Added true resume support in `trainer.py`:
  - resume starts from `checkpoint_epoch_best.pt`'s next epoch
  - carries forward the loaded best metric
  - writes the loaded best checkpoint into the new run directory
- Restarted both factorized runs from their saved `epoch 0` checkpoints with `num_workers: 0`.

### Results
- Current medium pair remains in progress.
- Best comparable snapshots seen so far:
  - `round2_medium_true_baseline`, epoch 1:
    - `val_acc = 0.693095`
    - `val_top_1_cost = 12.998212`
    - `val_macro_problem_top1_accuracy = 0.764278`
    - `val_macro_problem_oracle_ratio = 1.006580`
  - `round2_medium_balanced_stats`, epoch 1:
    - `val_acc = 0.700714`
    - `val_top_1_cost = 12.995359`
    - `val_macro_problem_top1_accuracy = 0.770333`
    - `val_macro_problem_oracle_ratio = 1.006301`
- Interpretation:
  - The heavier variant is no longer obviously worse.
  - It now looks slightly better on both accuracy and macro cost ratio.
  - The margin is still small, so the factorized ablations remain necessary.
- Medium pair final best-by-macro comparison:
  - `round2_medium_true_baseline`, epoch 9:
    - `val_acc = 0.725000`
    - `val_top_1_cost = 12.985547`
    - `val_macro_problem_top1_accuracy = 0.795500`
    - `val_macro_problem_oracle_ratio = 1.004276`
    - `val_macro_problem_gap_vs_best_single = -0.006960`
  - `round2_medium_balanced_stats`, epoch 8:
    - `val_acc = 0.728333`
    - `val_top_1_cost = 12.985430`
    - `val_macro_problem_top1_accuracy = 0.798722`
    - `val_macro_problem_oracle_ratio = 1.004246`
    - `val_macro_problem_gap_vs_best_single = -0.007540`
- Factorized epoch-0 evidence:
  - `round2_medium_balanced_baseline`, epoch 0:
    - `val_acc = 0.702143`
    - `val_top_1_cost = 12.996333`
    - `val_macro_problem_top1_accuracy = 0.771000`
    - `val_macro_problem_oracle_ratio = 1.006330`
  - `round2_medium_stats_no_balance`, epoch 0:
    - `val_acc = 0.695714`
    - `val_top_1_cost = 12.996359`
    - `val_macro_problem_top1_accuracy = 0.767333`
    - `val_macro_problem_oracle_ratio = 1.006334`
- Interim interpretation from the factorized pair:
  - `balance only` already matches or slightly beats `stats-only` at the first checkpoint.
  - The early signal suggests that balancing may explain more of the bundled gain than richer conditioning alone, but the resumed runs are still in progress.
- Resumed factorized epoch-1 evidence:
  - `round2_medium_balanced_baseline`, epoch 1:
    - `val_acc = 0.703810`
    - `val_top_1_cost = 12.991730`
    - `val_macro_problem_top1_accuracy = 0.772278`
    - `val_macro_problem_oracle_ratio = 1.005710`
    - `val_macro_problem_gap_vs_best_single = 0.000074`
  - `round2_medium_stats_no_balance`, epoch 1:
    - `val_acc = 0.667143`
    - `val_top_1_cost = 13.007476`
    - `val_macro_problem_top1_accuracy = 0.742667`
    - `val_macro_problem_oracle_ratio = 1.007276`
    - `val_macro_problem_gap_vs_best_single = 0.014152`
- Updated factorized interpretation:
  - `balance only` is clearly ahead of `stats-only` after the first resumed epoch.
  - The current best explanation of the bundled gain is that problem-balanced training is doing most of the work.
  - `stats/FiLM/solver-interaction` may still add value when combined with balancing, but they do not appear to help on their own.
- Systems finding:
  - `num_workers=8` and even `num_workers=2` were unstable under the container memory limit for two parallel medium runs.
  - `num_workers=0` is the robust setting for resumed factorized runs.

### Status
- Continuing with resumed factorized runs:
  - `arl_r2_med_balanced_base_resume`
  - `arl_r2_med_stats_nobalance_resume`

## Round 3 (2026-04-15 02:30 CST)

### Assessment (Summary)
- Score: 5.5/10
- Verdict: not ready
- Key criticisms:
  - The bundled gain over the true baseline is still real but very small.
  - The evidence now supports a narrower story: balancing looks useful, while richer conditioning alone is not convincingly helpful.
  - The paper case still requires completed factorized runs and fair `best-val -> test -> benchmark` comparison.
  - The simplest winning story should be preferred; a more complex bundled method needs a clearly repeatable edge.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

1. Score: `5.5/10`

2. Verdict: `not ready`

3. Remaining critical weaknesses:
- The gain is still tiny. Your best bundled win over the true baseline is real, but the margin is very small, so the claim is still fragile.
- The evidence now points to a narrower story than your original method story. Right now, `balancing` looks like the main useful ingredient, while `stats/FiLM/interaction` alone looks harmful.
- You still have not closed the loop on `test` and benchmark. Validation-only improvement, especially at this margin, is not enough.
- The factorized runs are not finished yet, so the “balance is the real cause” conclusion is strong but still technically provisional.
- For a paper claim, you need the simplest winning story. At the moment the bundled model may be slightly best on val, but it is not yet justified as the main method if almost all benefit comes from balance.

4. Minimum next fixes only:
- Finish the resumed `balance only` and `stats-only` runs.
- Compare all variants by the same rule: `best val_macro_problem_oracle_ratio -> test -> benchmark`.
- Promote only the top two candidates to full-scale runs:
  - `true baseline`
  - `balance only`
- Keep `balanced+stats` as a third candidate only if it still beats `balance only` on `test` or benchmark, not just on val.
- If `balance only` and `balanced+stats` are effectively tied, choose `balance only` as the main method because it is simpler and better identified.

5. Promotion target:
- `Provisional answer: yes, the next promotion target should be balance only.`
- Reason: the current evidence says the main positive effect is very likely from problem-balanced training, while richer conditioning without balancing is currently negative.
- The only reason not to commit fully yet is that `balance only` has not finished. So the clean decision rule is:
  - If `balance only` finishes within striking distance of the bundled model on `test/benchmark`, promote `balance only`.
  - Only promote `balanced+stats` if it shows a clear, repeatable edge after the factorized runs complete.

Most honest claim right now:
- You are starting to support `problem balancing helps unified cross-problem solver selection`.
- You are not yet supporting `richer conditioning helps` in a convincing way.
- The current evidence actually leans against that stronger claim unless balancing is present.

</details>

### Actions Taken
- Added explicit round-3 evaluation focus:
  - finish factorized runs
  - compare by `best val_macro_problem_oracle_ratio -> test -> benchmark`
  - prefer the simplest winning story
- Collected the completed medium-pair `test` and `benchmark` summaries for:
  - `round2_medium_true_baseline`
  - `round2_medium_balanced_stats`
- Confirmed that the bundled model still has a small edge on `test`, but both models remain weak on benchmark / LIB CVRP.
- Finished the fair medium comparison and launched the two winning full-scale candidates.
- Diagnosed another systems constraint:
  - two simultaneous full-data runs exceed the container memory limit even with `num_workers=0`.
- Switched the full-data plan from parallel to queued sequential execution:
  - run `balanced+stats` now
  - queue `true baseline` to start immediately after it finishes

### Results
- Completed medium-pair `test` comparison:
  - `round2_medium_true_baseline`
    - `selector_mean_cost = 13.037725`
    - `selector_top1_accuracy = 0.720476`
    - `macro_selector_top1_accuracy = 0.792722`
    - `macro_selector_oracle_ratio = 1.004429`
    - `macro_selector_gap_vs_best_single = -0.008641`
  - `round2_medium_balanced_stats`
    - `selector_mean_cost = 13.036776`
    - `selector_top1_accuracy = 0.730476`
    - `macro_selector_top1_accuracy = 0.801056`
    - `macro_selector_oracle_ratio = 1.004447`
    - `macro_selector_gap_vs_best_single = -0.009700`
- Completed medium-pair `benchmark` comparison:
  - Both runs are still worse than the best single method on CVRP LIB.
  - `round2_medium_true_baseline`: `macro_selector_gap_vs_best_single = 0.142024`
  - `round2_medium_balanced_stats`: `macro_selector_gap_vs_best_single = 0.124085`
- Factorized runs are still in progress, but current best validation snapshots are:
  - `round2_medium_balanced_baseline`: epoch 9, `val_macro_problem_oracle_ratio = 1.005076`
  - `round2_medium_stats_no_balance`: epoch 8, `val_macro_problem_oracle_ratio = 1.004574`
- Updated interpretation:
  - The earlier “balance explains almost everything” story was too strong.
  - `stats-only` eventually recovers and becomes much stronger than its early epochs suggested.
  - Finished factorized runs show that `balance only` is not the right promotion target.
  - The actual top-two candidates after the full medium comparison are still:
    - `true baseline`
    - `balanced+stats`
  - The bundled edge exists, but it remains very small.
- Fair medium factorized final results:
  - `round2_medium_balanced_baseline`
    - `val_macro_problem_oracle_ratio = 1.005076`
    - `test_macro_problem_oracle_ratio = 1.005105`
    - `benchmark_macro_problem_oracle_ratio = 1.011345`
  - `round2_medium_stats_no_balance`
    - `val_macro_problem_oracle_ratio = 1.004574`
    - `test_macro_problem_oracle_ratio = 1.005182`
    - `benchmark_macro_problem_oracle_ratio = 1.014899`
- Promotion decision from finished medium evidence:
  - `balance only` is worse than both promoted candidates.
  - `stats-only` is competitive on val/test but clearly worse on benchmark.
  - The top two overall remain `true baseline` and `balanced+stats`.
- Full-data execution state:
  - `arl_r3_full_bundle` finished.
  - `arl_queue_r3_full_base` successfully launched the full true-baseline retry after the bundle run exited.
- Full-data final comparison:
  - `round2_full_true_baseline-0415-040256`
    - `val_macro_problem_oracle_ratio = 1.003591`
    - `test_macro_problem_oracle_ratio = 1.003554`
    - `benchmark_macro_problem_oracle_ratio = 1.017672`
  - `round2_full_balanced_stats-0415-024024`
    - `val_macro_problem_oracle_ratio = 1.003738`
    - `test_macro_problem_oracle_ratio = 1.003723`
    - `benchmark_macro_problem_oracle_ratio = 1.017560`
- Updated full-scale interpretation:
  - The medium bundled edge did not survive clearly at full scale.
  - The simpler true baseline is now better on full `val` and `test`.
  - The bundled model has only a tiny benchmark edge, and both models remain clearly worse than the best single method on benchmark / LIB CVRP.
  - The current best paper story is shifting toward the simpler unified selector rather than the richer-conditioned bundle.

### Status
- Fair medium comparison is complete.
- Promoted top two candidates to full-data runs.
- Due to full-data memory limits, they are now scheduled sequentially:
  - completed: `arl_r3_full_bundle`
  - completed: `arl_queue_r3_full_base`
- Waiting for updated reviewer assessment on the finished full comparison.

## Round 4 (2026-04-15 04:55 CST)

### Assessment (Summary)
- Score: 6.0/10
- Verdict: almost
- Key criticisms:
  - The richer `balanced+stats` story is not supported by the final full-scale evidence.
  - The main remaining scientific weakness is benchmark / OOD performance on `CVRP/LIB`.
  - The result margins are coherent but small, so the paper must avoid aggressive claims.
  - The correct final story is the simpler unified selector, not the richer-conditioned bundle.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

1. Score: `6/10`

2. Verdict: `almost`

3. Remaining critical weaknesses:
- The main method story has changed. Full-scale evidence does not support the richer `balanced+stats` model as the real winner; the simpler baseline is better on full `val/test`, with only a tiny benchmark edge for the bundled model.
- The benchmark/OOD story is still weak. Both models are clearly worse than the best single method on benchmark `CVRP/LIB`, and that is now the biggest scientific hole.
- The margins are small. You have a coherent ordering, but not a decisive one, so the paper must avoid overclaiming.
- The strongest claim is no longer “richer conditioning helps.” If you keep that claim, the paper will read as contradicted by its own final results.

4. Minimum next fixes only:
- Freeze the main method as the `true baseline`.
- Recast `balanced+stats` as an ablation/negative result, not the headline method.
- Add paired significance or bootstrap confidence intervals on full `test` and `benchmark` for:
  - `true baseline` vs `balanced+stats`
  - `true baseline` vs `best single`
- Add one focused benchmark shift analysis for `CVRP/LIB`:
  - where the cost gap comes from
  - whether it is concentrated in a subset of instances/scales
- Do not start another broad model sweep before that analysis.

5. Paper story:
- `Yes.` The correct story is now the simpler one.
- The main paper should be about a shared unified selector with:
  - unified input representation
  - global solver pool + feasible mask
  - masked ranking objective
  - one shared NSS-style encoder across problems
- The richer-conditioned bundled method should stay in the paper as an informative ablation:
  - it looked promising at medium scale
  - it did not clearly survive at full scale
- “Balanced training analysis” is worth keeping, but as analysis/ablation, not as the core method contribution.

Most honest claim now:
- A simple unified supervised selector works surprisingly well across 18 routing problems and beats best-single baselines on many standard test splits.
- Extra conditioning and balancing do not yield a robust full-scale win.
- Benchmark `CVRP/LIB` generalization remains the main open failure mode.

</details>

### Actions Taken
- Finished the full-scale comparison for the two promoted candidates.
- Consolidated the evidence into a run-level summary table:
  - `train_logs/round3_full_summary.csv`
- Froze the method conclusion:
  - main method = simple unified selector / `true baseline`
  - `balanced+stats` = ablation / informative negative result
- Stopped further broad architecture sweeps.
- Switched the next recommended work from model search to statistical testing and benchmark shift analysis.

### Results
- Full-data final comparison:
  - `round2_full_true_baseline-0415-040256`
    - `val_macro_problem_oracle_ratio = 1.003591`
    - `test_macro_problem_oracle_ratio = 1.003554`
    - `benchmark_macro_problem_oracle_ratio = 1.017672`
    - `test_macro_problem_gap_vs_best_single = -0.018162`
  - `round2_full_balanced_stats-0415-024024`
    - `val_macro_problem_oracle_ratio = 1.003738`
    - `test_macro_problem_oracle_ratio = 1.003723`
    - `benchmark_macro_problem_oracle_ratio = 1.017560`
    - `test_macro_problem_gap_vs_best_single = -0.015164`
- Final interpretation:
  - The simple unified selector is the better main method because it wins on full `val/test`.
  - The bundled model's only advantage is a very small benchmark edge, which is not enough to justify making it the headline method.
  - Both methods still fail on benchmark `CVRP/LIB` against the best single solver.

### Status
- Stopping the autonomous model-sweep loop at a positive threshold (`6/10`, `almost`).
- Recommended next work is narrow and analysis-heavy rather than another broad training sweep.

## Method Description
The final method is a simple unified supervised selector trained across 18 routing problems with one shared NSS-style hierarchical graph encoder. Each instance is converted into a unified representation, paired with an explicit problem identity / feature descriptor, and scored against a global solver pool under a feasible mask so that unsupported solvers are excluded per instance.

Training uses a masked ranking objective rather than separate per-problem classifier heads. The key result of the review loop is that this simple shared selector is the most defensible final method: it generalizes well across standard validation and test splits, while richer conditioning, FiLM-style modulation, and bundled balancing tricks do not produce a robust full-scale win. The main remaining weakness is benchmark `CVRP/LIB` shift robustness.

## Final Summary
The review loop converged to a simpler and more honest story than the one we started with. A unified supervised selector with shared encoding, global solver pooling, feasible masking, and ranking loss is already strong across the multi-problem setting. The more elaborate `balanced+stats` variant looked promising at medium scale, but did not survive clearly at full scale.

The highest-value next steps are no longer broad model changes. They are:
- paired significance or bootstrap confidence intervals on the full `test` and `benchmark` comparisons
- a focused `CVRP/LIB` shift analysis explaining why the selector still loses to the best single solver there

## Round 5 (2026-04-15 10:00 CST)

### Assessment (Summary)
- Score: 7.0/10
- Verdict: almost
- Key criticisms:
  - No major experimental blocker remains.
  - The only real remaining risk is paper framing discipline: the simple unified selector must be the main method, and the benchmark limitation must be stated plainly.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

1. Score: `7/10`

2. Verdict: `almost`

3. Truly critical blocker left:
- No major experimental blocker is left.
- The only real blocker now is framing discipline: the paper must center the simple unified selector, keep `balanced+stats` as ablation/negative result, and state the `CVRP/LIB` benchmark weakness plainly without overclaiming.

4. Enough to freeze without further sweep?
- `Yes.`
- The package now looks experimentally closed: standard-test wins are supported, the bundled variant has been fairly ruled out as the main method, and the benchmark limitation is quantified and visualized.
- I would stop sweeping models and move to paper packaging, unless you want to pivot the whole paper toward benchmark/OOD robustness.

</details>

### Actions Taken
- Implemented reviewer-requested paired bootstrap analyses:
  - full `test`: simple baseline vs bundled, simple baseline vs best single, bundled vs best single
  - full `benchmark`: same comparisons
- Implemented focused `CVRP/LIB` shift analysis:
  - per-instance paired rows
  - per-solver gap summaries
  - worst-case instance table
  - coarse size-bin gap summary
- Added paper-ready packaging artifacts:
  - `train_logs/analysis_followup/paper_ready_summary.csv`
  - `train_logs/analysis_followup/benchmark_cvrp_size_gap.png`
  - `train_logs/analysis_followup/README.md`
- Added a persistent background monitor:
  - `live_monitor.sh`
  - tmux session `arl_live_monitor`
  - log file `monitor_logs/live_monitor.log`

### Results
- Bootstrap summary highlights:
  - Full `test`, simple baseline vs bundled:
    - mean diff `-0.002998`
    - 95% CI `[-0.004537, -0.001494]`
    - simple baseline significantly better
  - Full `test`, simple baseline vs best single:
    - mean diff `-0.018162`
    - 95% CI `[-0.020755, -0.015696]`
    - simple baseline significantly better
  - Full `benchmark`, simple baseline vs bundled:
    - mean diff `+0.008016`
    - 95% CI `[-0.514634, +0.507342]`
    - no reliable difference
  - Full `benchmark`, simple baseline vs best single:
    - mean diff `+0.850015`
    - 95% CI `[+0.367633, +1.482379]`
    - simple baseline significantly worse
- `CVRP/LIB` shift findings:
  - The benchmark weakness is concentrated in `CVRP/LIB`, not `TSP/LIB`.
  - For the main method, the mean gap vs best single grows with size:
    - `<300`: `+0.700`
    - `300-599`: `+1.213`
    - `600-899`: `+2.311`
    - `>=900`: `+2.648`
  - The worst failures often come from choosing `MVMOE` on large `CVRPLIB` instances.

### Status
- The experimental package is now considered frozen.
- Further broad model sweeps are no longer recommended.
- The remaining work is paper packaging and claim framing.
