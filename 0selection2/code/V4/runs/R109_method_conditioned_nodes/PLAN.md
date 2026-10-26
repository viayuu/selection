# R109: Method-Conditioned Node Computation

## Research Question

Can candidate-specific graph computation, before pooling, learn useful solver
advantages that a tied graph transformation does not? This follows the earlier
Astra representation-modularity research, not a claim that gradient interference
has already been diagnosed. R105's input-gated latent experts did not establish
the value of a separate evolving node state for every legal solver.

R106-R108 have completed. Their locked full18 results do not show a breakthrough;
we do not continue their donor fine-tuning, connected-path or density-ratio
configurations, nor start a coefficient/temperature/rare-class quota search.

## One Mechanism, Two Sharing Rules

Both arms retain the original R88B/R77 four shared instance layers, geometry,
directed distance bias, actual TSP physical-node0 role, solver ID plus34 features,
two synchronous joint layers, shared score MLP, utility differences and maximin.

For every legal method m, start a separate node state from
`LayerNorm(H4 + Linear(s_m))`, then apply two graph attention/FFN layers with
the original input-derived distance bias. Every candidate gets its own evolving
nodes, not a shared vector broadcast after pooling. Its masked mean/max summary
is added to its solver token and directly occupies the graph-summary input of
the final score MLP. The common decoder context remains unchanged.

- A_tied_late: all candidate branches use the same two graph-layer weights.
- B_private_late: a separate two-layer bank for each of21 active global solver
  identities. Deleted RELD_MTL has no private bank. Candidate column positions,
  costs, winners or routes never select a bank.

Both use the same original fresh seed2 tensors. All new graph banks start as
copies of the same original fresh layers2/3; the new pooling map copies the
original fresh pool. CPU evaluation verifies identical initial functions.
Total parameters differ: A2,305,033; B10,220,553; original1,842,953.
This is NOT a parameter-matched or FLOP-matched causal isolation of sharing.
More parameters, candidate computation and inference cost must be reported.
Non-reentrant activation checkpointing preserves dropout RNG during replay.

## Frozen Data And Training

Use the frozen NEW90k scenario,18 problems, each TRAIN81,000/VAL4,500/TEST4,500.
Keep the single original MOEL deployment named reld_mt and frozen B exclusions.
No solver reruns, new labels, route targets, current-query outcomes in forward,
historical-TRAIN label context, new dataset split, or R58 operations.

Seed2, natural data/task schedule, logical batch1024. Two dropout forwards and
the original `.35 CE + .35 comparison + .02 relative risk + ramped RDrop`.
AdamW peak1e-4/end1e-6, weight decay1e-4, norm clip1, dropout.1.
57,600 successful updates, warmup4,320, full VAL every1,440, TRAIN every7,200.
Same fixed-update LR schedule and sample order for A/B; no automatic retries of
terminal failures or microcap changes during a formal run.

Main's input-only two-view B128/1024 witness passes the largest TRAIN inputs:
TSP n499 peak allocated16,726MiB, CVRP n50019,117MiB, ATSP n10011,339MiB,
VRPBLTW n10110,165MiB. Common microcaps: TSP/CVRP128, others1024, eval128.
Logicalbatch remains1024. Witness has no optimizer steps or outcome-based cap
selection. Reuse the warm R84 inputs and frozen TRAIN geometry in place.

## Full-System Evidence

Select the strict minimum whole18 VAL macro_vs_sbs_pct, first exact tie, using
common full-TRAIN-fitted SBS. This is the existing mean-ratio selection rule,
not a claim it equals the mean of per-instance actual regret. Lock one model
and inference rule per arm, then full TEST, timing and both LIBs, with PDFs/CSV.

Report all18 and ALL Top1, mean_cost, SBS/Oracle costs, their mean-ratio metrics,
actual regret, fixed36 true rescues/false picks/net costs, NSS4 and distributions.
The fixed targets are the frozen R84A36 groups, not a moving zero-count list.
Model selection and native-winner bars share one axis, titled with modelTop1.
Full version completion triggers one curated GitHub publication, not a commit
per small change. Existing TEST has informed research; it is not blind.

R110 uses the identical two graph models with pure CE as the second half of a
four-GPU2x2 learning study. It is a separate fixed objective, not a search grid.
Cross-version comparisons must disclose one/two dropout forwards and compute.
Only locked full18 gains without increased cost/regret and with controlled
false picks support continuation; edge dynamics or nonzero picks are insufficient.

## Execution

The two original R107 slots on allocated gpu03/job1465 are used only after
their complete pipeline exits successfully. Original UUID mutexes and container
cgroups are retained. Root `preparation/dispatch` holds logs and receipts.
All engineering, protocols, review and execution use gpt-6.1-sol/max. Astra/max
is restricted to literature and mechanism ideas. No unrelated runs are stopped.
