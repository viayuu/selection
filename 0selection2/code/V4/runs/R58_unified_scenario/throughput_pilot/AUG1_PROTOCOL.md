# RF Aug1 Timing Candidate

This is one explicitly new protocol, NOT equivalent to the frozen augmentation8 recipe. No quality ranking, val/test solving, deployment lock, full labels or training is allowed by this pilot.

- Local `Aug1RouteFinder` subclass only. No main four files or shared bridge edits.
- No coordinate augmentation call. Assert original coordinates at the actual policy input.
- Same strict original checkpoint loading and alias validation, same constraint masks and clocks.
- All physical N starts, FP32 weights and inference, singleton input, original greedy decoding and stable tie rule.
- VRPBLTW sizes50/75/100 and CVRP50/277/499, covering checkpoint buckets50 and100 and genuinely large CVRP.
- Twelve fixed TRAIN indices per problem, four at each scale, drawn from the original pilot's preselected indices without reading costs. Same disjoint warmup indices.
- Workers1/2/4 directly measured twice with fresh processes. One warmup per scale per worker; retain cold costs. Compute speedups from summed repeat wall times, never a best/min repeat or a multiplied prior speedup.
- Every concurrent repetition must exactly match its aug1 serial repetition's paid FP64 costs and physical tour sequence. No comparison with aug8 cost ranks.
- Keep original failed BQ/LEHD W4 logs. Neither failed W4 run is credited or rerun.
- Original combined wall consumption586.033948s before this branch. The branch has a3600s limit, also capped by the remaining original10800s allowance.

Method-name note: the current frozen registry exposes `RouteFinder`, `MoSES_CaDA`, `MoSES_RF`. It does not expose `MoSES_Sparse`; that name cannot silently select a different checkpoint. MoSES_CaDA is implemented with attention sparse ratio0.5, whereas the registered MoSES_RF uses MultiLoRAPolicy.

The user explicitly confirmed `MoSES_RF` as the third method. The initial two-method run finished all24 phases successfully, consuming629.788858s (combined1215.822807s). Its executed source is preserved in `aug1_source/`. The sequential MoSES_RF supplement uses an independent `aug1_moses_rf_selection.json`, summary/state/controller log and tmux handle `r58_aug1_moses_rf`; the original manifest and phases are never overwritten. It measures identical indices/warmups, two repetitions and workers1/2/4, with an1800s phase-group cap and the remaining original10800s cumulative allowance. No MPS use or new recipe branch is authorized.
