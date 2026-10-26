# Round 01: Narrow Prelaunch Review

**Rating: READY. No evident material blocker to first R109 training found.**

Scope: `code/V4/r109_model.py`, `r109_experiment.py`, `r110_experiment.py`, and `r110_objective.py` only. R110 review is limited to objective hooks and arm mapping. No source edits, tests, training, GPU use, or commits were performed.

## Findings

No actionable blocking findings in the reviewed source.

- Model: shared four-layer encoder remains differentiable through candidate refinement, pooling, `index_copy`, joint layers, and scoring. Private banks use legal global solver IDs, not candidate columns or winners (`r109_model.py:78-135`). Adding each summary before joint attention and directly to scoring matches the intended cross-score coupling, not candidate independence.
- Initialization/restore: original seed2 trainables are hash-checked; late prototypes and pooling copy fresh original weights; strict state loading restores parameters and buffers. Activation checkpointing is non-reentrant with dropout RNG preservation (`r109_model.py:138-211`).
- R109 hooks: original RDrop objective, stateless restore checks, logical1024/fixed-cap update delegation, and explicit integration handoff are consistent at the reviewed boundary (`r109_experiment.py:116-140,295-331,495-496`).
- R110 hooks/map: A_tied_ce -> A_tied_late; B_private_ce -> B_private_late throughout factory, initialization, and checkpoint validation. The objective makes one `inputs_only` forward and masked raw-utility CE against local `ind`; no risk/pair/KL term is added. The private engine sets consistency weight to zero (`r110_objective.py:11-44`; `r110_experiment.py:119-167,324-330`).
- Both runners declare seed2, logical1024, 57600 updates, 4320 warmup, and fixed whole18 VAL checkpoint selection. No query-label read was found in the new model forward or R110 model-call boundary.

## Evidence And Limits

Main reports frozen protocols, 66 R109 + 12 R110 CPU PASS, and the stated actual-private-bank memory witnesses; accepted as supplied, not rerun. Imported R84 update/normalization and input-filter implementations were outside this four-file review. Full training/resume stability and completed VAL/TEST results are not yet established by this review. A/B are not parameter/FLOP matched; cross-score gradients are intentional. No broad causal or performance claim is made.

Reviewer provenance: the current session does not independently expose a verifiable Sol/max model-and-effort identity; this note does not certify that identity.
