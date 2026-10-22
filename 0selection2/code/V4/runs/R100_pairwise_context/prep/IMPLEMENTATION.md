# R100 Pairwise Frozen Context Implementation

Base commit: d57aa452a3ae51da2482c0c53a6b57cc149c6892.
Scope: only r100_{data,model,experiment,reporting}.py, test_r100_*.py and
R100_pairwise_context preparation/protocol files. No GPU launch or Git changes.

## Contract And Assumptions

| ID | Decision | Source |
| --- | --- | --- |
| A1 | Reuse immutable R97 TRAIN features, base IDs, constant-column drop, classifier weights and isolated public runtime. | user |
| A2 | Each globally ordered method pair retains all strict FP64 TRAIN comparisons, including rows won by a third method; exact pair ties alone are excluded. | user |
| A3 | Store pair labels as int8 (-1 tie, 0 first wins, 1 second wins), with row/pair ordering and support counts identity-bound. | implementation interface |
| A4 | Reject any query base ID in the complete TRAIN source-ID vector before inspecting pair masks; no TRAIN diagnostic mode. | user |
| A5 | Mean opponent win probability (expected-rank/Borda score), ascending global-ID exact ties, fixed before VAL. No coupling, calibration or aggregator search. | user |
| A6 | A formal log-softmax(mean-win) is saved only for descriptive CE interoperability; it is neither winner probability nor the ranking rule. | user |
| A7 | First full TSP VAL inference is the unlabeled resource witness, not an extra subset call; every pair is independently refit for each named dataset, one model resident at a time. | implementation interface |
| A8 | LIB input-only decoding is privately adapted from R98, so no label-opening loader runs before prediction. Saved native evaluation helpers remain read-only. | implementation interface |
| A9 | prepare writes prepare_identity.json and protocol_lock.json; run requires exact agreement, source hashes and VAL fingerprints before CUDA work. TEST input binding follows the unique full18 VAL bank lock. | user |

## Verification Ladder

1. CPU strict-label/provenance preparation and dummy-classifier isolation tests.
2. CLI prepare/run/report help and frozen scientific protocol readiness sidecar.
3. CPU saved-prediction/report checks, source-ID isolation and deterministic global-ID ties.

No result or method-success claim is authorized by these implementation checks.
No training accuracy, epochs, SGD updates or logical-context batch1024 claim.
The first full run and final report remain the main task's responsibility.
