# R45 Default Configuration Support

Base commit: `307a9b6dfda6797dc782550bca1017d3e8b01421`.
Scope: code and offline corpus preparation only; no training or paid API call.

Use the user's completed source manifest without overwriting it. Derive an
experimental manifest that fills missing semantic parameters from local source
defaults or explicit TSP templates. Preserve all historical gaps and verified
values. Allow this derived corpus to proceed only with explicit acceptance of
assumptions; never relabel assumed parameters as verified historical facts.

Outputs: a derived manifest, per-deployment default sources/field lists, an
assumption report and an upload-ready experimental corpus. Carry assumption
metadata through embedding bundles and model checkpoints. Missing actual core
source, invalid identity, leaked labels and missing real API embeddings remain
blocking errors.

Acceptance: R45 unit/regression checks and offline CLI preparation of all 22
solvers. No data labels need to be read and no solver needs to be executed.
