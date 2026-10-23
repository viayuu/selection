# R106 Runner Preparation

Preparation only. No training, GPU probe, allocation, launch, Git action,
main readiness receipt or protocol lock is created by this worker.
Only the R106 experiment/reporting modules, their two tests and this note
are owned here. Existing Rxx code, training, models and data stay unchanged.

## Stable Interfaces

Activate `easynco` before every shell command. Run modules from the repository
root. The following are interface examples, NOT instructions to launch now.

```bash
conda activate easynco
python -m code.V4.r106_experiment describe --arm A_frozen --large-micro-cap 256 --atsp-micro-cap 128 --other-micro-cap 1024 --eval-batch-size 128
python -m code.V4.r106_experiment train --arm A_frozen --protocol-lock /path/to/main/protocol_lock.json --large-micro-cap 256 --atsp-micro-cap 128 --other-micro-cap 1024 --eval-batch-size 128
python -m code.V4.r106_experiment train --arm B_finetune --protocol-lock /path/to/main/protocol_lock.json --large-micro-cap 256 --atsp-micro-cap 128 --other-micro-cap 1024 --eval-batch-size 128
python -m code.V4.r106_experiment test --arm B_finetune --protocol-lock /path/to/main/protocol_lock.json
python -m code.V4.r106_experiment timing --arm B_finetune --protocol-lock /path/to/main/protocol_lock.json
python -m code.V4.r106_reporting lib --arm B_finetune --device cuda:0
python -m code.V4.r106_reporting report --arm B_finetune --device cpu
python -m code.V4.r106_reporting compare --device cpu
```

All experiment commands retain `--root`, `--prepared-root`, `--data-root`,
`--cache-root`, `--device`, `--gpu-type`, `--protocol-lock`,
`--state-hash` (alias `--expected-state-hash`); train retains `--resume`,
`--wandb-mode {auto,online,offline}` and `--log-every`.
Default root: `code/V4/runs/R106_native_encoder_adaptation`.
Default cache: `/tmp/shiys_R84_new90k_cache`, read in place.

Main/admin can import the CPU-only contract immediately:

```python
from code.V4.r106_experiment import required_lock_fields
scientific = required_lock_fields(
    large_micro_cap=256, atsp_micro_cap=128,
    other_micro_cap=1024, eval_batch_size=128)
# Main adds its final data_root, prepared_root and primary_cache paths,
# bounded hardware witness and any descriptive provenance, then writes its lock.
```

The function returns the complete required scientific subset, not a lock
file. Final caps may be replaced once by main from its necessary input-only
hardware witness; both arms consume that one lock. There is no automatic cap
shrink or outcome-driven cap search. Defaults remain tentative until main locks.

Final main update: main reports `final256_1024/complete.json` PASS and has written
its scientific `protocol_lock.json`: `large_micro_cap=256`, `atsp_micro_cap=128`,
`other_micro_cap=1024`, `eval_batch_size=128`. This worker did not run that GPU
witness. The real final lock and all eleven actual stage CLI shapes were
consumed successfully on CPU. Formal TRAIN must pass these final lock values;
TEST/timing restore them from the selected model's same embedded lock rather
than reverting to CLI defaults. A science lock is NOT main readiness or launch.

## Implementation Decisions

| Contract | Implementation | Source |
| --- | --- | --- |
| Optimizer LR multiplier | Private torch.optim facade and AdamW subclass; native marker grouping, core1/native0.1 at step, common LR restored in finally | User |
| Training objective | Thin RDropObjective subclass; input-only native_pair_context around unchanged super forward, batchNone unchanged | User update |
| Graph lifetime | Only one microbatch context; model validates signatures and clears on exit; no disk/cross-batch cache | Model-worker interface |
| Evaluation cap | Private evaluator uses locked eval cap independently of training caps | Explicit eval128 contract |
| Model construction | Lazy r106_model interface import; CLI/lock contract needs no donor construction | Interface decision |
| Selected VAL identity | Nullable capture SHA requires exact native-inclusive model state, protocol state and successful update under parent lock; archive untouched | User |
| Historical R102B | Read completed recovery report view, not the old failed entry | User |

The raw-cache engine is a private R84 module. It reuses original row-count
microbatch normalization, two dropout forwards/loss terms, RNG-preserving AMP
same-batch retries and successful-update budget. Native remains eval; B has
native gradients from update1, A never updates its native parameters.
Optimizer checkpoints record roles/scales/shapes/trainability, successful-step
count and a bounded first-step implementation witness, not efficacy evidence.
Every model checkpoint contains the native parameters and buffers; model restore
passes the saved complete state to the R106 factory, not a native checkpoint path.
Resume also restores its initialization-stage model from the complete R106
checkpoint rather than rereading the original native pretrained weight files.

Reports use private stable R80/R84/R85/R87 helpers. Required outputs are all five
PDFs, full18 TRAIN/VAL/TEST metrics, VAL/TEST side-by-side CSV, native LIB49/100,
same-current-model NSS4, fixed36 true recovery/false picks/aligned net cost and
all18 actual online native+selector batch128 amortized timing. This is neither
singleton latency nor I/O-inclusive end-to-end timing. R94B and the designated
completed R102B recovery view are historical non-matched references.
The NSS CSV has explicit release/candidate-pool columns and UTF-8 without BOM
so plain administrative CSV readers see the `dataset` header correctly.
Separate selected TRAIN and VAL capture proofs bind the exact parent state and
update without rewriting archived nullable identities.

## Verification

Final existing suite: **15/15 passed in 5.361 seconds**, GPU hidden, two CPU
threads. It covers fresh complete-state/count equality; real small native pair
wrapper and exit cleanup; strict checkpoint parameter/buffer restore with all
external native weight reads forbidden; native LR scaling despite R84 resets,
optimizer-state resume and finally restoration; true-row normalization with a
synthetic uneven tail and same-batch nonfinite retry; nullable capture binding
and idempotent TEST; input-only cache opening; online donor hook/FP32 timing
adapter; native49/100 gate; NSS4 columns; strict first VAL tie; and same-axis
distribution titles using actual per-instance Top1. Synthetic curve/distribution
PDF checks use an installed CJK font. Only existing Pyparsing deprecations were
reported, not missing glyphs. None of these checks is selection efficacy.

```bash
conda activate easynco
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -B -m unittest code.V4.test_r106_experiment code.V4.test_r106_reporting -v
```

Final additional read-only CPU check consumed the REAL main lock, not a copied
fixture: both arms' train/test/timing (6), lib/report (4), and compare (1) CLI
contracts passed. Observed lock SHA256:

`76ab14852d72f2420daffc64fa5e7005533a627b1f78b18d12df3e3f6a6d8d3d`

No model was constructed by that final lock/CLI check. No GPU, allocation,
formal optimizer update, TRAIN/TEST/LIB execution, Git, ready receipt, lock write,
publication or launch was performed by this worker. The five owned files are
the only manual edits. The complete scientific PDFs require main's future
same-lock trained/evaluated evidence; they are not fabricated here.

Implementation handoff complete; main owns the parallel Sol review, source
freeze, explicit readiness and any authorized formal launch. No further audit
or automatic retry was started.
