# R88 controlled execution-role input comparison

Runner/reporter ownership is limited to the four R88 integration files. R84,
R85 and R87 sources remain unchanged. Main/resource owner queue both arms
after the R85A/B deliveries finish; this module never reserves or launches GPUs.

## Exact CLI

Activate `easynco` and run from `/public/home/shiys/0selection2`:

```bash
python -B -m code.V4.r88_experiment describe --arm A_canonical_anchor
python -B -u -m code.V4.r88_experiment train --root RUN --arm A_canonical_anchor --prepared-root PREPARED --data-root RELEASE --cache-root CACHE --protocol-lock RUN/protocol_lock.json --gpu-type 3090 --large-micro-cap 352 --eval-batch-size 256 --device cuda:0 --wandb-mode offline
python -B -u -m code.V4.r88_experiment test --root RUN --arm A_canonical_anchor --prepared-root PREPARED --data-root RELEASE --cache-root CACHE --device cuda:0
python -B -u -m code.V4.r88_reporting lib --root RUN --arm A_canonical_anchor --device cuda:0 --lib-batch-size 2
python -B -u -m code.V4.r88_reporting report --root RUN --arm A_canonical_anchor --device cpu
python -B -m code.V4.r88_experiment status --root RUN --arm A_canonical_anchor
python -B -m code.V4.r88_reporting compare --root RUN --device cpu
```

Second arm: `B_execution_anchor`. Default RUN:
`code/V4/runs/R88_execution_role_inputs`. PREPARED defaults to the frozen
`code/V4/runs/R84_new90k_attention/prepared`. RELEASE defaults to
`/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`.
CACHE is the existing GPU03 node-local R84 cache supplied by the launcher.
There is no preprocessing, cache copy, solver execution or label regeneration.

`train` locks its unique VAL checkpoint and completes TEST. `test` is idempotent
for that lock. `lib` runs the same R88 factory and reuses R84's exact4500 NSS
predictions. `report`/`compare` read saved predictions and render PDF on CPU.

## Model and fresh initialization

`code.V4.r88_model.make_model(params, arm, state_dict=None)` constructs the
original R77 full selector. It adds no parameters or buffers. `role_arm` and
`role_spec` in `model.params` describe strict checkpoint reconstruction; the
original score MLP, selection head and maximin inference stay intact.

Only TSP's existing padded node-input column3 changes: A marks the input-only
canonical node (minimum x, then minimum y); B marks physical input node0.
If multiple valid nodes share the exact canonical coordinate, A marks them all;
it never resolves that tie by row index. B always marks one physical node.
Slot3 leaves the encoder's time-window width statistics (columns7-6) unchanged.
The role is known before solving. All other17 tasks retain their original
input and mathematical forward. Supervision never supplies the role.

The initializer uses the existing R84 `initialize_shared` callback with the
R88 factory, copies every common fresh seed2 parameter and restores R84's
TRAIN-only geometry buffers. It verifies identical parameter names, shapes
and hashes and records zero added/removed parameters. No R87 residual or R85
mutable objective state is loaded; the original stateless RDropObjective trains
both complete full18 models from scratch.

## Main lock and controls

Main supplies `RUN/protocol_lock.json`: `experiment: R88`, `locked: true`,
`arms: [A_canonical_anchor, B_execution_anchor]`, `seed: 2`,
`batch_size: 1024`, `max_updates: 57600`, `warmup_updates: 4320`,
`eval_every: 1440`. Any optional numeric optimizer/hardware fields must agree
with the prelocked controls. Missing/changed locks prevent training/resume.

The frozen NEW90k split has81000 TRAIN /4500 VAL /4500 TEST per task, all18
tasks, unchanged legal pools and labels. Both arms use3090, TSP/CVRP microcap352
and eval256, logical batch1024, AdamW LR1e-4 cosine to1e-6, weight decay1e-4,
dropout.1, clipping1. The complete original objective remains .35 native CE,
.35 comparison, .02 risk and ramped R-Drop. There is no sampling, frequency,
head, calibration or loss change and no TSP-only training shortcut.

Checkpoint selection strictly minimizes full18 VAL macro_vs_sbs_pct with the
same TRAIN-fitted SBS; first exact tie wins. The one locked checkpoint serves
every task, TEST and both LIBs. Historical TEST has influenced development.

## Reports

Per-arm PDFs show model selections and native FP64 winners on the same axes,
with model Top1 titles, all18 task/ALL costs, TRAIN SBS and Oracle, and the four
TSP TEST/CVRP TEST/TSPLIB/CVRPLIB NSS comparisons. Saved TRAIN/VAL/TEST counts,
zero/near-zero winner mass and the frozen R8436-target rescues, losses,
precision/recall and new false picks are exported to CSV/PDF. Both completed
locks enable `compare` to produce paired A/B tables and `COMPARISON.pdf`.
TEST never changes a checkpoint or inference rule.

This tests whether explicitly representing a known execution role helps under
the controlled setup. It does not establish a cause for the entire historical
Top1 plateau.

Focused CPU checks:

```bash
python -B -m unittest code.V4.test_r88_experiment
```

Runner readiness receipt: this command actually completed on CPU with exit0:

```text
.......
----------------------------------------------------------------------
Ran 7 tests in 2.070s

OK
```

Completed execution session77042/output47ad2e. The first run exposed two
generic test fixtures with random values in TSP's reserved zero slots; only
those fixtures were corrected to the native zero padding before the passing
run. Production inputs, model checks and frozen modules were not changed.
The additional CLI invocation (session14123/outpute20950, exit0) parsed all
four train/test commands and four lib/report commands, and validated main's
actual R88 protocol lock. No GPU launch or PDF rendering was required.
