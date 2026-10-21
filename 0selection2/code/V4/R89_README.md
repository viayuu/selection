# R89 optimizer geometry comparison

Only R89 integration files are owned here. No R84/R85/R87/R88 source or shared
Torch object is changed. Main/resource owner launches the two4090 jobs; the
two R88 workers on3090 are untouched. No environment upgrade is required.

## Exact CLI

Activate `easynco`; work from `/public/home/shiys/0selection2`:

```bash
python -B -m code.V4.r89_experiment describe --arm A_adamw
python -B -u -m code.V4.r89_experiment train --root RUN --arm A_adamw --prepared-root PREPARED --data-root RELEASE --cache-root CACHE --protocol-lock RUN/protocol_lock.json --gpu-type 4090 --large-micro-cap 352 --eval-batch-size 256 --device cuda:0 --wandb-mode offline
python -B -u -m code.V4.r89_experiment test --root RUN --arm A_adamw --prepared-root PREPARED --data-root RELEASE --cache-root CACHE --device cuda:0
python -B -u -m code.V4.r89_reporting lib --root RUN --arm A_adamw --device cuda:0 --lib-batch-size 2
python -B -u -m code.V4.r89_reporting report --root RUN --arm A_adamw --device cpu
python -B -m code.V4.r89_experiment status --root RUN --arm A_adamw
python -B -m code.V4.r89_reporting compare --root RUN --device cpu
```

Second arm: `B_muon_hybrid`. Default RUN:
`code/V4/runs/R89_optimizer_geometry`. PREPARED is the completed
`code/V4/runs/R84_new90k_attention/prepared`. RELEASE is
`/public/home/shiys/selection_data_new90k_B_filtered_single_reld_mt_seed2_20261018`.
Use the existing node-local R84 cache. No preprocessing, copying, new labels,
solver runs or NSS retraining is introduced.

## Model, optimizer and adapter

Both groups call exactly
`r88_model.make_model(params, 'B_execution_anchor', state_dict=...)`.
The complete original R77 score MLP/maximin and physical-node0 TSP role in
slot3 are retained. Both groups have identical trainable names/shapes and
fresh seed2 values, with zero new neural parameters. The role choice comes
from the execution protocol, not from ongoing R88 evaluation results.

Optimizer owner supplies:

```python
optimizer = r89_optim.build_optimizer(selector, arm, lr, weight_decay)
```

A must be the actual original `torch.optim.AdamW`. B uses Muon on the selected
hidden Linear weight matrices and AdamW on the remaining original parameters.
Fixed Muon settings: five Newton-Schulz iterations, momentum.95, Nesterov,
BF16 Newton-Schulz, Moonshot scaling `.2*sqrt(max(matrix_dimensions))`.
Both groups use the same scheduled LR and weight decay. No5x LR or Keller
aspect-ratio scale is applied. Model forward still uses the original FP16 AMP.

`r89_engine_adapter` loads the immutable R84 engine in a private module
namespace. Only that private module's `torch` reference is replaced by a
delegating facade whose `optim.AdamW(...)` returns the optimizer-owner factory
bound to the exact current selector. Every other Torch attribute delegates to
real Torch. Shared `torch.optim.AdamW`, R84 and R88 remain the original objects.
The already prepared optimizer is reused when the engine reaches its normal
constructor, retaining the original engine's single logical-batch unscale,
clip and step. AMP failures never call optimizer.step or advance optimizer state.

Actual class, group names, parameter names/shapes, source/provenance and state
step ranges are recorded in initialization/checkpoints; optimizer.state_dict
is saved/restored by the unchanged engine. Objective state is stateless and
absent. Inference loads only the same R88 model and never needs an optimizer.

## Lock and common controls

Main supplies `RUN/protocol_lock.json`: experimentR89, lockedtrue,
arms`[A_adamw,B_muon_hybrid]`, seed2, batch_size1024, max_updates57600,
warmup_updates4320, eval_every1440. Optional fixed numerical fields must
agree with the implemented optimizer and training protocol. The full lock and
digest are saved; changes prevent resume.

All18 tasks use the frozen81000/4500/4500 split and unchanged legal candidate
pools. AdamW/Muon base LR follows warmup4320 then cosine1e-4 to1e-6, weight
decay1e-4, dropout.1, clipping1, logical batch1024,4090 microcap352/eval256.
Original .35 native CE/.35 comparison/.02 risk/original R-Drop ramp remain.
Full18 VAL macro_vs_sbs_pct strictly selects one checkpoint, first tie wins.
Then complete TEST and both LIBs use that checkpoint. TEST is not blind after
earlier research and never changes this version's checkpoint or rule.

## Reports and CPU checks

Same-axis model-choice/native-winner PDF bars, full18 and per-task TRAIN/VAL/
TEST counts/cost/SBS/Oracle, exact4500 NSS four-dataset comparisons, frozen36
target recovery and matched A/B PDF/CSV are retained. R84A is historical,
same-data, rather than a newly trained no-role control. Report rendering is
not a prerequisite for training launch.

```bash
python -B -m unittest code.V4.test_r89_experiment
```
