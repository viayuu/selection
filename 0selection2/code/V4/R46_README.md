# R46A: instance-conditioned source-token interaction

Only A is implemented and trained. B is intentionally not scheduled.

The locked R45 Atlas bundle supplies complete source-view vectors. Each role /
deployment variant stays a separate128-dimensional token. Two synchronous
node/source cross-attention layers precede instance-conditioned pooling inside
each solver. The original R43A decoder, shared utility head, score differences,
maximin decision, comparison objective and R-Drop are unchanged.

No trainable solver ID, solver-specific parameter, candidate position embedding,
handcrafted feature branch, API call or new embedding is used. The old averaged
role vectors are checkpoint provenance only; forward reads `chunk_vectors`.
Already mixed content inside one external vector cannot be split retroactively.

## Run

Activate `easynco`. On the compute node with the authorized idle3090:

```bash
R46_GPU_UUID=GPU-cb24f168-9f18-f0d5-6f4a-6d2406958f50 \
    bash code/V4/run_v4_r46.sh --stage all
```

The launcher refuses a non-idle / non-3090 GPU and masks all other GPUs. It runs
the focused tests, disposable real128-example preflight, fresh training, locked
validation-best test, and analysis. W&B is retained in offline mode, as in R45.
All output is under `runs/R46_code_token_interaction/`.

Training: all18 tasks, seed2, batch128, full coverage with tail batches, AdamW
1e-4 / weight decay1e-4, dropout0.1, warmup3 epochs, max40. R43A loss is reused:
0.35CE +0.35comparison +0.02scaled relative risk, plus R-Drop ramping to0.35.
The unchanged CostController halves LR after3 ineffective validation epochs and
can stop after epoch15 /8 ineffective epochs. Best is the strict minimum of
validation macro_vs_sbs_pct. Test does not select epochs.

## Known limits

The existing full token packages are identical for DIFUSCO / DIFUSCO500 and
T2T / T2T500. Without identity inputs these pairs are indistinguishable; no
position encoding is added to hide this limitation. All candidates remain in
their original order, with ascending global ID breaking exact score ties.

The26 assumed historical deployment records from R45 remain unverified. A/B
semantic-correspondence evidence is unavailable because R46B is not run.
Historical R45A/B are reused without changing their checkpoints or results.
