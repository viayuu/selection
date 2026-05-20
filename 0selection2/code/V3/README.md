# V3 Naive Attention Selector

V3 implements the simplest version of the proposed direction:

```text
Q = problem / instance embedding
K = solver token
logits = Q K^T / temperature
prob = softmax(logits over the available solver pool)
```

Compared with V2, solver tokens do not cross-attend to instance tokens.  The
attention weights over solver keys are the selector probabilities directly.

## Smoke Run

```bash
conda activate easynco_zhoucl
python -m code.V3.train \
  --save-dir code/V3/runs/_smoke \
  --problems TSP,CVRP \
  --epochs 1 \
  --batch-per-problem 32 \
  --num-workers 0 \
  --device cuda:0
```

## Full Run Example

```bash
conda activate easynco_zhoucl
python -m code.V3.train \
  --save-dir code/V3/runs/V3_naive_seed2 \
  --epochs 20 \
  --batch-per-problem 256 \
  --coord-augment 8 \
  --num-workers 2 \
  --seed 2 \
  --device cuda:0
```

## Test A Checkpoint

```bash
python -m code.V3.evaluate \
  --ckpt code/V3/runs/V3_naive_seed2/best.pt \
  --out code/V3/runs/V3_naive_seed2/test_best \
  --split test \
  --batch-size 512 \
  --device cuda:0
```

