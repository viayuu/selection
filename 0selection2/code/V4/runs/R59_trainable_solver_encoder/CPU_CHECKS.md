# R59 focused CPU checks

2026-10-11, easynco, CUDA_VISIBLE_DEVICES empty. This is implementation evidence,
not training or a performance result. No CUDA, solver decoding or test data used.

Command:

```bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest code.V4.test_r59_transfer code.V4.test_r59_experiment code.V4.test_r59_analysis -v
```

Result:26 tests passed in3.873 seconds. Coverage includes native encoder FP32
parity and weight identity, padding/masks, all18 legal pools, missing-field
preservation, label isolation, transferred/common gradients, paired parameters
and schedules, LR ratio, successful-update counting, epoch-atomic crash recovery,
validation-only checkpoint locking, FP64 replay, macro weighting and paired cost
changes when both selections are wrong. A fixture-only warning arose when casting
an uninitialized synthetic array to FP32; the dtype-rejection test passed.

Separately replayed historical R45A validation from original FP64 labels and saved
predictions:18 tasks,18,000 instances; macro Top1=0.49116666666666664,
actual_regret_pct=0.9260483583286789. All10 reported macro fields matched the
saved historical evaluation within absolute1e-10.

Real full-batch CUDA memory/runtime/gradient checks have not run yet. Both model
and experiment readiness reviews are pending; these CPU checks do not authorize
an unreviewed launch or establish any improvement.
