# R117: frozen heterogeneous native computation

Status: implementation, not a result. Scientific basis: the completed Astra
report `docs/research/post_r116_major_direction_20261027/REPORT.md`.

## One Decisive Comparison

- A: original whole18 selector plus the common pretrained ReLD static nodes and
  added native computation with randomly initialized, frozen weights.
- B: exactly the same trainable selector and reference computations; the added
  native modules use their existing pretrained weights, also frozen.
- No selector checkpoint is used as initialization. Both reuse the fresh R84
  seed2 trainable state and TRAIN geometry statistics. Added projections have
  identical isolated initialization. The shared ReLD encoder is pretrained in
  both arms, including when its decoder is randomized in A.

Added computation is LEHD's six heavy decoder layers on the input-defined
initial boundary for TSP/CVRP; ReLD's state-conditioned decoder hidden nodes for
the other 15 routing variants; and MatNet's directed encoder plus decoder
glimpses for ATSP. These are node-aligned, separately normalized typed features,
not pooled solver descriptions. TSP uses physical node0 as both initial
boundaries. CVRP uses an empty customer history, depot boundaries, capacity1
because demand is already capacity-normalized. ReLD uses each physical node as
an independent reference current node, load1/time0, supplied L or zero, and the
known open flag. All non-self nodes remain available. MatNet uses the existing
fixed FROM-zero/TO-identity roles and physical node0 as first-node reference.
None of these independent reference states is claimed to be a realized route
prefix or an independent feasibility certificate.

No environment stepping, action selection, query route/cost/winner input,
complete solution, new labels, or candidate-pool change. DIFUSCO's diffusion
edge computation is NOT covered by this finite version. Existing weight
identities are recorded; unavailable historical training manifests and exact
label-generation recipes remain uncertified. This is a finite test of the
research direction, not a claim to transfer every candidate's entire policy.

## Fixed Training And Decision Protocol

- Data: frozen new90k, each problem 81000 TRAIN / 4500 VAL / 4500 TEST; all18
  original legal pools. Deleted B candidates and RELD_MTL stay deleted.
- Model: R88B/R94 original raw-node4 + joint2 dual stream, hidden128, hand34/ID,
  original score MLP / score differences / maximin. Extra features enter before
  the first trainable instance layer. No new output head or auxiliary loss.
- Objective: unchanged R77 CE .35 + comparison .35 + risk .02, symmetric R-Drop
  ramped to .35. Two stochastic selector forwards share the frozen features.
- Seed2; natural problem and sample streams, logical batch1024, 28440 successful
  updates, VAL every1422, complete TRAIN evaluation every7110 and at selected
  checkpoint. This is 19.9743 passes in sample exposures, not 40 epochs.
- AdamW peak1e-4, cosine end1e-6, warmup4266, weight decay1e-4, dropout.1,
  global clip1. Row-weighted microbatches. One bounded capacity witness fixes
  common per-task caps before formal training. No coefficient or seed search.
- Checkpoint: strict maximum full18 VAL macro Top1; ties lower full18 actual
  regret; exact double ties retain the first. Inference remains original
  maximin argmax, stable global-ID ties; no calibration or task stitching.
- Two authorized RTX3090s, one arm per card. Hard maximum14 GPU-hours per arm
  including feature preparation and final inference. Incomplete runs remain
  incomplete. No 4090 use, new allocation, automatic retry or silent shrinking.
- Frozen TRAIN/VAL native features may be cached; TEST features only after
  that arm's unique VAL checkpoint lock. Shared base caches are input-only.

## Acceptance And Delivery

Predeclared comparison: B versus matched A must improve VAL Top1 by at least
0.20 percentage points, with actual regret and vs_SBS not increasing and no
task regret harm above0.05 percentage points. Also report the historical R94B
reference without claiming a matched comparison. This is an engineering
continuation gate, not a significance threshold or expected benefit.

After each whole-model VAL lock, report the same checkpoint on full TEST and
TSPLIB/CVRPLIB, never reselect using these results. Deliver complete task/ALL
cost/SBS/Oracle/regret, fourdataset NSS comparison, paired selection/winner PDFs,
learning curves and the frozen R84 36-group true recovery/false-pick analysis.
Record online native-computation-plus-selector timing separately from the
cached selector timing. Caching is not deployed latency. Existing holdouts have
participated in research and are not claimed to be new blind tests.
