# Model Preparation

- Factory: code.V4.r109_model.make_model / initialize_shared.
- Original seed2 trainable SHA256:
  dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93.
- A_tied_late2,305,033 parameters; B_private_late10,220,553. Original1,842,953.
- Four original shared node layers plus two per-candidate graph refinement
  layers. A ties graph weights; B keys its21 banks by actual global method ID.
- New scalar-choice head, calibration, class quotas, donor or route inputs: none.
  The original shared scorer consumes each candidate's own node summary.
- Common original tensors and graph prototype values are identical, not the
  unequal complete state dictionaries or active parameter counts.
- Seven bounded CPU tests pass: full18 initial function equality, common seed2,
  labels isolated, candidate permutation/padding, preserved physical start under
  node relabeling, private/shared gradients, dropout checkpoint replay and strict
  in-memory restoration. These establish wiring, not efficacy.
- Input-only B128/1024 two-view GPU witness passes; complete evidence is in
  preparation/kernel_witness/B_128_1024/complete.json. No optimizer step taken.
