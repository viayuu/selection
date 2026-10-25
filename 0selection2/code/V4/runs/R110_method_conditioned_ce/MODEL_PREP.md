# Model Preparation

- Reuses code.V4.r109_model without a forked model or prior buffers.
- A_tied_ce -> A_tied_late; B_private_ce -> B_private_late.
- Original fresh seed2 tensor SHA256:
  dad87d8598f7560e6f170ef048ba1bf515edeb03cef7ec42e16839a36af4ef93.
- Parameter counts2,305,033 and10,220,553; initial function/prototype match,
  not total parameter or full hash matching.
- Shared model's seven bounded CPU tests cover full18, label isolation,
  candidate identity/permutation, preserved physical role, strict restoration,
  active gradients and checkpointed dropout replay.
- Actual one-forward B256/1024 witness passes on the assigned4090. See
  preparation/kernel_witness/B_256_1024/complete.json. No optimizer step or
  TEST/LIB input opened. Formal seed2 training still starts from scratch.
