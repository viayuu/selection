# Auto Review Loop — Solver Selector

- **Topic**: Unified supervised neural solver selector across 18 routing problems (TSP / CVRP / ATSP / 15 MVRP)
- **Difficulty**: hard (Reviewer Memory + Debate Protocol, oracle-pro browser backend)
- **MAX_ROUNDS**: 10 · **POSITIVE_THRESHOLD**: 10/10 · skip >10 GPU-h · WANDB: on
- **Reviewer**: oracle-pro (GPT-5.4 Pro, browser) — per project convention in `~/.claude/CLAUDE.md`
- **Started**: 2026-04-18 (continuation of prior session's Round 1 review)

## Round 1 — Score: 7.2/10  (from prior session; reconstructed from conversation log)

### Assessment (Summary)
- **Score**: 7.2/10
- **Verdict**: almost — "basic approach is sound, 3 concrete fixes will close the gap"
- **Key criticisms**:
  1. Selector sometimes under-performs SBS → needs a confidence gate
  2. Regret-soft CE alone is not cost-aware enough → needs combined loss (CE + regret + switch-hinge)
  3. Head MLP can't capture all per-problem-per-solver priors → needs `problem_solver_bias` + targeted oversampling
- **Baseline state at Round 1 end**: R1-seed0 (2 epochs, soft-label CE only) → `macro_vs_sbs_pct = +0.156%`, beats SBS on 8/18 problems, CVRP badly behind (+0.94%).

### Actions taken (before Round 2)
- Implemented `gate.py` (SBS-gate calibration): confirmed +0.160% → +0.014% on R1 ckpt.
- Implemented `combined_cost_loss` in `model.py`: CE (regret-soft) + expected-regret + switch-hinge.
- Added `problem_solver_bias: nn.Parameter[P × M]` to `UnifiedSelector`.
- Added `--oversample` flag with `PROBLEM_WEIGHT` (CVRP×4, VRPB/OVRPB×3, OVRP×2, {VRPBL,VRPBTW,OVRPBL,OVRPBTW,VRPBLTW,OVRPBLTW}×1.5) in `train.py`.
- Added `--loss {soft,combined}` and `--no-problem-solver-bias` flags.

### Status
- Continuing to Round 2 after launching new runs with the three fixes.
