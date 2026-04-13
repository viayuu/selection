# 2cmab2 References

This directory archives the paper materials and code snapshots referenced by the `2cmab2` research line.

## Papers

- `papers/xu2022/`
  - `xu2022.pdf`
  - `xu2022.txt`
  - Source: Pan Xu et al., "Neural Contextual Bandits with Deep Representation and Shallow Exploration"
- `papers/zhou2020/`
  - `zhou20a.pdf`
  - `zhou20a.txt`
  - Source: Dongruo Zhou et al., "Neural Contextual Bandits with UCB-based Exploration"
- `papers/nss/`
  - `论文原文.md`
  - `PAPER_IMPLEMENTATION_DETAILS.md`
  - `README.md`
  - Source: local NSS paper/code materials already stored in this repo under `9nss论文/`
- `papers/autosaea/`
  - `Surrogate-Assisted_Evolutionary_Algorithm_With_Model_and_Infill_Criterion_Auto-Configuration (1).pdf`
  - `paper.md`
  - `AutoSAEA2.md`
  - `AutoSAEA_flowchart.md`
  - Source: local AutoSAEA paper/code materials already stored in this repo under `0AutoSAEA/`

## Code

- `code/neuralucb_official/`
  - `README.md`
  - `train.py`
  - `learner_diag.py`
  - `data_multi.py`
  - Source: local clone snapshot at `/tmp/NeuralUCB_repo`
- `code/nss_snapshot/`
  - `run.py`
  - `trainer.py`
  - `model.py`
  - `dataset.py`
  - `loss.py`
  - `utils.py`
  - `config_TSP.yml`
  - `config_CVRP.yml`
  - `README.md`
  - `PAPER_IMPLEMENTATION_DETAILS.md`
  - Source: repo-local NSS code under `9nss论文/neural-solver-selection/`
- `code/autosaea_snapshot/`
  - Full snapshot copied from `0AutoSAEA/AutoSAEA/`
  - Includes `RUN_AutoSAEA.m`, `TL_UCB.m`, `Low_level_r.m`, and related surrogate/model files

## Notes

- `NSS` is archived here as a reference implementation for encoder structure and manual features. It is a supervised selector codebase, not the training paradigm for `2cmab2`.
- `NeuralUCB` official code here is the practical diagonal approximation version that was referenced during plan design.
- `AutoSAEA` is included because the rank-reward formulation and TL-style bandit ideas were repeatedly referenced in the surrounding design discussion.
