# Gap Map (v2, post oracle-pro review)

- **G1** — No supervised neural *selector* with variable/overlapping solver pools across ≥3 routing problems, including constraint-compositional zero-shot. (NSS: TSP+CVRP separately. CTS/GINES: TSP only. Classical AS for CVRP/VRPTW exists — Asín-Achá 2024, Gutiérrez-Rodríguez 2019 — but is not neural / not multi-variant.) [unresolved]
- **G2** — NCO-specific imbalance of overlapping solver IDs across 18 problems. Classical AS masking exists, but interaction with neural solver pools + long-tail-per-problem is unstudied. [unresolved]
- **G3'** — Constraint-compositional zero-shot solver selection: known solvers, unseen constraint combinations. Stronger framing than "disambiguate from identical features" (which a one-hot ID trivializes). [unresolved]
- **G4** — Solver cold-start in neural solver selection (add a new solver without retraining). Not addressed by NSS / CTS / GINES. [unresolved]
- **G5** — Cost-aware per-instance selection under runtime budget across heterogeneous VRP variants. FrugalML / RouterBench exist in ML-systems; no NCO analogue. [unresolved]
