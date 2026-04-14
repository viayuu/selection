# Dataset and Label Tracker

| ID | Task | Status | Output | Notes |
|----|------|--------|--------|-------|
| A1 | Freeze exact 16 CVRP variants | TODO | variant list | should match the first paper scope |
| A2 | Build problem-method coverage table | TODO | coverage md/csv | use 平台方法统计统计.md as source |
| B1 | Confirm NSS TSP/CVRP paths | TODO | path manifest | no regeneration needed |
| B2 | Extend MVRPGenerator with coordinate distributions | TODO | code + config | keep EasyNCO-native |
| B3 | Extend PCTSPGenerator with diversity regimes | TODO | code + config | distribution + prize/penalty mix |
| B4 | Extend ATSPGenerator with structured asymmetric families | TODO | code + config | keep current generator as one family |
| C1 | Generate smoke datasets | TODO | small pt/pkl files | 200 per selected problem |
| C2 | Create init-only label configs | TODO | yaml files | one per method if needed |
| C3 | Run smoke labels via eval.py | TODO | instance_results.jsonl | verify per-instance export |
| D1 | Generate full 10k datasets | TODO | full dataset files | per problem |
| D2 | Run full labels in tmux | TODO | jsonl outputs | split jobs by problem/method |
| D3 | Merge labels into supervision tables | TODO | merged dataset | include feasible mask and ranks |
