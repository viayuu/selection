# Bounded Input/Execution Fact Trace

This is the Sol worker's narrow code/profile trace, not a new route audit.
No GPU, solver rerun, cost change or historical protocol substitution was used.

- Current TSP data loading takes xy coordinates in
  `/public/home/shiys/0selection2/code/unified_selector/data.py:102`.
  `/public/home/shiys/0selection2/code/V4/V4Model.py:181` pads the remaining
  channels with zeros; CLS is not a physical node0 role.
- BQ's generation record is
  `/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/labels/TSP/BQ/profile.json:55`.
  The bound adapter duplicates original node0 as the final endpoint for one
  greedy execution. The actual first/last handling is in
  `/public/home/shiys/methods论文_mineru_output/source_code/bq-nco/learning/tsp/decoding.py:73`.
- LEHD's generation profile is
  `/public/home/shiys/easynco_v3_bridge/exports/train_100k_20261002/labels/TSP/LEHD/profile.json:217`.
  It records starts=1. Although a setting is named first_mode=random, the
  executed first step uses arange(pomo_size), so that single start is node0,
  in `/public/home/shiys/EasyNCO/neural_solvers/utils/special_selected.py:47`.

The worker reported agreement between the bound source hashes and the
referenced implementations, not merely current-default behavior. These are
generation-record/source facts; individual routes were not rechecked. The
amount of selection loss attributable to the missing role is unmeasured.

Implementation uses TSP's zero channel3 rather than channel7: existing global
statistics interpret channels6/7 as time-window bounds. Local geometric
features depend only on xy and keep their frozen TRAIN normalization.

This trace covers the new90k generation records and their resplit populations.
Historical LIB execution protocols were not re-audited here. The predeclared
role rule transfers unchanged to LIB; reported LIB performance must not be
presented as fresh certification of every historical deployment condition.
