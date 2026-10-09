# R45 Default Configuration Assumptions

ASK mode: never. Effort: lite.

| ID | Unspecified or contested decision | Chosen | Class | Source |
| --- | --- | --- | --- | --- |
| A-001 | Missing historical parameters | Use source-backed defaults, explicitly labelled assumed; do not claim historical verification | semantic | user request, disclosed implementation |
| A-002 | Whether to replace the completed manifest | Keep it unchanged; write a derived experimental manifest and corpus | interface | default |
| A-003 | Default precedence | Verified values always win; default fields record exact local source and hash | semantic | default |
| A-004 | Unknown historical revisions, native binaries and RNG state | Preserve unknowns; waive historical certification for this experimental representation, not source availability or label isolation | semantic | default |
| A-005 | Different deployment configurations | Keep existing scope splits, including GLOP insertion-only, NSS CVRP MVMOE and OMNI val | semantic | default |
| A-006 | Voyage calls and selector training | Neither is authorized by this configuration-completion turn; prepare offline artifacts only | interface | default |

The corpus's `ready` means usable with the recorded experimental assumptions,
not that old solver runs have been reproduced. Explicit opt-in and persistent
provenance are required so downstream reports can distinguish these meanings.
