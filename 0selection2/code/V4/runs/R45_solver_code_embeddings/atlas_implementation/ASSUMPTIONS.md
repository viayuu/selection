# Assumption Ledger
<!-- ASK mode: never -->

| ID | Underdetermined item | Choice | Class | Source |
|----|----------------------|--------|-------|--------|
| A-001 | Credential variable | Keep `VOYAGE_API_KEY`, now containing an Atlas Model API Key; never fall back to another provider | interface | default |
| A-002 | Exact 32K bound | 32,000 tokens including request-text context, verified in MongoDB documentation | interface | user |
| A-003 | Overflow aggregation | Retain every chunk; role-shared trainable attention plus max-summary fusion before role fusion; average distinct deployment variants only | semantic | default |
| A-004 | Artifact compatibility | New provider/chunking schema and separate corpus/cache/bundle paths; legacy bundles remain loadable | interface | default |
| A-005 | Oversized atomic expression | Fail with symbol/range instead of silently truncating or cutting arbitrary characters | interface | default |
| A-006 | Missing history | Reuse explicitly accepted source-backed defaults; historical uncertainties stay recorded | semantic | user |
| A-007 | API batch size | Default 16 texts, bounded by Atlas 320,000-token request limit and configured TPM; no 4K per-text restriction | interface | default |
| A-008 | Call analysis scope | Selected AST callable names and suffix matching only, not a sound resolved dependency graph; dynamic aliases/imports may be unresolved | semantic | sweep |
| A-009 | Oversized class context | Repeat constructor/class state and method signatures; include all repeated context in the actual token budget | interface | default |

The new component aggregation is part of the feature representation, not a new
selector loss or decision rule. A/B/C training remains a future explicit action.
