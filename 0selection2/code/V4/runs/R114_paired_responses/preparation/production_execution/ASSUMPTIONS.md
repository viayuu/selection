# Fixed Production Preparation

Base commit: 370e0b49a6783036ba0899fdbc995f3d2230a27f.
ASK=never. Only R114-owned code and preparation artifacts may change.

| ID | Decision | Class | Source |
| --- | --- | --- | --- |
| P01 | Reuse the exact existing selection SHA, 512 TRAIN bases, 16 tasks x32. No new selection, no replacement of failures, no holdout inputs. | semantic | user |
| P02 | CPU conversion on admin01, numerical threads=1, affinity restricted to two allowed CPUs, tmux r114_prepare_production. No gpu05 CPU budget or existing locks are changed. | interface | user |
| P03 | Preserve cpu_smoke and its source lock; production_inputs and the production source inventory/lock are separate. Keep the existing data implementation unchanged. | interface | user |
| P04 | Only save a waiting queue now. Future GPU binding is explicit, never inferred from device availability. No GPU poller, waiter, automatic launch, process kill, or edits to old GPU locks. | interface | user |
| P05 | Preflight one representative original/changed pair per distinct candidate family plus deployment protocol signature before labeling; additionally keep each problem/candidate deployment blocked until its own representative pair passes. A family pass never certifies missing deployments. | semantic | default |
| P06 | Runtime probes are fresh real solver executions using exactly the locked source, budget, singleton batch and seed contract. Keep probe receipts separate from label receipts; no reuse of probe costs as production labels. | semantic | default |
| P07 | Retain each failed preflight/label receipt and timeout log in an immutable attempt directory. No automatic retry, base replacement, or masked value promoted to a success. | semantic | default |
| P08 | Pin the first future explicit host/GPU/CPU/timeout binding for the whole queue. Runtime proofs do not transfer to another binding, and paired endpoints cannot silently switch hardware. A different binding needs an independent queue and fresh gates. | semantic | default |

Checks: existing CPU suite, queue/gate CPU tests, complete production manifest,
independent current source lock, and a saved queue whose GPU binding is null.
External designated-model review is unavailable; no model-review pass is claimed.
