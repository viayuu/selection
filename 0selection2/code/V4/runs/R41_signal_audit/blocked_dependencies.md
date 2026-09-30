# R41 blocked original solver dependencies/provenance

Available weights alone do not prove that they generated the existing labels. No substitute configuration was run.

| Problem | Solver | Status | Missing evidence / dependency |
| --- | --- | --- | --- |
| TSP | BQ | blocked_provenance | Original solver checkpoint binding, decoding budget and invocation are not fully recoverable.; Result ancestry is verified but does not establish a solver-side generation recipe. |
| TSP | DIFUSCO | blocked_provenance | Original solver checkpoint binding, decoding budget and invocation are not fully recoverable.; Result ancestry is verified but does not establish a solver-side generation recipe. |
| TSP | DIFUSCO500 | blocked_provenance | Original solver checkpoint binding, decoding budget and invocation are not fully recoverable.; Result ancestry is verified but does not establish a solver-side generation recipe. |
| TSP | ELG | blocked_provenance | Original solver checkpoint binding, decoding budget and invocation are not fully recoverable.; Result ancestry is verified but does not establish a solver-side generation recipe. |
| TSP | LEHD | blocked_provenance | Original solver checkpoint binding, decoding budget and invocation are not fully recoverable.; Result ancestry is verified but does not establish a solver-side generation recipe. |
| TSP | OMNI | blocked_provenance | Original solver checkpoint binding, decoding budget and invocation are not fully recoverable.; Result ancestry is verified but does not establish a solver-side generation recipe.; OMNI train has an EasyNCO recipe; validation came from NSS and is not the same proven recipe. |
| TSP | T2T | blocked_provenance | Original solver checkpoint binding, decoding budget and invocation are not fully recoverable.; Result ancestry is verified but does not establish a solver-side generation recipe. |
| TSP | T2T500 | blocked_provenance | Original solver checkpoint binding, decoding budget and invocation are not fully recoverable.; Result ancestry is verified but does not establish a solver-side generation recipe. |
| OVRPTW | MTPOMO | blocked_provenance | Historical shard config has top-level greedy but omits module decoder_strategy; effective historical sampling/greedy remains unverified. |
| OVRPTW | MVMOE | blocked_provenance | Historical shard config has top-level greedy but omits module decoder_strategy; effective historical sampling/greedy remains unverified. |
| OVRPTW | MoSES_CaDA | blocked_provenance | Historical rf_easnco_env/rl4co dependency binding and effective decode/RNG behavior are not yet verified. |
| OVRPTW | MoSES_RF | blocked_provenance | Historical rf_easnco_env/rl4co dependency binding and effective decode/RNG behavior are not yet verified. |
| OVRPTW | RouteFinder | blocked_provenance | Historical rf_easnco_env/rl4co dependency binding and effective decode/RNG behavior are not yet verified. |
