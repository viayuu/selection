# R74 explicit CPU randperm, direct execution repair

SAME actual GPT-6.1-Sol/max repair thread 01a134a7-bc56-7ec2-9563-076ec70a0a96. Parent authorization: code/V4/runs/R74_adaptive_incomplete_acquisition/runtime_repair5_authorization.json. Previous actual turn completed; no concurrent executor owns these source files now.

Fix ONLY the demonstrated randperm device omission in r74_acquisition.py: existing CPU Generator(seed2) is invoked within `with torch.device(cuda)`, so torch.randperm must explicitly specify `device='cpu'` before its existing .to(self.device). Native imports, device/batch_size adapters and _reset now passed. Keep all scientific choices unchanged. No new generator, seed, algorithm, permutation or checkpoint.

Create continuation5 from the existing continuation implementation, chaining the five actual preserved failed steps and cumulative Slurm1487s. Complete20kTSP/CVRP plus verified16ATSPbaseoutputs remain untouched/reused; no new base forwards in continuations3/4. Only change necessary continuation record/authorization/prior-step constants and this one randperm line. Existing handoffs are evidence: do not rescan/rehash all artifacts or repeat old audits. You are not alone; preserve all old files and don't edit delivery sources.

One cheap syntax-only invocation, then DIRECT formal continuation5 on existing1465/gpu03/sole GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d, owned tmux r74_continuation5. Only normal assigned-UUID idle check. No real-data/neural/GPU preflight, old task inference, package installation, solver completion, labels, TEST, Git or watcher restart. Retain original seed2 policy/base/probe snapshots and unchanged full18/40E/9600updates/batch256tasktriples. If a different runtime error appears, preserve it and stop; no automatic retry/fallback.

Return actual first finite ATSP batch/live step and current generation counters, or actual failure/terminal status. Report actual base reuse/forward work, Slurm charge and syntax receipt. Distinguish feature generation from optimizer updates. Do not spend another broad review on this one-line execution defect.
