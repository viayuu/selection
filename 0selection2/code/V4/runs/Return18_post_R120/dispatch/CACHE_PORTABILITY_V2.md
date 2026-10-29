# Narrow Cache Portability Fix

Only six metadata keys may differ: directory, cache_root, created_utc,
completed_utc, seconds, test_lock_root. Every other field, including source file
identities, pools/order, scenario, data root, array specifications and geometry
settings/moments remains exact. TEST access-lock files must still exist.

When metadata bytes differ but canonical metadata agrees, costs.npy and
winner.npy are compared by full-file SHA256. All other declared arrays are
compared exactly at deterministic rows 0, 1, N//2, N-2, N-1, with matching
shapes/dtypes. This is deliberately a bounded input sample, NOT a claim of full
input-array hash identity. Existing header/file-size checks remain in place.
Proof details are preserved as cache.portable_metadata_checks in the new attempt.

No old failure, frozen runner, source data, cache, model or protocol is edited.
The original v1 plan is invalid under the changed transport source hash and its
terminal receipt stays untouched. Parent must create an explicitly approved new
plan name with updated source hashes. The plan JSON schema and CLI are unchanged.
No dispatch or training is performed by the implementation worker. Closeout is
still paused until the parent confirms actual TRAIN startup.
