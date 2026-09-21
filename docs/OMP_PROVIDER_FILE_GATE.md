# Provider-owned project boundary

Engineering stage D2; native project dispatch remains BLOCKED.
The optional provider callback runs after a validated worker terminal and before pending tool events or completion are published.
`DRIFT_PROVIDER_BOUNDARY_CONFIG` and `DRIFT_PROVIDER_BOUNDARY_SHA256` opt into a private, hash-pinned file gate.
Both variables must be present, and their binding cannot change within a provider pool.
Normal registrations without the variables retain their existing behavior.

## Binding and failure

The file gate checks the actual fixed WorkerRoutes identity, OMP session, canonical task directory, config digest and absolute deadline.
A parent task launch can precede the first rendezvous; once the child is ready, both actors wait for releases bound to both ready-file hashes.
Empty-tool final completion validates the same identity, signal and lifetime but does not create another rendezvous.
Cancellation, invalid release, peer failure or expiry poisons the gate and writes a typed failure marker.
Any existing ready record is renamed to an invalidated record, preserving its bytes.
The provider core independently discards pending tool events and closes the transport on failure.

This is a single rendezvous gate, not a continuous memory coordinator.
An exchange digest binds release metadata; this module does not establish that an exchange occurred or that its contents are meaningful.
Hard process death cannot execute an invalidation callback, so an external owner must reject dead workers and independently verify cleanup.
Process disappearance is separate from graceful backend close and native server/GPU cleanup.
The previous tool_call extension remains an unqualified failure reproduction and is not used as the provider gate.

## Verification

Baseline: `sh scripts/check_all.sh` passed 824 reference tests with two environment skips, 155 next-runtime tests with one known xfail, and 128 plugin tests plus TypeScript.
The no-tool completion regression failed before the fix, then passed with the three file-gate tests and the provider-core tests: 12 tests, 80 assertions.
`tests/test_native_room.py` and `tests/test_development_api.py` passed all 11 checks for staging and existing project guards.
Integrated `sh scripts/check_all.sh` passed 824 reference tests with two environment skips in 47.46 seconds, 155 next-runtime tests with one known xfail in 8.97 seconds, and 140 plugin tests with 569 assertions plus TypeScript in 2.15 seconds.
Skipped native adapter gates remain BLOCKED.

Log SHA256: baseline `d9fe8a09bba1c5d8585c4b1c53baadfcc1a3e61ea078826957ef93953a015963`; no-tool regression `7c76f02be9b36f4d7e2136f3b56848d14101ccc35db120f19e4eea505bfb1f38`; integrated suite `8e21e48b4f6669e3ea3681e4fd504056ee35823991ebae55ffa73c8e0c772629`.
Public installed-OMP qualification is recorded separately; local tests alone do not qualify a native session or source ownership.
No native inference, host operation, model-weight update or publication was performed; monetary and energy costs were not measured.
Claude owns the Studio/GLM research runs, and this gate does not authorize competing jobs.
