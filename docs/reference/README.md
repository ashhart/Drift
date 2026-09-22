# Technical reference

These focused guides document interface and safety contracts. Dated measurements
inside them describe the stated run, not today's deployment inventory; use
[current status](../STATUS.md) for the current qualification scope.

- [Engineering contracts](CONTRACTS.md): cache boundaries, ownership, isolation and evidence requirements.
- [Exchange](exchange/README.md): persistent sessions, translators and staged application.
- [GLM integration](glm/README.md): connector, cache ownership, taps and receipts.
- [Worker protocol](worker/README.md): private sessions, activation ownership and lifecycle.
- [OMP integration](omp/README.md): provider boundaries, room/task protocols and probes.
- [Cancellation](cancellation/README.md): admission, observation and independent cleanup.
- [Reference service](service/README.md): authenticated control protocol and serving contracts.

Use the [operator guide](../guides/NATIVE_OWNER_EXCHANGE.md) before a native
run; the presence of a protocol or probe does not authorize starting services.
