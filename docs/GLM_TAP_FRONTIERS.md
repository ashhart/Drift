# Accepted GLM tap positions

The supported asynchronous scheduler advances `num_computed_tokens` when it
schedules work, before the corresponding result arrives. Its
`num_in_flight_tokens` counts that outstanding work. A scheduled cursor alone
does not prove that cache rows are accepted.

The connector retains the scheduler's request reference locally and copies only
numeric cursor metadata into `LiveStep`. No request text or token IDs enter the
activation transport. The verified frontier at scheduling is
`before - num_in_flight_tokens`; missing, negative, boolean, inconsistent or
stale-output counters fail closed. The worker may retain a private candidate
through the scheduled end, but publishes only through the verified frontier.
Rollback past any already published row poisons the session.

At successful request completion, the final frontier is
`min(num_computed_tokens - num_in_flight_tokens, num_tokens)`.
Outstanding speculative work is not itself a failed request, but its rows must
not be exported. The finalizer checks the existing contiguous publication chain
and private candidate coverage, then copies only the accepted candidate prefix.
It never reads device pages after the scheduler frees them. Aborted requests,
stale output, missing candidates, gaps and inconsistent coverage still fail.

Scheduler finalization and worker capture share a per-request filesystem lock.
The scheduler holds it until both the final tap and completion marker exist;
a worker that acquires it later sees the marker and performs no capture.
This prevents already scheduled asynchronous work from overwriting the final
tap with an earlier frontier after the scheduler has declared completion.
The lock coordinates processes, not just threads, and closes on exceptions.

This qualifies a cached prefix, not every sampled output token. A final sampled
token may never have acquired a KV row. Collector receipts therefore retain
`final_tail: UNKNOWN` and `full_completion: false`; source attribution and recall
need separate model evidence.

## Regression coverage

`tests/test_live_tap_async.py` reproduces asynchronous scheduling through the
real connector metadata builder, cache-capture path and finish callback, using
synthetic device arrays. It proves that in-flight rows are not published and
that completion trims to the processed prefix. Invalid counters must fail
before cache application. The existing tail tests retain speculative trimming,
freed-page avoidance, contiguous sequences, corruption and failed-rank checks.
`tests/test_live_tap_terminal_race.py` reproduces the late-worker overwrite and
checks that the lock remains held between final-file creation and the marker.

These tests do not replace native execution or establish two-way recall.
