# Cancellation observation boundary

`parse_metrics(text, model, engine)` reads only the exact running-request,
waiting-request and KV-cache-usage metrics for the selected labels, requiring
one finite, nonnegative sample of each and integral request counts. It ignores
`waiting_by_reason`, unrelated metric names and other valid target identities;
missing, duplicate or malformed required samples are invalid, never zero.

`CancellationObserver(timeout=180, baseline_samples=3, settled_samples=3)` exposes
`poll(sample, request_state)` and `report()`. The sampler accepts a keyword
`timeout` and returns `MetricsSnapshot(running, waiting, kv_usage, model, engine)`;
`request_state` is a nonblocking callback exposing boolean `started`, `active`,
`abort_sent`, `cancelled`, `completed` and `concurrent` fields.
The observer makes no network requests and never starts or aborts a model itself.

The report requests `action=start` after three distinct-time idle baseline
samples, and `action=abort` after local owned-request activity coincides with
exactly one running and no waiting request. After explicit local abort and
cancelled-exit evidence it requires three consecutive distinct-time samples with
zero running/waiting requests and KV usage at or below the minimum baseline
usage. Any natural completion, evidence regression, target change, concurrent
activity or missing evidence invalidates the observation. Idle samples without
local cancellation acknowledgement cannot pass.

One monotonic deadline covers baseline, activity and recovery. The sampler must
honor its supplied timeout; callbacks that return late cannot pass. A sample cap
also prevents a frozen injected clock from creating an unbounded polling loop.
The driver owns all polling intervals and start/abort operations and must apply
the same remaining budget to them. Terminal statuses are PASSED, FAILED and
INVALID; RUNNING means more observation is needed.

Reports contain only metrics snapshots, counters, timing, fixed reason codes and
boolean local evidence, without exposition text or model output. Aggregate
metrics cannot independently establish exclusive engine ownership or eliminate
a race with natural completion; the client must supply reliable owned-process
and explicit cancellation evidence. PASSED means these observational conditions
were met, not proof that allocator GPU memory was freed or that linked drift,
source ownership or causal collaboration works. The observer launches no trial.

Verification used isolated base `a1d4eea32428730e14f06121641d8c4d238b6adf`.
The new tests first failed collection because both required modules were absent,
then 31 focused parser/state-machine tests passed in 0.04 seconds.
`PYTHONPATH="$PWD" ./scripts/check_all.sh` passed 350 reference tests with two
existing skips in 19.49 seconds, 155 next-runtime tests with the known xfail,
and 29 plugin tests in 68 milliseconds plus TypeScript checks. Both interpreters
resolved the isolated worktree package. No inference or remote mutation occurred;
GPU/energy costs are unmeasured. The next step is integration with the bounded
owned-session evidence client and the separately authorized qualification driver.
