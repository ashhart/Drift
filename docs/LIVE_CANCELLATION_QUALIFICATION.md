# First live cancellation gate

This engineering probe follows the verified two-rank deployment receipt at `a1d4eea`.
It qualifies only scoped request cancellation; it does not advance a scientific stage
or establish working two-model collaboration. Limits are frozen in
`configs/qualification.cancel-glm-v1.json` before any request is made.

The probe uses the running GLM service on the Sparks, one request, at most 256 newly
generated tokens, at most 256 own prompt tokens, eight placeholder rows and a shared
180-second deadline, with ten seconds reserved for bounded cleanup. It runs no
model on the MacBook and does not use the Studio.
There is no activation link, no training, no serving restart, no MCDMA and no retry.
These are conservative execution caps chosen within the requested qualification work,
not new measured costs or a budget for the later Duo comparison.

The qualification driver and exact runner/helper are staged in a fresh private
directory, with source hashes checked before execution. The existing head-host
runner is older and lacks the supervisor helper, so it cannot substitute silently.
The service key remains on the host. Only aggregate evidence returns to the
coordinator; generated text is neither retained nor printed by the qualifier.

Require three complete idle metric snapshots for the exact model and engine before
starting. Tokenization's `started` event is insufficient: require actual output from
the owned stream plus an observed single running request and no waiting requests.
Send abort to that supervisor only, require its typed cancellation acknowledgement
and expected exit, then require three subsequent idle snapshots with KV occupancy
returned to baseline. Missing, invalid, late or ambiguous evidence cannot pass.
Observed concurrent work invalidates attribution and prevents launching another run.

Do not infer token counts from streamed chunks, request identity from aggregate zero,
or allocator memory release from KV occupancy. Aggregate metrics only support this
scoped result under the idle-engine assumption; they cannot exclude every short-lived
concurrent request between samples. Preserve failures and report that limitation.
If the stream finishes naturally before the required cancellation evidence, the
attempt is not a cancellation pass and is not silently retried.

The first request's latency remains in the report. Claude's earlier 36-second cold
cache-write observation is not reproduced by a no-link cancellation probe; when
linked writes are later qualified, cold and warm timings must be reported separately.
This probe neither benchmarks speed nor hides setup cost behind warm-up.

Stop after this attempt if cancellation, metrics or ownership evidence is missing.
Perform read-only health/metric diagnosis and preserve its report; do not restart
serving or start another model request as an automatic recovery.

The first launcher at `03425704b8d6ccc2f8de0869660c543395716585` stopped in credential preflight before opening a report, creating a session or issuing inference. The existing service has an empty API key, and read-only unauthenticated GET /v1/models returned 200 without inspecting its body. The profile now explicitly records that existing authentication mode; this changes no server controls and does not add an inference retry. The failed launch and its aggregate diagnosis are preserved outside the repository, with a new package and source hash required before the single permitted model request.
