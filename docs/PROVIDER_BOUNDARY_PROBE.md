# Local provider-boundary qualification

The sibling `scripts/omp/provider_boundary_probe.py` stages the actual installed OMP executable, unchanged stock Duo source, public deterministic workers and the optional provider hook. It keeps the older `boundary_probe.py` red reproducer unchanged. No model, remote host or activation channel is used.

Run it with `--omp`, `--source`, `--duo`, `--candidate` and `--mode timeout|release|invalid|deadline|abort`; an optional `--provider-source` permits separately reviewed core code. Only `DRIFT_PROVIDER_BOUNDARY_CONFIG` and its SHA256 environment variable enable the new hook; the old boundary extension is never loaded. The worker owner configuration is private outside the task directory, with explicit linked labels, fixed sessions and declared text/artifact channels. These labels are synthetic route checks, not evidence of activation exchange.

Every CLI handler timeout is 150 ms; timeout mode holds the completed pair for 500 ms before publishing matching releases. Failure modes send an invalid nonce or withhold releases. Abort mode uses actual OMP RPC prompt/abort commands and requires its explicit successful abort acknowledgement; killing a process never substitutes for that acknowledgement. Abort reaches both parked workers and is acknowledged, and every mode now records a graceful close at both workers.

The independent outer owner drains and hashes both output streams, caps output, reserves two seconds of its eighteen-second bound for cleanup, and verifies the OMP process group and separately recorded worker PIDs disappeared. Reports retain source/executable hashes, readiness/release/failure hashes, aggregate terminal/tool counters and cleanup evidence. They never retain stdout/stderr bodies. These are local process facts, not native GPU/cache-release evidence.

## Observations

The preserved initial run found the provider hook rejected a valid final turn with no tool calls. Parent fixed that assertion; this harness does not alter provider or Duo source. The final short-timeout run passed release ordering and independent process cleanup in 2.960 seconds, with both close callbacks observed. An earlier scope-identical run observed Qwen callback zero despite all worker PIDs disappearing; callback delivery is therefore reported separately and never inferred from process exit.

Invalid release took 2.277 seconds and deadline expiry 3.554 seconds: both proved zero gated Hub admissions, readiness invalidation and disappearance of both worker PIDs and their owned process group. Graceful callbacks were initially unobserved because the poisoned transport terminated workers, so their composite records were BLOCKED with the boundary and process-cleanup fields PASSED. That difference between a close callback and independent process cleanup still stands, and the requirement that workers disappear was never relaxed; the callback itself is now delivered, for the reasons in the graceful-close section below.

The earlier RPC reproduction took 0.700 seconds and remained BLOCKED before the pair formed: GLM opened without an own prompt, then reported `stream:REMOTE_PROTOCOL`. No abort was sent or acknowledged. Its two-worker cleanup criterion was not established because Qwen never opened, although the OMP group disappeared. A discarded fixture-only headless bootstrap attempt encountered fixed-session rejection; no such shim is committed and neither failed attempt qualified cancellation.

That failure was a fixture startup-ordering defect, not a worker or provider defect. `main.ts` passes an initial message only to interactive and print mode, so an RPC session starts with no user turn. The fixture opened the Duo room from `session_start`, so stock Duo's admit control became a turn whose only message was the control itself, and the worker correctly refused a first stream carrying no own prompt. A traced run confirmed both halves: with the prompt delayed, the control-only turn failed first and the later user turn inherited the dead client; with the prompt sent immediately, it was rejected as `Agent is already processing`. The fixture now opens the room from `before_agent_start`, which carries the prompt, so the room opens on a real user turn exactly as an interactive `/duo` would. No worker own-prompt requirement, session ownership rule or provider binding was relaxed.

The fixed abort mode reaches the parked pair in 1.783 seconds: both readiness records present, the actual unmodified OMP RPC abort sent and explicitly acknowledged, `[stream:CANCELLED]` and `[context:CLOSED]` observed, zero gated Hub admissions, both readiness records invalidated, both roles marked failed, and both worker PIDs and the owned process group reaped. Boundary and process-cleanup fields are PASSED. Graceful close remains BLOCKED: the cancelled transport terminates the workers before their close callbacks, as in the invalid and deadline modes, so the composite verdict stays BLOCKED. The close callback is also non-deterministic in the passing modes; one timeout run in the same session recorded the child callback as 1 and a later identical run recorded 0, so a close callback is still never inferred from process exit.


## Graceful close

Two independent defects kept the close callback from both workers, and both were in this repository rather than in OMP or stock Duo.

On the cancellation and failure paths the provider went straight from the caught error to `client.abort()`, so the worker only ever saw its transport disappear and had nothing to record. The provider now attempts one bounded graceful close first, skipped when the client is already dead and always followed by the abort. See `docs/OMP_WORKER_TERMINAL_GATE.md` for the interface and its budget.

On the success path the parent closed reliably and the child did not, in roughly a quarter of runs. A traced release run showed why: two `session_shutdown` events fire, the owning session's first and the Duo peer subagent's four to eighteen milliseconds later, each closing only its own pool entries. The first teardown is awaited and ends the process, so the subagent's later handler starts but its awaited close is not guaranteed to finish. The probe extension now records the first session to bind a route as the pool's owner, and that session's teardown closes every remaining entry, because once it ends no worker in the process can survive. A subagent teardown still closes only its own entries, so a peer that finishes early cannot close a worker the owning session is still using.

After both changes all five modes passed three consecutive times each, with both workers closing in every run: timeout about 2.9 s, release 2.5 s, invalid 2.2 s, deadline 3.6 s and abort 1.7 s. Process cleanup is still verified separately and a close callback is still never inferred from process exit.

| Receipt | SHA256 |
| --- | --- |
| provider-boundary-close-timeout.json | `3eae1cc996dd094b9b028a6edd2dea8477c5135dbbb47fa2cdee4e6a6363e44f` |
| provider-boundary-close-release.json | `0ce9c5d47a75bed4bfdbdbebbe370c2a655ae72c6dd306d6eac131c7699f81d9` |
| provider-boundary-close-invalid.json | `45fc46a6bde350095037802a24022cd66888e8456477a4e66bfb65127caac762` |
| provider-boundary-close-deadline.json | `2c083767b5b8b5557ec764a6b5adcc2258734f59a82aa0217c1965069a77c067` |
| provider-boundary-close-abort.json | `c0c31cfdb3181ad545707c3618599f54944b7018f6acc00c238bdfd4c9aed16a` |

These receipts are under `/home/example/.codex/private-audits/drift-provider-boundary-probe/graceful-close-20260921`.

| Receipt | SHA256 |
| --- | --- |
| provider-boundary-pinned-timeout.json | `4a638afdf6be8f0934fd2eb0215cb7641ffad18f8b48b956c769ba51c818f89d` |
| provider-boundary-pinned-invalid.json | `4c057b545d48b8575c3f36bfc4b98a24b96a807168e78fb790e9a6bf0b8b7053` |
| provider-boundary-pinned-deadline.json | `c14cad5224999348312be1b0dfa8bc19bd19d5c7cb066d4630ea1bfdb619dfbc` |
| provider-boundary-pinned-abort.json | `09d22fff89ca1becbd226de65c65f351cfbfcfc0ae93c4c35104709e55280912` |
| provider-boundary-rpc-abort.json | `c1324855ddf469020ffb16a49ec7e2bbd7df6e6fedc38ac3e5bd6cbe144ff0fc` |
| provider-boundary-rpc-timeout.json | `da893b71aa3cff4564eb8871a59b4849f28e3faa0394421480d005c649870036` |

The installed-OMP provider-boundary and cancellation qualification on these public synthetic fixtures is now PASSED in all five modes: a real acknowledged abort reaches both parked workers, readiness is invalidated, both workers close gracefully and both processes are independently reaped, without weakening session ownership. Native project Drift qualification remains BLOCKED for unrelated reasons: no native linked run has been made, `activation_supplement` stays disabled, and no source ownership, semantic transfer or full project collaboration is established by these local fixtures.

## Reproduce and inspect evidence

From the isolated checkout, the final invocation was:

```sh
PYTHONPATH=. .venv/bin/python scripts/omp/provider_boundary_probe.py \
  --omp /home/example/.local/bin/omp \
  --source /opt/drift \
  --duo /home/example/Documents/Codex/2026-08-25/i-added-omp-harness-and-want/omp-local-duo \
  --candidate /opt/drift \
  --mode timeout
```

The same invocation with `release`, `invalid`, `deadline`, and `abort` produced the separate receipts. The two RPC-era receipts are under `/home/example/.codex/private-audits/drift-provider-boundary-probe/cancellation-20260921`. The candidate source hashes in each receipt pin the exact bytes, including parent fixes that were still being integrated; rerunning against another candidate is a new observation. Owner-only receipts and preserved failed attempts are under `/home/example/.codex/private-audits/drift-provider-boundary-probe/qualification-20260921`; they are outside the repository and are not publication artifacts.

Validation: the pre-change full suite passed; initial focused regressions failed seven cases before helpers existed; final focused suite passed 17 tests. The final full suite passed 835 reference tests (2 skips), 155 next-runtime tests (1 known xfail), 128 plugin tests and TypeScript checks. Full-suite log SHA256: `e9390980b8eed2542b36d7fa3dd0cb3f42ebafcc71609060fd2909b08cd2e722`. Skipped real adapter gates remain BLOCKED.
