# Current engineering record

Read [current status](STATUS.md) for user-facing readiness and
[the roadmap](ROADMAP.md) for the next gate. Detailed earlier tickets, including
failed attempts, costs and evidence hashes, are preserved in the
[historical engineering record](history/engineering-2026-09-22.md).
Private receipts and model-inaccessible evidence remain outside the repository.

## 2026-09-22: documentation consolidation

Stage: documentation maintenance; no scientific-stage advancement.
Base development commit: `54bd96fad3e0c8ce94538292ca300ce789845f4b`.
Published source base: `e334afc589bee2386b9ad41732af447dce985605`.
Implementation commit: `209a7a7db0c0e653bab74785c3f97239e272e8d5`.

Changes: one current-status page and documentation index; focused user guides
and subsystem reference indexes; superseded status pages and SSH instructions
clearly marked historical; the stale FRONTIER placeholder retired with its
valid caveats preserved in STATUS. Earlier evidence and failure records remain
available, rather than being erased or relabelled as passes.

The engineering handoff's hashed source appendix is unchanged. Frozen
preregistrations retain their original citation strings and bytes. No corpus,
token IDs, translator, checkpoint or scoring artifact was regenerated; future
repository-text corpus generation changes with documentation and must pin its
source checkout, as described in [evaluation guidance](evaluation/README.md).

Baseline: `.venv/bin/python -m pytest -q`, 1419 passed, three BLOCKED environment
skips and one existing warning in 64.87 seconds. Documentation regressions
started with two failures for the missing reader index and historical labels.
Final validation: PASSED, 1430 Python tests in 63.56 seconds; three required
environment checks remain BLOCKED, with one existing warning. The plugin suite
passed 183 tests and 775 assertions in 2.72 seconds; `bunx tsc --noEmit` passed.
Focused documentation and GLM deployment checks passed 19 tests. The first full
run caught a pinned connector hash change caused by a moved-path comment; the
original module bytes were restored and the deployment checks passed unchanged.
Concurrent licence, CLI, scoring and CPU-demo work was preserved.

Evidence SHA-256:

- Final Python log: `bb52f51cea503d851eed206a3ab9f1a2f8307b7cec3009eb05c33228ba35a464`.
- Plugin log: `5f33b6fba2d7a056071dbc86937cca464bb0976f38db07b7dadc6a431d8c663a`.
- Private pre-cleanup archive: `ffd2809e488a0f5420046e6adbd329311e788fca3d0e387f8cf8dc784733bbc3`.

The GitHub skill guided preservation and staged-content inspection; the scoped
secret scan found no leaks. This is not a full publication clearance: original
development history and the separate publication checkout were not uploaded.
No native inference, training or host mutation; money and energy were not measured.

Next product ticket: operator-independent native startup, with the research
and reliability limits in STATUS retained; no model budget is granted here.

## 2026-09-22: publication preparation

Stage: publication maintenance; no scientific-stage advancement.
Source snapshot: `6bc72fcc9ce3f9e7b74709f5dcbf34d3b6f04d47`.
The documentation implementation is `209a7a7db0c0e653bab74785c3f97239e272e8d5`.

The candidate combines the completed documentation cleanup, Apache-2.0 licence,
CI, CLI help, citation, note-versus-memory re-scoring and CPU mechanism demo.
Licence and citation attribution use the approved public handle; private
original development history is not imported. Ongoing prefill work is excluded.
The evidence checksum list now uses the standard two-space separator so
`shasum -a 256 -c SHA256SUMS` works; all six payload hashes are unchanged.

Candidate QA: PASSED, 1430 Python tests in 63.94 seconds, three environment gates
BLOCKED and one existing warning; 183 plugin tests and 775 assertions passed in
1.75 seconds; TypeScript passed. Focused checks passed 27 tests. Re-scoring
validates recorded results only; it is not a fresh native model qualification.

Evidence SHA-256:

- Python: `68df3fcbd4fd64578e1242b8d3777047c7ebe4e81db4132b9cdd0b848fe408fc`.
- Plugin: `e4ab1ee12c3c7a6ce82bec09f6591c0cfff1b499d63d6aeded4d92ca3b09e8c0`.
- TypeScript: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

No inference, training, deployment or host mutation occurred; money and energy
were not measured. Current scientific and native workflow limits remain in
STATUS. The exact publication commit, object audit and transfer result are
recorded privately; passing tests alone do not authorize publication.

## 2026-09-22: CI environment repair

Stage: CI portability; no scientific-stage advancement.
Implementation: `3122a34a7844681183679dae4c04a5636c916f48`.
Failed public run: `35731775153`, source `1d220c49aee5b0d7175eee6af2370e35b21a5cbd`.

The Linux Python jobs failed on eleven eager optional-tokenizer imports and
three fixtures hardcoding a macOS interpreter. Five plugin tests require actual
macOS Seatbelt, so Ubuntu correctly refused their sandbox calls. Local installs
had hidden these environment assumptions.

Recipe loading now imports tokenizers only when the live tokenization operation
is requested; its eleven regressions explicitly remove that optional package.
Pure staging fixtures use the running interpreter. Plugin CI runs on macOS 15
with its required Python 3.11.9 framework and explicit executable preflight.
No sandbox guard, security policy, assertion or required test was disabled.
The Linux Python 3.11, 3.12 and 3.13 matrix remains.

Red-capable checks reproduced eleven missing-tokenizer failures, three missing
framework-path failures and DRIFT_TASK_SCOPE for a Linux platform. The added
CI/runtime contract checks plus recipe regressions first failed thirteen tests.
After repair, twenty focused tests and seven simulated missing-framework tests
passed. Baseline candidate suite: 1430 passed; final: 1432 passed in 64.33
seconds, three required environments still BLOCKED and one existing warning.
Plugin: 183 passed, 775 assertions in 2.73 seconds; TypeScript passed.

Evidence SHA-256:

- Failed CI log: `6e95e7972a69b8f74a40622e36591f4855864dac16a9962ddeb4a3e624af1984`.
- Python final: `1402623647dd9d3154bd9ba98044963db18d4d3ccd49e19479f81f3c7e5a241b`.
- Plugin final: `315c3a95790a08b24ee640d55365864639259ef75c6bc5b824766753fe35c495`.

No native inference, training or host mutation; energy and money not measured.
Next gate: verify the replacement GitHub Actions run on its actual runners;
local checks do not claim remote CI success or native model qualification.

## 2026-09-22: CI cold-process follow-up

Stage: CI fixture lifecycle; no native-stage advancement.
Implementations: `24167258abb67807c991b2a3d9a5fe04f1d3dd88` and
`4cbafc8a21fad16bcd352049077da7332c26d49f`.

Run `35736018257` cleared all original failures; Python 3.11 and 3.13 passed.
It exposed two timing failures: the plugin worker fixture stopped at its
one-second deadline while launching the system Python, and Python 3.12's Node
reconnect process exceeded the outer four-second bound without diagnostics.
Those are recorded failures, not a successful replacement CI run.

A 1.1-second delay on the system-Python launch reproduced the provider error.
The fixture now uses the same pinned framework interpreter as the sandbox
tests, with its original 1000 ms deadline unchanged and an explicit assertion
for any allowlisted error diagnostic. That fault-injected test now passes;
twenty independent repeats also passed.

A 4.5-second Node startup delay reproduced the subprocess timeout. The reconnect
test now waits for an explicit ready signal before starting the coordinator and
its protocol clock; startup has a separate ten-second bound. Its original
one-second request timeout, four-second exchange-process timeout and five-second
coordinator deadline are unchanged. The delayed-start case is a permanent
regression, and the original injected-delay command now passes too.
This isolates startup from protocol timing without claiming to have measured
the precise source of the earlier GitHub scheduling delay.

Final local QA: PASSED, 1434 Python tests in 67.81 seconds, three required
environment gates BLOCKED and one existing warning; 183 plugin tests with
776 assertions in 2.18 seconds; TypeScript passed.
Python-log SHA-256: `06d8b69e57a33d81b0b13c5400016bba0c3ab82c0274588674fc63d43e429a84`.
Plugin-log SHA-256: `27827869da65ec7131f62204d9ed748a3d7cde1c4e0115a3dca1a2a01253a0f5`.
No model inference, host mutation or native-runtime change; costs in money and
energy were not measured. Next gate remains completion on actual CI runners.

## 2026-09-22: opt-in transcript-backed memory

Stage: M-1 input/provenance contract and local integration mechanics; no scientific
stage advancement. Implementation commit: `83f89d1552a5164bedaacdf43284c37957c64230`.
This is a separate text-ingestion feature, not recovered provider cache or a
replacement for continuous native OMP exchange.

Added focused `drift/transcript` modules and `drift transcript import`, `validate`,
`capture`, `receive` and `answer`. The importer preserves declared sources, message
roles and tool-call relationships, refuses unsupported fields and bounds input.
GLM capture checks the export's session and exact prompt IDs locally, then writes
only selected latent tensors to the transport artifact; raw headers stay local.
The MCDMA receiver verifies size, digest and layout before local publication.
Qwen translation and capacity checks precede cache mutation; the reader sees the
latent artifact and its question, not the transcript receipt. CLI native child
lifetimes are bounded and no shared daemon is killed. A timed-out GLM request's
server-side release remains explicitly UNVERIFIED.

Baseline command: `.venv/bin/python -m pytest -q`, 1438 passed, three environment
skips and one existing warning in 72.29 seconds. New schema/codec tests first
failed at collection because the package was absent, then all 19 passed; the
following full suite passed 1457 tests. The new expired-deadline reader test
initially exposed cache mutation before timeout, then passed after checking the
deadline before translation and again before cache creation.

Final focused command: `.venv/bin/python -m pytest -q tests/test_transcript* tests/test_tap_cli.py tests/test_documentation.py`,
60 passed in 2.83 seconds. Final full command: `.venv/bin/python -m pytest -q`,
1487 passed in 69.07 seconds, three required native environments BLOCKED and one
existing warning. An earlier command named a nonexistent documentation test
file and exited 4 without running tests; it was corrected to the existing file.
`git diff --cached --check` passed after removing trailing blank lines.

Full-suite log SHA-256:
`80dee921e0b2af29c9ec02be58a617692687292fa2e2e4830e503c976cbcd920`.
Evidence is retained in the private external transcript-memory QA directory,
not in this repository. SERIES rules inspected at SHA-256
`48e7919b29a1a663bb72de979f072678bd6074968780e617c08875ef9d62991f`;
PASSED here is an engineering label, not a demonstrated-improvement verdict.

Costs: local CPU/fixture tests only; zero native inference, training, research-host
operations or cloud API calls. Money and energy unmeasured. No publication was
requested or performed for this feature; the development history is not a
publication candidate.

Next gate: qualify the new transcript workload on owned GLM/Qwen runtimes against
full-transcript, no-memory and wrong-memory controls with isolated scoring and
explicit compute limits. Native startup, recall, attribution and total latency
for this feature remain BLOCKED pending that evidence; local tests do not supply it.

## 2026-09-22: transcript native capture and transport gate

Stage: M-1 integration qualification, with no scientific stage advancement.
Implementation commit: `e064da770db22592baa54bb860da7aca915c83cd`.
Baseline `.venv/bin/python -m pytest -q`: 1487 passed, three required native
environment gates BLOCKED, one existing warning, 67.84 seconds.

The first failing regression reproduced the capture preflight's rejection of an
owned shared export parent. Capture now allows a canonical owned parent without
group/other write permission while keeping request and output directories 0700.
Two additional tests reject writable parents before inference. Focused transcript
tests passed, then the full suite passed 1490 tests in 71.17 seconds.

Native GLM capture PASSED on one synthetic transcript: 130 prefill tokens, one
generated export-completion token, 130 latent rows and 2,928,652 output bytes.
The capture reported 0.7458743499591947 seconds, excluding staging and startup.
The actual MCDMA receive FAILED on the source control connection; retry failed
to connect. Qwen was never loaded and end-to-end recall remains BLOCKED.
Read-only checks found the source peer down on the receiver, a retained source
control connection absent on the receiver, and successful source reachability.
The original disconnect's cause is not established; these observations are not
evidence of payload corruption or model recall failure.

Three new regressions reproduced uncaught transport errors. The receiver now
returns a redacted FAILED report for daemon and socket errors, publishes no
artifact and makes no text/SSH fallback. README adds CLI and Python importer
examples; the guide records the exact native qualification limit.

Final `.venv/bin/python -m pytest -q tests/test_transcript*`: 55 passed in
0.81 seconds. Documentation and CLI checks: 17 passed in 0.18 seconds.
Final `.venv/bin/python -m pytest -q`: 1493 passed, three native environment
gates BLOCKED, one existing warning, 71.84 seconds. `git diff --check` passed.
Full-suite log SHA-256:
`269a2de171b60ff24bf3ca2917246c52881e93dafeaef4c55c440fdffe0cf461`.
Private native capture receipt SHA-256:
`cff95d5642c3b0a49e7fa71a74524ecbef526bbbe04c0f2bba52b1563b823bb7`.

Evidence, synthetic inputs and runtime staging remain outside the publication
candidate. No shared service restart, driver change or daemon termination was
performed; no native Qwen inference, training or cloud API call was made.
Money and energy were not measured. Native work stopped at the transport gate,
well within the ten-minute inference allowance.

Next ticket: coordinated recovery of the source MCDMA control connection, then
verified tensor receive and isolated Qwen recall controls. DO NOT PUSH: the
user's end-to-end prerequisite is unresolved and the new publication snapshot
has not passed the required full GitHub privacy audit.

## 2026-09-22: transcript native smoke recovered

Stage: M-1 integration mechanics; no scientific stage advancement.
Documented implementation and launch correction:
`83dc35414766267fa18bc83a167c4c0bbbdf3399`.
The earlier transport failure remains recorded above, but no longer blocks the
bounded transcript smoke path using a retained control session.

A direct TCP PING/SERVE/RELEASE on the source succeeded. An isolated diagnostic
Studio daemon reported connect errno 65, before any source handler or RDMA read.
A minimal TCP-only executable reproduced the launch-dependent fault: connected
under a live SSH session, failed after detached launch, and failed as a user
launchd job. Keeping the SSH session alive restored transfers with the original,
unmodified daemon binary. This localizes the observed fault to process/network
launch context, not translator weights or Spark registration; it does not prove
the exact operating-system policy responsible.

The recovery used an isolated daemon socket and shared-memory name, retaining
its control session, without restarting shared daemons, models or drivers.
All diagnostic owners were stopped via SHUTDOWN, not signals. The original
shared daemon is unchanged; its detached source route still needs an
operator-approved lifecycle correction for unattended operation. No privacy
control was disabled. The guide now explains the qualified session-bound launch
and the separate unattended-service gate.

Native CLI capture/receive/answer completed. Two fresh GLM snapshots each had
130 prefill rows and one export-completion token; both 2,928,652-byte latent
frames crossed MCDMA with matching digests. CLI Qwen returned the queried fact
with no source transcript on its activation input. A separate twelve-answer
process used a fresh Qwen cache for each question: target memory 3/3, complete
transcript 3/3, no memory 0/3, other-conversation memory 0/3 target facts and 3/3
alternative facts. The time question was already observed in the pilot, so this
is smoke evidence, not a preregistered held-out recall qualification.

Source capture wall times: 0.746 and 0.718 seconds. Verified receive wall times:
6.33 and 5.52 milliseconds. CLI loading plus answer: 9.42 seconds, including
31.4 milliseconds translation/append. The twelve-answer process took 14.59
seconds including loading. These component timings exclude diagnosis, staging
and failed attempts; no latency distribution or text advantage is claimed.
The native model work remained below the ten-minute allowance; no training,
cloud inference or backbone modification occurred. Money and energy unmeasured.
No model process remains from these runs; private scorer/evidence files were
kept outside model workspaces and publication candidates.

Baseline `.venv/bin/python -m pytest -q`: 1493 passed, three environment gates
BLOCKED, one existing warning in 75.23 seconds. Focused transcript/documentation
tests: 58 passed in 0.83 seconds; `git diff --check` passed.
Baseline log SHA-256:
`d9fb4b3c754010f6d9df91ef2ecc1c78b6110aa22d8ff1767dc72f32cec8e34b`.
Private control-result SHA-256:
`d3fc853aefdc81b133f9343460ace1788bb7b0fb41330fe5f4598a54bc18e9d1`.

Next gates: full GitHub publication inspection of the transcript snapshot;
broader held-out recall/attribution and unattended daemon lifecycle qualification
remain separate. Per-invocation execution receipts cannot assert scored recall.

Publication follow-up: the isolated public snapshot was committed and pushed as
`daa6c90645cfed7fe5aed1fd8189b2dd066a3654`, tree
`c62b86dbb252c654be1f823684fb866a77a67e7c`; remote main was verified at that SHA.
The GitHub skill inspection covered all 810 candidate files, stored/history
objects, metadata, identities, encoded payloads and ignored inventory, with
unchanged banner provenance retained by exact hash. Four scanner matches were
reviewed non-secret fixture/digest expressions; no confirmed unintended private
or employer content remained in the inspected transfer. Private native artifacts
and development history were excluded. Candidate QA passed 1489 Python tests
with three BLOCKED environment gates, 183 plugin tests and TypeScript; four
development-only prefill-cost tests were not part of that snapshot. CI run
35744832885 was triggered and is recorded in the external publication evidence.

## 2026-09-22: Retire the standalone engineering bundle

Stage M-1 repository contract maintenance, PASSED locally at implementation
commit `37cea7c1972cf28849ca2843e4c1feaa35d4c197`.
Removed the obsolete standalone handoff and its source-extraction script after
privately preserving exact copies; published history remains recoverable.
Moved the required cache, ownership, transport, training and isolation contracts
to `docs/reference/CONTRACTS.md`, retained M-1 through M6 in the roadmap, updated
agent instructions and links, and removed the retired input from corpus creation.
Existing corpus artifacts, translators and runtime behavior remain unchanged.

Baseline `.venv/bin/python -m pytest -q`: 1493 passed, three environment gates
BLOCKED, one existing warning in 72.84 seconds. Both new regression cases failed
before the changes; focused documentation/corpus checks then passed 5/5 in
0.07 seconds. Full suite after changes: 1495 passed, three environment gates
BLOCKED, one existing warning in 70.39 seconds; `git diff --check` passed.
Full-suite log SHA-256:
`099750984463bcbbea31ac50d627a0afefd82cfb36fcbd0811de8ab696fab8e5`.
Archived handoff SHA-256:
`8b628c22da21392ae66ecf8b617133e113e70c7fd27a875ae057d3fbf93e7caa`.
Current contract SHA-256:
`955b9db795528665604320a19f0c7cadba685a03d5d1db83ae5e41b92127bddb`.
No native host actions, model inference or training; money and energy unmeasured.
Next ticket is the isolated candidate QA and GitHub inspection for this cleanup;
general recall and unattended service qualification remain unchanged.

## 2026-09-22: Repair the proposed daemon shutdown controls

Stage M-1 control-path repair, local tests PASSED at
`7974c2e2063cf6e837fd427b62f5df69346dab80`; native qualification BLOCKED.
Reviewed the earlier external-daemon proposal without deploying it.
Its copied signal/socket helpers reproduced a pending worker-directed signal,
a blocked sender after stop, and queued-command acceptance after stop, ten
times each; the process-directed signal control passed ten times.
Those private diagnostic receipts do not constitute native daemon evidence.

The replacement has focused signal/wake and control-I/O headers, atomic stop
and worker flags, nonblocking listeners and socket operations, stop-aware
command admission, bounded sends/connects and cancellation checks in transfer
and NEXT loops. Existing worker joins still precede endpoint teardown.
An explicit compatibility restriction now requires numeric peer IP addresses
to avoid unbounded synchronous name resolution. Driver/provider teardown
behavior has not been qualified by the local changes.

The preparer accepts only the pinned upstream source digest, refuses existing
output directories, and records a private staged-source/header manifest.
It neither builds nor launches a daemon. Preparation against the reviewed source
passed; the whole candidate compiled with C11, pthread, -Wall -Wextra -Werror
and an explicitly pinned macOS 26.2 SDK, linked to librdma, without execution.
The original owner source and all deployed services remain untouched.

Retained checks use the real candidate helpers with no verbs:
`python -m pytest -q tests/test_handoffd_shutdown.py tests/test_documentation.py`,
14 passed in 2.78 seconds, including nine control scenarios repeated ten times.
The worker-directed-signal fixture now waits for the target worker to confirm
its inherited signal mask before delivering the signal; this tests the stated
steady-state boundary rather than thread bootstrap timing.
Final `python -m pytest -q`: 1505 passed, three required native environment
gates BLOCKED, one existing warning in 71.21 seconds; `git diff --check` passed.
Full-suite log SHA-256:
`7cf5df629eca9cc13a697a630a17fd364e34194457532e3329c1707de8b775e0`.
Patch SHA-256:
`36c0134a1f19ee3c3ebaab8f23f008fe42590dc865e913500f6b35a683875fea`.

The proposal's unsupported native count and source claims were corrected, its
private host identifier removed, and new commits use the approved public
identity. The earlier development commits still contain identifying metadata
and must remain private; no history rewrite or public push occurred.
Contextual checks and Gitleaks scans found no new secret/employer match in the
reviewed repair, but a complete new publication-candidate audit is still required.
No host connections, model inference, training or native verbs execution;
money and energy unmeasured.
Next gate: isolated Linux coordinator/target shutdown and worker-quiescence
qualification before any Mac verbs run; stale-slot recovery, background retry
and unattended service qualification remain separate open gates.

## 2026-09-22: KV-only provider admission for the Drift subagent counterpart

Stage: M-1 integration prerequisite; no M0-M6 scientific promotion.
Code commit: `6a20ec568aa926ad89113089c4d24e3a6de64b44`.
Base: `d62bf87be3f067fa6c390bd515c12857f3f7390f`.
PASSED for the restricted local provider policy; BLOCKED for automatic
`/drift subagent enable` startup and native qualification of this new mode.

Read the current contracts, SERIES, the installed Duo command implementation,
MCDMA integration guidance and the existing provider/exchange modules before
editing. The first missing gate was explicit peer-text opt-in on fixed routes.
Added owner-pinned `kv_only`, requiring a linked fixed pair with repeated parking
and exchange configurations. It does not authorize the text-enabled Duo path.
The provider admits one own setup prompt, an immutable system prompt and only
the fixed `drift_sync {}` tool with its `ready` control receipt. Unexpected peer
developer messages, follow-up prompts, tools, arguments and results fail closed.

Parent-first sync now parks instead of using the text workflow's startup bypass;
both actors settle on completion. The pinned parent owns pool teardown even if
the child binds first. Exchange failures poison the provider without releasing
the pending tool. Focused modules retain single-line code comments.
The standard plugin still exposes only reference commands; no nonfunctional
native toggle was added and no existing Duo room is relabelled as promptless.

Baseline commands: `.venv/bin/python -m pytest -q` yielded 1505 passed,
three required environment gates BLOCKED and one existing warning in 69.09s;
`cd plugin/omp-drift && bun test && bun run typecheck` yielded 183 passed,
776 assertions in 2.71s and clean TypeScript. Five new cases failed before the
policy implementation. Final focused KV-policy/boundary tests: 10 passed,
88 assertions in 0.14s. Final plugin suite: 193 passed, 864 assertions in 2.77s;
TypeScript passed. Final full Python suite: 1505 passed, three environment skips
remain BLOCKED, one existing warning in 68.86s. `git diff --check` passed.

Evidence log SHA-256 values, retained outside the repository:
plugin `3dbd8da8afc0863a41fc29d81146ea0ae9394bfd41392fc9b427a0fbea7f2e30`;
Python `327e3777a9ba2801f875ed89e3c74a4be5a971c5805aa6ec1eb7dffac03bada9`.
Tests use synthetic worker replies and a private Unix coordinator socket,
including two successive paired exchanges and missing-peer cancellation.
They do not establish RDMA transport, native recall or operating-system isolation.
No model inference, training, host connections, deployments or native verbs;
monetary cost and energy unmeasured.

GitHub-skill review inspected the changed text, test fixtures, file types and
staged diff; contextual checks and Gitleaks reported no new confirmed findings
within that scope. The development history remains private and is not cleared
for publication; no push or modification of the public checkout occurred.
Next ticket: owner-controlled native launch, subagent admission and epoch release
without relaying model-generated task prompts or child answers, followed by
installed-OMP/native qualification and independent resource-release receipts.


## 2026-09-22: Appointment runner, non-oracle selection and strict street scoring

Stage: exploratory development tooling, no scientific promotion. Base commit `2755455d5eac65f3e4b9022ee38dede326f47a6b`.

The appointment demonstration sends only the rows of the appointment sentence. `demo_appointment.py` passes the fact text to the Studio loop as `--publish-text`, and `studio_mcdma_loop.py` selects the tokens that overlap it, tiles them 12 times and, with `--prompt-only`, publishes nothing else. What crosses MCDMA is KV, not text, but the controller picked the rows by matching the answer, so the demonstration does not show one model reading its partner's memory in general.

`--selection all` drops `--publish-text`, so Qwen publishes every row it holds and no step of the transfer uses the answer. `--reserve` sets Qwen's publish budget and GLM's reserved span together. A reverse publication skipped for lack of reserve now rejects the arm with `REVERSE_PUBLICATION_SKIPPED` instead of being scored as a failed recall. `glm_named_only_true_street` requires the true street and no other listed street after the marker. The existing `glm_recalled_true_street` accepts an answer naming several streets and stays for comparison with earlier runs. Reports now record selection, copies and reserve.

Commands: `.venv/bin/python -m pytest -q tests/test_demo_appointment_bounds.py` 12 passed. With the runner change stashed the five new tests failed, 5 failed and 7 passed, and the previous runner scored a hedged two-street answer as a correct recall. Full reference suite 1514 passed with 3 environment skips. `gitleaks git --pre-commit --staged` 8.30.1 exited 0. No host, model, inference or daemon was touched, so compute was zero.

Next ticket: an exploratory ablation on the six development scenarios, oracle against all at 12, 3 and 1 copies. That is about 48 arms at the roughly 25 seconds per arm the last run took. It needs the session-attached Studio daemon replacement and owner approval for host time. These are development scenarios and none may be relabelled as held out.

## 2026-09-22: Live loop repairs, selection ablation, Drift start-up for GLM and the Studio daemon

Stage: exploratory development evidence and host operations, no scientific promotion. Base commit `37c78fa`.

Correction to the previous entry: the ablation did not need the Studio handoffd replacement. The appointment loop moves memory through the dedicated MCDMA targets on the Sparks, not through handoffd.

Code changes:

- `7a04621` The publication cleanup had put documentation addresses into the three Studio scripts, so every linked run stopped at startup. The coordinators now read `DRIFT_MCDMA_LINKS`, refuse a linked run without it and pass it to the Studio worker. A test fails if a Studio script gains a literal address.
- `f2ed3eb` The rephase from the frontier checkout called `clear_index_blocks`, which the deployed oMLX QSA cache lacks, and crashed the first decode step after a forward tap. The cache shim now finds the receiver's real reset call. `studio_rephase_check.py` passed 11 of 11 checks on the Studio under mlx 0.32.2 and mlx-vlm 0.6.3 in the oMLX 0.7.0.dev2 bundle. A control with the reset disabled failed the two block-bank checks.
- `14c5f34` The QA pull named its handoffd peer with a placeholder. It now uses `DRIFT_SPARK`.
- `9fc878f` The stock mlx-vlm 0.7.1 native rephase test is back beside the oMLX check.

Tests: the main suite gave 1534 passed and 4 skipped. In `.venv-next` with transformers 5.17.0, mlx 0.32.2 and mlx-vlm 0.7.1, `tests/test_adapters_next.py`, `tests/test_adapters_mlx.py` and `tests/test_mcdma_positions.py` gave 81 passed, 1 skipped and 1 expected failure. The skip is the oMLX-only check, which passed on the Studio instead. The expected failure is the recorded sparse-indexer tie in `test_cross_runtime_stock_parity[16-qwen4_exp]`. The adapter gates that skip in `.venv` therefore pass in their qualified environment.

Ablation: results and report hashes are in `docs/evaluation/APPOINTMENT_SELECTION_ABLATION.md`. Under the rules fixed in advance the outcome is undecided, because the no-link arm scored 3 of 6.

Host changes, approved by the owner:

- On the head Spark, `/etc/systemd/system/glm53-vllm.service.d/drift.conf` makes the unit source `/root/drift-live/glm53-drift-env.sh` before `start.sh`. A plain unit start had previously dropped the connector. `systemctl restart glm53-vllm` ran from 20:08:42Z to 20:14:16Z and came up with `DriftGlm53Connector`, `fp8_ds_mla`, `--max-model-len 131072` and TP=2, with boot-shape warmup 24 of 24. The Drift env limits context to 131,072 tokens against the recipe's 400,000.
- Studio handoffd: `SHUTDOWN` returned `BYE`, and the degraded daemon, which reported the head Spark down, logged a clean exit. Both Spark daemons logged the control connection closed. A replacement started at 20:11:32Z inside a kept-alive SSH session and reported both peers up. A 64 MiB random file pulled from each Spark matched its SHA-256, at 18.7 Gbit/s.
- The export probe after the restart returned HTTP 200 with the export ready and rank 0 writing 67 tensors.

Costs: 11 min 48 s of GLM and Studio time for the 30 ablation arms, plus about a minute across two failed attempts. The GLM restart took 5 min 34 s, the export probe was one 4-token request and the native rephase check ran under 10 s.

Unresolved: Qwen3.8 and DeepSeek V4 on the Sparks have no Drift connector, so their units cannot start Drift-enabled yet. The generic `DriftConnector` is tested only against a mocked vLLM.

Next ticket: score the preregistered prefill-cost run started at 20:15Z, then the Qwen3.8 and DeepSeek V4 connectors with an identity gate on the Sparks.

## 2026-09-22: Drift connectors for Qwen3.8 and DeepSeek V4 on the Sparks

Stage: engineering, unit-tested, not yet run against a live server. Commits `32b2587` and `4bd81cd`.

The owner asked for every model the Sparks serve to start Drift-enabled. GLM already does, since the unit drop-in above. Qwen3.8 and DeepSeek V4 had no Drift connector, and the generic `DriftConnector` is tested only against a mocked vLLM.

Layouts were read from each image's own vLLM source, copied out of the images on the head Spark without running them:

- Qwen3.8 in `vllm/vllm-openai:qwen38-flash-next`: full-attention pages are `[blocks, kv_heads_per_rank, slots, K then V]` in bf16, and the QSA backend accepts only a bf16 main cache. The QSA indexer stores RoPE of the k_layernorm of the mean raw key over groups of four. This vLLM has `cache_salt` but no write-side no-store flag.
- DeepSeek V4 in `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`: layers with compress ratio 4 or 128 keep compressed MLA entries of 584 bytes each, block_size divided by ratio per block. The ratio-4 layers also keep a compressed indexer cache. The recipe runs `nvfp4_ds_mla` with a 256-token block, prefix caching and a 1024-token long-prefill threshold.

What was built:

- `DriftQwen38Connector` extends the owner's `Qwen38HandoffConnector` from the split-inference project. It writes a span's full-attention K and V for each rank's head, rotating canonical keys or taking pre-rotated ones, plus the compressed selector keys when supplied. Recurrent layers keep the state the placeholders gave them.
- `DriftRawRowConnector` taps and injects compressed MLA entries as raw bytes in the cache's own format, for DeepSeek V4. It never touches sliding-window or compressor-state caches.
- `drift_spans` holds the scheduler rules both share. Every refusal writes an error file per rank instead of dropping silently.

Tests: 1567 passed and 4 skipped in the main suite. They include a round trip from the Qwen page writer through the owner's export format and back through `qwen38_export`, and a byte-exact tap-to-inject round trip for the raw-row connector.

Deployment findings: the Qwen recipe's TP=2 launch ignores `EXTRA_DOCKER_ARGS`, so `start_drift.py` derives an untracked `start-drift.sh` that overlays the connector through the recipe's own `add_overlay`. The derivation adds 12 lines and removes none from the live recipe. Containers run with `--ipc host`, which shares the host's `/dev/shm`, as the running GLM container confirmed. The DeepSeek compose command has no extra-arguments hook, but the launcher copies `COMPOSE_FILE` to the worker, so a derived compose file can carry the connector.

Plan for the live Qwen gate, criteria fixed in `scripts/live/qwen38_drift_qualify.py` before any run:

1. Stop GLM through its unit.
2. Start Qwen from `start-drift.sh` with `qwen38-drift-env.sh`.
3. Run 12 validation passages, 24 questions, from `gen_val.jsonl`, not Test A's evaluation set.
4. WRITE passes when an export of the inject request equals the injected rows bit for bit. RECALL passes when inject exact match is at least 0.8 of native and no_memory is at most 0.2.
5. Install the unit drop-in only after the gate, then restore GLM.

## 2026-09-22: Why memory trails text, and the recurrent state fix

Stage: diagnosis recorded; engineering fix unit-tested, not yet deployed. Results from this session are in `docs/evaluation/`: prefill cost SUPPORTED at 4.59 times (`8a34c87`), reverse recall PARTLY at 16 of 24 (`e1eec4d`), joint code NOT SUPPORTED at 0 of 16 against text's 16 of 16 (`c3d1e58`), and the forward diagnostic (`8398535`). Query capture for attention distillation is committed (`9906e39`, `8f615ae`) but not deployed; its bundle was staged and passed `check-staged`, then superseded by this change before cutover.

Placeholder dilution, development only, 12 fresh balanced scenarios from seed 4242, linked arm, at `c3d1e58` and `8f615ae`:

| Run | Copies into reserve | GLM names the true street |
| --- | --- | --- |
| a3-1536 | 3 into 1,536 | 9 of 12 |
| b3-384 | 3 into 384 | 9 of 12 |
| c12-1344 | 12 into 1,344 | INVALID, Qwen's context overflowed the reserve at scenario 1 and nothing was published |

Filler rows do not explain the misses. The prompts were about 1,728 tokens, below the 2,048 the sparse selector keeps, so selection dropped nothing either. Report SHA-256, private local copies: a3 `6a9a0903e23a55ae946554ef1d2407632fc9e09afbe954dd56ad5d0a15c7b1bc`, b3 `2e27f85afd9a59f6858dcfe9c2e68dfe70e85e920422655ea3e7206ca9741a24`, c12 `6b36e9700838fee2fc9b3fbf3ad5455eeeedc0cf2b24d9084e0ecb34f5febfc8`.

What the diagnosis found, with the details in `docs/reference/glm/GLM_RECURRENT_STATE.md`:

- Memory only ever reached GLM's 11 MLA layers. The 34 KDA layers keep a recurrent state that saw placeholder tokens.
- The live server ran without the Drift scheduler, so GLM computed its whole prompt, question included, before memory landed. Only generated tokens read memory. The earlier scheduler could not have worked here: it deferred to vLLM's align-mode split, which stops only at 3,584-token blocks on this server.
- The owner's handoff export already carries each rank's KDA state pages. Only the write side was missing.

Changes: `glm_prefill_boundary.split` ends the first prefill chunk exactly at the boundary when it lies in the current block, `DriftScheduler` applies it, and `glm_state_inject` copies an export's prompt-end state pages into the live request's running state block at the reserve's end, before the first receipt. `scripts/live/glm_state_gate.py` is the own-cache gate. The bundle cap rose from 16 to 24 modules, since the state writer is the sixteenth.

Tests: 1609 passed and 4 skipped, the skips being the MLX and transformers gates, which remain BLOCKED.

Costs: 10 min 47 s of GLM and Studio time for the dilution runs, including the invalid one. One 130-token export probe.

Unresolved: the fix is not deployed. Qualification needs the re-pinned bundle on both ranks and a restart with `--scheduler-cls glm_prefill_scheduler.DriftScheduler`. Cross-model state translation, Qwen to GLM, does not exist yet. The same gap on Qwen's side, its Gated DeltaNet layers, is untouched.

Next ticket: deploy, restart with the scheduler, and run the own-cache gate. Own rows plus own state should answer as text does, and own rows alone should fall short.

## 2026-09-22: Recurrent state deployed; own-cache gate PASSED; causal loop mode

Stage: engineering gate PASSED. Commits `a959fd6` and `5a11cc1`; this entry's commit adds the causal loop mode.

Deployment on both Sparks: bundle `c8e2870c…` passed `check-staged`, and both ranks' 16 modules matched the manifest. Rollback copies are kept under the handoff cache's `rollback/` folder. The Drift env now adds `--scheduler-cls glm_prefill_scheduler.DriftScheduler`, with the previous env kept beside it. `systemctl restart glm53-vllm` at 22:36:36Z came up at 22:42:52Z with the custom scheduler logged and no errors.

The first gate attempt stopped GLM at 22:43:37Z. Rank 1's export had gone into the handoff daemon's arena, the writer only read files, and a failed live step stops the engine by design. `5a11cc1` reads arena exports under their lease, and the driver checks both ranks' exports before sending a state. After replacing that one module on both ranks and matching all 16 hashes, GLM restarted at 22:48:55Z and came up at 22:54:57Z.

Own-cache gate, 24 validation questions: text 24, none 0, own rows 23, own rows plus own state 24, with 13 outputs identical to text. Details in `docs/evaluation/GLM_OWN_STATE_GATE.md`, results SHA-256 `219e14600c5201ea5df222af41c9397addbca420358fa64e5623d15690eeb761`.

Loop finding: the MCDMA loop releases GLM and Qwen together, and GLM's session runs in tap mode without a boundary, so in every appointment run GLM computed its question before any memory landed. `--causal` now makes GLM's session wait for the first publication and stop its prefill at the reserve's end. The head sees a publication only after every rank staged it, so one local file is enough to wait on.

Tests: 1618 passed and 4 skipped, the skips being the BLOCKED MLX and transformers gates.

Costs: 4 min 45 s for the gate, two GLM restarts of 6 min 16 s and 6 min 2 s, and one failed attempt of under a minute.

Next ticket: the appointment dev comparison with `--causal`, same 12 scenarios as a3-1536, to measure the ordering fix across the two models. Then a translated recurrent state, Qwen to GLM.

## 2026-09-23: Causal confirmations SUPPORTED; Qwen gate; translator data captured

Stage: two preregistered confirmations SUPPORTED; state translators being fitted.

Results:

- Causal recall confirmation, `docs/evaluation/CAUSAL_RECALL_CONFIRMATION.md`: SUPPORTED. Linked named only the true street in 24 of 24, text 24 of 24, no-link 6 of 24, p = 4e-06. Report SHA-256 `5fa7096499646576f1102e96c8b3f1a9efeaac07f48cc8ff9e5395d9f2cfdfa6`.
- Joint code causal confirmation, `docs/evaluation/JOINT_CODE_CAUSAL_CONFIRMATION.md`: SUPPORTED. Linked passed 15 of 16, text 16 of 16, no-link 0 of 16, p = 3.1e-05. Report SHA-256 `2a59d80b27c147a97ec3b774f71711f0e9772fbf023266f15291ebbe027214d2`.
- Qwen own-cache gate, `docs/evaluation/QWEN_OWN_STATE_GATE.md`: PASSED, 21 of 24 outputs identical to text with its own state, 12 without.
- Causal development runs, `docs/evaluation/CAUSAL_LOOP_DEVELOPMENT.md`: GLM 12 of 12 against 9 of 12 on the same scenarios, and joint code 14 of 16 exploratory, report SHA-256 `37a5b28ea7baf2fa07c2fa8dfd8a9f345f52cca2efce01f1f377ac0e67e7fa87`.

Engineering since the last entry:

- A capture now spans engine steps (`9e27fd8`). This server's scheduler works in 64-token blocks, and with the drafter it runs a prompt's last 64 or more tokens as a separate step, the rule behind the old inject path's layout.
- GLM ran in an eager capture session: env `glm53-drift-env.capture.sh` with `--enforce-eager` and `drift_hidden_capture`, bundle `ff2b1d39…`, restarted 00:15:11Z to 00:20:51Z. It captured KDA inputs and MLA latents for 41 validation and 380 training passages. 25 passages with non-ASCII characters were skipped, because byte-level token pieces only map one to one onto ASCII. The normal env was restored and GLM restarted 00:30:57Z to 00:37:22Z.
- Qwen tapped its full-attention features and linear-attention inputs for the same passages on the Studio in under two minutes.
- The GLM captures reached the Studio through a stream relayed by the laptop, at 85 MiB/s, never written to its disk. The Studio and the Sparks have no SSH trust between them, and adding it is the owner's call.
- The state compute path ran live for the first time: GLM's state rebuilt from its own captured layer inputs gave the same answer as its own exported state. The first call took 41 s for kernel compilation, later calls 0.1 s.
- `f941a69` was committed with 8 failing tests, because a pipe hid pytest's exit status. `13fe322` fixed them. Commits now check pytest's own exit code.

Costs: 26 min 47 s and 14 min 11 s for the confirmations, about 16 minutes of GLM restarts, about 9 minutes of GLM capture, 2 minutes of Studio taps.

Next ticket: fit both state translators on the Studio, measure the translated state on the gate questions, then preregister a forward-direction confirmation. DeepSeek V4 follows.

## 2026-09-23: Translated state, forward framing, DeepSeek V4 live

Stage: exploratory results recorded; DeepSeek V4 engineering gate PASSED.

- Translated state, Qwen to GLM (`docs/evaluation/TRANSLATED_STATE_DEVELOPMENT.md`): 24 of 24 gate answers with a state built from Qwen's translated layer inputs, against 20 of 24 from translated rows alone and 23 of 24 from text. The translators were fitted on the Studio from 37,275 aligned token pairs. Ridge 0.1 gave the best validation fit, R-squared 0.37 Qwen to GLM and 0.40 GLM to Qwen.
- Forward direction: offline, translated rows plus translated state in a framed block named GLM's recommendation in 22 of 22 against text's 18. The loop's old layout capped text at 9 of 22. The live loop gained `--state-at`, `--followup-block` and a framing line with no content, and reached 5 of 12 from 2.
- DeepSeek V4 (`docs/evaluation/DSV4_DRIFT_GATE.md`): the first start crash-looped, because vLLM refuses a KV connector with expandable CUDA segments. The derived compose file now turns them off. The gate then passed WRITE and RECALL, inject 1.000 against native 0.958 and no_memory 0.042. The persistent drop-in `/etc/systemd/system/dspark-vllm.service.d/drift.conf` points `COMPOSE_FILE` at the derived file. DeepSeek's compressed rows were captured over 41 validation and 380 training passages on whole 256-token pages, since a page keeps its scales at its end. The captures are on the Studio.
- The Sparks swapped back to GLM at 02:15:56Z. The first attempt failed the memory preflight while DeepSeek was still stopping; GLM was healthy at 02:21:44Z.

Costs: DeepSeek start 15 min including the failed start, gate 7 min 14 s, captures 11 min; GLM restarts about 12 minutes; forward development loops 5 runs of about 5 minutes; offline Studio gates about 10 minutes.

Unresolved: the forward live gap; DeepSeek translators, which need an entry-level decoder for its cache pages; Qwen3.8's Sparks connector gate; the owner-run held-out set.

Next ticket: the forward live gap first. Refit the forward translators on GLM latents from its generation context rather than from reading passages, then preregister a forward confirmation with a framed text arm. In parallel, decode DeepSeek's pages and fit group-level translators.

## 2026-09-23: Forward gap traced to GLM's live latents; DeepSeek V4 decoder and first translators

Stage: exploratory development; no gate advanced.

- Forward direction (`docs/evaluation/TRANSLATED_STATE_DEVELOPMENT.md`). i3, a translator fitted on GLM writing each passage as its own reply, gave 3 of 12 live. Offline rebuilds of i3's 12 live messages showed the cause is context: the same reply tokens from GLM's live prompt translate far worse than read as a document (3 against 10 of 12), and the connector also sends about 144 rows of GLM's own prompt, which help (8 of 12). A state translator fitted on latents exported in the live prompt layout (R-squared 0.44) lifted the reply-only block from 3 to 9 of 12.
- j3 ran live with that translator and `--save-taps`: 5 of 12. Replaying the saved taps offline gave the same 5, with 11 of 12 answers identical to live, so the loop's mechanics are faithful. The same messages exported in the live layout gave 8 of 12. Live taps and exports agree closely (middle-layer cosine 0.89 to 0.94) and still lose 3 of 12: live GLM reads Qwen's translated rows in its reserve. `glm_export_live_taps.py` now collects training rows through the connector's own taps, with translated memory staged as the loop stages it.
- DeepSeek V4 (`docs/evaluation/DSV4_TRANSLATOR_DEVELOPMENT.md`). `drift/translate/dsv4_pages.py` decodes and encodes its cache pages; all 421 captures decode. Linear translators from its grouped entries are weak: DeepSeek to Qwen state R-squared 0.14, rows 0.33; per-position maps did not help. Qwen answered 2 of 20 gate questions from DeepSeek's translated rows and state, GLM 3 of 20.
- Tooling: `glm_export_live_block.py`, `glm_export_live_taps.py`, `replay_forward_items.py`, `dsv4_token_features.py`, `drift/translate/ridge_map.py` (translators with per-position phases), the loop's `--save-taps`, the Qwen gate's `t_state` and DeepSeek arms, and `export_passages.py`, which produced earlier documented results but lived only on the head Spark.

Tests: 1652 passed and 4 skipped, the skips being the BLOCKED MLX and transformers gates.

Costs: j3 about 25 minutes of both models; GLM exports about 40 minutes; Studio fits about 20 minutes; about a dozen offline Qwen gate runs of 2 to 4 minutes; the GLM gate with DeepSeek memory 7 min 43 s.

Unresolved: the forward live gap (5 of 12 live against text's 12); DeepSeek translators need a nonlinear reader; the push decision on the audited candidate.

Next ticket: fit the forward state translator on the connector's own taps, test it on the j3 replay, then run live.

## 2026-09-23: Answer-level training, and a fresh model dropped into a real project

Stage: exploratory development; no gate advanced. Owner goal for this stretch: Drift as a telepathic layer that beats text, tested on a real project, including whether a model dropped into ongoing work can skip its initial prompt processing by attaching to the shared cache over MCDMA.

- Forward direction: a 48-case set built by running the GLM half of the live loop offline (`glm_export_live_taps.py --generate`) put the translated channel at 28 to 31 of 48 against text's 47, whichever translator; the 12-case differences before were noise.
- Answer-level training (`studio_train_memory_answer.py`): teacher Qwen reading the text, student Qwen given the sender's translated cache, KL on the teacher's answer tokens with gradients through Qwen's own layers, low-rank corrections of rows and state on fixed translators. DeepSeek to Qwen went from 10 to 19 to 22 of 60 after the reduced components were standardised; the first run diverged without that. GLM task training on 300 appointment scenarios fell from 34 to 20 of 48 held-out cases and was stopped: it fitted the training names.
- Drop-in on this repository's own code (`docs/evaluation/DROPIN_REAL_PROJECT.md`): a joining Qwen answered 27 of 32 code questions from GLM's cache against text's 32, its first token 4.0 times sooner after 96 prefilled tokens instead of 1,413.
- Past Qwen's 2,048-token sparse-attention budget, appended memory has no usable selector keys: even Qwen's own cache answered 0 of 6 at 4,603 tokens. Raising the joining model's budget (`--budget`) restored its own cache to 4 of 4; GLM's translated cache then reached 2 of 4, so translation precision over thousands of rows is what limits long contexts.
- Stockledger, the owner-selected benchmark project, is used read-only: `stockledger_task.py` asks for its event reader from the specification alone, graded by 33 hidden checks derived from the specification in isolated processes.

Tests: 1666 passed and 4 skipped, the skips being the BLOCKED MLX and transformers gates.

Costs: about 2 hours of Studio GPU for training runs, about 1.5 hours for gates, about 1.5 hours of GLM exports and generation on the Sparks.

Unresolved: the drop-in over MCDMA, the Stockledger grades and code training are running; long-context translation precision; the push decision.

Later the same day: the MCDMA drop-in ran end to end (exports of 150 to 218 MB crossed at about 49 Gbit/s; first token 3.3 to 4.9 times sooner than text). With Qwen's own cache the drop-in matched or beat text on every set, including a Stockledger validator passing all 33 hidden checks; with GLM's translated cache it reached 76 to 84% of text. Refitting the per-token maps on code did not move that (26 against 27 of 32), so a contextual reader (`drift/translate/context_reader.py`) is being trained against Qwen's own rows on 921 windows of this repository's code, test modules held out.

Next ticket: train the contextual reader, gate it on the short, balanced long and def-line sets, then fine-tune it on answers and run the Stockledger task with it.


## 2026-09-23: A contextual reader, and the spread that squared error takes out

Stage: exploratory development; no gate advanced. Base commit `6fa24e9`.

- Contextual reader (`drift/translate/context_reader.py`, `studio_train_context_reader.py`): a four-layer bidirectional transformer over GLM's latents for the whole context, correcting the code-fitted rows map, trained on 921 windows of this repository's code (1,179,064 aligned rows, 3,000 steps, 209 s). Validation R-squared rose from 0.578 to 0.645, yet on the 32 short questions its rows answered 9, and 19 with the state, against the map's 27.
- Cause: squared error shrank the rows toward their mean, to 0.71 to 0.86 of Qwen's own spread, which flattens Qwen's attention over the memory. Ruled out first: unaligned positions (99.8% of GLM's tokens align), the difference between the gate's and the validation export of the same contexts (both translators fit either equally well), and lost token identity (each K row finds its own token as often as the map's does). The loop's per-token maps already carry a gain for this.
- Fix: `drift/translate/spread.py` fits a per-dimension gain on training windows (`studio_reader_gain.py`; the trainer now saves `gain.npz` itself). `ContextRows` applies it; the answer-level trainer keeps it fixed and exports it.
- With the spread restored (budget 16,384, code-fitted state): short contexts 29 of 32 (stack 27, text 32); balanced long set 24 of 25, equal to text (stack 19); def lines 18 callable and 18 exact of 25 (stack 16 and 13, text 25).
- Over MCDMA with the reader (`studio_dropin.py --context-reader`, fresh GLM exports pulled at 47 to 49 Gbit/s, translation 62 to 259 ms per context): long context 66 of 70 against text's 69 (the stack's run: 42), first token 0.97 s against 4.62 s; short contexts 28 of 32 against 32. All 19 `decode` questions answered, where the stack answered 1.
- Also: `--arms` on the Qwen gate; `drift/translate/qwen_rows.py` and `drift/translate/context_pairs.py`, shared by the gate, the drop-in and the trainers; `drift/eval/code_questions.py` now holds the question logic of `project_questions.py`, reproduces all 131 stored question sets exactly, and adds a recall kind for training; `code_triples.py` builds training triples from the stored contexts.
- Stopped: chain10's gates and fine-tuning, which built on the reader before its spread was restored. Its balanced long run was stopped part way and is not used.

Tests: 1678 passed and 4 skipped, the skips being the BLOCKED MLX and transformers gates.

Costs: about 1 hour of Studio GPU (reader training 4 minutes, gates about 30 minutes, the MCDMA drop-in 18 minutes, diagnostics); GLM exports on the Sparks for the drop-in's five contexts, and 1,000 standard-library windows for a second reader (about 35 minutes, running).

Unresolved: how much of the MCDMA gain is the rows, since that run also switched to the code-fitted state (a control gate is queued); the def-line gap to text, 18 against 25; the head Spark's root disk is 99% full (13 GB free), and its 13 GB `corpus_latents` duplicates the Studio's copy.

Next ticket: Stockledger with the reader (running), answer-level fine-tuning on QA and recall triples (queued), a second reader on this repository plus 1,000 standard-library windows (queued).

## 2026-09-23: More readers, reading in windows, and fine-tuning on exact copying

Stage: exploratory development; no gate advanced. Base commit `ee252f1`; nothing committed since, at the owner's instruction, until the short contexts, the def lines and Stockledger reach text.

- Second reader: 1,000 standard-library windows added (1,919 windows, 2,497,244 aligned rows, 9,000 steps); held-out R-squared 0.670 against the first reader's 0.645; short contexts 31 of 32 from rows alone.
- Third reader: GLM's full latents (5,632 per token) instead of the rows map's 2,048 principal directions, which keep only 74 to 85% of GLM's first three MLA layers; R-squared 0.674; best on the balanced long set and def lines.
- Reading in windows (`ContextRows(window=...)`, `--reader-window 1536`): the readers train on windows of about 1,500 tokens; reading the 4,603-token context in overlapping windows lifts the balanced long set to 25 of 25 for two readers, text 24.
- Stockledger diagnosis: every module written from any translated cache, stack included, required `quantity` and `timestamp` where the specification names `delta` and `occurred_at`, so it rejected every valid event. The def-line misses and neighbour confusions are the same failure: meaning carried, exact token lost.
- Fine-tuning on exact copying (`drift/eval/window_questions.py`, `code_triples.py --windows`, `studio_train_memory_answer.py` with a reader of full latents, a capped passage cache, the text arm read once, a shuffle seed): 7,716 questions from text GLM had read, none from the test modules, including 899 line quotations from public Markdown documentation. After 125 updates (102 minutes): def lines 25 of 25, callable and exact, as text; balanced long set 25 of 25, text 24; short contexts 31 of 32 from rows alone; Stockledger modules accept valid events and pass 28 to 30 of 33 checks, against text's 33, 23 and 33 on the same seeds.
- Also: `--init`, `--contrast` and an identity score in `studio_train_context_reader.py`; `--arms`-limited runs scored without a text arm (`score_state_gate.py`); MLX tests run on the Studio through a pytest stand-in kept outside the repository.

Tests: 1682 passed and 5 skipped here, the skips being the BLOCKED MLX and transformers gates; the MLX tests (`tests/test_context_rows.py`) pass on the Studio.

Costs: about 5 hours of Studio GPU (taps 33 minutes, three readers 40 minutes, gates about 1.5 hours, fine-tuning 102 minutes, Stockledger 25 minutes); GLM exports on the Sparks (1,000 standard-library and 300 Markdown windows, about 50 minutes). Two session restarts left the Studio idle for about 2.5 hours and cleared the scratch directory; working scripts and data copies now live under the ignored `local/`.

Unresolved: the short contexts' last miss (`final_frontier` against `processed_frontier`, messages that differ by one word); Stockledger's dense rule lists (invalid UTF-8, byte order mark, dots in SKUs, leap seconds); a second fine-tuning phase on 9,923 questions with eight-line quotations is running.

## 2026-09-24: Held-out project, shared space, DeepSeek as receiver, MCDMA session fix, long-context reads

Stage: exploratory development; no gate advanced. Base commit `ee252f1`; nothing committed since, at the owner's instruction, until the translator reaches text on the owner's sets.

- Held-out project (`docs/evaluation/HELD_OUT_PROJECT.md`): packaging, idna and filelock files, 40 code questions and 25 def lines over 3 seeds, with the averaged translator fixed beforehand. Code questions FAILED the bar of one question: rows and state 34, 35 and 33 against text's 37 on every seed. Def lines: rows alone 23, 24 and 24 against 24; rows and state 23, 22 and 23. The misses name a neighbouring function for errors raised deep in long methods. The coding task is INVALID: text's module hit the 3,500-token limit on every seed.
- Coding task harness: at 12,000 answer tokens the gate crashed with a Metal "failed to create shared event" error. Qwen's selector cache concatenates its raw index keys once per token, and under a dense budget nothing reads them, so the lazy graph grew until Metal ran out of events. `studio_qwen_state_gate.py` now evaluates every cache array each 256 generated tokens; values are unchanged.
- Fine-tuning, phases 2 and 3 and the average (`DROPIN_REAL_PROJECT.md`): the average of phases 1 and 2 stays the final translator (short contexts 32 of 32 both arms, Stockledger rows 29.0 against text's 30.5, no zero); phase 3, with 368 table questions, fell to 30 of 32 and wrote one module that passed nothing.
- Second dev project (`local/live/dev2`, not committed): 40 code questions and 25 def lines on fsspec, urllib3, yaml, networkx, jinja2 and pytest files, none in training or the held-out set, for choosing among translators on unseen code. The averaged translator matched text on the code questions (35 each) and trailed on def lines (20 against 22); phase 3 matched both (35 and 22); a 3,072-token reader window lifted rows alone from 33 to 35 on the code questions.
- Shared space (`drift/translate/hub.py`, `scripts/live/hub_add_model.py`, `docs/guides/SHARED_SPACE.md`, `SHARED_SPACE_MEMBER.md`): Qwen joined a GLM-anchored hub with one command (fit 208 s, gate 598 s); GLM to Qwen through it answered 29 of 32 against 26 for the direct linear translator.
- DeepSeek as receiver (`drift/translate/dsv4_member.py`, `scripts/live/hub_fit_dsv4.py`, `dsv4_drift_qualify.py --translated --align 256`, `encode_index_keys`, `SIX_DIRECTIONS.md`): one grouped decoder in the shared space serves GLM and Qwen; held-out R-squared 0.17 for ratio-4 content; DeepSeek's gate answered 20 of 20 from its own rows, 1 of 20 from GLM's cache (a guess) and 0 of 20 from Qwen's. The Sparks were switched to DeepSeek and back.
- MCDMA (`MCDMA_RECOVERY.md`): after ten idle hours every pull from the head Spark failed at once with `connect`. The Studio daemon's launching SSH session had ended, and macOS blocks a process's new LAN connections once its session is gone (Apple's `nc` still connected from such a process; Python got EHOSTUNREACH). Build 3 logs connect errors; the daemon was stopped with SHUTDOWN and restarted in a kept-alive session; the recovery test passed again on both Sparks.
- Long contexts: GLM's handoff read of 121,586 tokens never finished, twice, because the connector computes a full export in one engine step. `spark_dropin_export.py --split-at` reads the first 60,928 tokens with a full export and the rest from the prefix cache with a delta export (77.3 s in all); `drift/serving/glm53_delta.py` reads and stitches the two, since the deployed `glm53_handoff.py` is pinned by the deploy bundle.
- Payoff (`PAYOFF_LONG_CONTEXT.md`, `scripts/live/payoff_contexts.py`, `scripts/live/studio_payoff.py`): five joiners over 32,919, 64,426 and 131,290 Qwen tokens. A joiner attaching GLM's translated cache had its first token in 0.65, 1.08 and 2.90 s against text's 43.2, 114.0 and 380.8 s, prefilling 92 tokens; five of them were all answering in 31, 59 and 119 s counting GLM's read, the pull and the translation, against 217, 571 and 1,904 s for five text reads. Every way found its needles; with no memory Qwen named 1 to 3 of 5, the code being the standard library.
- Qwen to GLM (`SIX_DIRECTIONS.md`): reverse pairing (`reverse_windows`), `--direction qwen-to-glm` in the reader trainer, and `translate_passages.py --context-reader` or a flat shared-space pair. On the 32 short-context code questions, from rows alone: reverse reader 32, text 32; prose stack 24; shared-space pair 4. On the one held-out context that fits the receiver, 1 of 5 code questions (refusals) and 3 of 4 def lines.
- GLM's live receiver stopped its engine twice, each time on a limit it enforces inside the engine instead of refusing the request: a translated state of 367 MB against a 128 MiB publication bound (a poisoned session), and GLM's own rows for 4,641 tokens against a 4,096-row reserve limit. Both causes are explained; `glm_state_gate.py` now skips any arm that would break either limit and records why, the scorer leaves skipped arms out, and the server was restarted through its unit each time.
- Coding task: with the gate's cache flush, text ran to 12,000 tokens and wrote 49,167 characters of planning comments; told to write the code directly, 31,470 and 33,622 characters; 0 of 33 each time. A variant with thinking allowed (`iniparse_task.py --think`, grading only what follows the thinking) is queued on text alone.
- The average of the averaged translator and phase 3 matched or beat text with rows and state on every dev set (short contexts 32, def lines 25, balanced long 25, second dev project 35 and 22), but its rows-and-state Stockledger modules passed 23, 23, 0 and 29. Phase 4 trains from it on 4,143 questions, half from 600 windows of other installed packages exported through GLM (`third_windows.json`, not committed). The final translator is chosen by `choose_final.py` (local), the rule set before phase 4's results: most rows-and-state dev hits among candidates with no Stockledger zero, else most hits.

Tests: 1703 passed and 5 skipped here, the skips being the BLOCKED MLX and transformers gates.

Costs so far: about 4 hours of Studio GPU overnight (held-out 1.8 hours, the shared-space gate 10 minutes, phase 3 with its gates and Stockledger 2.3 hours) and about 2 hours this morning (dev project gates, the coding-task check); on the Sparks, GLM exports of the payoff contexts, the dev project and 600 new training windows (about 25 minutes), and the DeepSeek switch and gate (about 30 minutes).

Unresolved: the held-out code questions and Stockledger; a coding task text can finish; Qwen to GLM on unseen code and past the receiver's limits (4,096 rows, 128 MiB), where answer-level fine-tuning is blocked because GLM runs only inside vLLM; the GLM receiver stopping its engine on a refused request, which needs a change on the Sparks.

Next ticket: phase 4's dev sets and Stockledger, the final choice, the held-out run with it, and the coding task if text can finish it with thinking allowed.

