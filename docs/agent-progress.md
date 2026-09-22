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

## 2026-09-22: transcript-backed memory

Stage: M-1 integration mechanics, with no scientific stage advancement.
Source implementation: `83f89d1552a5164bedaacdf43284c37957c64230`;
native permission/error repairs: `e064da770db22592baa54bb860da7aca915c83cd`;
native evidence and launch guidance: `83dc35414766267fa18bc83a167c4c0bbbdf3399`.
Development history and private runtime artifacts are excluded from publication.

Added transcript import, validate, capture, receive and answer commands plus
focused modules and regressions; README includes CLI and Python entry points.
Owner-only request/output directories remain required under a non-writable
shared export parent. Unavailable MCDMA control connections fail explicitly,
with no text fallback or server restart.

Native smoke PASSED: two GLM snapshots crossed MCDMA with matching digests;
Qwen recalled 3/3 target facts with target memory and full text, 0/3 without
memory, and the other snapshot's 3/3 alternative facts with different memory.
One question was already used in the pilot, so this is not held-out evaluation.
Capture times were 0.746/0.718 seconds, verified receive 6.33/5.52 milliseconds,
CLI loading-plus-answer 9.42 seconds and the twelve-answer process 14.59 seconds.
Setup and failed attempts are excluded from those component timings; native
model work stayed below the ten-minute allowance. Money/energy unmeasured.
The guide records the verified session-bound launch correction and the separate
unattended-service limitation; no shared model/daemon restart or driver change.

Publication-tree QA: `python -m pytest -q`, 1489 passed in 70.56 seconds,
three required environment gates BLOCKED and one existing warning;
`bun test`, 183 passed / 776 assertions in 2.66 seconds; TypeScript passed.
Python-log SHA-256:
`68110f3cce5405ea23f2a94f89753e2515d2de232215d851de818c1a1add0e05`.
Private native-control result SHA-256:
`d3fc853aefdc81b133f9343460ace1788bb7b0fb41330fe5f4598a54bc18e9d1`.
General recall, source attribution and unattended lifecycle remain unqualified;
per-invocation execution receipts cannot assert a scored recall result.

## 2026-09-22: Retire the standalone engineering bundle

Stage M-1 contract maintenance, PASSED locally, based on development commit
`37cea7c1972cf28849ca2843e4c1feaa35d4c197`.
Removed the obsolete standalone handoff and source-extraction script after
privately preserving exact copies; published Git history remains recoverable.
Current requirements now live in `docs/reference/CONTRACTS.md` and the roadmap,
with agent instructions and documentation links updated.
The corpus builder no longer requires the retired file; existing corpus
artifacts, model weights, translators and runtime behavior remain unchanged.
Reproducing an older corpus requires its original source revision and hashes.

Two new documentation/corpus regressions failed before repair; focused checks
then passed 5/5 in 0.07 seconds. Development baseline: 1493 passed, three
environment gates BLOCKED in 72.84 seconds; after changes: 1495 passed, three
BLOCKED in 70.39 seconds. Publication candidate `python -m pytest -q`: 1491
passed, three environment gates BLOCKED, one existing warning in 68.08 seconds;
`git diff --check` passed. Four development-only tests remain excluded.
Candidate full-suite log SHA-256:
`5625ed0a6c47904ab6c5ac44eb907428f392e713dea93d157d57994a6a930087`.
Archived handoff SHA-256:
`8b628c22da21392ae66ecf8b617133e113e70c7fd27a875ae057d3fbf93e7caa`.
Current contract SHA-256:
`955b9db795528665604320a19f0c7cadba685a03d5d1db83ae5e41b92127bddb`.
No native host actions, model inference or training; money and energy unmeasured.
Next gate is GitHub publication inspection of this cleanup; general recall,
source attribution and unattended service qualification remain unchanged.
