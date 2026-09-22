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
