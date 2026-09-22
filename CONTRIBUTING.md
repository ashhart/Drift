# Contributing to Drift

Thanks for looking. Drift is experimental research code, and its rules exist because
the claims it makes are easy to break by accident.

## Before you start

Read [AGENTS.md](AGENTS.md). It is the execution contract for this repository and it
applies to people as well as agents. The short version:

- Keep backbone weights frozen. Never change model weights to make a test pass.
- Keep task text and token IDs off the designated live activation channel.
- A skipped environment-gated test is a blocked gate, not a pass.
- Record engineering verdicts as `PASSED`, `FAILED`, `BLOCKED` or `INVALID`.
- Keep modules focused on one responsibility, and keep code comments to one line.

## Running the checks

The reference suite needs no model downloads and no special hardware:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

The OMP plugin is TypeScript under Bun:

```bash
cd plugin/omp-drift
bun test
bunx tsc -p tsconfig.json
```

Both run in CI on every pull request. Some suites skip when their runtime is absent
(MLX, the transformers 5.17 environment). Those skips are blocked gates; please do not
report them as passes or delete them to get a clean run.

## What makes a good change

- A regression that fails before your fix and passes after it. If you are fixing a bug,
  show the red first.
- A concrete failure scenario in the pull request description: inputs, state, and the
  wrong behaviour that results.
- No new god files. If a module is growing a second responsibility, split it.

## What needs discussion first

Open an issue before:

- Adding a model family adapter. Start from `drift adapter scaffold` and the
  [adapter contract](docs/guides/ADAPTERS.md); a matching tensor shape is not compatibility.
- Changing anything in `drift/exchange/` contracts, the worker protocol, or the
  provider boundary. These carry qualification evidence and their guards are load-bearing.
- Relaxing a guard, cap, or fail-closed path to make something run.

## Evidence and claims

Do not add a performance, recall or capability claim to documentation without the run
that supports it, including what was fixed before the run and what the result does not
establish. A green local suite does not establish native recall, source ownership or
successful agent collaboration.

## Security and private data

Never commit private activations, credentials, scorer output, answer keys or host
configuration. Public examples use `.invalid` hostnames and example paths; keep it
that way.
