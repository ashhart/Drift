# Explicit forward recipes in the exploratory live loop

The existing default remains `--forward-recipe legacy`: the one-to-one reader from
`local/live/stacked2.npz` (or `--forward`) with gain power 1.5 unless overridden.
No default reader, reverse recipe, trained weights or frozen evaluation files were changed.

The live coordinator now accepts `--forward-recipe base`, `fanout` or `v4` together
with `--forward-manifest PATH`; those explicit modes reject `--forward` to avoid
ambiguous artifact selection. The manifest supplies `artifacts`, `sha256` and
`forward_gain_power` using the existing frozen-manifest keys:

| Recipe | Required artifact keys |
| --- | --- |
| base | forward_base |
| fanout | forward_base, forward_fanout |
| v4 | forward_base, forward_fanout, correction |

Artifact paths resolve from the process working directory, matching the existing
frozen evaluation manifests. Each selected file is checked against its SHA-256 pin
before loading and again after loading; the fan-out artifact must also name the
matching base hash. The run records the selected recipe, artifact pins, manifest
hash and effective gain. An explicit `--gain-forward` must match the pinned value.
Legacy runs record their observed base hash but do not become a pinned experiment.
No-link runs load no translator artifacts, including when explicit recipe flags exist.

Selecting the previously frozen correction requires both flags, for example
`--forward-recipe v4 --forward-manifest configs/frozen.v4-answer-level.json`
in addition to the normal scenario/output arguments, from a checkout containing
the pinned local artifacts. This is an invocation reference, not authorization to
launch a worker or a claim that these artifacts have been deployed.

## Source and emitted rows

The GLM publication cursor continues to use GLM source positions: a source block
of one row advances the cursor by one even when fan-out emits two Qwen rows.
Before constructing translated output, the coordinator predicts the emitted count
and checks the publication row cap, remaining configured reserve and wire-byte
bound. Every K/V layer must then have exactly that count and the declared layout,
with finite values representable on the float16 wire; the complete payload is
validated before saving or sending any layer. Queued deliveries and forward
reserve consumption count emitted rows, and timeline entries record both counts.

The configured forward reserve is a conservative coordinator cap. This ticket
preserves the legacy unframed Studio worker's existing position allocation rather
than silently changing its default reserve; the framed path already uses the CLI
reserve. Native cache offset, source cursor and emitted foreign row count remain
distinct, and exhausted capacity invalidates a session rather than dropping data.

## Qualification boundary

Synthetic regressions exercise the actual tap function with expanding translations,
source-cursor continuity, cumulative reserve accounting and malformed later outputs.
Tiny generated NPZ/safetensors fixtures exercise all explicit recipe variants,
artifact tampering and malformed fan-out parameters without using experiment data.
These mechanical checks do not transfer any lookup evaluation score to continuous
drift, establish source ownership, qualify remote cancellation or prove prior-epoch
causality; run metadata explicitly marks live qualification as BLOCKED.
No real activation, held-out item, model load, remote deployment or inference run
was used to implement this ticket.

## Local verification for this change

The isolated worktree started at `c74ae70a03762d24041190e9158dac3090fff701`.
Before edits the reference environment passed 251 tests with three existing skips.
The expanding-publication regression failed on the original row-shape check, and
the explicit-v4 loading regression failed on the original unsupported keyword.
Final focused checks passed 65 tests (0.60 seconds), and the full reference suite
passed 279 with three existing skips (15.15 seconds); the next-runtime suite passed
155 with the existing sparse-indexer-tie xfail (7.98 seconds). Plugin checks passed
29 tests and TypeScript compilation; missing worktree-local Node declarations were
resolved by pointing the already installed compiler at its existing type roots.

All Python commands ran with this worktree as cwd and `PYTHONPATH=.` while using
`/opt/drift/.venv/bin/python` or the parallel
`.venv-next/bin/python`; `drift.__file__` was verified inside this worktree. The
reference command was `python -m pytest -q`, and the focused selection was
`tests/test_live_forward.py tests/test_live_recipe.py tests/test_live_recipe_options.py tests/test_live_no_link.py`.
The next-runtime selection matched `scripts/check_all.sh`; plugin tests ran with
`bun test`, followed by the installed `tsc -p tsconfig.json --typeRoots` pointing at
`/opt/drift/plugin/omp-drift/node_modules/@types`.
Syntax/single-line-docstring checks and `git diff --check` passed. No environment
installation, primary-checkout write or model workload occurred; energy was not measured.
