# Selected public comparison project

Project choice is resolved: a fresh isolated **Stockledger** benchmark at
`/home/example/Documents/Projects/Personal/Drift-Benchmark`.
The directory did not exist when discovered; no parent-directory AGENTS.md was found, and no existing repository was overwritten.

## Frozen public starting state

- Initial commit: `9059e94ad1fc47c29d25f29f9e568e7d588dd07b`.
- Git tree: `f8e31939e427f2d01e9d4a63e2ef43f9f3c759ae`.
- Branch: `main`; exactly one local starter commit, using the already configured identity.
- No remote, publication or user configuration change; working tree clean after commit.
- Contents: 12 public files, 254 lines, standard-library Python only; no implementation, hidden tests or answer store.

The task is to implement a persistent warehouse-event CLI with strict validation, atomic/idempotent imports,
filtered listing, deterministic daily reports and concurrency-safe SQLite writes.
SPEC.md freezes objective acceptance categories: CLI/output, validation, persistence, query/report correctness,
concurrency/contended writes, and safe failures/repeatability; final independent scoring weights remain undecided.
AGENTS.md requires focused modules, single-line comments/docstrings, preserved public tests and honest reporting.
The CLI parser is supplied; validation, import, listing and reporting remain explicit NotImplementedError stubs.
The public examples are fictional and visible to both arms.

## Local starter verification

`python3 --version` reported Python 3.11.1; the public task permits Python 3.11 or newer.
`python3 -m stockledger --help` succeeded without creating a database.
`python3 -m unittest discover -s tests -v` ran four public smoke tests in 0.081 seconds:
one help test passed, and three implementation checks failed because commands deliberately returned the unimplemented code 4.
These are expected starting deficiencies, not waived tests or a passing application gate.
AST parsing, the single-line docstring check, the under-100-lines-per-Python-file check and `git diff --cached --check` passed.
No GPU, model, network inference, package installation or hidden evaluator was used; energy and compute cost were not measured.

## Next boundary

Use clean run clones at the exact starter commit for both normal Duo and Duo-drift, with a shared repository inside each arm
and no access across arms or to runtime-private/evaluator state.
Project selection is no longer a blocker; budgets, host windows, the real OMP bridge, cancellation/causality/provenance gates,
and independent hidden evaluation remain pending.
This public task is already visible to the engineering agents, so a result is project-specific development performance,
not evidence on an unseen task family; later fresh task variants require a separate freeze and independent evaluation process.
Nothing here advances D2/D6 or authorizes inference.

## Public file hashes

- `SPEC.md`: `653acdea9367105d020b6590340a1f0f1ecdc64ad5f86fbdd8b310aa04926e74`.
- `AGENTS.md`: `5a5d57352ae1509b21797bea970efc5532d4757e77b4a5a4b442a68b6cfcaa07`.
- `tests/test_public_smoke.py`: `e63ed6010921f598460f60460567762f893ae948e23d118bae14f705da3acc98`.
