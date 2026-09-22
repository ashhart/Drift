# Agent execution contract

Keep modules focused on one responsibility; do not create or retain god files.
Use single-line code comments; put longer explanations in documentation.

Read docs/reference/CONTRACTS.md and docs/STATUS.md before modifying the project.
Treat reference code as a correctness aid, not evidence of qualified model recall.

Start at M-1. Preserve the user's component names and M0-M6 research sequence.
Read existing MCDMA/OMP source and SERIES.md before writing integrations. Locate
files with read-only commands; ask for paths only after discovery cannot resolve them.
Do not invent provider APIs, GPU memory guarantees, model revisions, benchmark
scores, energy values, paper novelty, or driver compatibility.

Run the included tests before changing code. A skipped HF test is a BLOCKED real
adapter gate, not a pass. Keep all backbone parameters frozen. Never change model
weights to get identity, transfer, or mailbox tests to pass. Keep task text/token
IDs off the designated live activation transport. Local token generation and
private token-conditioned mail are allowed and must be counted.

Implement the first failing/unimplemented gate only. After each meaningful change,
run focused tests and then the full suite. Do not advance a scientific stage without
its evidence. Use PASSED, FAILED, BLOCKED, INVALID for engineering records; reconcile
these with the actual SERIES.md rules before formal trials.

Put model-inaccessible evidence outside agent-readable repos/worktrees. Do not expose
answer keys, probe output, peer private files, hidden tests or scorer stdout to models.
Use real process/filesystem isolation, not prompt instructions as an access boundary.
Do not auto-install kernel drivers, disable security controls, run privileged changes,
publish private activations, or spend unapproved compute budgets.

Maintain docs/agent-progress.md with stage, exact commit, changes, commands, test
results, evidence hashes, measured costs, unresolved blockers and the next ticket.
Distinguish a measured failure from invalid evidence. Stop on unexplained native
parity errors, forbidden channel access, corrupted activations or poisoned sessions.
