# Retained Qwen native tool continuation

Status: PASSED for the bounded two-turn native echo at source `e0e3844`.
The three earlier failed attempts below remain preserved.
The successful public-note cache capture is a separate, shorter prefill-only gate.
All attempts below use the existing frozen Studio checkpoint with no activation
link, no tools beyond one non-I/O echo, two turns, at most 64 output tokens per
turn and a 60-second supervisor envelope. Every attempt has fresh owner pins and
separate private evidence; earlier failures remain in the record.

| Private stage suffix | Native outcome | Wall time |
| --- | --- | --- |
| 65916956c53d | Context failed before a tool call | 7.388 s |
| b3cedbfd5fbf | Context accepted, native stream failed before a tool call | 10.149 s |
| 4032dcffd723 | Diagnostic reproduction: first decode step raised RuntimeError | 10.089 s |
| baa30f1a0c2d | Two assistant turns, one echo tool, retained native cache, clean exit | 15.307 s |

CPU-only checks using the real checkpoint tokenizer reproduced the first context
failure: initial system messages needed ordered coalescing, and the runtime's
tokenizer returned BatchEncoding where the worker expected a list. The regression
fix is committed at `e0e92e3`; it does not relax the exact cached-prefix check.
Another CPU test qualified explicitly lower-authority control serialization for
this checkpoint at `7b31f56`. Synthetic output through the real parser succeeds;
neither tokenizer nor parser checks establish native generation.

The diagnostic attempt pins runtime `36e03f26d9ac731a47293e33bb7c0dfcc6042bdc`
and owner configuration `739953d8ab52b96410df984ea4361ee2e9c8275c03ce56c1cb8fb6357e465e04`.
It records 2,674 own-prefill tokens, one generation attempt and zero successfully
returned generated IDs, plus a separately counted four-token startup prefill.
The failed generation step may still have performed GPU work. There is no terminal
usage result, so zero reported terminal tokens is not zero inference cost.

The owner-only diagnostic contains fixed phases, numeric counters and allowlisted
stack locations, without exception strings, model text, token IDs or arrays.
It identifies native generation at actual runtime language.py lines 3076, 2558
and 2905, reached from the worker's decode step. This identifies the failing path,
not by itself its root cause. A guarded, 128 MiB, random recurrent-component
fixture subsequently reproduced unfinished cache state crossing threads; settling
all cache arrays on their producing thread fixed it with maximum absolute output
difference 0.0 against the same-model main-thread reference.

The narrow fix at `67f3be9` materializes recurrent, convolution, PLE and selector
state under the existing native lock, without changing model math or threading.
Actual tokenizer/parser tests also exposed string-valued tool arguments and loss
of native tool-call whitespace during canonical rendering. Commit `e0e3844` keeps
parsed semantics separate from raw assistant markup and preserves the exact native
token-prefix guard. Internal whitespace and thinking-block variants pass; outer
leading/trailing whitespace stripped by this template remains unsupported and
fails closed. The confirmation therefore does not qualify every possible output.

Cleanup is independently recorded: input EOF, child exit 2, reaped process and
no surviving group, without TERM/KILL escalation, after 9.970 seconds in the failed
diagnostic attempt. Its guard lock was available and OS reclaimable memory was 247.076 GiB;
that figure does not prove complete allocator release. No serving configuration,
model weight, MCDMA daemon or existing oMLX process was changed. Money and energy
are unmeasured, and no heavy model work occurred on the MacBook.

Evidence root: private-audits/drift-qwen-echo-stage, outside agent task repositories.
Each stage retains its exact ready-command.sh, owner configuration and source pins.

| Failed diagnostic attempt artifact | SHA-256 |
| --- | --- |
| Native aggregate | `ea9b32c622cce89b9e0889ea8177e1f36b7b2ddbe325b6017ce727ddb506e33e` |
| Sanitized diagnostic | `86f3085e231279f050a4edd5db3a956c84f2a302e2c2300671dc87b9ffd264c2` |
| Termination receipt | `3740402b7d5bf7671bf665bc3b6bb21e57ed26741a4ef3b644feb925a1e77a9e` |
| Source package | `0b0045491a6511b885ceb0140738bd32151ca9389d4b7f285c1591431cca74cc` |

## Native confirmation

The single fresh confirmation at `e0e384478d3ea598b616c6f74deece9e26094c25` passed:
two assistant turns, one non-I/O echo and no other tools, 5,055 incremental input
tokens and 33 output tokens, 15.307 seconds, no diagnostic error. Four startup
prefill tokens are separately measured, giving 5,059 input tokens including startup.
Input usage is reset after each completed turn, so it does not recount the retained
cache; changed control snapshots can add new input, but no per-turn/control split
was recorded and no token-saving claim follows.

The independent supervisor receipt records input EOF, exit 0, child reaped,
process group gone and no TERM/KILL escalation, in 15.278 seconds. The diagnostic
file is empty, the guard is free, and OS reclaimable memory is 247.092 GiB.
Money, energy, full weight-shard hashes and complete allocator release remain
unmeasured. The profile and limits are unchanged; only the frozen code and fresh
owner/evidence identifiers differ from the preceding failed diagnostic attempt.

Exact command is preserved in private-audits/drift-qwen-echo-stage/
qwen-confirm-stage-baa30f1a0c2d/ready-command.sh and invokes the same native_probe.py
with owner SHA-256 `6b40f7226440ce80641c70e781a34d9643b48e6f6294d73f3d19ccd8ea70ea4b`.
Report hash: `6134a3b644ef2da1c21bd53076553222d240a97ef3735cff41bae870cd727147`;
termination hash: `7cb5d8f1d4b02718901581381a10aa53569d45d3683276cb0f1666f42bb35f48`;
package hash: `2ba001e2a0817339659e48973a64ad1c19e402c7ad77f5d9bb08677db3fc4dd8`.

Integrated local checks passed 622 reference tests with two skips, 155 next-runtime
tests with one existing expected failure, 59 plugin tests and TypeScript compilation.
Skipped real-adapter gates remain BLOCKED. Continuous bidirectional memory exchange,
simultaneous cleanup, source ownership and the requested Duo project comparison
are still unqualified; no synthetic result substitutes for those native gates.
